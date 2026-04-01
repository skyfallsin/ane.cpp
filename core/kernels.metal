#include <metal_stdlib>
using namespace metal;

// Fused SwiGLU: out_f16[i] = (half)(silu(gate_f32[i]) * up_f32[i])
// where silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
kernel void swiglu_fused(
    device const float* gate [[buffer(0)]],
    device const float* up   [[buffer(1)]],
    device half*        out  [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    float g = gate[id];
    float s = 1.0f / (1.0f + exp(-g));
    out[id] = half(g * s * up[id]);
}

// Residual add in-place: x[i] += y[i]  (both f32)
kernel void residual_add_f32(
    device float*       x [[buffer(0)]],
    device const float* y [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    x[id] += y[id];
}

// RMSNorm with f16 output:
//   For each row r of N rows, each of width `dim`:
//     rms = sqrt(mean(x[r]^2) + eps)
//     out_f16[r][j] = (half)(x[r][j] / rms * weight[j])
//
// Params buffer layout: { uint dim, float eps }
// Uses threadgroup reduction for the row-wise sum-of-squares.
// One threadgroup per row, threads cooperate on the reduction.
kernel void rmsnorm_f16out(
    device const float*  x      [[buffer(0)]],   // [N, dim] input
    device const float*  weight [[buffer(1)]],   // [dim] norm weight
    device half*         out    [[buffer(2)]],   // [N, dim] output fp16
    device const uint*   params [[buffer(3)]],   // [dim_uint, eps_as_uint]
    uint2 gid   [[thread_position_in_grid]],     // (j, row)
    uint2 gsz   [[threads_per_grid]])            // (threads_per_row, N)
{
    uint dim = params[0];
    float eps = as_type<float>(params[1]);

    uint row = gid.y;
    uint j = gid.x;
    uint threads_per_row = gsz.x;

    device const float* row_ptr = x + (uint64_t)row * dim;

    // Each thread accumulates partial sum-of-squares
    float partial_ss = 0.0f;
    for (uint i = j; i < dim; i += threads_per_row) {
        float v = row_ptr[i];
        partial_ss += v * v;
    }

    // Threadgroup reduction
    threadgroup float shared_ss[1024];
    uint lid = j;  // local thread index within the row's threadgroup
    shared_ss[lid] = partial_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction
    for (uint stride = threads_per_row / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared_ss[lid] += shared_ss[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rms_inv = rsqrt(shared_ss[0] / float(dim) + eps);

    // Write normalized output
    device half* out_row = out + (uint64_t)row * dim;
    for (uint i = j; i < dim; i += threads_per_row) {
        out_row[i] = half(row_ptr[i] * rms_inv * weight[i]);
    }
}

// f32 → f16 conversion kernel
kernel void f32_to_f16_kernel(
    device const float* input  [[buffer(0)]],
    device half*        output [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = half(input[id]);
}

// ============ INT8 × FP16 GEMM (simdgroup matrix multiply) ============
//
// C[M,N] = A[M,K]_fp16 @ B[N,K]_int8^T * scales[N]_fp16
//
// A:      [M, K] row-major fp16 (activations)
// B:      [N, K] row-major int8 (weights, per-row symmetric quantized)
// scales: [N]    fp16 (one scale per output row of B)
// C:      [M, N] row-major fp32 (output)
//
// Uses simdgroup_multiply_accumulate to tap into Apple Silicon's hardware
// matrix multiply units. Each 32-thread simdgroup computes 8×8 output tiles.
//
// 4 simdgroups (128 threads) in a 2×2 layout, each handling 32×32 output.
// B's per-row scale is pre-multiplied during int8→half conversion in TG memory,
// so the simdgroup matmul output is the final result (no post-scale needed).
// Transposed simdgroup_load on B handles the B^T operation.
//
// Tile: BM=64, BN=64, BK=32. TG memory: 4KB A + 4KB B = 8KB.

kernel void gemm_int8(
    device const half*   A       [[buffer(0)]],   // [M, K] fp16
    device const char*   B       [[buffer(1)]],   // [N, K] int8
    device const half*   scales  [[buffer(2)]],   // [N] fp16
    device float*        C       [[buffer(3)]],   // [M, N] fp32
    device const uint*   params  [[buffer(4)]],   // { M, N, K }
    uint2 tg_pos  [[threadgroup_position_in_grid]],
    uint  tid     [[thread_index_in_threadgroup]],
    uint  simd_id [[simdgroup_index_in_threadgroup]],
    uint  lane_id [[thread_index_in_simdgroup]])
{
    constexpr uint BM = 64, BN = 64, BK = 32;
    constexpr uint SG_M = 32, SG_N = 32;  // per-simdgroup output tile
    constexpr uint NTH = 128;              // 4 simdgroups × 32 threads

    const uint M = params[0];
    const uint N = params[1];
    const uint K = params[2];

    const uint row0 = tg_pos.y * BM;
    const uint col0 = tg_pos.x * BN;

    // 2×2 simdgroup layout within the threadgroup
    const uint sg_row = simd_id / 2;  // 0 or 1
    const uint sg_col = simd_id % 2;  // 0 or 1

    // 4×4 grid of 8×8 accumulators per simdgroup = 32×32 output tile
    simdgroup_float8x8 acc[4][4];
    for (uint i = 0; i < 4; i++)
        for (uint j = 0; j < 4; j++)
            acc[i][j] = simdgroup_float8x8(0);

    // TG memory for A and B tiles
    threadgroup half tg_A[BM * BK];   // 64×32 = 4 KB
    threadgroup half tg_B[BN * BK];   // 64×32 = 4 KB (int8 → half*scale)

    for (uint k0 = 0; k0 < K; k0 += BK) {
        // --- cooperative load A tile (fp16 direct copy) ---
        for (uint idx = tid; idx < BM * BK; idx += NTH) {
            uint r = idx / BK, c = idx % BK;
            uint gr = row0 + r, gc = k0 + c;
            tg_A[idx] = (gr < M && gc < K) ? A[gr * K + gc] : half(0);
        }

        // --- cooperative load B tile (int8 → half with pre-scaling) ---
        // Pre-multiply by per-row scale so matmul output is final
        for (uint idx = tid; idx < BN * BK; idx += NTH) {
            uint r = idx / BK, c = idx % BK;
            uint gn = col0 + r, gc = k0 + c;
            if (gn < N && gc < K) {
                tg_B[idx] = half(B[gn * K + gc]) * scales[gn];
            } else {
                tg_B[idx] = half(0);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- simdgroup matrix multiply with transposed B ---
        // C[M,N] = A[M,K] @ B[N,K]^T
        // tg_A stored [BM, BK] row-major, stride = BK
        // tg_B stored [BN, BK] row-major, stride = BK
        // Load A normally: a_tile = tg_A[8×8] block
        // Load B transposed: b_tile = tg_B[8×8]^T (the transpose flag does B^T)
        // Then: acc += a_tile @ b_tile = A_sub @ B_sub^T  ✓
        for (uint kk = 0; kk < BK; kk += 8) {
            // Load 4 A sub-tiles along M for this simdgroup
            simdgroup_half8x8 a_tile[4];
            for (uint i = 0; i < 4; i++) {
                simdgroup_load(a_tile[i],
                    &tg_A[(sg_row * SG_M + i * 8) * BK + kk], BK);
            }

            // Load 4 B sub-tiles along N with transpose
            simdgroup_half8x8 b_tile[4];
            for (uint j = 0; j < 4; j++) {
                simdgroup_load(b_tile[j],
                    &tg_B[(sg_col * SG_N + j * 8) * BK + kk], BK,
                    ulong2(0, 0), true);
            }

            // 4×4 = 16 multiply-accumulate operations
            for (uint i = 0; i < 4; i++)
                for (uint j = 0; j < 4; j++)
                    simdgroup_multiply_accumulate(acc[i][j], a_tile[i], b_tile[j], acc[i][j]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // --- store results to device memory ---
    // Use TG memory as staging for bounds-checked writes.
    // Reuse tg_A (no longer needed) as float staging: 4KB = 1024 floats.
    // Each simdgroup has a 256-float (8×32) staging region.
    threadgroup float* tg_staging = reinterpret_cast<threadgroup float*>(tg_A);
    threadgroup float* my_stage = &tg_staging[simd_id * 64];  // 64 floats per simdgroup

    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
            uint base_row = row0 + sg_row * SG_M + i * 8;
            uint base_col = col0 + sg_col * SG_N + j * 8;

            // Check if this entire 8×8 tile is in-bounds
            if (base_row + 8 <= M && base_col + 8 <= N) {
                // Full tile: store directly to device memory
                simdgroup_store(acc[i][j], C + base_row * N + base_col, N);
            } else {
                // Edge tile: store via TG staging with bounds checking
                simdgroup_store(acc[i][j], my_stage, 8);
                // 64 elements / 32 lanes = 2 elements per thread
                for (uint e = lane_id; e < 64; e += 32) {
                    uint lr = e / 8;
                    uint lc = e % 8;
                    uint gr = base_row + lr;
                    uint gc = base_col + lc;
                    if (gr < M && gc < N) {
                        C[gr * N + gc] = my_stage[e];
                    }
                }
            }
        }
    }
}
