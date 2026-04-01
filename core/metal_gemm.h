#pragma once

#include <cstdint>
#include <cstddef>

namespace ane_lm {

// Initialize Metal device, command queue. Call once at startup.
// Returns true if Metal + MPS is available.
bool metal_init();

// Shut down Metal resources.
void metal_shutdown();

// Is Metal initialized and available?
bool metal_available();

// Allocate a Metal shared buffer and return a CPU-accessible pointer.
// The buffer is retained internally and can be looked up by pointer.
// fp16_data: if non-null, copy fp16 data into the buffer.
// Returns CPU pointer (unified memory), or nullptr on failure.
void* metal_alloc(size_t bytes, const void* init_data = nullptr);

// Wrap an existing mmap'd (or malloc'd) region as a Metal shared buffer.
// The region must be page-aligned and the caller must keep it alive.
// The Metal buffer does NOT own the memory — metal_free will unregister but not deallocate.
// Returns the same pointer on success, nullptr on failure.
void* metal_alloc_nocopy(void* ptr, size_t bytes);

// Register an interior pointer as belonging to an existing Metal buffer.
// The parent buffer must already be registered via metal_alloc or metal_alloc_nocopy.
// This allows find_buffer() to locate the MTLBuffer for sub-regions.
void metal_register_subptr(void* subptr, void* parent_ptr);

// Free a Metal buffer by its CPU pointer.
void metal_free(void* ptr);

// Perform C = alpha * A @ B^T + beta * C using MPSMatrixMultiplication.
// All pointers must come from metal_alloc().
// A: [M x K] row-major fp16
// B: [N x K] row-major fp16 (transposed: computes A @ B^T)
// C: [M x N] row-major fp32
// This is the main GEMM used for prefill projections.
bool metal_sgemm_f16(
    const void* A,    // [M, K] fp16
    const void* B,    // [N, K] fp16 (weight matrix, row-major)
    void* C,          // [M, N] fp32 output
    int M, int N, int K,
    float alpha = 1.0f, float beta = 0.0f);

// Synchronous flush: commit current command buffer and wait for GPU completion.
// Call after a batch of metal_sgemm_f16 calls to ensure results are in C buffers.
void metal_sync();

// Initialize compute kernels from kernels.metal shader file.
// Call after metal_init(). Returns true if compilation succeeded.
bool metal_init_compute(const char* metallib_path);

// Fused SwiGLU on GPU: out_f16[i] = silu(gate_f32[i]) * up_f32[i]
// gate, up: [count] fp32 Metal buffers  |  out: [count] fp16 Metal buffer
// Dispatches on current command buffer (call metal_sync() when ready).
bool metal_swiglu_fused(void* gate_f32, void* up_f32, void* out_f16, int count);

// Residual add in-place on GPU: x[i] += y[i]
// Both x, y: [count] fp32 Metal buffers
bool metal_residual_add(void* x_f32, const void* y_f32, int count);

// RMSNorm with fp16 output on GPU:
// x: [N, dim] fp32, weight: [dim] fp32, out: [N, dim] fp16
bool metal_rmsnorm_f16(const void* x_f32, const void* weight_f32, void* out_f16,
                       int N, int dim, float eps);

// f32 → f16 conversion on GPU
bool metal_f32_to_f16(const void* in_f32, void* out_f16, int count);

// Stats
uint64_t metal_total_alloc_bytes();

} // namespace ane_lm
