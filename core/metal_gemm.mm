#include "metal_gemm.h"
#include <ane_lm/common.h>

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <unordered_map>
#include <mutex>

namespace ane_lm {

// --- Globals ---

static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_queue = nil;
static id<MTLCommandBuffer> g_cmdbuf = nil;
static bool g_initialized = false;
static uint64_t g_total_alloc = 0;

// Map CPU pointer → MTLBuffer for shared-memory buffers
static std::unordered_map<void*, id<MTLBuffer>> g_buffer_map;
static std::mutex g_mutex;

// Compute pipeline states for custom kernels
static id<MTLComputePipelineState> g_swiglu_pso = nil;
static id<MTLComputePipelineState> g_residual_add_pso = nil;
static id<MTLComputePipelineState> g_rmsnorm_f16_pso = nil;
static id<MTLComputePipelineState> g_f32_to_f16_pso = nil;
static bool g_compute_initialized = false;

// --- Init / Shutdown ---

bool metal_init() {
    if (g_initialized) return true;

    g_device = MTLCreateSystemDefaultDevice();
    if (!g_device) {
        fprintf(stderr, "[metal] No Metal device available\n");
        return false;
    }

    // Check MPS support
    if (![g_device supportsFamily:MTLGPUFamilyApple7]) {
        fprintf(stderr, "[metal] Device does not support Apple7 GPU family (required for MPS fp16)\n");
        return false;
    }

    g_queue = [g_device newCommandQueue];
    if (!g_queue) {
        fprintf(stderr, "[metal] Failed to create command queue\n");
        return false;
    }

    g_initialized = true;
    LOG("[metal] Initialized: %s (%.0f MB shared)\n",
        [[g_device name] UTF8String],
        [g_device recommendedMaxWorkingSetSize] / 1048576.0);
    return true;
}

void metal_shutdown() {
    if (!g_initialized) return;

    // Wait for any in-flight work
    if (g_cmdbuf) {
        [g_cmdbuf commit];
        [g_cmdbuf waitUntilCompleted];
        g_cmdbuf = nil;
    }

    std::lock_guard<std::mutex> lock(g_mutex);
    g_buffer_map.clear();
    g_queue = nil;
    g_device = nil;
    g_initialized = false;
    g_total_alloc = 0;
}

bool metal_available() {
    return g_initialized;
}

// --- Buffer Management ---

void* metal_alloc(size_t bytes, const void* init_data) {
    if (!g_initialized || bytes == 0) return nullptr;

    id<MTLBuffer> buf = [g_device newBufferWithLength:bytes
                                              options:MTLResourceStorageModeShared];
    if (!buf) {
        fprintf(stderr, "[metal] Failed to allocate %zu bytes\n", bytes);
        return nullptr;
    }

    void* ptr = [buf contents];

    if (init_data) {
        memcpy(ptr, init_data, bytes);
    } else {
        memset(ptr, 0, bytes);
    }

    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_buffer_map[ptr] = buf;
        g_total_alloc += bytes;
    }

    return ptr;
}

void* metal_alloc_nocopy(void* ptr, size_t bytes) {
    if (!g_initialized || !ptr || bytes == 0) return nullptr;

    // newBufferWithBytesNoCopy requires page-aligned pointer and length
    size_t page_size = getpagesize();
    uintptr_t addr = (uintptr_t)ptr;
    if (addr % page_size != 0) {
        fprintf(stderr, "[metal] metal_alloc_nocopy: pointer %p not page-aligned (page=%zu)\n", ptr, page_size);
        return nullptr;
    }
    // Round up length to page boundary
    size_t aligned_bytes = (bytes + page_size - 1) & ~(page_size - 1);

    id<MTLBuffer> buf = [g_device newBufferWithBytesNoCopy:ptr
                                                    length:aligned_bytes
                                                   options:MTLResourceStorageModeShared
                                               deallocator:nil];
    if (!buf) {
        fprintf(stderr, "[metal] metal_alloc_nocopy failed for %zu bytes at %p\n", bytes, ptr);
        return nullptr;
    }

    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_buffer_map[ptr] = buf;
        // Don't count toward g_total_alloc since we don't own the memory
    }

    return ptr;
}

void metal_register_subptr(void* subptr, void* parent_ptr) {
    if (!subptr || !parent_ptr) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_buffer_map.find(parent_ptr);
    if (it != g_buffer_map.end()) {
        g_buffer_map[subptr] = it->second;
    }
}

void metal_free(void* ptr) {
    if (!ptr) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_buffer_map.find(ptr);
    if (it != g_buffer_map.end()) {
        // Only decrement total_alloc for buffers we allocated (not nocopy)
        id<MTLBuffer> buf = it->second;
        if ([buf contents] == ptr) {
            // This is the base pointer for a buffer we may own
            // Check if any other entries share this buffer (nocopy sub-ptrs)
            // Only count it once for alloc tracking
            g_total_alloc -= [buf length];
        }
        g_buffer_map.erase(it);
    }
}

static id<MTLBuffer> find_buffer(const void* ptr) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_buffer_map.find(const_cast<void*>(ptr));
    if (it != g_buffer_map.end()) return it->second;
    return nil;
}

// --- GEMM ---

// Get or create a command buffer for batching
static id<MTLCommandBuffer> get_cmdbuf() {
    if (!g_cmdbuf) {
        g_cmdbuf = [g_queue commandBuffer];
    }
    return g_cmdbuf;
}

bool metal_sgemm_f16(
    const void* A,   // [M, K] fp16
    const void* B,   // [N, K] fp16
    void* C,         // [M, N] fp32
    int M, int N, int K,
    float alpha, float beta)
{
    if (!g_initialized) return false;

    id<MTLBuffer> bufA = find_buffer(A);
    id<MTLBuffer> bufB = find_buffer(B);
    id<MTLBuffer> bufC = find_buffer(C);

    if (!bufA || !bufB || !bufC) {
        fprintf(stderr, "[metal] sgemm: buffer not found (A=%p B=%p C=%p)\n", A, B, C);
        return false;
    }

    // A: [M, K] fp16, row-major → rowBytes = K * 2
    MPSMatrixDescriptor* descA = [MPSMatrixDescriptor
        matrixDescriptorWithRows:M
                         columns:K
                        rowBytes:(NSUInteger)K * 2
                        dataType:MPSDataTypeFloat16];

    // B: [N, K] fp16, row-major → rowBytes = K * 2
    // We want A @ B^T, so B is "right" with transpose
    MPSMatrixDescriptor* descB = [MPSMatrixDescriptor
        matrixDescriptorWithRows:N
                         columns:K
                        rowBytes:(NSUInteger)K * 2
                        dataType:MPSDataTypeFloat16];

    // C: [M, N] fp32, row-major → rowBytes = N * 4
    MPSMatrixDescriptor* descC = [MPSMatrixDescriptor
        matrixDescriptorWithRows:M
                         columns:N
                        rowBytes:(NSUInteger)N * 4
                        dataType:MPSDataTypeFloat32];

    NSUInteger offA = (NSUInteger)((const uint8_t*)A - (const uint8_t*)[bufA contents]);
    NSUInteger offB = (NSUInteger)((const uint8_t*)B - (const uint8_t*)[bufB contents]);
    NSUInteger offC = (NSUInteger)((const uint8_t*)C - (const uint8_t*)[bufC contents]);

    MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:bufA offset:offA descriptor:descA];
    MPSMatrix* matB = [[MPSMatrix alloc] initWithBuffer:bufB offset:offB descriptor:descB];
    MPSMatrix* matC = [[MPSMatrix alloc] initWithBuffer:bufC offset:offC descriptor:descC];

    // C = alpha * A @ B^T + beta * C
    MPSMatrixMultiplication* mm = [[MPSMatrixMultiplication alloc]
        initWithDevice:g_device
           transposeLeft:NO
          transposeRight:YES
              resultRows:M
           resultColumns:N
         interiorColumns:K
                   alpha:(double)alpha
                    beta:(double)beta];

    id<MTLCommandBuffer> cmdbuf = get_cmdbuf();
    [mm encodeToCommandBuffer:cmdbuf leftMatrix:matA rightMatrix:matB resultMatrix:matC];

    return true;
}

void metal_sync() {
    if (!g_cmdbuf) return;
    [g_cmdbuf commit];
    [g_cmdbuf waitUntilCompleted];
    g_cmdbuf = nil;
}

// --- Compute Kernels ---

bool metal_init_compute(const char* metallib_path) {
    if (g_compute_initialized) return true;
    if (!g_initialized) return false;

    @autoreleasepool {
        NSError* error = nil;

        // Load the compiled metallib
        NSString* path = [NSString stringWithUTF8String:metallib_path];
        NSURL* url = [NSURL fileURLWithPath:path];
        id<MTLLibrary> lib = [g_device newLibraryWithURL:url error:&error];
        if (!lib) {
            fprintf(stderr, "[metal] Failed to load metallib from %s: %s\n",
                    metallib_path, [[error localizedDescription] UTF8String]);
            return false;
        }

        // Create pipeline states for each kernel
        auto make_pso = [&](const char* name) -> id<MTLComputePipelineState> {
            id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
            if (!fn) {
                fprintf(stderr, "[metal] Kernel '%s' not found in metallib\n", name);
                return nil;
            }
            id<MTLComputePipelineState> pso = [g_device newComputePipelineStateWithFunction:fn error:&error];
            if (!pso) {
                fprintf(stderr, "[metal] Failed to create PSO for '%s': %s\n",
                        name, [[error localizedDescription] UTF8String]);
                return nil;
            }
            return pso;
        };

        g_swiglu_pso = make_pso("swiglu_fused");
        g_residual_add_pso = make_pso("residual_add_f32");
        g_rmsnorm_f16_pso = make_pso("rmsnorm_f16out");
        g_f32_to_f16_pso = make_pso("f32_to_f16_kernel");

        if (!g_swiglu_pso || !g_residual_add_pso || !g_rmsnorm_f16_pso || !g_f32_to_f16_pso) {
            return false;
        }

        g_compute_initialized = true;
        LOG("[metal] Compute kernels loaded: swiglu, residual_add, rmsnorm_f16, f32_to_f16\n");
        return true;
    }
}

// Helper: encode a 1D compute dispatch on the current command buffer
static bool dispatch_1d(id<MTLComputePipelineState> pso,
                        id<MTLBuffer> bufs[], NSUInteger offsets[], int nbuf,
                        int count) {
    id<MTLCommandBuffer> cmdbuf = get_cmdbuf();
    id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
    if (!enc) return false;

    [enc setComputePipelineState:pso];
    for (int i = 0; i < nbuf; i++) {
        [enc setBuffer:bufs[i] offset:offsets[i] atIndex:i];
    }

    NSUInteger tw = [pso threadExecutionWidth];
    MTLSize grid = MTLSizeMake((NSUInteger)count, 1, 1);
    MTLSize tg = MTLSizeMake(tw, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    return true;
}

bool metal_swiglu_fused(void* gate_f32, void* up_f32, void* out_f16, int count) {
    if (!g_compute_initialized || count <= 0) return false;
    id<MTLBuffer> b0 = find_buffer(gate_f32);
    id<MTLBuffer> b1 = find_buffer(up_f32);
    id<MTLBuffer> b2 = find_buffer(out_f16);
    if (!b0 || !b1 || !b2) return false;

    NSUInteger off0 = (NSUInteger)((uint8_t*)gate_f32 - (uint8_t*)[b0 contents]);
    NSUInteger off1 = (NSUInteger)((uint8_t*)up_f32 - (uint8_t*)[b1 contents]);
    NSUInteger off2 = (NSUInteger)((uint8_t*)out_f16 - (uint8_t*)[b2 contents]);

    id<MTLBuffer> bufs[] = {b0, b1, b2};
    NSUInteger offsets[] = {off0, off1, off2};
    return dispatch_1d(g_swiglu_pso, bufs, offsets, 3, count);
}

bool metal_residual_add(void* x_f32, const void* y_f32, int count) {
    if (!g_compute_initialized || count <= 0) return false;
    id<MTLBuffer> b0 = find_buffer(x_f32);
    id<MTLBuffer> b1 = find_buffer(y_f32);
    if (!b0 || !b1) return false;

    NSUInteger off0 = (NSUInteger)((uint8_t*)x_f32 - (uint8_t*)[b0 contents]);
    NSUInteger off1 = (NSUInteger)((const uint8_t*)y_f32 - (const uint8_t*)[b1 contents]);

    id<MTLBuffer> bufs[] = {b0, b1};
    NSUInteger offsets[] = {off0, off1};
    return dispatch_1d(g_residual_add_pso, bufs, offsets, 2, count);
}

bool metal_rmsnorm_f16(const void* x_f32, const void* weight_f32, void* out_f16,
                       int N, int dim, float eps) {
    if (!g_compute_initialized || N <= 0 || dim <= 0) return false;
    id<MTLBuffer> bx = find_buffer(x_f32);
    id<MTLBuffer> bw = find_buffer(weight_f32);
    id<MTLBuffer> bo = find_buffer(out_f16);
    if (!bx || !bw || !bo) return false;

    // Create params buffer with { dim, eps_as_uint }
    uint32_t params[2];
    params[0] = (uint32_t)dim;
    memcpy(&params[1], &eps, sizeof(float));

    // Use a small Metal buffer for params
    id<MTLBuffer> bp = [g_device newBufferWithBytes:params
                                             length:sizeof(params)
                                            options:MTLResourceStorageModeShared];
    if (!bp) return false;

    NSUInteger offx = (NSUInteger)((const uint8_t*)x_f32 - (const uint8_t*)[bx contents]);
    NSUInteger offw = (NSUInteger)((const uint8_t*)weight_f32 - (const uint8_t*)[bw contents]);
    NSUInteger offo = (NSUInteger)((uint8_t*)out_f16 - (uint8_t*)[bo contents]);

    id<MTLCommandBuffer> cmdbuf = get_cmdbuf();
    id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
    if (!enc) return false;

    [enc setComputePipelineState:g_rmsnorm_f16_pso];
    [enc setBuffer:bx offset:offx atIndex:0];
    [enc setBuffer:bw offset:offw atIndex:1];
    [enc setBuffer:bo offset:offo atIndex:2];
    [enc setBuffer:bp offset:0 atIndex:3];

    // 2D grid: (threads_per_row, N) — one threadgroup per row
    // Use up to 1024 threads per row for the reduction, capped at dim
    NSUInteger tpr = [g_rmsnorm_f16_pso maxTotalThreadsPerThreadgroup];
    if (tpr > 1024) tpr = 1024;
    // Round down to power of 2 for clean reduction
    NSUInteger tpr_po2 = 1;
    while (tpr_po2 * 2 <= tpr && tpr_po2 * 2 <= (NSUInteger)dim) tpr_po2 *= 2;

    MTLSize grid = MTLSizeMake(tpr_po2, (NSUInteger)N, 1);
    MTLSize tg = MTLSizeMake(tpr_po2, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    return true;
}

bool metal_f32_to_f16(const void* in_f32, void* out_f16, int count) {
    if (!g_compute_initialized || count <= 0) return false;
    id<MTLBuffer> b0 = find_buffer(in_f32);
    id<MTLBuffer> b1 = find_buffer(out_f16);
    if (!b0 || !b1) return false;

    NSUInteger off0 = (NSUInteger)((const uint8_t*)in_f32 - (const uint8_t*)[b0 contents]);
    NSUInteger off1 = (NSUInteger)((uint8_t*)out_f16 - (uint8_t*)[b1 contents]);

    id<MTLBuffer> bufs[] = {b0, b1};
    NSUInteger offsets[] = {off0, off1};
    return dispatch_1d(g_f32_to_f16_pso, bufs, offsets, 2, count);
}

uint64_t metal_total_alloc_bytes() {
    return g_total_alloc;
}

} // namespace ane_lm
