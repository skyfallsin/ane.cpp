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

void metal_free(void* ptr) {
    if (!ptr) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_buffer_map.find(ptr);
    if (it != g_buffer_map.end()) {
        g_total_alloc -= [it->second length];
        g_buffer_map.erase(it);
    }
}

static id<MTLBuffer> find_buffer(const void* ptr) {
    // The pointer might be inside a buffer (offset), not exactly at the start.
    // But for our usage, we always pass the base pointer.
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

uint64_t metal_total_alloc_bytes() {
    return g_total_alloc;
}

} // namespace ane_lm
