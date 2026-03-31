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

// Stats
uint64_t metal_total_alloc_bytes();

} // namespace ane_lm
