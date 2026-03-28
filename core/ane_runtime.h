#pragma once

#include <cstdint>
#include <cstddef>
#include <string>

namespace ane_lm {

// ANE minimum spatial dimension required by hardware
constexpr int ANE_SPATIAL = 32;

// Opaque kernel handle
struct ANEKernel;

// Chunked FFN: N kernels that chain accumulation on ANE
struct ChunkedFFN {
    ANEKernel** chunks = nullptr;
    int num_chunks = 0;
    int dim = 0;
    int chunk_inter = 0;
};

// Per-layer ANE kernels
struct LayerANEKernels {
    ANEKernel* first_proj = nullptr;
    ANEKernel* o_proj = nullptr;
    ANEKernel* fused_ffn = nullptr;
    ChunkedFFN chunked_ffn = {};
    ANEKernel* fused_oproj_norm = nullptr;  // conv(O_proj) → add(res) → RMSNorm [2 outputs: x_norm, x_updated]
    ANEKernel* fused_oproj_ffn = nullptr;   // conv(O_proj) → add(res) → RMSNorm → SwiGLU FFN [2 outputs: ffn_out, x_updated]
    ANEKernel* oproj_add = nullptr;         // conv(O_proj) → add(res) [1 output: x_updated] — no redundant norm
    ANEKernel* ffn_resadd = nullptr;        // SwiGLU FFN + residual add [2 inputs: x_norm, x_res → 1 output: x_res + ffn(x_norm)]
};

// Global state
void ane_init();
bool ane_available();
void ane_set_persist_cache(bool enabled);
int ane_compile_count();
int ane_cache_loads();

// Kernel compilation (from BF16 weights)
ANEKernel* ane_compile_matmul(const uint16_t* bf16_weights, int out_dim, int in_dim);
ANEKernel* ane_compile_fused_2(const uint16_t* bf16_a, int a_out,
                                const uint16_t* bf16_b, int b_out,
                                int in_dim);
ANEKernel* ane_compile_fused_3(const uint16_t* bf16_a, int a_out,
                                const uint16_t* bf16_b, int b_out,
                                const uint16_t* bf16_c, int c_out,
                                int in_dim);
ANEKernel* ane_compile_fused_ffn(const uint16_t* gate_bf16, const uint16_t* up_bf16,
                                  const uint16_t* down_bf16, int dim, int inter_ch);

// Fused FFN + residual add: output = x_residual + SwiGLU_FFN(x_norm)
// 2 inputs: x_norm [dim], x_residual [dim]
// 1 output: y [dim]
ANEKernel* ane_compile_fused_ffn_resadd(const uint16_t* gate_bf16, const uint16_t* up_bf16,
                                         const uint16_t* down_bf16, int dim, int inter_ch);
ANEKernel* ane_compile_fused_ffn_resadd_blob(const std::string& gate_path, const std::string& up_path,
                                              const std::string& down_path, int dim, int inter_ch);
bool ane_eval_fused_ffn_resadd(ANEKernel* k, float* output,
                                const float* x_norm, const float* x_residual, int dim);
bool ane_compile_chunked_ffn(ChunkedFFN* out, const uint16_t* gate_bf16,
                              const uint16_t* up_bf16, const uint16_t* down_bf16,
                              int dim, int inter_ch, int num_chunks);

// Kernel compilation (from pre-converted ANE blob files)
ANEKernel* ane_compile_matmul_blob(const std::string& blob_path, int out_dim, int in_dim);
ANEKernel* ane_compile_fused_2_blob(const std::string& a_path, int a_out,
                                     const std::string& b_path, int b_out,
                                     int in_dim);
ANEKernel* ane_compile_fused_3_blob(const std::string& a_path, int a_out,
                                     const std::string& b_path, int b_out,
                                     const std::string& c_path, int c_out,
                                     int in_dim);
ANEKernel* ane_compile_fused_ffn_blob(const std::string& gate_path, const std::string& up_path,
                                       const std::string& down_path, int dim, int inter_ch);
bool ane_compile_chunked_ffn_blob(ChunkedFFN* out, const std::string& gate_path,
                                   const std::string& up_path, const std::string& down_path,
                                   int dim, int inter_ch, int num_chunks);

// Fused O_proj + residual add + RMSNorm kernel
// 2 inputs: attn_out [in_dim], x_residual [out_dim]
// 2 outputs: x_norm [out_dim], x_updated [out_dim] (residual after add, before norm)
ANEKernel* ane_compile_fused_oproj_norm(const uint16_t* oproj_bf16,
                                         const float* norm_weight,
                                         int out_dim, int in_dim, float eps);
ANEKernel* ane_compile_fused_oproj_norm_blob(const std::string& oproj_path,
                                              const float* norm_weight,
                                              int out_dim, int in_dim, float eps);

// Eval fused O_proj+norm: returns x_norm and x_updated
bool ane_eval_fused_oproj_norm(ANEKernel* k, float* x_norm, float* x_updated,
                                const float* attn_out, const float* x_residual,
                                int in_dim, int out_dim);

// Simplified O_proj + residual add (no RMSNorm — CPU handles norm)
// 2 inputs: attn_out [in_dim], x_residual [out_dim]
// 1 output: x_updated [out_dim] = x_residual + conv(O_proj, attn_out)
ANEKernel* ane_compile_oproj_add(const uint16_t* oproj_bf16, int out_dim, int in_dim);
ANEKernel* ane_compile_oproj_add_blob(const std::string& oproj_path, int out_dim, int in_dim);
bool ane_eval_oproj_add(ANEKernel* k, float* x_updated,
                         const float* attn_out, const float* x_residual,
                         int in_dim, int out_dim);

// Fused O_proj + residual add + RMSNorm + SwiGLU FFN kernel
// 2 inputs: attn_out [in_dim], x_residual [out_dim]
// 2 outputs: ffn_out [out_dim], x_updated [out_dim]
ANEKernel* ane_compile_fused_oproj_ffn(const uint16_t* oproj_bf16,
                                        const uint16_t* gate_bf16,
                                        const uint16_t* up_bf16,
                                        const uint16_t* down_bf16,
                                        const float* norm_weight,
                                        int dim, int in_dim, int inter_ch, float eps);

bool ane_eval_fused_oproj_ffn(ANEKernel* k, float* ffn_out, float* x_updated,
                               const float* attn_out, const float* x_residual,
                               int in_dim, int dim);

// Kernel execution
bool ane_matvec(ANEKernel* k, float* output, const float* input, int in_dim, int out_dim);
bool ane_eval_chunked_ffn(const ChunkedFFN* cffn, float* output, const float* input);

// Kernel cleanup
void ane_free(ANEKernel* k);
void ane_free_chunked_ffn(ChunkedFFN* cffn);
void ane_free_layer(LayerANEKernels* lk);

// Auto-determine chunk count for FFN
int ane_ffn_chunk_count(int dim, int inter_ch);

// Generic MIL compile
ANEKernel* ane_compile_mil(const char* mil_text, int n_inputs, size_t* input_sizes,
                            int n_outputs, size_t* output_sizes);

// Generic MIL compile with named BF16 weight blobs
// weight_names: array of N names (e.g. "weight", "wa")
// weight_bf16:  array of N bf16 data pointers
// weight_numel: array of N element counts
struct MILWeight {
    const char* name;         // e.g. "weight" → @model_path/weights/weight.bin
    const uint16_t* bf16;     // BF16 data pointer
    int numel;                // number of elements
};
ANEKernel* ane_compile_mil_weighted(const char* mil_text,
                                     int n_inputs, size_t* input_sizes,
                                     int n_outputs, size_t* output_sizes,
                                     MILWeight* weights, int n_weights);

// Generic MIL compile with raw weight blobs (no BF16→FP16 conversion)
// Each weight entry is raw blob bytes written verbatim to disk.
// Use this for INT8 quantized weights, custom formats, etc.
struct MILRawWeight {
    const char* name;         // e.g. "wq" → @model_path/weights/wq.bin
    const void* data;         // Raw blob data (header + payload)
    size_t size;              // Total blob size in bytes
};
ANEKernel* ane_compile_mil_raw(const char* mil_text,
                                int n_inputs, size_t* input_sizes,
                                int n_outputs, size_t* output_sizes,
                                MILRawWeight* weights, int n_weights);

// Dynamic-weight conv
ANEKernel* ane_compile_dynamic_conv(int out_dim, int in_dim);
bool ane_dynamic_conv_eval(ANEKernel* k, float* output, const float* input,
                            const uint16_t* fp16_weights, int in_dim, int out_dim);

// Dynamic-weight fused FFN
ANEKernel* ane_compile_dynamic_ffn(int dim, int inter_ch);
bool ane_dynamic_ffn_eval(ANEKernel* k, float* output, const float* input,
                           const uint16_t* gate_fp16, const uint16_t* up_fp16,
                           const uint16_t* down_fp16, int dim, int inter_ch);

// Generic multi-input eval
bool ane_eval_multi(ANEKernel* k,
                    float** inputs, int* input_channels,
                    float** outputs, int* output_channels);

// Write raw uint16 fp16 data into input IOSurface (dense, no striding)
bool ane_write_surface_raw(ANEKernel* k, int input_idx, const uint16_t* data, size_t bytes);

// Write raw uint16 fp16 data with SP striding
bool ane_write_surface_strided(ANEKernel* k, int input_idx, const uint16_t* data, int channels);

// IOSurface sizes
size_t ane_get_input_size(ANEKernel* k, int input_idx);
size_t ane_get_output_size(ANEKernel* k, int output_idx);

// Counts
int ane_get_n_inputs(ANEKernel* k);
int ane_get_n_outputs(ANEKernel* k);

// Raw eval: dispatch without writing inputs, read SP-strided outputs
bool ane_eval_raw_outputs(ANEKernel* k, float** outputs, int* output_channels);

// Read ALL fp16 values (dense) from output surface into float array
bool ane_read_output_raw(ANEKernel* k, int output_idx, float* data, int count);

// Write float data into input surface with tiled [N,C,H,W] layout
bool ane_write_input_tiled(ANEKernel* k, int input_idx, const float* data,
                           int N, int C, int H, int W);

// ============ API Tuning ============

// Access internal ObjC objects (for direct API exploration)
void* ane_get_model(ANEKernel* k);    // Returns id (_ANEInMemoryModel*)
void* ane_get_request(ANEKernel* k);  // Returns id (_ANERequest*)

// Evaluate with a specific QoS level (default is 21 = userInteractive)
bool ane_eval_qos(ANEKernel* k, int qos);

// Print model info: queueDepth, state, perfStatsMask, attributes
void ane_print_model_info(ANEKernel* k);

// Set queue depth on the model (for pipelining)
void ane_set_queue_depth(ANEKernel* k, int depth);

// Enable perf stats collection (mask=0xFFFFFFFF for all)
void ane_set_perf_stats_mask(ANEKernel* k, unsigned int mask);

} // namespace ane_lm
