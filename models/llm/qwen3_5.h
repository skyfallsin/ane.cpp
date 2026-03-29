#pragma once

#include "../../core/model_loader.h"
#include "../../core/ane_runtime.h"
#include "../../core/sampling.h"
#include <nlohmann/json.hpp>
#include <memory>
#include <string>
#include <vector>

namespace ane_lm {

enum class LayerType {
    LinearAttention,
    FullAttention,
};

// Abstract base class for LLM models
class LLMModel {
public:
    virtual ~LLMModel() = default;
    virtual bool load(const std::string& model_dir) = 0;
    virtual float* forward(int token_id, int pos) = 0;
    virtual float* prefill(const std::vector<int>& token_ids, int start_pos = 0) {
        float* logits = nullptr;
        for (int i = 0; i < (int)token_ids.size(); i++) {
            logits = forward(token_ids[i], start_pos + i);
            if (!logits) return nullptr;
        }
        return logits;
    }
    virtual bool supports_speculative_decode() const { return false; }
    virtual bool init_speculative(const std::vector<int>& prompt_tokens, int start_pos = 0) {
        (void)prompt_tokens;
        (void)start_pos;
        return false;
    }
    virtual int speculative_batch_size() const { return 0; }
    virtual bool draft_speculative_tokens(int max_tokens, int sampler_vocab,
                                          const SamplingParams& sampling,
                                          const std::vector<int>& recent_tokens,
                                          std::vector<int>& drafted_tokens,
                                          int seed_token = -1) {
        (void)max_tokens;
        (void)sampler_vocab;
        (void)sampling;
        (void)recent_tokens;
        (void)seed_token;
        drafted_tokens.clear();
        return false;
    }
    virtual const float* draft_logits_at(int position) const {
        (void)position;
        return nullptr;
    }
    virtual bool verify_speculative(const int* token_ids, int batch, int start_pos,
                                    float** logits_batch, int* logits_stride) {
        (void)token_ids;
        (void)batch;
        (void)start_pos;
        if (logits_batch) *logits_batch = nullptr;
        if (logits_stride) *logits_stride = 0;
        return false;
    }
    virtual void finalize_speculative(int accepted_tokens) {
        (void)accepted_tokens;
    }
    virtual bool accept_speculative_token(int token_id, int pos) {
        (void)token_id;
        (void)pos;
        return false;
    }
    virtual bool supports_external_draft_decode() const { return false; }
    virtual bool init_external_draft_verify(const std::vector<int>& prompt_tokens, int start_pos = 0) {
        (void)prompt_tokens;
        (void)start_pos;
        return false;
    }
    virtual void finalize_external_draft_verify(int accepted_tokens) {
        (void)accepted_tokens;
    }
    virtual void reset() = 0;
    virtual int vocab_size() const = 0;
};

// Model-owned config (mirrors mlx-lm.cpp style)
struct Qwen35Args {
    int hidden_size = 1024;
    int num_hidden_layers = 24;
    int num_attention_heads = 8;
    int num_key_value_heads = 2;
    int head_dim = 256;
    int intermediate_size = 3584;
    int vocab_size = 248320;
    int full_attention_interval = 4;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 10000000.0f;
    float partial_rotary_factor = 0.25f;
    int linear_num_key_heads = 16;
    int linear_key_head_dim = 128;
    int linear_value_head_dim = 128;
    int linear_num_value_heads = 16;
    int linear_conv_kernel_dim = 4;
    bool tie_word_embeddings = true;
    bool attn_output_gate = true;

    // Derived
    int key_dim() const { return linear_key_head_dim * linear_num_key_heads; }
    int value_dim() const { return linear_value_head_dim * linear_num_value_heads; }
    int conv_dim() const { return 2 * key_dim() + value_dim(); }
    int rotation_dim() const { return static_cast<int>(head_dim * partial_rotary_factor); }

    std::vector<LayerType> layer_types;

    static Qwen35Args from_json(const nlohmann::json& config);
};

// Qwen3.5 model implementation
class Qwen35Model : public LLMModel {
public:
    struct Session {
        std::vector<float*> delta_ssm_state;
        std::vector<float*> delta_conv_state;
        std::vector<int> delta_conv_pos;
        std::vector<float*> kv_k_cache;
        std::vector<float*> kv_v_cache;
        std::vector<int> kv_len;
        std::vector<int> kv_start;
        float* x = nullptr;
        float* x_norm = nullptr;
        float* logits = nullptr;
        float* scratch_qkv = nullptr;
        float* scratch_conv = nullptr;
        float* scratch_y = nullptr;
        float* scratch_attn = nullptr;
        float* scratch_tmp = nullptr;

        Session() = default;
        Session(const Session&) = delete;
        Session& operator=(const Session&) = delete;
        Session(Session&&) = delete;
        Session& operator=(Session&&) = delete;
        ~Session();
    };

    ~Qwen35Model() override;
    bool load(const std::string& model_dir) override;
    float* forward(int token_id, int pos) override;
    float* prefill(const std::vector<int>& token_ids, int start_pos = 0) override;
    float* forward(Session& session, int token_id, int pos);
    bool forward_batch(Session** sessions, const int* token_ids, const int* positions, int batch);
    float* prefill(Session& session, const std::vector<int>& token_ids, int start_pos = 0);
    std::unique_ptr<Session> create_session() const;
    void reset_session(Session& session) const;
    void reset() override;
    int vocab_size() const override { return vocab_size_; }

private:
    // Config
    int hidden_size_ = 0;
    int intermediate_size_ = 0;
    int vocab_size_ = 0;
    int num_layers_ = 0;
    int num_q_heads_ = 0;
    int num_kv_heads_ = 0;
    int head_dim_ = 0;
    int rot_dim_ = 0;
    float rope_theta_ = 0;
    float rms_eps_ = 0;
    int lin_num_heads_ = 0;
    int lin_num_val_heads_ = 0;
    int lin_key_dim_ = 0;
    int lin_val_dim_ = 0;
    int lin_total_key_ = 0;
    int lin_total_val_ = 0;
    int lin_qkv_dim_ = 0;
    int conv_kernel_ = 0;
    int full_q_dim_ = 0;
    int full_kv_dim_ = 0;
    int full_out_dim_ = 0;
    bool attn_output_gate_ = true;
    bool tie_word_embeddings_ = true;

    static constexpr int MAX_SEQ_LEN = 4096;
    static constexpr int KV_CACHE_CAPACITY = 2048;
    static constexpr int LM_HEAD_ANE_CHUNK_MAX = 65536;

    std::vector<LayerType> layer_types_;

    // DeltaNet weights
    struct DeltaNetWeights {
        float* in_proj_a = nullptr;
        float* in_proj_b = nullptr;
        float* conv1d_w = nullptr;
        float* A = nullptr;
        float* dt_bias = nullptr;
        float* norm_w = nullptr;
    };

    // Full attention weights
    struct FullAttnWeights {
        float* q_norm = nullptr;
        float* k_norm = nullptr;
    };

    // Layer weights
    struct LayerWeights {
        LayerType type;
        DeltaNetWeights deltanet;
        FullAttnWeights full_attn;
        float* input_layernorm = nullptr;
        float* post_attention_layernorm = nullptr;
    };

    // Model data
    std::vector<LayerWeights> layers_;
    float* embed_tokens_ = nullptr;
    float* lm_head_ = nullptr;
    float* final_norm_ = nullptr;

    std::vector<LayerANEKernels> ane_layers_;

    // LM head ANE kernels
    std::vector<ANEKernel*> lm_head_kernels_;
    int lm_head_chunk_ = LM_HEAD_ANE_CHUNK_MAX;
    bool ane_lm_head_enabled_ = false;

    std::unique_ptr<Session> default_session_;
    float* rope_cos_ = nullptr;
    float* rope_sin_ = nullptr;

    bool init_session(Session& session) const;
    void apply_args(const Qwen35Args& args);
    bool load_weights(ModelWeights* sf);
    bool compile_ane(ModelWeights* sf, const std::string& blob_dir);
    bool compile_lm_head_ane(ModelWeights* sf, const std::string& blob_dir);
    void free_lm_head_ane();

    bool forward_deltanet_core(Session& session, int L, float* x, float* pre_oproj);
    bool forward_full_attn_core(Session& session, int L, float* x, float* pre_oproj, int pos);
};

} // namespace ane_lm
