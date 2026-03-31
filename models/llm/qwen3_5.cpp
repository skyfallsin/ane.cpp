#include "qwen3_5.h"
#include "../../core/cpu_ops.h"
#include <cmath>
#include <fstream>
#include <sys/stat.h>
#include <mutex>
#include <dispatch/dispatch.h>

namespace ane_lm {

using json = nlohmann::json;

static bool qwen35_profile_serve_batch() {
    static bool enabled = getenv("PROFILE_SERVE_BATCH") != nullptr;
    return enabled;
}

struct Qwen35BatchProfile {
    std::mutex mu;
    uint64_t calls = 0;
    double rms1_ms = 0.0;
    double first_proj_ms = 0.0;
    double core_cpu_ms = 0.0;
    double oproj_ms = 0.0;
    double rms2_ms = 0.0;
    double ffn_ms = 0.0;
    double final_norm_ms = 0.0;
    double lm_head_ms = 0.0;

    void add(double rms1, double first_proj, double core_cpu, double oproj, double rms2, double ffn, double final_norm, double lm_head) {
        std::lock_guard<std::mutex> lock(mu);
        calls++;
        rms1_ms += rms1;
        first_proj_ms += first_proj;
        core_cpu_ms += core_cpu;
        oproj_ms += oproj;
        rms2_ms += rms2;
        ffn_ms += ffn;
        final_norm_ms += final_norm;
        lm_head_ms += lm_head;
        if (calls % 50 == 0) {
            double total = rms1_ms + first_proj_ms + core_cpu_ms + oproj_ms + rms2_ms + ffn_ms + final_norm_ms + lm_head_ms;
            fprintf(stderr,
                    "Qwen35 forward_batch profile: %llu calls avg %.3f ms [rms1 %.3f | first_proj %.3f | core_cpu %.3f | oproj %.3f | rms2 %.3f | ffn %.3f | final_norm %.3f | lm_head %.3f]\n",
                    (unsigned long long)calls,
                    total / calls,
                    rms1_ms / calls,
                    first_proj_ms / calls,
                    core_cpu_ms / calls,
                    oproj_ms / calls,
                    rms2_ms / calls,
                    ffn_ms / calls,
                    final_norm_ms / calls,
                    lm_head_ms / calls);
        }
    }
};

static Qwen35BatchProfile g_qwen35_batch_profile;

static int qwen35_prefill_batch_size() {
    static int batch = [] {
        const char* env = getenv("ANE_PREFILL_BATCH");
        int v = env ? atoi(env) : 4;
        if (v < 1) v = 1;
        if (v > ANE_SPATIAL) v = ANE_SPATIAL;
        return v;
    }();
    return batch;
}

static bool qwen35_use_merged_ffn_gu() {
    static bool enabled = getenv("ANE_FFN_MERGED_GU") != nullptr;
    return enabled;
}

static bool qwen35_linear_z_on_cpu() {
    static bool enabled = getenv("ANE_LINEAR_Z_ON_CPU") != nullptr;
    return enabled;
}

static bool qwen35_batch_rmsgate() {
    static bool enabled = getenv("ANE_BATCH_RMSGATE") != nullptr;
    return enabled;
}

static void rmsnorm_gated_repeated(float* out,
                                   const float* x,
                                   const float* z,
                                   const float* weight,
                                   int heads,
                                   int dim,
                                   float* tmp) {
    for (int h = 0; h < heads; h++) {
        float* out_h = out + (size_t)h * dim;
        const float* x_h = x + (size_t)h * dim;
        const float* z_h = z + (size_t)h * dim;
        float ss = 0.0f;
        vDSP_svesq(x_h, 1, &ss, (vDSP_Length)dim);
        ss = 1.0f / sqrtf(ss / dim + 1e-6f);
        vDSP_vsmul(x_h, 1, &ss, out_h, 1, (vDSP_Length)dim);
        vDSP_vmul(out_h, 1, weight, 1, out_h, 1, (vDSP_Length)dim);
        mul_sigmoid_inplace(out_h, z_h, dim, tmp);
        vDSP_vmul(out_h, 1, z_h, 1, out_h, 1, (vDSP_Length)dim);
    }
}

static void pack_w_lanes(std::vector<float>& packed,
                         const float* batch_data,
                         int batch,
                         int dim) {
    packed.assign((size_t)dim * ANE_SPATIAL, 0.0f);
    for (int c = 0; c < dim; c++) {
        size_t base = (size_t)c * ANE_SPATIAL;
        for (int b = 0; b < batch; b++) {
            packed[base + b] = batch_data[(size_t)b * dim + c];
        }
    }
}

static void unpack_w_lanes(float* batch_data,
                           const std::vector<float>& raw,
                           int batch,
                           int dim) {
    for (int c = 0; c < dim; c++) {
        size_t base = (size_t)c * ANE_SPATIAL;
        for (int b = 0; b < batch; b++) {
            batch_data[(size_t)b * dim + c] = raw[base + b];
        }
    }
}

static bool ane_matvec_batch(ANEKernel* kernel,
                             float* output,
                             const float* input,
                             int batch,
                             int in_dim,
                             int out_dim,
                             std::vector<float>& packed_in,
                             std::vector<float>& raw_out) {
    pack_w_lanes(packed_in, input, batch, in_dim);
    if (!ane_write_input_tiled(kernel, 0, packed_in.data(), 1, in_dim, 1, ANE_SPATIAL)) return false;
    float dummy = 0.0f;
    float* outputs[1] = { &dummy };
    int out_chs[1] = { 1 };
    if (!ane_eval_raw_outputs(kernel, outputs, out_chs)) return false;
    raw_out.resize((size_t)out_dim * ANE_SPATIAL);
    if (!ane_read_output_raw(kernel, 0, raw_out.data(), (int)raw_out.size())) return false;
    unpack_w_lanes(output, raw_out, batch, out_dim);
    return true;
}

static bool ane_binary_batch(ANEKernel* kernel,
                             int input0_dim,
                             int input1_dim,
                             int output_dim,
                             float* output,
                             const float* input0,
                             const float* input1,
                             int batch,
                             std::vector<float>& packed0,
                             std::vector<float>& packed1,
                             std::vector<float>& raw_out) {
    pack_w_lanes(packed0, input0, batch, input0_dim);
    pack_w_lanes(packed1, input1, batch, input1_dim);
    if (!ane_write_input_tiled(kernel, 0, packed0.data(), 1, input0_dim, 1, ANE_SPATIAL)) return false;
    if (!ane_write_input_tiled(kernel, 1, packed1.data(), 1, input1_dim, 1, ANE_SPATIAL)) return false;
    float dummy = 0.0f;
    float* outputs[1] = { &dummy };
    int out_chs[1] = { 1 };
    if (!ane_eval_raw_outputs(kernel, outputs, out_chs)) return false;
    raw_out.resize((size_t)output_dim * ANE_SPATIAL);
    if (!ane_read_output_raw(kernel, 0, raw_out.data(), (int)raw_out.size())) return false;
    unpack_w_lanes(output, raw_out, batch, output_dim);
    return true;
}

// --- Qwen35Args::from_json ---

Qwen35Args Qwen35Args::from_json(const json& j) {
    Qwen35Args args;

    // Parse text_config if present, otherwise read from top level
    const json& tc = j.contains("text_config") ? j["text_config"] : j;

    args.hidden_size = tc.value("hidden_size", args.hidden_size);
    args.num_hidden_layers = tc.value("num_hidden_layers", args.num_hidden_layers);
    args.num_attention_heads = tc.value("num_attention_heads", args.num_attention_heads);
    args.num_key_value_heads = tc.value("num_key_value_heads", args.num_key_value_heads);
    args.head_dim = tc.value("head_dim", args.head_dim);
    args.intermediate_size = tc.value("intermediate_size", args.intermediate_size);
    args.vocab_size = tc.value("vocab_size", args.vocab_size);
    args.full_attention_interval = tc.value("full_attention_interval", args.full_attention_interval);
    args.rms_norm_eps = tc.value("rms_norm_eps", args.rms_norm_eps);
    args.tie_word_embeddings = tc.value("tie_word_embeddings", j.value("tie_word_embeddings", args.tie_word_embeddings));
    args.attn_output_gate = tc.value("attn_output_gate", args.attn_output_gate);
    args.linear_num_key_heads = tc.value("linear_num_key_heads", args.linear_num_key_heads);
    args.linear_key_head_dim = tc.value("linear_key_head_dim", args.linear_key_head_dim);
    args.linear_value_head_dim = tc.value("linear_value_head_dim", args.linear_value_head_dim);
    args.linear_num_value_heads = tc.value("linear_num_value_heads", args.linear_num_value_heads);
    args.linear_conv_kernel_dim = tc.value("linear_conv_kernel_dim", args.linear_conv_kernel_dim);

    // RoPE parameters
    if (tc.contains("rope_parameters")) {
        auto& rp = tc["rope_parameters"];
        args.rope_theta = rp.value("rope_theta", args.rope_theta);
        args.partial_rotary_factor = tc.value("partial_rotary_factor", args.partial_rotary_factor);
    } else {
        args.rope_theta = tc.value("rope_theta", args.rope_theta);
        args.partial_rotary_factor = tc.value("partial_rotary_factor", args.partial_rotary_factor);
    }

    // Layer types
    if (tc.contains("layer_types")) {
        for (auto& lt : tc["layer_types"]) {
            std::string s = lt.get<std::string>();
            if (s == "linear_attention") {
                args.layer_types.push_back(LayerType::LinearAttention);
            } else {
                args.layer_types.push_back(LayerType::FullAttention);
            }
        }
    } else {
        for (int i = 0; i < args.num_hidden_layers; i++) {
            if ((i + 1) % args.full_attention_interval == 0) {
                args.layer_types.push_back(LayerType::FullAttention);
            } else {
                args.layer_types.push_back(LayerType::LinearAttention);
            }
        }
    }

    return args;
}

// --- Qwen35Model ---

Qwen35Model::Session::~Session() {
    for (float* ptr : delta_ssm_state) free(ptr);
    for (float* ptr : delta_conv_state) free(ptr);
    for (float* ptr : kv_k_cache) free(ptr);
    for (float* ptr : kv_v_cache) free(ptr);
    free(x);
    free(x_norm);
    free(logits);
    free(scratch_qkv);
    free(scratch_conv);
    free(scratch_y);
    free(scratch_attn);
    free(scratch_tmp);
}

Qwen35Model::~Qwen35Model() {
    default_session_.reset();
    free(embed_tokens_);
    if (!tie_word_embeddings_) {
        free(lm_head_);
    }
    free(final_norm_);
    free(rope_cos_);
    free(rope_sin_);

    for (int L = 0; L < num_layers_; L++) {
        auto& lw = layers_[L];
        free(lw.input_layernorm);
        free(lw.post_attention_layernorm);

        if (lw.type == LayerType::LinearAttention) {
            free(lw.deltanet.in_proj_a);
            free(lw.deltanet.in_proj_b);
            free(lw.deltanet.in_proj_z);
            free(lw.deltanet.conv1d_w);
            free(lw.deltanet.A);
            free(lw.deltanet.dt_bias);
            free(lw.deltanet.norm_w);
        } else {
            free(lw.full_attn.q_norm);
            free(lw.full_attn.k_norm);
        }
    }

    for (int L = 0; L < num_layers_; L++) {
        ane_free_layer(&ane_layers_[L]);
    }

    free_lm_head_ane();
}

bool Qwen35Model::init_session(Session& session) const {
    session.delta_ssm_state.assign(num_layers_, nullptr);
    session.delta_conv_state.assign(num_layers_, nullptr);
    session.delta_conv_pos.assign(num_layers_, 0);
    session.kv_k_cache.assign(num_layers_, nullptr);
    session.kv_v_cache.assign(num_layers_, nullptr);
    session.kv_len.assign(num_layers_, 0);
    session.kv_start.assign(num_layers_, 0);

    session.x = (float*)calloc(hidden_size_, sizeof(float));
    session.x_norm = (float*)calloc(hidden_size_, sizeof(float));
    session.logits = (float*)calloc(vocab_size_, sizeof(float));
    session.scratch_qkv = (float*)calloc(lin_qkv_dim_ + lin_total_val_, sizeof(float));
    session.scratch_conv = (float*)calloc(lin_qkv_dim_, sizeof(float));
    session.scratch_y = (float*)calloc(lin_total_val_, sizeof(float));
    session.scratch_attn = (float*)calloc(full_out_dim_, sizeof(float));
    session.scratch_tmp = (float*)calloc((size_t)lin_num_val_heads_ * 2 + lin_qkv_dim_, sizeof(float));

    if (!session.x || !session.x_norm || !session.logits || !session.scratch_qkv ||
        !session.scratch_conv || !session.scratch_y || !session.scratch_attn || !session.scratch_tmp) {
        return false;
    }

    for (int L = 0; L < num_layers_; L++) {
        if (layer_types_[L] == LayerType::FullAttention) {
            session.kv_k_cache[L] = (float*)calloc((size_t)KV_CACHE_CAPACITY * num_kv_heads_ * head_dim_, sizeof(float));
            session.kv_v_cache[L] = (float*)calloc((size_t)KV_CACHE_CAPACITY * num_kv_heads_ * head_dim_, sizeof(float));
            if (!session.kv_k_cache[L] || !session.kv_v_cache[L]) return false;
        }
        if (layer_types_[L] == LayerType::LinearAttention) {
            session.delta_ssm_state[L] = (float*)calloc((size_t)lin_num_val_heads_ * lin_key_dim_ * lin_val_dim_, sizeof(float));
            session.delta_conv_state[L] = (float*)calloc((size_t)lin_qkv_dim_ * (conv_kernel_ - 1), sizeof(float));
            if (!session.delta_ssm_state[L] || !session.delta_conv_state[L]) return false;
        }
    }

    return true;
}

std::unique_ptr<Qwen35Model::Session> Qwen35Model::create_session() const {
    auto session = std::make_unique<Session>();
    if (!init_session(*session)) {
        return nullptr;
    }
    return session;
}

void Qwen35Model::reset_session(Session& session) const {
    for (int L = 0; L < num_layers_; L++) {
        if (layer_types_[L] == LayerType::FullAttention) {
            session.kv_len[L] = 0;
            session.kv_start[L] = 0;
            memset(session.kv_k_cache[L], 0, (size_t)KV_CACHE_CAPACITY * num_kv_heads_ * head_dim_ * sizeof(float));
            memset(session.kv_v_cache[L], 0, (size_t)KV_CACHE_CAPACITY * num_kv_heads_ * head_dim_ * sizeof(float));
        }
        if (layer_types_[L] == LayerType::LinearAttention) {
            memset(session.delta_ssm_state[L], 0, (size_t)lin_num_val_heads_ * lin_key_dim_ * lin_val_dim_ * sizeof(float));
            memset(session.delta_conv_state[L], 0, (size_t)lin_qkv_dim_ * (conv_kernel_ - 1) * sizeof(float));
            session.delta_conv_pos[L] = 0;
        }
    }
    memset(session.x, 0, hidden_size_ * sizeof(float));
    memset(session.x_norm, 0, hidden_size_ * sizeof(float));
    memset(session.logits, 0, vocab_size_ * sizeof(float));
}

void Qwen35Model::reset() {
    if (default_session_) {
        reset_session(*default_session_);
    }
}

void Qwen35Model::apply_args(const Qwen35Args& args) {
    hidden_size_ = args.hidden_size;
    intermediate_size_ = args.intermediate_size;
    vocab_size_ = args.vocab_size;
    num_layers_ = args.num_hidden_layers;
    num_q_heads_ = args.num_attention_heads;
    num_kv_heads_ = args.num_key_value_heads;
    head_dim_ = args.head_dim;
    rot_dim_ = args.rotation_dim();
    rope_theta_ = args.rope_theta;
    rms_eps_ = args.rms_norm_eps;
    lin_num_heads_ = args.linear_num_key_heads;
    lin_num_val_heads_ = args.linear_num_value_heads;
    lin_key_dim_ = args.linear_key_head_dim;
    lin_val_dim_ = args.linear_value_head_dim;
    lin_total_key_ = lin_num_heads_ * lin_key_dim_;
    lin_total_val_ = lin_num_val_heads_ * lin_val_dim_;
    lin_qkv_dim_ = lin_total_key_ * 2 + lin_total_val_;
    conv_kernel_ = args.linear_conv_kernel_dim;
    full_q_dim_ = num_q_heads_ * head_dim_ * 2;
    full_kv_dim_ = num_kv_heads_ * head_dim_;
    full_out_dim_ = num_q_heads_ * head_dim_;
    attn_output_gate_ = args.attn_output_gate;
    tie_word_embeddings_ = args.tie_word_embeddings;
    layer_types_ = args.layer_types;
}

bool Qwen35Model::load(const std::string& model_dir) {
    // 1. Read config.json and parse args
    std::string config_path = model_dir + "/config.json";
    std::ifstream f(config_path);
    if (!f.is_open()) {
        fprintf(stderr, "Cannot open %s\n", config_path.c_str());
        return false;
    }
    json j = json::parse(f);
    Qwen35Args args = Qwen35Args::from_json(j);
    apply_args(args);

    // 2. Open model weights (single-file or sharded)
    auto sf = ModelWeights::open(model_dir);
    if (!sf) {
        fprintf(stderr, "Failed to open model weights in %s\n", model_dir.c_str());
        return false;
    }

    // Infer dims from safetensors
    const SFTensor* embed = sf->find("model.language_model.embed_tokens.weight");
    if (!embed || embed->ndims != 2) {
        fprintf(stderr, "Cannot infer dims: missing or invalid embed_tokens.weight\n");
        return false;
    }
    const SFTensor* gate = sf->find("model.language_model.layers.0.mlp.gate_proj.weight");
    if (!gate || gate->ndims != 2) {
        fprintf(stderr, "Cannot infer dims: missing or invalid gate_proj.weight\n");
        return false;
    }

    hidden_size_ = (int)embed->shape[1];
    vocab_size_ = (int)embed->shape[0];
    intermediate_size_ = (int)gate->shape[0];

    LOG("Model dims: hidden=%d intermediate=%d vocab=%d layers=%d\n",
        hidden_size_, intermediate_size_, vocab_size_, num_layers_);

    // 3. Init ANE
    ane_init();

    rope_cos_ = (float*)calloc((size_t)MAX_SEQ_LEN * (rot_dim_ / 2), sizeof(float));
    rope_sin_ = (float*)calloc((size_t)MAX_SEQ_LEN * (rot_dim_ / 2), sizeof(float));

    // Precompute RoPE trig table
    if (rope_cos_ && rope_sin_) {
        int half_rot = rot_dim_ / 2;
        float inv_freq[half_rot];
        for (int j2 = 0, i = 0; i < rot_dim_; i += 2, j2++) {
            inv_freq[j2] = 1.0f / powf(rope_theta_, (float)i / (float)rot_dim_);
        }
        for (int pos = 0; pos < MAX_SEQ_LEN; pos++) {
            float* cos_row = rope_cos_ + (size_t)pos * half_rot;
            float* sin_row = rope_sin_ + (size_t)pos * half_rot;
            for (int j2 = 0; j2 < half_rot; j2++) {
                float angle = pos * inv_freq[j2];
                cos_row[j2] = cosf(angle);
                sin_row[j2] = sinf(angle);
            }
        }
    }

    // Initialize layers
    layers_.resize(num_layers_);
    ane_layers_.resize(num_layers_);

    // 4. Load weights + compile ANE kernels
    if (!load_weights(sf.get())) { return false; }
    // Detect pre-converted ANE blob directory
    std::string blob_dir = model_dir + "/ane_weights";
    struct stat st_blob;
    bool has_blobs = (stat(blob_dir.c_str(), &st_blob) == 0 && S_ISDIR(st_blob.st_mode));
    if (has_blobs) {
        LOG("Using pre-converted ANE blobs from %s\n", blob_dir.c_str());
    }

    if (!compile_ane(sf.get(), has_blobs ? blob_dir : "")) { return false; }

    default_session_ = create_session();
    if (!default_session_) {
        fprintf(stderr, "Failed to allocate default Qwen3.5 session\n");
        return false;
    }
    reset_session(*default_session_);

    return true;
}

bool Qwen35Model::load_weights(ModelWeights* sf) {
    char name[256];

    embed_tokens_ = sf->load_bf16_to_f32("model.language_model.embed_tokens.weight",
                                           (int64_t)vocab_size_ * hidden_size_);
    if (!embed_tokens_) return false;

    if (tie_word_embeddings_) {
        lm_head_ = embed_tokens_;
    } else {
        lm_head_ = sf->load_bf16_to_f32("lm_head.weight", (int64_t)vocab_size_ * hidden_size_);
        if (!lm_head_) return false;
    }

    final_norm_ = sf->load_norm_weight("model.language_model.norm.weight", hidden_size_);
    if (!final_norm_) return false;

    for (int L = 0; L < num_layers_; L++) {
        auto& lw = layers_[L];
        lw.type = layer_types_[L];

        snprintf(name, sizeof(name), "model.language_model.layers.%d.input_layernorm.weight", L);
        lw.input_layernorm = sf->load_norm_weight(name, hidden_size_);
        if (!lw.input_layernorm) return false;

        snprintf(name, sizeof(name), "model.language_model.layers.%d.post_attention_layernorm.weight", L);
        lw.post_attention_layernorm = sf->load_norm_weight(name, hidden_size_);
        if (!lw.post_attention_layernorm) return false;

        if (lw.type == LayerType::LinearAttention) {
            auto& dw = lw.deltanet;

            // Note: in_proj_a/b, A_log, dt_bias use linear_num_value_heads (not key_heads)
            snprintf(name, sizeof(name), "model.language_model.layers.%d.linear_attn.in_proj_a.weight", L);
            dw.in_proj_a = sf->load_bf16_to_f32(name, (int64_t)lin_num_val_heads_ * hidden_size_);

            snprintf(name, sizeof(name), "model.language_model.layers.%d.linear_attn.in_proj_b.weight", L);
            dw.in_proj_b = sf->load_bf16_to_f32(name, (int64_t)lin_num_val_heads_ * hidden_size_);

            snprintf(name, sizeof(name), "model.language_model.layers.%d.linear_attn.in_proj_z.weight", L);
            dw.in_proj_z = sf->load_bf16_to_f32(name, (int64_t)lin_total_val_ * hidden_size_);

            snprintf(name, sizeof(name), "model.language_model.layers.%d.linear_attn.conv1d.weight", L);
            dw.conv1d_w = sf->load_bf16_to_f32(name, (int64_t)lin_qkv_dim_ * conv_kernel_);

            snprintf(name, sizeof(name), "model.language_model.layers.%d.linear_attn.A_log", L);
            dw.A = sf->load_f32_direct(name, lin_num_val_heads_);
            if (dw.A) {
                for (int i = 0; i < lin_num_val_heads_; i++) dw.A[i] = expf(dw.A[i]);
            }

            snprintf(name, sizeof(name), "model.language_model.layers.%d.linear_attn.dt_bias", L);
            dw.dt_bias = sf->load_bf16_to_f32(name, lin_num_val_heads_);

            snprintf(name, sizeof(name), "model.language_model.layers.%d.linear_attn.norm.weight", L);
            dw.norm_w = sf->load_f32_direct(name, lin_val_dim_);

            if (!dw.in_proj_a || !dw.in_proj_b || !dw.in_proj_z || !dw.conv1d_w ||
                !dw.A || !dw.dt_bias || !dw.norm_w) {
                fprintf(stderr, "Failed to load DeltaNet weights for layer %d\n", L);
                return false;
            }
        } else {
            auto& fw = lw.full_attn;

            snprintf(name, sizeof(name), "model.language_model.layers.%d.self_attn.q_norm.weight", L);
            fw.q_norm = sf->load_norm_weight(name, head_dim_);

            snprintf(name, sizeof(name), "model.language_model.layers.%d.self_attn.k_norm.weight", L);
            fw.k_norm = sf->load_norm_weight(name, head_dim_);

            if (!fw.q_norm || !fw.k_norm) {
                fprintf(stderr, "Failed to load FullAttn weights for layer %d\n", L);
                return false;
            }
        }
    }

    // Load CPU float32 weight copies for GEMM-based prefill
    cpu_weights_.resize(num_layers_);
    for (int L = 0; L < num_layers_; L++) {
        auto& cw = cpu_weights_[L];
        char n1[256], n2[256], n3[256];

        if (layer_types_[L] == LayerType::LinearAttention) {
            snprintf(n1, sizeof(n1), "model.language_model.layers.%d.linear_attn.in_proj_qkv.weight", L);
            cw.first_proj = sf->load_bf16_to_f32(n1, (int64_t)lin_qkv_dim_ * hidden_size_);
            cw.first_proj_rows = lin_qkv_dim_;

            snprintf(n1, sizeof(n1), "model.language_model.layers.%d.linear_attn.in_proj_z.weight", L);
            cw.first_proj_b = sf->load_bf16_to_f32(n1, (int64_t)lin_total_val_ * hidden_size_);
            cw.first_proj_b_rows = lin_total_val_;

            snprintf(n1, sizeof(n1), "model.language_model.layers.%d.linear_attn.out_proj.weight", L);
            cw.o_proj = sf->load_bf16_to_f32(n1, (int64_t)hidden_size_ * lin_total_val_);
            cw.o_proj_in = lin_total_val_;
        } else {
            // Full attention: load q/k/v separately and concatenate
            snprintf(n1, sizeof(n1), "model.language_model.layers.%d.self_attn.q_proj.weight", L);
            snprintf(n2, sizeof(n2), "model.language_model.layers.%d.self_attn.k_proj.weight", L);
            snprintf(n3, sizeof(n3), "model.language_model.layers.%d.self_attn.v_proj.weight", L);
            int proj_rows = full_q_dim_ + 2 * full_kv_dim_;
            cw.first_proj = (float*)calloc((size_t)proj_rows * hidden_size_, sizeof(float));
            float* q_w = sf->load_bf16_to_f32(n1, (int64_t)full_q_dim_ * hidden_size_);
            float* k_w = sf->load_bf16_to_f32(n2, (int64_t)full_kv_dim_ * hidden_size_);
            float* v_w = sf->load_bf16_to_f32(n3, (int64_t)full_kv_dim_ * hidden_size_);
            if (q_w && k_w && v_w && cw.first_proj) {
                memcpy(cw.first_proj, q_w, (size_t)full_q_dim_ * hidden_size_ * sizeof(float));
                memcpy(cw.first_proj + (size_t)full_q_dim_ * hidden_size_, k_w, (size_t)full_kv_dim_ * hidden_size_ * sizeof(float));
                memcpy(cw.first_proj + (size_t)(full_q_dim_ + full_kv_dim_) * hidden_size_, v_w, (size_t)full_kv_dim_ * hidden_size_ * sizeof(float));
            }
            free(q_w); free(k_w); free(v_w);
            cw.first_proj_rows = proj_rows;

            snprintf(n1, sizeof(n1), "model.language_model.layers.%d.self_attn.o_proj.weight", L);
            cw.o_proj = sf->load_bf16_to_f32(n1, (int64_t)hidden_size_ * full_out_dim_);
            cw.o_proj_in = full_out_dim_;
        }

        snprintf(n1, sizeof(n1), "model.language_model.layers.%d.mlp.gate_proj.weight", L);
        snprintf(n2, sizeof(n2), "model.language_model.layers.%d.mlp.up_proj.weight", L);
        snprintf(n3, sizeof(n3), "model.language_model.layers.%d.mlp.down_proj.weight", L);
        cw.gate_proj = sf->load_bf16_to_f32(n1, (int64_t)intermediate_size_ * hidden_size_);
        cw.up_proj = sf->load_bf16_to_f32(n2, (int64_t)intermediate_size_ * hidden_size_);
        cw.down_proj = sf->load_bf16_to_f32(n3, (int64_t)hidden_size_ * intermediate_size_);

        if (!cw.first_proj || !cw.o_proj || !cw.gate_proj || !cw.up_proj || !cw.down_proj) {
            fprintf(stderr, "Warning: failed to load CPU weights for layer %d, CPU prefill unavailable\n", L);
        }
    }
    cpu_lm_head_ = lm_head_;  // already f32 from load_bf16_to_f32 above

    LOG("All weights loaded successfully (including CPU prefill copies)\n");
    return true;
}

// Convert tensor name to blob path: "a.b.c" → "<dir>/a/b/c.bin"
static std::string blob_path(const std::string& dir, const char* tensor_name) {
    std::string p = dir + "/";
    for (const char* c = tensor_name; *c; c++) {
        p += (*c == '.') ? '/' : *c;
    }
    p += ".bin";
    return p;
}

bool Qwen35Model::compile_ane(ModelWeights* sf, const std::string& blob_dir) {
    if (!ane_available()) {
        fprintf(stderr, "ANE not available, cannot run\n");
        return false;
    }

    bool use_blobs = !blob_dir.empty();
    LOG("Compiling ANE kernels%s...\n", use_blobs ? " (from blobs)" : "");
    char name[256], name2[256], name3[256];

    for (int L = 0; L < num_layers_; L++) {
        LOG("  Layer %d/%d (%s)...\r", L+1, num_layers_,
            layer_types_[L] == LayerType::LinearAttention ? "deltanet" : "full_attn");

        if (layer_types_[L] == LayerType::LinearAttention) {
            snprintf(name, sizeof(name), "model.language_model.layers.%d.linear_attn.in_proj_qkv.weight", L);
            snprintf(name2, sizeof(name2), "model.language_model.layers.%d.linear_attn.in_proj_z.weight", L);

            if (qwen35_linear_z_on_cpu()) {
                ane_layers_[L].first_proj = ane_compile_matmul(
                    sf->get_bf16_ptr(name), lin_qkv_dim_, hidden_size_);
            } else if (use_blobs) {
                ane_layers_[L].first_proj = ane_compile_fused_2_blob(
                    blob_path(blob_dir, name), lin_qkv_dim_,
                    blob_path(blob_dir, name2), lin_total_val_, hidden_size_);
            } else {
                ane_layers_[L].first_proj = ane_compile_fused_2(
                    sf->get_bf16_ptr(name), lin_qkv_dim_,
                    sf->get_bf16_ptr(name2), lin_total_val_, hidden_size_);
            }
        } else {
            snprintf(name, sizeof(name), "model.language_model.layers.%d.self_attn.q_proj.weight", L);
            snprintf(name2, sizeof(name2), "model.language_model.layers.%d.self_attn.k_proj.weight", L);
            snprintf(name3, sizeof(name3), "model.language_model.layers.%d.self_attn.v_proj.weight", L);

            if (use_blobs) {
                ane_layers_[L].first_proj = ane_compile_fused_3_blob(
                    blob_path(blob_dir, name), full_q_dim_,
                    blob_path(blob_dir, name2), full_kv_dim_,
                    blob_path(blob_dir, name3), full_kv_dim_, hidden_size_);
            } else {
                ane_layers_[L].first_proj = ane_compile_fused_3(
                    sf->get_bf16_ptr(name), full_q_dim_,
                    sf->get_bf16_ptr(name2), full_kv_dim_,
                    sf->get_bf16_ptr(name3), full_kv_dim_, hidden_size_);
            }
        }

        if (!ane_layers_[L].first_proj) {
            fprintf(stderr, "ANE first_proj compile failed for layer %d\n", L);
            return false;
        }

        // O projection + residual add
        int attn_dim;
        if (layer_types_[L] == LayerType::LinearAttention) {
            snprintf(name, sizeof(name), "model.language_model.layers.%d.linear_attn.out_proj.weight", L);
            attn_dim = lin_total_val_;
        } else {
            snprintf(name, sizeof(name), "model.language_model.layers.%d.self_attn.o_proj.weight", L);
            attn_dim = full_out_dim_;
        }
        if (use_blobs) {
            ane_layers_[L].oproj_add = ane_compile_oproj_add_blob(blob_path(blob_dir, name), hidden_size_, attn_dim);
        } else {
            ane_layers_[L].oproj_add = ane_compile_oproj_add(sf->get_bf16_ptr(name), hidden_size_, attn_dim);
        }
        if (!ane_layers_[L].oproj_add) {
            fprintf(stderr, "ANE oproj_add compile failed for layer %d\n", L);
            return false;
        }

        // Fused FFN — try single kernel first, fall back to chunked
        snprintf(name, sizeof(name), "model.language_model.layers.%d.mlp.gate_proj.weight", L);
        snprintf(name2, sizeof(name2), "model.language_model.layers.%d.mlp.up_proj.weight", L);
        snprintf(name3, sizeof(name3), "model.language_model.layers.%d.mlp.down_proj.weight", L);

        int ffn_chunks = ane_ffn_chunk_count(hidden_size_, intermediate_size_);
        if (ffn_chunks <= 1) {
            // Try fused FFN + residual add first
            if (qwen35_use_merged_ffn_gu()) {
                ane_layers_[L].ffn_resadd = ane_compile_fused_ffn_resadd_merged_gu(
                    sf->get_bf16_ptr(name), sf->get_bf16_ptr(name2),
                    sf->get_bf16_ptr(name3), hidden_size_, intermediate_size_);
            } else if (use_blobs) {
                ane_layers_[L].ffn_resadd = ane_compile_fused_ffn_resadd_blob(
                    blob_path(blob_dir, name), blob_path(blob_dir, name2),
                    blob_path(blob_dir, name3), hidden_size_, intermediate_size_);
            } else {
                ane_layers_[L].ffn_resadd = ane_compile_fused_ffn_resadd(
                    sf->get_bf16_ptr(name), sf->get_bf16_ptr(name2),
                    sf->get_bf16_ptr(name3), hidden_size_, intermediate_size_);
            }
            if (!ane_layers_[L].ffn_resadd) {
                if (qwen35_use_merged_ffn_gu()) {
                    ane_layers_[L].fused_ffn = ane_compile_fused_ffn_merged_gu(
                        sf->get_bf16_ptr(name), sf->get_bf16_ptr(name2),
                        sf->get_bf16_ptr(name3), hidden_size_, intermediate_size_);
                } else if (use_blobs) {
                    ane_layers_[L].fused_ffn = ane_compile_fused_ffn_blob(
                        blob_path(blob_dir, name), blob_path(blob_dir, name2),
                        blob_path(blob_dir, name3), hidden_size_, intermediate_size_);
                } else {
                    ane_layers_[L].fused_ffn = ane_compile_fused_ffn(
                        sf->get_bf16_ptr(name), sf->get_bf16_ptr(name2),
                        sf->get_bf16_ptr(name3), hidden_size_, intermediate_size_);
                }
            }
        }
        if (!ane_layers_[L].ffn_resadd && !ane_layers_[L].fused_ffn) {
            // Fall back to chunked FFN
            if (ffn_chunks <= 1) ffn_chunks = ane_ffn_chunk_count(hidden_size_, intermediate_size_);
            if (ffn_chunks <= 1) ffn_chunks = 2; // force chunking
            if (L == 0) LOG("  Using chunked FFN (%d chunks, inter=%d)\n", ffn_chunks, intermediate_size_);
            if (!use_blobs) {
                if (!ane_compile_chunked_ffn(&ane_layers_[L].chunked_ffn,
                        sf->get_bf16_ptr(name), sf->get_bf16_ptr(name2),
                        sf->get_bf16_ptr(name3), hidden_size_, intermediate_size_, ffn_chunks)) {
                    fprintf(stderr, "ANE chunked FFN compile failed for layer %d\n", L);
                    return false;
                }
            } else {
                if (!ane_compile_chunked_ffn_blob(&ane_layers_[L].chunked_ffn,
                        blob_path(blob_dir, name), blob_path(blob_dir, name2),
                        blob_path(blob_dir, name3), hidden_size_, intermediate_size_, ffn_chunks)) {
                    fprintf(stderr, "ANE chunked FFN blob compile failed for layer %d\n", L);
                    return false;
                }
            }
        }
    }

    int compiled = ane_compile_count();
    int cached = ane_cache_loads();
    LOG("  %d ANE layer kernels ready (compiled=%d, cached=%d)\n",
        compiled + cached, compiled, cached);

    // Compile LM head
    if (!compile_lm_head_ane(sf, blob_dir)) {
        LOG("ANE LM head disabled, falling back to CPU\n");
    } else {
        LOG("  LM head ANE enabled (%d chunks)\n", (int)lm_head_kernels_.size());
    }

    return true;
}

bool Qwen35Model::compile_lm_head_ane(ModelWeights* sf, const std::string& blob_dir) {
    bool use_blobs = !blob_dir.empty();
    const char* lm_name = tie_word_embeddings_ ? "model.language_model.embed_tokens.weight" : "lm_head.weight";

    const uint16_t* lm_bf16 = sf->get_bf16_ptr(lm_name);
    if (!lm_bf16) {
        fprintf(stderr, "ANE LM head: missing BF16 weights for %s\n", lm_name);
        return false;
    }

    int chunk = lm_head_chunk_;
    if (chunk > vocab_size_) chunk = vocab_size_;

    int chunks = (vocab_size_ + chunk - 1) / chunk;
    lm_head_kernels_.resize(chunks, nullptr);

    LOG("  LM head ANE: compiling %d chunks (chunk=%d)\n", chunks, chunk);
    for (int c = 0; c < chunks; c++) {
        int offset = c * chunk;
        int rows = vocab_size_ - offset;
        if (rows > chunk) rows = chunk;

        LOG("    LM head chunk %d/%d...\r", c + 1, chunks);

        if (use_blobs) {
            // LM head is chunked dynamically, so keep using the BF16 source tensor here.
            (void)use_blobs;
        }
        const uint16_t* chunk_w = lm_bf16 + (int64_t)offset * hidden_size_;
        lm_head_kernels_[c] = ane_compile_matmul(chunk_w, rows, hidden_size_);
        if (!lm_head_kernels_[c]) {
            fprintf(stderr, "\nANE LM head: compile failed at chunk %d/%d\n", c + 1, chunks);
            free_lm_head_ane();
            return false;
        }
    }
    LOG("    LM head chunk %d/%d done          \n", chunks, chunks);
    ane_lm_head_enabled_ = true;
    lm_head_chunk_ = chunk;
    return true;
}

void Qwen35Model::free_lm_head_ane() {
    for (auto* k : lm_head_kernels_) ane_free(k);
    lm_head_kernels_.clear();
    ane_lm_head_enabled_ = false;
}

bool Qwen35Model::forward_deltanet_core(Session& session, int L, float* x, float* pre_oproj, double* out_ane_ms) {
    auto& dw = layers_[L].deltanet;

    float* qkv_z = session.scratch_qkv;
    float* mixed_qkv = qkv_z;
    float* z = qkv_z + lin_qkv_dim_;
    Timer t_ane;
    if (qwen35_linear_z_on_cpu()) {
        if (!ane_matvec(ane_layers_[L].first_proj, mixed_qkv, x,
                        hidden_size_, lin_qkv_dim_)) {
            fprintf(stderr, "ANE first_proj eval failed at layer %d (DeltaNet qkv)\n", L);
            return false;
        }
        if (out_ane_ms) *out_ane_ms += t_ane.elapsed_ms();
        matvec(z, dw.in_proj_z, x, lin_total_val_, hidden_size_);
    } else {
        if (!ane_matvec(ane_layers_[L].first_proj, qkv_z, x,
                        hidden_size_, lin_qkv_dim_ + lin_total_val_)) {
            fprintf(stderr, "ANE first_proj eval failed at layer %d (DeltaNet)\n", L);
            return false;
        }
        if (out_ane_ms) *out_ane_ms += t_ane.elapsed_ms();
    }

    float* a_vec = session.scratch_tmp;
    float* b_vec = session.scratch_tmp + lin_num_val_heads_;
    matvec(a_vec, dw.in_proj_a, x, lin_num_val_heads_, hidden_size_);
    matvec(b_vec, dw.in_proj_b, x, lin_num_val_heads_, hidden_size_);

    float* conv_out = session.scratch_conv;
    conv1d_update(conv_out, session.delta_conv_state[L], &session.delta_conv_pos[L], mixed_qkv, dw.conv1d_w, lin_qkv_dim_, conv_kernel_);
    silu_vec_inplace(conv_out, lin_qkv_dim_, session.scratch_tmp + lin_num_val_heads_ * 2);

    float* Q = conv_out;
    float* K = conv_out + lin_total_key_;
    float* V = conv_out + lin_total_key_ * 2;

    float* y = session.scratch_y;
    float q_scale = 1.0f / sqrtf((float)lin_key_dim_);
    int val_heads_per_key = lin_num_val_heads_ / lin_num_heads_;

    for (int kh = 0; kh < lin_num_heads_; kh++) {
        float* qh = Q + kh * lin_key_dim_;
        float* kh_ptr = K + kh * lin_key_dim_;

        l2_normalize(qh, lin_key_dim_);
        l2_normalize(kh_ptr, lin_key_dim_);
        float qs = q_scale;
        vDSP_vsmul(qh, 1, &qs, qh, 1, (vDSP_Length)lin_key_dim_);

        for (int vsub = 0; vsub < val_heads_per_key; vsub++) {
            int vh = kh * val_heads_per_key + vsub;
            float* vh_ptr = V + vh * lin_val_dim_;
            float* yh = y + vh * lin_val_dim_;
            float* state = session.delta_ssm_state[L] + vh * lin_key_dim_ * lin_val_dim_;

            float beta = sigmoid_f(b_vec[vh]);
            float decay = expf(-dw.A[vh] * softplus_f(a_vec[vh] + dw.dt_bias[vh]));
            ssm_step(yh, state, qh, kh_ptr, vh_ptr, decay, beta, lin_key_dim_, lin_val_dim_);
        }
    }

    if (qwen35_batch_rmsgate()) {
        rmsnorm_gated_repeated(pre_oproj, y, z, dw.norm_w,
                               lin_num_val_heads_, lin_val_dim_,
                               session.scratch_tmp);
    } else {
        for (int h = 0; h < lin_num_val_heads_; h++) {
            rmsnorm_gated(pre_oproj + h * lin_val_dim_,
                          y + h * lin_val_dim_,
                          z + h * lin_val_dim_,
                          dw.norm_w, lin_val_dim_);
        }
    }
    return true;
}

bool Qwen35Model::forward_full_attn_core(Session& session, int L, float* x, float* pre_oproj, int pos, double* out_ane_ms) {
    auto& fw = layers_[L].full_attn;

    float* qkv_buf = session.scratch_qkv;
    Timer t_ane;
    if (!ane_matvec(ane_layers_[L].first_proj, qkv_buf, x,
                    hidden_size_, full_q_dim_ + full_kv_dim_ * 2)) {
        fprintf(stderr, "ANE first_proj eval failed at layer %d (FullAttn)\n", L);
        return false;
    }
    if (out_ane_ms) *out_ane_ms += t_ane.elapsed_ms();

    float* q_gate_raw = qkv_buf;
    float* k_raw = qkv_buf + full_q_dim_;
    float* v_raw = qkv_buf + full_q_dim_ + full_kv_dim_;

    for (int h = 0; h < num_q_heads_; h++) {
        float* qh = q_gate_raw + (size_t)h * head_dim_ * 2;
        rmsnorm(qh, qh, fw.q_norm, head_dim_, rms_eps_);
    }
    for (int h = 0; h < num_kv_heads_; h++) {
        rmsnorm(k_raw + h * head_dim_, k_raw + h * head_dim_, fw.k_norm, head_dim_, rms_eps_);
    }

    const float* rope_cos_row = nullptr;
    const float* rope_sin_row = nullptr;
    if (pos >= 0 && pos < MAX_SEQ_LEN && rope_cos_ && rope_sin_) {
        int half_rot = rot_dim_ / 2;
        rope_cos_row = rope_cos_ + (size_t)pos * half_rot;
        rope_sin_row = rope_sin_ + (size_t)pos * half_rot;
    }
    apply_rope_cached(q_gate_raw, k_raw, num_q_heads_, num_kv_heads_,
                      head_dim_, head_dim_ * 2, head_dim_, rot_dim_, pos, rope_theta_,
                      rope_cos_row, rope_sin_row);

    int slot;
    if (session.kv_len[L] < KV_CACHE_CAPACITY) {
        slot = session.kv_start[L] + session.kv_len[L];
        if (slot >= KV_CACHE_CAPACITY) slot -= KV_CACHE_CAPACITY;
        session.kv_len[L]++;
    } else {
        slot = session.kv_start[L];
        session.kv_start[L]++;
        if (session.kv_start[L] >= KV_CACHE_CAPACITY) session.kv_start[L] = 0;
    }
    size_t kv_stride = (size_t)num_kv_heads_ * head_dim_;
    memcpy(session.kv_k_cache[L] + (size_t)slot * kv_stride, k_raw, kv_stride * sizeof(float));
    memcpy(session.kv_v_cache[L] + (size_t)slot * kv_stride, v_raw, kv_stride * sizeof(float));

    gqa_attention(pre_oproj, q_gate_raw, session.kv_k_cache[L], session.kv_v_cache[L],
                  num_q_heads_, num_kv_heads_, head_dim_, head_dim_ * 2,
                  session.kv_start[L], session.kv_len[L], KV_CACHE_CAPACITY);

    if (attn_output_gate_) {
        for (int h = 0; h < num_q_heads_; h++) {
            float* oh = pre_oproj + h * head_dim_;
            const float* gh = q_gate_raw + (size_t)h * head_dim_ * 2 + head_dim_;
            mul_sigmoid_inplace(oh, gh, head_dim_, session.scratch_tmp);
        }
    }
    return true;
}

float* Qwen35Model::prefill(const std::vector<int>& token_ids, int start_pos) {
    if (!default_session_) return nullptr;
    return prefill(*default_session_, token_ids, start_pos);
}

float* Qwen35Model::prefill(Session& session, const std::vector<int>& token_ids, int start_pos) {
    if (token_ids.empty()) return nullptr;

    // Auto-select CPU prefill for long prompts (GEMM >> ANE matvec loops)
    static const char* cpu_env = getenv("ANE_CPU_PREFILL");
    bool cpu_prefill_enabled = cpu_env ? (atoi(cpu_env) != 0) : true;  // on by default
    int cpu_threshold = 32;  // tokens above this use CPU
    if (cpu_prefill_enabled && (int)token_ids.size() > cpu_threshold && !cpu_weights_.empty() && cpu_weights_[0].first_proj) {
        return prefill_cpu(session, token_ids, start_pos);
    }

    int batch_size = qwen35_prefill_batch_size();
    if (batch_size <= 1 || token_ids.size() == 1) {
        float* logits = nullptr;
        for (int i = 0; i < (int)token_ids.size(); i++) {
            logits = forward(session, token_ids[i], start_pos + i);
            if (!logits) return nullptr;
        }
        return logits;
    }

    const int max_proj_dim = std::max(lin_qkv_dim_ + lin_total_val_, full_q_dim_ + 2 * full_kv_dim_);
    const int max_attn_dim = std::max(lin_total_val_, full_out_dim_);
    std::vector<float> x_batch((size_t)batch_size * hidden_size_);
    std::vector<float> x_norm_batch((size_t)batch_size * hidden_size_);
    std::vector<float> proj_batch((size_t)batch_size * max_proj_dim);
    std::vector<float> pre_oproj_batch((size_t)batch_size * max_attn_dim);
    std::vector<float> mlp_batch((size_t)batch_size * hidden_size_);
    std::vector<float> packed0;
    std::vector<float> packed1;
    std::vector<float> raw_out;

    float* last_hidden = nullptr;

    for (int base = 0; base < (int)token_ids.size(); base += batch_size) {
        int batch = std::min(batch_size, (int)token_ids.size() - base);
        for (int b = 0; b < batch; b++) {
            memcpy(x_batch.data() + (size_t)b * hidden_size_,
                   embed_tokens_ + (int64_t)token_ids[(size_t)base + b] * hidden_size_,
                   (size_t)hidden_size_ * sizeof(float));
        }

        for (int L = 0; L < num_layers_; L++) {
            for (int b = 0; b < batch; b++) {
                rmsnorm(x_norm_batch.data() + (size_t)b * hidden_size_,
                        x_batch.data() + (size_t)b * hidden_size_,
                        layers_[L].input_layernorm, hidden_size_, rms_eps_);
            }

            const bool is_linear = layer_types_[L] == LayerType::LinearAttention;
            const int proj_dim = is_linear
                ? (qwen35_linear_z_on_cpu() ? lin_qkv_dim_ : (lin_qkv_dim_ + lin_total_val_))
                : (full_q_dim_ + 2 * full_kv_dim_);
            for (int b = 0; b < batch; b++) {
                if (!ane_matvec(ane_layers_[L].first_proj,
                                proj_batch.data() + (size_t)b * max_proj_dim,
                                x_norm_batch.data() + (size_t)b * hidden_size_,
                                hidden_size_, proj_dim)) {
                    fprintf(stderr, "ANE first_proj eval failed at layer %d\n", L);
                    return nullptr;
                }
            }

            for (int b = 0; b < batch; b++) {
                float* x_norm = x_norm_batch.data() + (size_t)b * hidden_size_;
                float* proj = proj_batch.data() + (size_t)b * max_proj_dim;
                float* pre_oproj = pre_oproj_batch.data() + (size_t)b * max_attn_dim;
                int pos = start_pos + base + b;

                if (is_linear) {
                    auto& dw = layers_[L].deltanet;

                    float* mixed_qkv = proj;
                    float* z = proj + lin_qkv_dim_;
                    if (qwen35_linear_z_on_cpu()) {
                        matvec(z, dw.in_proj_z, x_norm, lin_total_val_, hidden_size_);
                    }

                    float* a_vec = session.scratch_tmp;
                    float* b_vec = session.scratch_tmp + lin_num_val_heads_;
                    matvec(a_vec, dw.in_proj_a, x_norm, lin_num_val_heads_, hidden_size_);
                    matvec(b_vec, dw.in_proj_b, x_norm, lin_num_val_heads_, hidden_size_);

                    float* conv_out = session.scratch_conv;
                    conv1d_update(conv_out, session.delta_conv_state[L], &session.delta_conv_pos[L], mixed_qkv, dw.conv1d_w, lin_qkv_dim_, conv_kernel_);
                    silu_vec_inplace(conv_out, lin_qkv_dim_, session.scratch_tmp + lin_num_val_heads_ * 2);

                    float* Q = conv_out;
                    float* K = conv_out + lin_total_key_;
                    float* V = conv_out + lin_total_key_ * 2;
                    float* y = session.scratch_y;
                    float q_scale = 1.0f / sqrtf((float)lin_key_dim_);
                    int val_heads_per_key = lin_num_val_heads_ / lin_num_heads_;

                    for (int kh = 0; kh < lin_num_heads_; kh++) {
                        float* qh = Q + kh * lin_key_dim_;
                        float* kh_ptr = K + kh * lin_key_dim_;

                        l2_normalize(qh, lin_key_dim_);
                        l2_normalize(kh_ptr, lin_key_dim_);
                        float qs = q_scale;
                        vDSP_vsmul(qh, 1, &qs, qh, 1, (vDSP_Length)lin_key_dim_);

                        for (int vsub = 0; vsub < val_heads_per_key; vsub++) {
                            int vh = kh * val_heads_per_key + vsub;
                            float* vh_ptr = V + vh * lin_val_dim_;
                            float* yh = y + vh * lin_val_dim_;
                            float* state = session.delta_ssm_state[L] + (size_t)vh * lin_key_dim_ * lin_val_dim_;

                            float beta = sigmoid_f(b_vec[vh]);
                            float decay = expf(-dw.A[vh] * softplus_f(a_vec[vh] + dw.dt_bias[vh]));
                            ssm_step(yh, state, qh, kh_ptr, vh_ptr, decay, beta, lin_key_dim_, lin_val_dim_);
                        }
                    }

                    if (qwen35_batch_rmsgate()) {
                        rmsnorm_gated_repeated(pre_oproj, y, z, dw.norm_w,
                                               lin_num_val_heads_, lin_val_dim_,
                                               session.scratch_tmp);
                    } else {
                        for (int h = 0; h < lin_num_val_heads_; h++) {
                            rmsnorm_gated(pre_oproj + h * lin_val_dim_,
                                          y + h * lin_val_dim_,
                                          z + h * lin_val_dim_,
                                          dw.norm_w, lin_val_dim_);
                        }
                    }
                } else {
                    auto& fw = layers_[L].full_attn;

                    float* q_gate_raw = proj;
                    float* k_raw = proj + full_q_dim_;
                    float* v_raw = proj + full_q_dim_ + full_kv_dim_;

                    for (int h = 0; h < num_q_heads_; h++) {
                        float* qh = q_gate_raw + (size_t)h * head_dim_ * 2;
                        rmsnorm(qh, qh, fw.q_norm, head_dim_, rms_eps_);
                    }
                    for (int h = 0; h < num_kv_heads_; h++) {
                        rmsnorm(k_raw + h * head_dim_, k_raw + h * head_dim_, fw.k_norm, head_dim_, rms_eps_);
                    }

                    const float* rope_cos_row = nullptr;
                    const float* rope_sin_row = nullptr;
                    if (pos >= 0 && pos < MAX_SEQ_LEN && rope_cos_ && rope_sin_) {
                        int half_rot = rot_dim_ / 2;
                        rope_cos_row = rope_cos_ + (size_t)pos * half_rot;
                        rope_sin_row = rope_sin_ + (size_t)pos * half_rot;
                    }
                    apply_rope_cached(q_gate_raw, k_raw, num_q_heads_, num_kv_heads_,
                                      head_dim_, head_dim_ * 2, head_dim_, rot_dim_, pos, rope_theta_,
                                      rope_cos_row, rope_sin_row);

                    int slot;
                    if (session.kv_len[L] < KV_CACHE_CAPACITY) {
                        slot = session.kv_start[L] + session.kv_len[L];
                        if (slot >= KV_CACHE_CAPACITY) slot -= KV_CACHE_CAPACITY;
                        session.kv_len[L]++;
                    } else {
                        slot = session.kv_start[L];
                        session.kv_start[L]++;
                        if (session.kv_start[L] >= KV_CACHE_CAPACITY) session.kv_start[L] = 0;
                    }
                    size_t kv_stride = (size_t)num_kv_heads_ * head_dim_;
                    memcpy(session.kv_k_cache[L] + (size_t)slot * kv_stride, k_raw, kv_stride * sizeof(float));
                    memcpy(session.kv_v_cache[L] + (size_t)slot * kv_stride, v_raw, kv_stride * sizeof(float));

                    gqa_attention(pre_oproj, q_gate_raw, session.kv_k_cache[L], session.kv_v_cache[L],
                                  num_q_heads_, num_kv_heads_, head_dim_, head_dim_ * 2,
                                  session.kv_start[L], session.kv_len[L], KV_CACHE_CAPACITY);

                    if (attn_output_gate_) {
                        for (int h = 0; h < num_q_heads_; h++) {
                            float* oh = pre_oproj + h * head_dim_;
                            const float* gh = q_gate_raw + (size_t)h * head_dim_ * 2 + head_dim_;
                            mul_sigmoid_inplace(oh, gh, head_dim_, session.scratch_tmp);
                        }
                    }
                }
            }

            const int attn_dim = is_linear ? lin_total_val_ : full_out_dim_;
            if (ane_layers_[L].oproj_add) {
                if (!ane_binary_batch(ane_layers_[L].oproj_add,
                                      attn_dim, hidden_size_, hidden_size_,
                                      x_batch.data(),
                                      pre_oproj_batch.data(), x_batch.data(), batch,
                                      packed0, packed1, raw_out)) {
                    fprintf(stderr, "ANE oproj_add batched eval failed at layer %d\n", L);
                    return nullptr;
                }
            } else {
                for (int b = 0; b < batch; b++) {
                    float* x = x_batch.data() + (size_t)b * hidden_size_;
                    float* pre_oproj = pre_oproj_batch.data() + (size_t)b * max_attn_dim;
                    float* attn_out = mlp_batch.data() + (size_t)b * hidden_size_;
                    if (!ane_matvec(ane_layers_[L].o_proj, attn_out, pre_oproj, attn_dim, hidden_size_)) {
                        fprintf(stderr, "ANE o_proj eval failed at layer %d\n", L);
                        return nullptr;
                    }
                    for (int i = 0; i < hidden_size_; i++) x[i] += attn_out[i];
                }
            }

            for (int b = 0; b < batch; b++) {
                rmsnorm(x_norm_batch.data() + (size_t)b * hidden_size_,
                        x_batch.data() + (size_t)b * hidden_size_,
                        layers_[L].post_attention_layernorm, hidden_size_, rms_eps_);
            }

            if (ane_layers_[L].ffn_resadd) {
                if (!ane_binary_batch(ane_layers_[L].ffn_resadd,
                                      hidden_size_, hidden_size_, hidden_size_,
                                      x_batch.data(),
                                      x_norm_batch.data(), x_batch.data(), batch,
                                      packed0, packed1, raw_out)) {
                    fprintf(stderr, "ANE ffn_resadd batched eval failed at layer %d\n", L);
                    return nullptr;
                }
            } else if (ane_layers_[L].fused_ffn) {
                if (!ane_matvec_batch(ane_layers_[L].fused_ffn,
                                      mlp_batch.data(), x_norm_batch.data(),
                                      batch, hidden_size_, hidden_size_, packed0, raw_out)) {
                    fprintf(stderr, "ANE fused_ffn batched eval failed at layer %d\n", L);
                    return nullptr;
                }
                for (int b = 0; b < batch; b++) {
                    float* x = x_batch.data() + (size_t)b * hidden_size_;
                    float* mlp = mlp_batch.data() + (size_t)b * hidden_size_;
                    for (int i = 0; i < hidden_size_; i++) x[i] += mlp[i];
                }
            } else if (ane_layers_[L].chunked_ffn.num_chunks > 0) {
                for (int b = 0; b < batch; b++) {
                    float* x = x_batch.data() + (size_t)b * hidden_size_;
                    float* x_norm = x_norm_batch.data() + (size_t)b * hidden_size_;
                    float* mlp = mlp_batch.data() + (size_t)b * hidden_size_;
                    if (!ane_eval_chunked_ffn(&ane_layers_[L].chunked_ffn, mlp, x_norm)) {
                        fprintf(stderr, "ANE chunked_ffn eval failed at layer %d\n", L);
                        return nullptr;
                    }
                    for (int i = 0; i < hidden_size_; i++) x[i] += mlp[i];
                }
            } else {
                fprintf(stderr, "No FFN kernel for layer %d\n", L);
                return nullptr;
            }
        }

        last_hidden = x_batch.data() + (size_t)(batch - 1) * hidden_size_;
    }

    rmsnorm(session.x, last_hidden, final_norm_, hidden_size_, rms_eps_);

    if (ane_lm_head_enabled_ && !lm_head_kernels_.empty()) {
        bool ok = true;
        int chunks = (int)lm_head_kernels_.size();
        for (int c = 0; c < chunks; c++) {
            int offset = c * lm_head_chunk_;
            int rows = vocab_size_ - offset;
            if (rows > lm_head_chunk_) rows = lm_head_chunk_;
            if (!ane_matvec(lm_head_kernels_[c], session.logits + offset, session.x, hidden_size_, rows)) {
                fprintf(stderr, "ANE LM head eval failed at chunk %d/%d, falling back to CPU\n", c + 1, chunks);
                ok = false;
                break;
            }
        }
        if (!ok) {
            free_lm_head_ane();
            matvec(session.logits, lm_head_, session.x, vocab_size_, hidden_size_);
        }
    } else {
        matvec(session.logits, lm_head_, session.x, vocab_size_, hidden_size_);
    }

    return session.logits;
}



float* Qwen35Model::prefill_cpu(Session& session, const std::vector<int>& token_ids, int start_pos) {
    if (token_ids.empty()) return nullptr;
    const int N = (int)token_ids.size();

    if (cpu_weights_.empty() || !cpu_weights_[0].first_proj) {
        fprintf(stderr, "[cpu_prefill] CPU weights not loaded, falling back to ANE\n");
        return prefill(session, token_ids, start_pos);
    }

    Timer total_timer;

    const int H = hidden_size_;
    const int I = intermediate_size_;
    const int max_proj = std::max(lin_qkv_dim_ + lin_total_val_, full_q_dim_ + 2 * full_kv_dim_);
    const int max_attn = std::max(lin_total_val_, full_out_dim_);

    // Persistent batch buffers
    std::vector<float> X(     (size_t)N * H);
    std::vector<float> X_norm((size_t)N * H);
    std::vector<float> Proj(  (size_t)N * max_proj);
    std::vector<float> Oproj( (size_t)N * max_attn);
    std::vector<float> Attn(  (size_t)N * H);
    std::vector<float> Gate(  (size_t)N * I);
    std::vector<float> Up(    (size_t)N * I);
    std::vector<float> Down(  (size_t)N * H);

    // Pre-allocate batched Z/A/B projection buffers for DeltaNet layers
    std::vector<float> Z_batch((size_t)N * lin_total_val_);
    std::vector<float> A_batch((size_t)N * lin_num_val_heads_);
    std::vector<float> B_batch((size_t)N * lin_num_val_heads_);

    // Batch buffers for parallelized DeltaNet
    std::vector<float> conv_batch((size_t)N * lin_qkv_dim_);  // conv1d output per token
    std::vector<float> Y_batch((size_t)N * lin_total_val_);    // ssm output per token

    // Batch buffers for full-attention prefill
    // Q layout: [N, num_q_heads, head_dim*2] (includes gate), K: [N, num_kv_heads, head_dim], V: same
    std::vector<float> fullQ((size_t)N * full_q_dim_);
    std::vector<float> fullK((size_t)N * full_kv_dim_);
    std::vector<float> fullV((size_t)N * full_kv_dim_);

    // Merged gate+up weight (allocated once, populated per layer)
    std::vector<float> GateUp((size_t)N * 2 * I);

    dispatch_queue_t par_queue = dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0);

    // Embedding lookup
    for (int i = 0; i < N; i++) {
        memcpy(X.data() + (size_t)i * H,
               embed_tokens_ + (int64_t)token_ids[i] * H,
               (size_t)H * sizeof(float));
    }

    double total_gemm_ms = 0, total_attn_ms = 0, total_ffn_ms = 0;
    double delta_attn_ms = 0, full_attn_ms = 0;

    for (int L = 0; L < num_layers_; L++) {
        auto& cw = cpu_weights_[L];

        // 1. RMSNorm
        for (int i = 0; i < N; i++) {
            rmsnorm(X_norm.data() + (size_t)i * H,
                    X.data() + (size_t)i * H,
                    layers_[L].input_layernorm, H, rms_eps_);
        }

        const bool is_linear = layer_types_[L] == LayerType::LinearAttention;
        const int proj_rows = cw.first_proj_rows;

        // 2. Batched projections via GEMM
        Timer gemm_timer;

        // First projection: QKV
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    N, proj_rows, H, 1.0f,
                    X_norm.data(), H,
                    cw.first_proj, H,
                    0.0f, Proj.data(), max_proj);

        if (is_linear) {
            // Batch Z projection: Z_batch = X_norm @ W_z^T  [N×H] × [H×lin_total_val]
            if (cw.first_proj_b) {
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            N, cw.first_proj_b_rows, H, 1.0f,
                            X_norm.data(), H,
                            cw.first_proj_b, H,
                            0.0f, Z_batch.data(), lin_total_val_);
            }

            // Batch A projection: A_batch = X_norm @ W_a^T  [N×H] × [H×num_val_heads]
            auto& dw = layers_[L].deltanet;
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        N, lin_num_val_heads_, H, 1.0f,
                        X_norm.data(), H,
                        dw.in_proj_a, H,
                        0.0f, A_batch.data(), lin_num_val_heads_);

            // Batch B projection: B_batch = X_norm @ W_b^T  [N×H] × [H×num_val_heads]
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        N, lin_num_val_heads_, H, 1.0f,
                        X_norm.data(), H,
                        dw.in_proj_b, H,
                        0.0f, B_batch.data(), lin_num_val_heads_);
        }

        total_gemm_ms += gemm_timer.elapsed_ms();

        // 3. Attention core
        Timer attn_timer;
        if (is_linear) {
            auto& dw = layers_[L].deltanet;
            const int val_heads_per_key = lin_num_val_heads_ / lin_num_heads_;
            const float q_scale = 1.0f / sqrtf((float)lin_key_dim_);

            // Phase 1: Sequential conv1d + silu for all N tokens
            // (conv1d has causal state dependency across tokens)
            for (int i = 0; i < N; i++) {
                float* mixed_qkv = Proj.data() + (size_t)i * max_proj;
                float* conv_out_i = conv_batch.data() + (size_t)i * lin_qkv_dim_;
                conv1d_update(conv_out_i, session.delta_conv_state[L], &session.delta_conv_pos[L],
                              mixed_qkv, dw.conv1d_w, lin_qkv_dim_, conv_kernel_);
                silu_vec_inplace(conv_out_i, lin_qkv_dim_, session.scratch_tmp);
            }

            // Phase 2: Parallel across key heads — each head processes all N tokens
            // Each key head's Q/K/V slices are non-overlapping, states are independent
            const int num_kh = lin_num_heads_;
            const int key_dim = lin_key_dim_;
            const int val_dim = lin_val_dim_;
            const int total_key = lin_total_key_;
            const int total_val = lin_total_val_;
            const int num_vh = lin_num_val_heads_;
            const int qkv_dim = lin_qkv_dim_;
            float* conv_data = conv_batch.data();
            float* y_data = Y_batch.data();
            float* a_data = A_batch.data();
            float* b_data = B_batch.data();
            float* oproj_data = Oproj.data();
            float* z_data = Z_batch.data();
            float** ssm_states = session.delta_ssm_state.data();
            const float* dw_A = dw.A;
            const float* dw_dt_bias = dw.dt_bias;
            const float* dw_norm_w = dw.norm_w;

            dispatch_apply((size_t)num_kh, par_queue, ^(size_t kh_idx) {
                int kh = (int)kh_idx;
                for (int i = 0; i < N; i++) {
                    float* conv_out_i = conv_data + (size_t)i * qkv_dim;
                    float* qh = conv_out_i + kh * key_dim;
                    float* kh_ptr = conv_out_i + total_key + kh * key_dim;

                    l2_normalize(qh, key_dim);
                    l2_normalize(kh_ptr, key_dim);
                    float qs = q_scale;
                    vDSP_vsmul(qh, 1, &qs, qh, 1, (vDSP_Length)key_dim);

                    float* a_vec = a_data + (size_t)i * num_vh;
                    float* b_vec = b_data + (size_t)i * num_vh;

                    for (int vsub = 0; vsub < val_heads_per_key; vsub++) {
                        int vh = kh * val_heads_per_key + vsub;
                        float* vh_ptr = conv_out_i + total_key * 2 + vh * val_dim;
                        float* yh = y_data + (size_t)i * total_val + vh * val_dim;
                        float* state = ssm_states[L] + (size_t)vh * key_dim * val_dim;
                        float beta = sigmoid_f(b_vec[vh]);
                        float decay = expf(-dw_A[vh] * softplus_f(a_vec[vh] + dw_dt_bias[vh]));
                        ssm_step(yh, state, qh, kh_ptr, vh_ptr, decay, beta, key_dim, val_dim);
                    }

                    // rmsnorm_gated for this head's value heads (fused into parallel work)
                    float* pre_oproj = oproj_data + (size_t)i * max_attn;
                    float* z_ptr = z_data + (size_t)i * total_val;
                    for (int vsub = 0; vsub < val_heads_per_key; vsub++) {
                        int vh = kh * val_heads_per_key + vsub;
                        float* yh = y_data + (size_t)i * total_val + vh * val_dim;
                        rmsnorm_gated(pre_oproj + vh * val_dim,
                                      yh,
                                      z_ptr + vh * val_dim,
                                      dw_norm_w, val_dim);
                    }
                }
            });

        } else {
            // Batched full-attention prefill: GEMM-based Q@K^T with causal mask
            auto& fw = layers_[L].full_attn;
            const int hd = head_dim_;
            const int nqh = num_q_heads_;
            const int nkvh = num_kv_heads_;
            const int groups = nqh / nkvh;
            const float scale = 1.0f / sqrtf((float)hd);

            // Step 1: Batch Q/K norms + RoPE for all N tokens
            for (int i = 0; i < N; i++) {
                float* proj = Proj.data() + (size_t)i * max_proj;
                float* q_gate_raw = proj;
                float* k_raw = proj + full_q_dim_;
                int pos = start_pos + i;

                for (int h = 0; h < nqh; h++) {
                    float* qh = q_gate_raw + (size_t)h * hd * 2;
                    rmsnorm(qh, qh, fw.q_norm, hd, rms_eps_);
                }
                for (int h = 0; h < nkvh; h++) {
                    rmsnorm(k_raw + h * hd, k_raw + h * hd, fw.k_norm, hd, rms_eps_);
                }

                const float* rope_cos_row = nullptr;
                const float* rope_sin_row = nullptr;
                if (pos >= 0 && pos < MAX_SEQ_LEN && rope_cos_ && rope_sin_) {
                    int half_rot = rot_dim_ / 2;
                    rope_cos_row = rope_cos_ + (size_t)pos * half_rot;
                    rope_sin_row = rope_sin_ + (size_t)pos * half_rot;
                }
                apply_rope_cached(q_gate_raw, k_raw, nqh, nkvh,
                                  hd, hd * 2, hd, rot_dim_, pos, rope_theta_,
                                  rope_cos_row, rope_sin_row);
            }

            // Step 2: Gather K, V into contiguous [N, nkvh, hd] arrays and
            // populate KV cache. Q stays in Proj (stride max_proj, head stride hd*2).
            // fullK/fullV: [N, nkvh * hd] contiguous
            const size_t kv_stride = (size_t)nkvh * hd;
            for (int i = 0; i < N; i++) {
                float* proj = Proj.data() + (size_t)i * max_proj;
                float* k_raw = proj + full_q_dim_;
                float* v_raw = proj + full_q_dim_ + full_kv_dim_;
                memcpy(fullK.data() + (size_t)i * kv_stride, k_raw, kv_stride * sizeof(float));
                memcpy(fullV.data() + (size_t)i * kv_stride, v_raw, kv_stride * sizeof(float));

                // Insert into KV cache for future decode tokens
                int slot;
                if (session.kv_len[L] < KV_CACHE_CAPACITY) {
                    slot = session.kv_start[L] + session.kv_len[L];
                    if (slot >= KV_CACHE_CAPACITY) slot -= KV_CACHE_CAPACITY;
                    session.kv_len[L]++;
                } else {
                    slot = session.kv_start[L];
                    session.kv_start[L]++;
                    if (session.kv_start[L] >= KV_CACHE_CAPACITY) session.kv_start[L] = 0;
                }
                memcpy(session.kv_k_cache[L] + (size_t)slot * kv_stride, k_raw, kv_stride * sizeof(float));
                memcpy(session.kv_v_cache[L] + (size_t)slot * kv_stride, v_raw, kv_stride * sizeof(float));
            }

            // Step 3: Batched causal attention per KV group
            // For each KV head, we have `groups` Q heads attending to the same K/V.
            // Q layout in Proj: token i, head h → Proj[i*max_proj + h*hd*2], stride between tokens = max_proj
            // K layout in fullK: token i, kv head kvh → fullK[i*kv_stride + kvh*hd], stride = kv_stride
            // V same layout as K.
            //
            // Per KV head: process one Q head at a time (avoids gather into contiguous Q buffer).
            // For each Q head h in the group:
            //   scores[i, j] = scale * sum_d(Q[i,h,d] * K[j,kvh,d])  for j <= i  (causal)
            //   out[i, d] = sum_j(softmax(scores[i,:])[j] * V[j,kvh,d])
            //
            // We compute Q@K^T as sgemm, then apply causal mask + softmax + sgemm for output.

            // Allocate score matrix [N, N] and gather buffers per head
            std::vector<float> scores((size_t)N * N);
            std::vector<float> q_head_buf((size_t)N * hd);  // gathered Q for one head

            for (int kvh = 0; kvh < nkvh; kvh++) {
                // K for this KV head: fullK[:, kvh*hd : (kvh+1)*hd]
                // Laid out with stride kv_stride between rows. Need contiguous [N, hd].
                // Gather K into temp buffer
                std::vector<float> k_head_buf((size_t)N * hd);
                std::vector<float> v_head_buf((size_t)N * hd);
                for (int i = 0; i < N; i++) {
                    memcpy(k_head_buf.data() + (size_t)i * hd,
                           fullK.data() + (size_t)i * kv_stride + kvh * hd,
                           (size_t)hd * sizeof(float));
                    memcpy(v_head_buf.data() + (size_t)i * hd,
                           fullV.data() + (size_t)i * kv_stride + kvh * hd,
                           (size_t)hd * sizeof(float));
                }

                for (int g = 0; g < groups; g++) {
                    int h = kvh * groups + g;
                    // Gather Q for head h: Proj[i*max_proj + h*hd*2], take first hd floats (skip gate)
                    for (int i = 0; i < N; i++) {
                        memcpy(q_head_buf.data() + (size_t)i * hd,
                               Proj.data() + (size_t)i * max_proj + (size_t)h * hd * 2,
                               (size_t)hd * sizeof(float));
                    }

                    // scores = scale * Q @ K^T : [N, hd] × [hd, N] → [N, N]
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                                N, N, hd, scale,
                                q_head_buf.data(), hd,
                                k_head_buf.data(), hd,
                                0.0f, scores.data(), N);

                    // Apply causal mask + softmax per row
                    for (int i = 0; i < N; i++) {
                        float* row = scores.data() + (size_t)i * N;
                        // Mask: set positions j > i to -inf
                        for (int j = i + 1; j < N; j++) row[j] = -1e9f;
                        softmax(row, N);
                    }

                    // out = scores @ V : [N, N] × [N, hd] → [N, hd]
                    // Write directly into Oproj at the right head offset
                    // Oproj layout: token i → Oproj[i*max_attn + h*hd]
                    // But max_attn stride means we need a temp then scatter
                    std::vector<float> out_head((size_t)N * hd);
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                N, hd, N, 1.0f,
                                scores.data(), N,
                                v_head_buf.data(), hd,
                                0.0f, out_head.data(), hd);

                    // Scatter output into Oproj
                    for (int i = 0; i < N; i++) {
                        memcpy(Oproj.data() + (size_t)i * max_attn + (size_t)h * hd,
                               out_head.data() + (size_t)i * hd,
                               (size_t)hd * sizeof(float));
                    }
                }
            }

            // Step 4: Output gating
            if (attn_output_gate_) {
                for (int i = 0; i < N; i++) {
                    float* pre_oproj = Oproj.data() + (size_t)i * max_attn;
                    float* q_gate_raw = Proj.data() + (size_t)i * max_proj;
                    // Use a stack-allocated tmp for sigmoid
                    float tmp[256];  // head_dim_ = 256
                    for (int h = 0; h < nqh; h++) {
                        float* oh = pre_oproj + h * hd;
                        const float* gh = q_gate_raw + (size_t)h * hd * 2 + hd;
                        mul_sigmoid_inplace(oh, gh, hd, tmp);
                    }
                }
            }
        }
        double layer_attn = attn_timer.elapsed_ms();
        total_attn_ms += layer_attn;
        if (is_linear) delta_attn_ms += layer_attn;
        else full_attn_ms += layer_attn;

        // 4. O projection via GEMM
        Timer oproj_timer;
        const int attn_dim = cw.o_proj_in;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    N, H, attn_dim, 1.0f,
                    Oproj.data(), max_attn,
                    cw.o_proj, attn_dim,
                    0.0f, Attn.data(), H);

        // 5. Residual add: X += Attn
        vDSP_vadd(X.data(), 1, Attn.data(), 1, X.data(), 1, (vDSP_Length)(N * H));
        total_gemm_ms += oproj_timer.elapsed_ms();

        // 6. Post-attention RMSNorm
        for (int i = 0; i < N; i++) {
            rmsnorm(X_norm.data() + (size_t)i * H,
                    X.data() + (size_t)i * H,
                    layers_[L].post_attention_layernorm, H, rms_eps_);
        }

        // 7. FFN via GEMM
        Timer ffn_timer;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    N, I, H, 1.0f,
                    X_norm.data(), H,
                    cw.gate_proj, H,
                    0.0f, Gate.data(), I);

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    N, I, H, 1.0f,
                    X_norm.data(), H,
                    cw.up_proj, H,
                    0.0f, Up.data(), I);

        // SwiGLU: silu(gate) * up
        {
            const int total = N * I;
            float* gp = Gate.data();
            float* up = Up.data();
            for (int i = 0; i < total; i++) {
                float g = gp[i];
                float e = expf(-g);
                gp[i] = g * up[i] / (1.0f + e);
            }
        }

        // Down = Gate @ W_down^T  [N×I] × [I×H]^T = [N×H]
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    N, H, I, 1.0f,
                    Gate.data(), I,
                    cw.down_proj, I,
                    0.0f, Down.data(), H);

        // 8. Residual add: X += Down
        vDSP_vadd(X.data(), 1, Down.data(), 1, X.data(), 1, (vDSP_Length)(N * H));
        total_ffn_ms += ffn_timer.elapsed_ms();
    }

    // Final norm + LM head
    float* last_hidden = X.data() + (size_t)(N - 1) * H;
    rmsnorm(session.x, last_hidden, final_norm_, H, rms_eps_);
    matvec(session.logits, cpu_lm_head_, session.x, vocab_size_, H);

    double elapsed = total_timer.elapsed_ms();
    fprintf(stderr, "[cpu_prefill] %d tokens in %.1f ms (%.1f tok/s) | gemm=%.0fms attn=%.0fms (delta=%.0f full=%.0f) ffn=%.0fms\n",
            N, elapsed, N / (elapsed / 1000.0), total_gemm_ms, total_attn_ms, delta_attn_ms, full_attn_ms, total_ffn_ms);

    return session.logits;
}

bool Qwen35Model::forward_batch(Session** sessions, const int* token_ids, const int* positions, int batch) {
    if (!sessions || !token_ids || !positions || batch <= 0) return false;
    if (batch == 1) {
        return forward(*sessions[0], token_ids[0], positions[0]) != nullptr;
    }
    if (batch > ANE_SPATIAL) {
        for (int base = 0; base < batch; base += ANE_SPATIAL) {
            int chunk = std::min(ANE_SPATIAL, batch - base);
            if (!forward_batch(sessions + base, token_ids + base, positions + base, chunk)) {
                return false;
            }
        }
        return true;
    }

    const int max_proj_dim = std::max(lin_qkv_dim_ + lin_total_val_, full_q_dim_ + 2 * full_kv_dim_);
    const int max_attn_dim = std::max(lin_total_val_, full_out_dim_);

    thread_local std::vector<float> x_batch;
    thread_local std::vector<float> x_norm_batch;
    thread_local std::vector<float> proj_batch;
    thread_local std::vector<float> pre_oproj_batch;
    thread_local std::vector<float> mlp_batch;
    thread_local std::vector<float> packed0;
    thread_local std::vector<float> packed1;
    thread_local std::vector<float> raw_out;
    thread_local std::vector<float> lm_head_batch;

    x_batch.resize((size_t)batch * hidden_size_);
    x_norm_batch.resize((size_t)batch * hidden_size_);
    proj_batch.resize((size_t)batch * max_proj_dim);
    pre_oproj_batch.resize((size_t)batch * max_attn_dim);
    mlp_batch.resize((size_t)batch * hidden_size_);

    for (int b = 0; b < batch; b++) {
        memcpy(x_batch.data() + (size_t)b * hidden_size_,
               embed_tokens_ + (int64_t)token_ids[b] * hidden_size_,
               (size_t)hidden_size_ * sizeof(float));
    }

    double rms1_ms = 0.0;
    double first_proj_ms = 0.0;
    double core_cpu_ms = 0.0;
    double oproj_ms = 0.0;
    double rms2_ms = 0.0;
    double ffn_ms = 0.0;
    double final_norm_ms = 0.0;
    double lm_head_ms = 0.0;
    const bool do_profile = qwen35_profile_serve_batch();

    for (int L = 0; L < num_layers_; L++) {
        Timer t_stage;
        for (int b = 0; b < batch; b++) {
            rmsnorm(x_norm_batch.data() + (size_t)b * hidden_size_,
                    x_batch.data() + (size_t)b * hidden_size_,
                    layers_[L].input_layernorm, hidden_size_, rms_eps_);
        }
        if (do_profile) rms1_ms += t_stage.elapsed_ms();

        const bool is_linear = layer_types_[L] == LayerType::LinearAttention;
        const int proj_dim = is_linear
            ? (qwen35_linear_z_on_cpu() ? lin_qkv_dim_ : (lin_qkv_dim_ + lin_total_val_))
            : (full_q_dim_ + 2 * full_kv_dim_);
        t_stage.reset();
        if (!ane_matvec_batch(ane_layers_[L].first_proj,
                              proj_batch.data(), x_norm_batch.data(),
                              batch, hidden_size_, proj_dim, packed0, raw_out)) {
            fprintf(stderr, "ANE first_proj batched eval failed at layer %d\n", L);
            return false;
        }
        if (do_profile) first_proj_ms += t_stage.elapsed_ms();

        t_stage.reset();
        for (int b = 0; b < batch; b++) {
            Session& session = *sessions[b];
            float* x_norm = x_norm_batch.data() + (size_t)b * hidden_size_;
            float* proj = proj_batch.data() + (size_t)b * max_proj_dim;
            float* pre_oproj = pre_oproj_batch.data() + (size_t)b * max_attn_dim;
            int pos = positions[b];

            if (is_linear) {
                auto& dw = layers_[L].deltanet;

                float* mixed_qkv = proj;
                float* z = proj + lin_qkv_dim_;
                if (qwen35_linear_z_on_cpu()) {
                    matvec(z, dw.in_proj_z, x_norm, lin_total_val_, hidden_size_);
                }

                float* a_vec = session.scratch_tmp;
                float* b_vec = session.scratch_tmp + lin_num_val_heads_;
                matvec(a_vec, dw.in_proj_a, x_norm, lin_num_val_heads_, hidden_size_);
                matvec(b_vec, dw.in_proj_b, x_norm, lin_num_val_heads_, hidden_size_);

                float* conv_out = session.scratch_conv;
                conv1d_update(conv_out, session.delta_conv_state[L], &session.delta_conv_pos[L], mixed_qkv, dw.conv1d_w, lin_qkv_dim_, conv_kernel_);
                silu_vec_inplace(conv_out, lin_qkv_dim_, session.scratch_tmp + lin_num_val_heads_ * 2);

                float* Q = conv_out;
                float* K = conv_out + lin_total_key_;
                float* V = conv_out + lin_total_key_ * 2;
                float* y = session.scratch_y;
                float q_scale = 1.0f / sqrtf((float)lin_key_dim_);
                int val_heads_per_key = lin_num_val_heads_ / lin_num_heads_;

                for (int kh = 0; kh < lin_num_heads_; kh++) {
                    float* qh = Q + kh * lin_key_dim_;
                    float* kh_ptr = K + kh * lin_key_dim_;

                    l2_normalize(qh, lin_key_dim_);
                    l2_normalize(kh_ptr, lin_key_dim_);
                    float qs = q_scale;
                    vDSP_vsmul(qh, 1, &qs, qh, 1, (vDSP_Length)lin_key_dim_);

                    for (int vsub = 0; vsub < val_heads_per_key; vsub++) {
                        int vh = kh * val_heads_per_key + vsub;
                        float* vh_ptr = V + vh * lin_val_dim_;
                        float* yh = y + vh * lin_val_dim_;
                        float* state = session.delta_ssm_state[L] + (size_t)vh * lin_key_dim_ * lin_val_dim_;

                        float beta = sigmoid_f(b_vec[vh]);
                        float decay = expf(-dw.A[vh] * softplus_f(a_vec[vh] + dw.dt_bias[vh]));
                        ssm_step(yh, state, qh, kh_ptr, vh_ptr, decay, beta, lin_key_dim_, lin_val_dim_);
                    }
                }

                if (qwen35_batch_rmsgate()) {
                    rmsnorm_gated_repeated(pre_oproj, y, z, dw.norm_w,
                                           lin_num_val_heads_, lin_val_dim_,
                                           session.scratch_tmp);
                } else {
                    for (int h = 0; h < lin_num_val_heads_; h++) {
                        rmsnorm_gated(pre_oproj + h * lin_val_dim_,
                                      y + h * lin_val_dim_,
                                      z + h * lin_val_dim_,
                                      dw.norm_w, lin_val_dim_);
                    }
                }
            } else {
                auto& fw = layers_[L].full_attn;

                float* q_gate_raw = proj;
                float* k_raw = proj + full_q_dim_;
                float* v_raw = proj + full_q_dim_ + full_kv_dim_;

                for (int h = 0; h < num_q_heads_; h++) {
                    float* qh = q_gate_raw + (size_t)h * head_dim_ * 2;
                    rmsnorm(qh, qh, fw.q_norm, head_dim_, rms_eps_);
                }
                for (int h = 0; h < num_kv_heads_; h++) {
                    rmsnorm(k_raw + h * head_dim_, k_raw + h * head_dim_, fw.k_norm, head_dim_, rms_eps_);
                }

                const float* rope_cos_row = nullptr;
                const float* rope_sin_row = nullptr;
                if (pos >= 0 && pos < MAX_SEQ_LEN && rope_cos_ && rope_sin_) {
                    int half_rot = rot_dim_ / 2;
                    rope_cos_row = rope_cos_ + (size_t)pos * half_rot;
                    rope_sin_row = rope_sin_ + (size_t)pos * half_rot;
                }
                apply_rope_cached(q_gate_raw, k_raw, num_q_heads_, num_kv_heads_,
                                  head_dim_, head_dim_ * 2, head_dim_, rot_dim_, pos, rope_theta_,
                                  rope_cos_row, rope_sin_row);

                int slot;
                if (session.kv_len[L] < KV_CACHE_CAPACITY) {
                    slot = session.kv_start[L] + session.kv_len[L];
                    if (slot >= KV_CACHE_CAPACITY) slot -= KV_CACHE_CAPACITY;
                    session.kv_len[L]++;
                } else {
                    slot = session.kv_start[L];
                    session.kv_start[L]++;
                    if (session.kv_start[L] >= KV_CACHE_CAPACITY) session.kv_start[L] = 0;
                }
                size_t kv_stride = (size_t)num_kv_heads_ * head_dim_;
                memcpy(session.kv_k_cache[L] + (size_t)slot * kv_stride, k_raw, kv_stride * sizeof(float));
                memcpy(session.kv_v_cache[L] + (size_t)slot * kv_stride, v_raw, kv_stride * sizeof(float));

                gqa_attention(pre_oproj, q_gate_raw, session.kv_k_cache[L], session.kv_v_cache[L],
                              num_q_heads_, num_kv_heads_, head_dim_, head_dim_ * 2,
                              session.kv_start[L], session.kv_len[L], KV_CACHE_CAPACITY);

                if (attn_output_gate_) {
                    for (int h = 0; h < num_q_heads_; h++) {
                        float* oh = pre_oproj + h * head_dim_;
                        const float* gh = q_gate_raw + (size_t)h * head_dim_ * 2 + head_dim_;
                        mul_sigmoid_inplace(oh, gh, head_dim_, session.scratch_tmp);
                    }
                }
            }
        }
        if (do_profile) core_cpu_ms += t_stage.elapsed_ms();

        int attn_dim = is_linear ? lin_total_val_ : full_out_dim_;
        t_stage.reset();
        if (ane_layers_[L].oproj_add) {
            if (!ane_binary_batch(ane_layers_[L].oproj_add,
                                  attn_dim, hidden_size_, hidden_size_,
                                  x_batch.data(),
                                  pre_oproj_batch.data(), x_batch.data(), batch,
                                  packed0, packed1, raw_out)) {
                fprintf(stderr, "ANE oproj_add batched eval failed at layer %d\n", L);
                return false;
            }
        } else {
            for (int b = 0; b < batch; b++) {
                float* x = x_batch.data() + (size_t)b * hidden_size_;
                float* pre_oproj = pre_oproj_batch.data() + (size_t)b * max_attn_dim;
                float* attn_out = mlp_batch.data() + (size_t)b * hidden_size_;
                if (!ane_matvec(ane_layers_[L].o_proj, attn_out, pre_oproj, attn_dim, hidden_size_)) {
                    fprintf(stderr, "ANE o_proj eval failed at layer %d\n", L);
                    return false;
                }
                for (int i = 0; i < hidden_size_; i++) x[i] += attn_out[i];
            }
        }
        if (do_profile) oproj_ms += t_stage.elapsed_ms();

        t_stage.reset();
        for (int b = 0; b < batch; b++) {
            rmsnorm(x_norm_batch.data() + (size_t)b * hidden_size_,
                    x_batch.data() + (size_t)b * hidden_size_,
                    layers_[L].post_attention_layernorm, hidden_size_, rms_eps_);
        }
        if (do_profile) rms2_ms += t_stage.elapsed_ms();

        t_stage.reset();
        if (ane_layers_[L].ffn_resadd) {
            if (!ane_binary_batch(ane_layers_[L].ffn_resadd,
                                  hidden_size_, hidden_size_, hidden_size_,
                                  x_batch.data(),
                                  x_norm_batch.data(), x_batch.data(), batch,
                                  packed0, packed1, raw_out)) {
                fprintf(stderr, "ANE ffn_resadd batched eval failed at layer %d\n", L);
                return false;
            }
        } else if (ane_layers_[L].fused_ffn) {
            if (!ane_matvec_batch(ane_layers_[L].fused_ffn,
                                  mlp_batch.data(), x_norm_batch.data(),
                                  batch, hidden_size_, hidden_size_, packed0, raw_out)) {
                fprintf(stderr, "ANE fused_ffn batched eval failed at layer %d\n", L);
                return false;
            }
            for (int b = 0; b < batch; b++) {
                float* x = x_batch.data() + (size_t)b * hidden_size_;
                float* mlp = mlp_batch.data() + (size_t)b * hidden_size_;
                for (int i = 0; i < hidden_size_; i++) x[i] += mlp[i];
            }
        } else if (ane_layers_[L].chunked_ffn.num_chunks > 0) {
            for (int b = 0; b < batch; b++) {
                float* x = x_batch.data() + (size_t)b * hidden_size_;
                float* x_norm = x_norm_batch.data() + (size_t)b * hidden_size_;
                float* mlp = mlp_batch.data() + (size_t)b * hidden_size_;
                if (!ane_eval_chunked_ffn(&ane_layers_[L].chunked_ffn, mlp, x_norm)) {
                    fprintf(stderr, "ANE chunked_ffn eval failed at layer %d\n", L);
                    return false;
                }
                for (int i = 0; i < hidden_size_; i++) x[i] += mlp[i];
            }
        } else {
            fprintf(stderr, "No FFN kernel for layer %d\n", L);
            return false;
        }
        if (do_profile) ffn_ms += t_stage.elapsed_ms();
    }

    Timer t_stage;
    for (int b = 0; b < batch; b++) {
        rmsnorm(x_norm_batch.data() + (size_t)b * hidden_size_,
                x_batch.data() + (size_t)b * hidden_size_,
                final_norm_, hidden_size_, rms_eps_);
        memcpy(sessions[b]->x, x_norm_batch.data() + (size_t)b * hidden_size_,
               (size_t)hidden_size_ * sizeof(float));
    }
    if (do_profile) final_norm_ms += t_stage.elapsed_ms();

    t_stage.reset();
    if (ane_lm_head_enabled_ && !lm_head_kernels_.empty()) {
        bool ok = true;
        int chunks = (int)lm_head_kernels_.size();
        for (int c = 0; c < chunks; c++) {
            int offset = c * lm_head_chunk_;
            int rows = vocab_size_ - offset;
            if (rows > lm_head_chunk_) rows = lm_head_chunk_;
            lm_head_batch.resize((size_t)batch * rows);
            if (!ane_matvec_batch(lm_head_kernels_[c],
                                  lm_head_batch.data(), x_norm_batch.data(),
                                  batch, hidden_size_, rows, packed0, raw_out)) {
                fprintf(stderr, "ANE LM head batched eval failed at chunk %d/%d, falling back to CPU\n", c + 1, chunks);
                ok = false;
                break;
            }
            for (int b = 0; b < batch; b++) {
                memcpy(sessions[b]->logits + offset,
                       lm_head_batch.data() + (size_t)b * rows,
                       (size_t)rows * sizeof(float));
            }
        }
        if (!ok) {
            free_lm_head_ane();
            for (int b = 0; b < batch; b++) {
                matvec(sessions[b]->logits, lm_head_, x_norm_batch.data() + (size_t)b * hidden_size_, vocab_size_, hidden_size_);
            }
        }
    } else {
        for (int b = 0; b < batch; b++) {
            matvec(sessions[b]->logits, lm_head_, x_norm_batch.data() + (size_t)b * hidden_size_, vocab_size_, hidden_size_);
        }
    }
    if (do_profile) {
        lm_head_ms += t_stage.elapsed_ms();
        g_qwen35_batch_profile.add(rms1_ms, first_proj_ms, core_cpu_ms, oproj_ms, rms2_ms, ffn_ms, final_norm_ms, lm_head_ms);
    }

    return true;
}

static bool profile_forward_enabled() {
    static bool enabled = getenv("PROFILE_FORWARD") != nullptr;
    return enabled;
}

struct ForwardProfile {
    uint64_t calls = 0;
    double rms1_ms = 0.0;
    double first_proj_ms = 0.0;
    double attn_core_ms = 0.0;
    double oproj_ms = 0.0;
    double rms2_ms = 0.0;
    double ffn_ms = 0.0;
    double final_norm_ms = 0.0;
    double lm_head_ms = 0.0;

    void print_and_reset() {
        double total = rms1_ms + first_proj_ms + attn_core_ms + oproj_ms + rms2_ms + ffn_ms + final_norm_ms + lm_head_ms;
        fprintf(stderr,
            "\n=== Qwen3.5 forward profile (%llu tokens, avg %.2f ms/tok) ===\n"
            "  rms1 (input norm):    %6.2f ms (%4.1f%%)\n"
            "  first_proj (ANE):     %6.2f ms (%4.1f%%)\n"
            "  attn_core (CPU):      %6.2f ms (%4.1f%%)\n"
            "  oproj_add (ANE):      %6.2f ms (%4.1f%%)\n"
            "  rms2 (post-attn):     %6.2f ms (%4.1f%%)\n"
            "  ffn_resadd (ANE):     %6.2f ms (%4.1f%%)\n"
            "  final_norm (CPU):     %6.2f ms (%4.1f%%)\n"
            "  lm_head (ANE):        %6.2f ms (%4.1f%%)\n"
            "  TOTAL:                %6.2f ms → %.1f tok/s\n",
            (unsigned long long)calls, total / calls,
            rms1_ms / calls, 100.0 * rms1_ms / total,
            first_proj_ms / calls, 100.0 * first_proj_ms / total,
            attn_core_ms / calls, 100.0 * attn_core_ms / total,
            oproj_ms / calls, 100.0 * oproj_ms / total,
            rms2_ms / calls, 100.0 * rms2_ms / total,
            ffn_ms / calls, 100.0 * ffn_ms / total,
            final_norm_ms / calls, 100.0 * final_norm_ms / total,
            lm_head_ms / calls, 100.0 * lm_head_ms / total,
            total / calls, 1000.0 * calls / total);
        calls = 0;
        rms1_ms = rms2_ms = first_proj_ms = attn_core_ms = oproj_ms = ffn_ms = final_norm_ms = lm_head_ms = 0.0;
    }
};

static ForwardProfile g_fwd_profile;

float* Qwen35Model::forward(int token, int pos) {
    if (!default_session_) return nullptr;
    return forward(*default_session_, token, pos);
}

float* Qwen35Model::forward(Session& session, int token, int pos) {
    const bool do_profile = profile_forward_enabled();
    memcpy(session.x, embed_tokens_ + (int64_t)token * hidden_size_, hidden_size_ * sizeof(float));

    float* pre_oproj = session.scratch_attn;
    double rms1 = 0, first_proj = 0, attn_core = 0, oproj = 0, rms2 = 0, ffn = 0;

    for (int L = 0; L < num_layers_; L++) {
        Timer t;
        rmsnorm(session.x_norm, session.x, layers_[L].input_layernorm, hidden_size_, rms_eps_);
        if (do_profile) rms1 += t.elapsed_ms();

        t.reset();
        if (layer_types_[L] == LayerType::LinearAttention) {
            if (!forward_deltanet_core(session, L, session.x_norm, pre_oproj, do_profile ? &first_proj : nullptr)) return nullptr;
        } else {
            if (!forward_full_attn_core(session, L, session.x_norm, pre_oproj, pos, do_profile ? &first_proj : nullptr)) return nullptr;
        }
        if (do_profile) attn_core += t.elapsed_ms();

        int attn_dim = (layer_types_[L] == LayerType::LinearAttention) ? lin_total_val_ : full_out_dim_;
        t.reset();
        if (ane_layers_[L].oproj_add) {
            if (!ane_eval_oproj_add(ane_layers_[L].oproj_add, session.x, pre_oproj, session.x, attn_dim, hidden_size_)) {
                fprintf(stderr, "ANE oproj_add eval failed at layer %d\n", L);
                return nullptr;
            }
        } else {
            float* attn_out = session.x_norm;
            if (!ane_matvec(ane_layers_[L].o_proj, attn_out, pre_oproj, attn_dim, hidden_size_)) {
                fprintf(stderr, "ANE o_proj eval failed at layer %d\n", L);
                return nullptr;
            }
            for (int i = 0; i < hidden_size_; i++) session.x[i] += attn_out[i];
        }
        if (do_profile) oproj += t.elapsed_ms();

        t.reset();
        rmsnorm(session.x_norm, session.x, layers_[L].post_attention_layernorm, hidden_size_, rms_eps_);
        if (do_profile) rms2 += t.elapsed_ms();

        t.reset();
        if (ane_layers_[L].ffn_resadd) {
            if (!ane_eval_fused_ffn_resadd(ane_layers_[L].ffn_resadd, session.x, session.x_norm, session.x, hidden_size_)) {
                fprintf(stderr, "ANE ffn_resadd eval failed at layer %d\n", L);
                return nullptr;
            }
        } else {
            float* mlp_out = session.scratch_attn;
            if (ane_layers_[L].fused_ffn) {
                if (!ane_matvec(ane_layers_[L].fused_ffn, mlp_out, session.x_norm, hidden_size_, hidden_size_)) {
                    fprintf(stderr, "ANE fused_ffn eval failed at layer %d\n", L);
                    return nullptr;
                }
            } else if (ane_layers_[L].chunked_ffn.num_chunks > 0) {
                if (!ane_eval_chunked_ffn(&ane_layers_[L].chunked_ffn, mlp_out, session.x_norm)) {
                    fprintf(stderr, "ANE chunked_ffn eval failed at layer %d\n", L);
                    return nullptr;
                }
            } else {
                fprintf(stderr, "No FFN kernel for layer %d\n", L);
                return nullptr;
            }

            for (int i = 0; i < hidden_size_; i++) session.x[i] += mlp_out[i];
        }
        if (do_profile) ffn += t.elapsed_ms();
    }

    Timer t;
    rmsnorm(session.x, session.x, final_norm_, hidden_size_, rms_eps_);
    double final_norm = do_profile ? t.elapsed_ms() : 0.0;

    t.reset();
    if (ane_lm_head_enabled_ && !lm_head_kernels_.empty()) {
        bool ok = true;
        int chunks = (int)lm_head_kernels_.size();
        for (int c = 0; c < chunks; c++) {
            int offset = c * lm_head_chunk_;
            int rows = vocab_size_ - offset;
            if (rows > lm_head_chunk_) rows = lm_head_chunk_;
            if (!ane_matvec(lm_head_kernels_[c], session.logits + offset, session.x, hidden_size_, rows)) {
                fprintf(stderr, "ANE LM head eval failed at chunk %d/%d, falling back to CPU\n", c + 1, chunks);
                ok = false;
                break;
            }
        }
        if (!ok) {
            free_lm_head_ane();
            matvec(session.logits, lm_head_, session.x, vocab_size_, hidden_size_);
        }
    } else {
        matvec(session.logits, lm_head_, session.x, vocab_size_, hidden_size_);
    }
    double lm_head_time = do_profile ? t.elapsed_ms() : 0.0;

    if (do_profile) {
        g_fwd_profile.calls++;
        g_fwd_profile.rms1_ms += rms1;
        g_fwd_profile.first_proj_ms += first_proj;
        g_fwd_profile.attn_core_ms += attn_core - first_proj;
        g_fwd_profile.oproj_ms += oproj;
        g_fwd_profile.rms2_ms += rms2;
        g_fwd_profile.ffn_ms += ffn;
        g_fwd_profile.final_norm_ms += final_norm;
        g_fwd_profile.lm_head_ms += lm_head_time;
        if (g_fwd_profile.calls % 50 == 0) {
            g_fwd_profile.print_and_reset();
        }
    }

    return session.logits;
}

} // namespace ane_lm
