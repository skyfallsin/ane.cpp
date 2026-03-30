#include "cpu_ops.h"
#include <alloca.h>
#include <cstdlib>

namespace ane_lm {

void silu_vec_inplace(float* x, int n, float* tmp) {
    int vn = n;
    float one = 1.0f;
    vDSP_vneg(x, 1, tmp, 1, (vDSP_Length)n);
    vvexpf(tmp, tmp, &vn);
    vDSP_vsadd(tmp, 1, &one, tmp, 1, (vDSP_Length)n);
    vDSP_vdiv(tmp, 1, x, 1, x, 1, (vDSP_Length)n);
}

void mul_sigmoid_inplace(float* y, const float* z, int n, float* tmp) {
    int vn = n;
    float one = 1.0f;
    vDSP_vneg(z, 1, tmp, 1, (vDSP_Length)n);
    vvexpf(tmp, tmp, &vn);
    vDSP_vsadd(tmp, 1, &one, tmp, 1, (vDSP_Length)n);
    vDSP_vdiv(tmp, 1, y, 1, y, 1, (vDSP_Length)n);
}

void rmsnorm(float* out, const float* x, const float* weight, int dim, float eps) {
    float ss = 0.0f;
    vDSP_svesq(x, 1, &ss, (vDSP_Length)dim);
    ss = 1.0f / sqrtf(ss / dim + eps);
    vDSP_vsmul(x, 1, &ss, out, 1, (vDSP_Length)dim);
    vDSP_vmul(out, 1, weight, 1, out, 1, (vDSP_Length)dim);
}

void rmsnorm_gated(float* out, const float* x, const float* z,
                   const float* weight, int dim) {
    rmsnorm(out, x, weight, dim);
    float* tmp = (float*)alloca((size_t)dim * sizeof(float));
    mul_sigmoid_inplace(out, z, dim, tmp);
    vDSP_vmul(out, 1, z, 1, out, 1, (vDSP_Length)dim);
}

void apply_rope_cached(float* q, float* k, int n_q_heads, int n_kv_heads,
                       int head_dim, int q_head_stride, int k_head_stride,
                       int rot_dim, int pos, float theta,
                       const float* cos_row, const float* sin_row) {
    for (int h = 0; h < n_q_heads + n_kv_heads; h++) {
        float* v = (h < n_q_heads) ? q + h * q_head_stride : k + (h - n_q_heads) * k_head_stride;
        for (int i = 0, j = 0; i < rot_dim; i += 2, j++) {
            float cos_a, sin_a;
            if (cos_row && sin_row) {
                cos_a = cos_row[j];
                sin_a = sin_row[j];
            } else {
                float freq = 1.0f / powf(theta, (float)i / (float)rot_dim);
                float angle = pos * freq;
                cos_a = cosf(angle);
                sin_a = sinf(angle);
            }
            float v0 = v[i];
            float v1 = v[i + 1];
            v[i]     = v0 * cos_a - v1 * sin_a;
            v[i + 1] = v0 * sin_a + v1 * cos_a;
        }
    }
}

void softmax(float* x, int n) {
    // vDSP-accelerated softmax
    float max_val;
    vDSP_maxv(x, 1, &max_val, (vDSP_Length)n);
    float neg_max = -max_val;
    vDSP_vsadd(x, 1, &neg_max, x, 1, (vDSP_Length)n);  // x -= max
    int vn = n;
    vvexpf(x, x, &vn);                                    // x = exp(x)
    float sum;
    vDSP_sve(x, 1, &sum, (vDSP_Length)n);                 // sum(x)
    float inv = 1.0f / sum;
    vDSP_vsmul(x, 1, &inv, x, 1, (vDSP_Length)n);         // x /= sum
}

void matvec(float* y, const float* W, const float* x, int out_dim, int in_dim) {
    cblas_sgemv(CblasRowMajor, CblasNoTrans, out_dim, in_dim, 1.0f,
                W, in_dim, x, 1, 0.0f, y, 1);
}

void l2_normalize(float* x, int dim) {
    float norm = 0.0f;
    vDSP_svesq(x, 1, &norm, (vDSP_Length)dim);
    norm = 1.0f / sqrtf(norm + 1e-12f);
    vDSP_vsmul(x, 1, &norm, x, 1, (vDSP_Length)dim);
}

void conv1d_update(float* y, float* conv_state, int* state_pos, const float* x,
                   const float* w, int channels, int kernel_size) {
    int state_len = kernel_size - 1;
    int pos = *state_pos;

    if (kernel_size == 4) {
        int p0 = pos;
        int p1 = (pos + 1);
        if (p1 == 3) p1 = 0;
        int p2 = (p1 + 1);
        if (p2 == 3) p2 = 0;

        for (int c = 0; c < channels; c++) {
            const int sbase = c * 3;
            const int wbase = c * 4;
            float s0 = conv_state[sbase + p0];
            float s1 = conv_state[sbase + p1];
            float s2 = conv_state[sbase + p2];
            float xc = x[c];
            y[c] = s0 * w[wbase] + s1 * w[wbase + 1] + s2 * w[wbase + 2] + xc * w[wbase + 3];
            conv_state[sbase + p0] = xc;
        }
    } else {
        for (int c = 0; c < channels; c++) {
            float sum = 0.0f;
            int base = c * state_len;
            for (int j = 0; j < state_len; j++) {
                int idx = pos + j;
                if (idx >= state_len) idx -= state_len;
                sum += conv_state[base + idx] * w[c * kernel_size + j];
            }
            float xc = x[c];
            y[c] = sum + xc * w[c * kernel_size + state_len];
            conv_state[base + pos] = xc;
        }
    }

    pos++;
    if (pos == state_len) pos = 0;
    *state_pos = pos;
}

void ssm_step(float* y, float* state, const float* q, const float* k,
              const float* v, float decay, float beta, int key_dim, int value_dim) {
    static bool use_fused = getenv("ANE_SSM_OLD") == nullptr;
    if (!use_fused) {
        float* Sk = (float*)alloca(value_dim * sizeof(float));
        cblas_sgemv(CblasRowMajor, CblasTrans, key_dim, value_dim, 1.0f,
                    state, value_dim, k, 1, 0.0f, Sk, 1);

        float* delta = (float*)alloca(value_dim * sizeof(float));
        vDSP_vsub(Sk, 1, v, 1, delta, 1, (vDSP_Length)value_dim);

        cblas_sscal(key_dim * value_dim, decay, state, 1);
        cblas_sger(CblasRowMajor, key_dim, value_dim, beta,
                   k, 1, delta, 1, state, value_dim);

        cblas_sgemv(CblasRowMajor, CblasTrans, key_dim, value_dim, 1.0f,
                    state, value_dim, q, 1, 0.0f, y, 1);
        return;
    }

    float* Sk = (float*)alloca(value_dim * sizeof(float));
    float* delta = (float*)alloca(value_dim * sizeof(float));
    memset(Sk, 0, (size_t)value_dim * sizeof(float));

    for (int i = 0; i < key_dim; i++) {
        const float ki = k[i];
        const float* row = state + (size_t)i * value_dim;
        for (int j = 0; j < value_dim; j++) {
            Sk[j] += row[j] * ki;
        }
    }

    for (int j = 0; j < value_dim; j++) {
        delta[j] = v[j] - Sk[j];
        y[j] = 0.0f;
    }

    for (int i = 0; i < key_dim; i++) {
        const float q_i = q[i];
        const float beta_k_i = beta * k[i];
        float* row = state + (size_t)i * value_dim;
        for (int j = 0; j < value_dim; j++) {
            const float updated = row[j] * decay + beta_k_i * delta[j];
            row[j] = updated;
            y[j] += updated * q_i;
        }
    }
}

void gqa_attention(float* out, const float* q,
                   const float* k_cache, const float* v_cache,
                   int n_heads, int n_kv_heads, int head_dim, int q_head_stride,
                   int cache_start, int cache_len, int cache_capacity) {
    if (cache_len <= 0) {
        memset(out, 0, (size_t)n_heads * head_dim * sizeof(float));
        return;
    }

    int groups = n_heads / n_kv_heads;
    float scale = 1.0f / sqrtf((float)head_dim);
    size_t kv_step = (size_t)n_kv_heads * head_dim;

    int first_span = cache_capacity - cache_start;
    if (first_span > cache_len) first_span = cache_len;
    int second_span = cache_len - first_span;

    // Batch all Q heads in a KV group via sgemm: [groups, head_dim] × [cache_len, head_dim]^T
    // Score buffer: [groups, cache_len] per KV head group
    float* group_scores = (float*)alloca((size_t)groups * cache_len * sizeof(float));

    for (int kv_h = 0; kv_h < n_kv_heads; kv_h++) {
        int h_start = kv_h * groups;
        const float* q_group = q + (size_t)h_start * q_head_stride;
        float* out_group = out + (size_t)h_start * head_dim;
        size_t kv_head_off = (size_t)kv_h * head_dim;

        // Q·K^T scores via sgemm: [groups, cache_len] = Q[groups, head_dim] × K[cache_len, head_dim]^T
        if (first_span > 0) {
            const float* k_span1 = k_cache + (size_t)cache_start * kv_step + kv_head_off;
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        groups, first_span, head_dim,
                        scale,
                        q_group, q_head_stride,
                        k_span1, (int)kv_step,
                        0.0f,
                        group_scores, cache_len);
        }
        if (second_span > 0) {
            const float* k_span2 = k_cache + kv_head_off;
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        groups, second_span, head_dim,
                        scale,
                        q_group, q_head_stride,
                        k_span2, (int)kv_step,
                        0.0f,
                        group_scores + first_span, cache_len);
        }

        // Softmax per Q head (each row of group_scores)
        for (int g = 0; g < groups; g++) {
            softmax(group_scores + (size_t)g * cache_len, cache_len);
        }

        // Attention·V via sgemm: out[groups, head_dim] = scores[groups, cache_len] × V[cache_len, head_dim]
        if (first_span > 0) {
            const float* v_span1 = v_cache + (size_t)cache_start * kv_step + kv_head_off;
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        groups, head_dim, first_span,
                        1.0f,
                        group_scores, cache_len,
                        v_span1, (int)kv_step,
                        0.0f,
                        out_group, head_dim);
        }
        if (second_span > 0) {
            const float* v_span2 = v_cache + kv_head_off;
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        groups, head_dim, second_span,
                        1.0f,
                        group_scores + first_span, cache_len,
                        v_span2, (int)kv_step,
                        first_span > 0 ? 1.0f : 0.0f,
                        out_group, head_dim);
        }
    }
}

} // namespace ane_lm
