#include "sampling.h"
#include "cpu_ops.h"
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace ane_lm {

void compute_sampling_probs(float* probs_out, const float* logits, int vocab_size,
                            const SamplingParams& params,
                            const std::vector<int>& recent_tokens) {
    memcpy(probs_out, logits, vocab_size * sizeof(float));

    std::unordered_map<int, int> freq;
    if (!recent_tokens.empty()) {
        int start = std::max(0, (int)recent_tokens.size() - params.repetition_context_size);
        for (int j = start; j < (int)recent_tokens.size(); j++) {
            int tok = recent_tokens[j];
            if (tok >= 0 && tok < vocab_size) {
                freq[tok]++;
            }
        }

        for (auto& [tok, count] : freq) {
            if (params.repetition_penalty > 1.0f) {
                if (probs_out[tok] > 0.0f) {
                    probs_out[tok] /= params.repetition_penalty;
                } else {
                    probs_out[tok] *= params.repetition_penalty;
                }
            }
            if (params.presence_penalty > 0.0f) {
                probs_out[tok] -= params.presence_penalty;
            }
            if (params.frequency_penalty > 0.0f) {
                probs_out[tok] -= params.frequency_penalty * count;
            }
        }
    }

    if (params.temperature <= 0.0f) {
        int max_i = 0;
        for (int i = 1; i < vocab_size; i++) {
            if (probs_out[i] > probs_out[max_i]) max_i = i;
        }
        memset(probs_out, 0, vocab_size * sizeof(float));
        probs_out[max_i] = 1.0f;
        return;
    }

    std::vector<int> candidates(vocab_size);
    std::iota(candidates.begin(), candidates.end(), 0);

    int top_k = params.top_k;
    if (top_k > 0 && top_k < vocab_size) {
        auto cmp = [&](int a, int b) { return probs_out[a] > probs_out[b]; };
        std::nth_element(candidates.begin(), candidates.begin() + top_k, candidates.end(), cmp);
        candidates.resize(top_k);
        std::sort(candidates.begin(), candidates.end(), cmp);
    }

    float inv_t = 1.0f / params.temperature;
    std::vector<float> candidate_probs(candidates.size());
    float max_logit = -INFINITY;
    for (int tok : candidates) {
        float scaled = probs_out[tok] * inv_t;
        if (scaled > max_logit) max_logit = scaled;
    }

    float sum = 0.0f;
    for (size_t i = 0; i < candidates.size(); i++) {
        float value = expf(probs_out[candidates[i]] * inv_t - max_logit);
        candidate_probs[i] = value;
        sum += value;
    }
    if (sum <= 0.0f) {
        memset(probs_out, 0, vocab_size * sizeof(float));
        probs_out[candidates.front()] = 1.0f;
        return;
    }
    for (float& p : candidate_probs) p /= sum;

    if (params.top_p > 0.0f && params.top_p < 1.0f) {
        float cumulative = 0.0f;
        size_t keep = 0;
        for (; keep < candidate_probs.size(); keep++) {
            cumulative += candidate_probs[keep];
            if (cumulative >= params.top_p) {
                keep++;
                break;
            }
        }
        if (keep == 0) keep = 1;
        if (keep < candidate_probs.size()) {
            float kept_sum = 0.0f;
            for (size_t i = 0; i < keep; i++) kept_sum += candidate_probs[i];
            candidates.resize(keep);
            candidate_probs.resize(keep);
            if (kept_sum > 0.0f) {
                for (float& p : candidate_probs) p /= kept_sum;
            }
        }
    }

    memset(probs_out, 0, vocab_size * sizeof(float));
    for (size_t i = 0; i < candidates.size(); i++) {
        probs_out[candidates[i]] = candidate_probs[i];
    }
}

int sample_token(const float* logits, int vocab_size,
                 const SamplingParams& params,
                 const std::vector<int>& recent_tokens) {
    float* adjusted = (float*)malloc(vocab_size * sizeof(float));
    compute_sampling_probs(adjusted, logits, vocab_size, params, recent_tokens);

    float r = (float)drand48();
    float cum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        cum += adjusted[i];
        if (cum >= r) { free(adjusted); return i; }
    }
    free(adjusted);
    return vocab_size - 1;
}

} // namespace ane_lm
