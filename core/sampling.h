#pragma once

#include <vector>

namespace ane_lm {

struct SamplingParams {
    float temperature = 0.7f;
    float repetition_penalty = 1.0f;
    int repetition_context_size = 256;
    float frequency_penalty = 0.0f;
    float presence_penalty = 1.5f;
    int top_k = 20;
    float top_p = 0.8f;
};

void compute_sampling_probs(float* probs_out, const float* logits, int vocab_size,
                            const SamplingParams& params,
                            const std::vector<int>& recent_tokens = {});

int sample_token(const float* logits, int vocab_size,
                 const SamplingParams& params,
                 const std::vector<int>& recent_tokens = {});

} // namespace ane_lm
