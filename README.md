# ane.cpp

LLM inference on Apple Neural Engine (ANE) using private `AppleNeuralEngine.framework` APIs. This project was originally forked from [johnmai-dev/ANE-LM](https://github.com/johnmai-dev/ANE-LM).

ane.cpp focuses on running dense Qwen-family models directly on ANE, then improving throughput through measurement-driven changes to kernel structure, dispatch count, and batching.

## Companion field guide

This repo has a companion research repo:

- 📚 **Field guide:** [apple-neural-engine-field-guide](https://github.com/skyfallsin/apple-neural-engine-field-guide)
- 🧪 **Experiment log:** [results.tsv](https://github.com/skyfallsin/apple-neural-engine-field-guide/blob/main/results.tsv)
- 🔬 **Test programs:** [tests/README.md](https://github.com/skyfallsin/apple-neural-engine-field-guide/blob/main/tests/README.md)
- ♻️ **Autoresearch workflow:** [program.md](https://github.com/skyfallsin/apple-neural-engine-field-guide/blob/main/program.md)

The field guide documents the reverse-engineering work behind the runtime and follows a [karpathy/autoresearch](https://github.com/karpathy/autoresearch)-style loop: write a focused probe, measure it, keep or discard it, then port the findings that hold up into ane.cpp.

## Current status

Current measurements on the M3 Max test machine:

| Model | Mode | Prompt throughput | Generate throughput |
|---|---|---:|---:|
| **Qwen3.5-4B** | fp16 | **18.93 tok/s** | **9.21 tok/s** |
| **Qwen3.5-4B** | int8 | **30.08 tok/s** | **11.66 tok/s** |
| **Qwen3.5-9B** | fp16 | **6.49 tok/s** | **4.27 tok/s** |
| **Qwen3.5-9B** | int8 | **7.39 tok/s** | **7.01 tok/s** |

Single-user numbers are warmed 5-run medians at 500 generated tokens with Qwen thinking defaults (`--enable-thinking`, `top_p=0.95`, `top_k=20`, `presence_penalty=1.5`, `repeat_penalty=1.0`, `temp=1.0`).

Shared-process serve mode (8 requests, 100 tokens/request, 3 repeats, `--sessions 4`) — aggregate generation throughput:

| Model | Mode | c1 | c2 | c4 |
|---|---|---:|---:|---:|
| Qwen3.5-4B | fp16 | 8.62 | 12.59 | 20.75 |
| Qwen3.5-4B | int8 | 10.53 | 15.95 | 28.62 |
| Qwen3.5-9B | fp16 | 3.62 | 4.44 | 4.89 |
| Qwen3.5-9B | int8 | 5.76 | 5.81 | 6.57 |

These numbers come from the companion field guide's current measurements and should be treated as hardware- and setup-specific rather than general benchmarks.

Single-stream tok/s on current M3 Max hardware is still modest. The reason this path is interesting is not just foreground chat speed: if ANE inference can handle useful work at low energy cost, and if one warm process can absorb multiple low-rate requests concurrently, it becomes a plausible building block for background or always-on workloads. That is why this repo tracks both shared-process concurrency and relative energy use, not just peak tok/s.

For relative power / tokens-per-joule measurements, see `scripts/energy_benchmark.sh`.

## Supported models

- **Qwen3** (dense) — tested up to 4B parameters
- **Qwen3.5** (dense, text-only) — including the hybrid DeltaNet + full-attention variants

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Usage

```bash
# Single-shot generation
./build/ane.cpp generate --model /path/to/Qwen3-4B --prompt "Hello" --max-tokens 100

# Interactive chat
./build/ane.cpp chat --model /path/to/Qwen3-4B

# OpenAI-compatible API server
./build/ane.cpp serve --model /path/to/Qwen3.5-4B --port 8088 --sessions 4

# Pre-convert weights (BF16 -> FP16, speeds up subsequent loads)
./build/ane.cpp convert --model /path/to/Qwen3-4B
```

Download one of the supported safetensors model directories from HuggingFace and pass that directory to `--model`.

If you do not already have the Hugging Face CLI installed:

```bash
python3 -m pip install "huggingface_hub[cli]"
```

If you want faster downloads, Hugging Face also supports transfer acceleration via `hf_transfer`:

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download Qwen/Qwen3-4B --local-dir Qwen3-4B
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download Qwen/Qwen3.5-4B --local-dir Qwen3.5-4B
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download Qwen/Qwen3.5-9B --local-dir Qwen3.5-9B
```

These commands create local model directories such as `./Qwen3-4B`, which can then be used directly with `--model ./Qwen3-4B`.

### Serve mode

`serve` starts an OpenAI-compatible HTTP server at `http://127.0.0.1:<port>/v1`. It implements:

- `POST /v1/chat/completions` — streaming (SSE) and non-streaming responses
- `GET /v1/models` — lists the loaded model

```bash
./build/ane.cpp serve --model /path/to/Qwen3.5-4B --port 8088 --sessions 4

# Non-streaming
curl http://localhost:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "stream": false, "max_tokens": 100}'

# Streaming (default)
curl -N http://localhost:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 100}'
```

Supported request fields: `messages`, `stream`, `max_tokens`, `temperature`, `top_p`, `top_k`, `presence_penalty`, `frequency_penalty`, `repetition_penalty`, `enable_thinking`, `chat_template_kwargs.enable_thinking`.

**Use with [pi](https://github.com/badlogic/pi)** — add to `~/.pi/agent/models.json`:

```json
{
  "providers": {
    "ane": {
      "baseUrl": "http://localhost:8088/v1",
      "api": "openai-completions",
      "apiKey": "local",
      "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": false,
        "maxTokensField": "max_tokens",
        "thinkingFormat": "qwen-chat-template"
      },
      "models": [
        {
          "id": "Qwen3.5-4B",
          "name": "Qwen 3.5 4B (ANE)",
          "reasoning": true,
          "input": ["text"],
          "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
          "contextWindow": 32768,
          "maxTokens": 8192
        }
      ]
    }
  }
}
```

Then select it with `/model` in pi.

### Options

```text
--model <path>       Path to model directory (required)
--prompt <text>      Input prompt (generate mode, default: "Hello")
--max-tokens N       Max tokens to generate (default: unlimited)
--temp T             Temperature (default: 0.6)
--repeat-penalty P   Repetition penalty (default: 1.2, 1.0=off)
--enable-thinking    Enable thinking/reasoning mode
--port N             Server port for serve mode (default: 8088)
--sessions N         Session pool size for serve mode (default: 4)
--no-ane-cache       Disable persistent ANE compile cache
-v, --verbose        Show detailed initialization info
```

### Experimental environment flags

```bash
ANE_PREFILL_BATCH=4                # W-lane prefill batching (default 4, max 32)
ANE_USE_DIRECT_EVAL=1              # use _ANEClient direct eval path when available (set 0 to force daemon path)
ANE_SPECULATIVE_DECODE=1           # experimental speculative decode for Qwen3
ANE_SPECULATIVE_BATCH=32           # speculative verify batch size / W-lanes to use (default 32)
ANE_SPECULATIVE_DRAFT_LAYERS=2     # truncated self-draft depth (default 2)
ANE_SPECULATIVE_STATS=1            # print speculative accept/attempt stats at the end of generation
```

Speculative decode is still experimental. In the measured Qwen3 draft-model paths so far, it has remained slower than the baseline decode path.

## How it works

ane.cpp compiles each weight matrix into an ANE convolution kernel with weights baked in at compile time. For a single token, the runtime alternates between ANE kernels and CPU-side operations:

1. **Embedding lookup** (CPU)
2. **Per layer**
   - RMSNorm (CPU)
   - Fused QKV projection (ANE)
   - QK norm + RoPE (CPU)
   - Attention + KV cache update (CPU)
   - O projection + residual add (ANE)
   - RMSNorm (CPU)
   - SwiGLU FFN + residual add (ANE)
3. **Final norm** (CPU)
4. **LM head** (ANE)

The current performance comes from a few main changes relative to the original codebase:

- **chunked FFN support**, which made larger models feasible on ANE
- **fused kernels** such as `oproj_add` and `ffn_resadd`
- **larger LM-head chunks** where they improved throughput
- **W-lane batching**, which improved prompt throughput and shared-process serving throughput

Persistent compile cache stores compiled ANE programs on disk, so first-run compilation cost only needs to be paid once per kernel variant.

## Performance notes

The current bottleneck appears to be a combination of:

- repeated weight streaming from unified memory
- per-dispatch fixed overhead
- lower effective bandwidth on smaller kernels than on the largest FFN / LM-head kernels

The companion field guide includes a dispatch scaling benchmark suggesting an eval-only fit of approximately:

```text
latency ≈ 119µs + bytes / 78 GB/s
```

That model has been more useful than treating the remaining overhead as a single opaque dispatch cost.

The practical implication is that future gains may matter most in aggregate-throughput and background-work settings. If newer ANE generations improve bandwidth, compiler behavior, or multi-program runtime limits, the same batching structure that already helps at 2–4 concurrent requests on M3 Max could become more useful than the current single-stream tok/s alone suggests.

## ANE findings that shape the runtime

A few constraints from the field guide have directly shaped ane.cpp:

- runtime IOSurfaces needed `W >= 32` on the tested working paths
- `tile` was unsafe in the tested runtime path and was avoided
- const-weight conv behaved reliably, while runtime-weight conv did not produce a usable path on the tested M3 Max setup
- fp16 RMSNorm on ANE showed an error floor that was too large for the norm-containing mega-kernels we wanted
- W-lane batching produced useful throughput improvements and became part of the prefill and serve paths

Full details and reproductions live in the companion field guide:

- [apple-neural-engine-field-guide](https://github.com/skyfallsin/apple-neural-engine-field-guide)

## Requirements

- macOS 13.0+
- Apple Silicon (M1/M2/M3/M4/M5)

## Acknowledgments

- [johnmai-dev/ANE-LM](https://github.com/johnmai-dev/ANE-LM) — original project: ANE runtime, Qwen3/3.5 inference, safetensors loader, tokenizer, chat template engine
- [maderix/ANE](https://github.com/maderix/ANE) — training neural networks on Apple Neural Engine via reverse-engineered private APIs
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — LLM inference in C/C++
