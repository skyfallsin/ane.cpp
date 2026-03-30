# ane.cpp

LLM inference on Apple Neural Engine (ANE) using private `AppleNeuralEngine.framework` APIs. This project was originally forked from [johnmai-dev/ANE-LM](https://github.com/johnmai-dev/ANE-LM).

ane.cpp focuses on running dense Qwen-family models directly on ANE, then improving throughput through measurement-driven changes to kernel structure, dispatch count, and batching.

A companion repo, [apple-neural-engine-field-guide](https://github.com/skyfallsin/apple-neural-engine-field-guide), documents the reverse-engineering work behind the runtime. That repo follows a [karpathy/autoresearch](https://github.com/karpathy/autoresearch)-style loop: write a focused probe, measure it, keep or discard it, then port the findings that hold up into ane.cpp. The workflow is described in [program.md](https://github.com/skyfallsin/apple-neural-engine-field-guide/blob/main/program.md).

## Current status

Current measurements on the M3 Max test machine:

| Model | Prompt throughput | Generate throughput | Notes |
|---|---:|---:|---|
| **Qwen3-4B** | **26.59 tok/s** | **7.27 tok/s** | Batched prefill path enabled |
| **Qwen3.5-4B** | **13.68 tok/s** | **8.37 tok/s** | Fastest single-user path measured so far |
| **Qwen3.5-9B** | **3.97 tok/s** | **4.27 tok/s** | Runs end-to-end on ANE |

Shared-process serve mode on **Qwen3.5-4B** reached:

- **7.18 tok/s** aggregate at 1 active request
- **8.55 tok/s** aggregate at 2 active requests
- **12.9–13.1 tok/s** aggregate at 4 active requests

These numbers come from the companion field guide's current measurements and should be treated as hardware- and setup-specific rather than general benchmarks.

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

# Pre-convert weights (BF16 -> FP16, speeds up subsequent loads)
./build/ane.cpp convert --model /path/to/Qwen3-4B
```

Download models in safetensors format from HuggingFace, for example `Qwen/Qwen3-4B` or `Qwen/Qwen3.5-4B`.

### Options

```text
--model <path>       Path to model directory (required)
--prompt <text>      Input prompt (generate mode, default: "Hello")
--max-tokens N       Max tokens to generate (default: unlimited)
--temp T             Temperature (default: 0.6)
--repeat-penalty P   Repetition penalty (default: 1.2, 1.0=off)
--enable-thinking    Enable thinking/reasoning mode
--no-ane-cache       Disable persistent ANE compile cache
-v, --verbose        Show detailed initialization info
```

### Experimental environment flags

```bash
ANE_PREFILL_BATCH=4                # W-lane prefill batching (default 4, max 32)
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
