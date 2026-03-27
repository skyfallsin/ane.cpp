# ANE-LM

LLM inference on Apple Neural Engine (ANE) using private `AppleNeuralEngine.framework` APIs.

Every other inference engine on Mac (llama.cpp, MLX, etc.) runs on CPU and GPU. The Neural Engine sits idle. ANE-LM is the first to run full LLM inference directly on the ANE — including 4B+ parameter models.

## What's New: 4B+ Model Support

The original project ran small models (0.6B–0.8B). We've added **chunked FFN compilation** that breaks large feed-forward layers into multiple ANE kernels, enabling models like **Qwen3-4B** (intermediate_size=9728) that previously exceeded ANE single-kernel limits.

**Qwen3-4B** (2560 hidden, 36 layers, 32 Q-heads / 8 KV-heads):
- ~6 tok/s generation on Apple Silicon
- ~8s cached init (216 ANE kernels + 10 LM head chunks)
- First run compiles in ~28s, subsequent runs load from persistent ANE cache

Also includes **dynamic-weight matrix-vector multiply** APIs — the building blocks for weight-swapping without recompilation. See our [Apple Neural Engine field guide](https://github.com/skyfallsin/apple-neural-engine) for the full reverse-engineering findings.

## Supported Models

- **Qwen3** (dense) — tested up to 4B parameters
- **Qwen3.5** (dense, text-only) — hybrid DeltaNet + full attention

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Usage

```bash
# Single-shot generation
./build/ane-lm generate --model /path/to/Qwen3-4B --prompt "Hello" --max-tokens 100

# Interactive chat
./build/ane-lm chat --model /path/to/Qwen3-4B

# Pre-convert weights (BF16 -> FP16, speeds up subsequent loads)
./build/ane-lm convert --model /path/to/Qwen3-4B
```

Download models in safetensors format from HuggingFace (e.g. `Qwen/Qwen3-4B`, `Qwen/Qwen3.5-0.8B`).

### Options

```
--model <path>       Path to model directory (required)
--prompt <text>      Input prompt (generate mode, default: "Hello")
--max-tokens N       Max tokens to generate (default: unlimited)
--temp T             Temperature (default: 0.6)
--repeat-penalty P   Repetition penalty (default: 1.2, 1.0=off)
--enable-thinking    Enable thinking/reasoning mode
--no-ane-cache       Disable persistent ANE compile cache
-v, --verbose        Show detailed initialization info
```

## Performance

| Model | Params | Gen tok/s | Init (cached) | Init (first run) | ANE Kernels | FFN Strategy |
|-------|--------|-----------|---------------|-------------------|-------------|--------------|
| Qwen3.5-0.8B | 0.8B | ~17 t/s | ~2s | ~5s | 72 | Single fused |
| Qwen3-4B | 4B | ~6 t/s | ~8s | ~28s | 216 | 4-chunk |

The bottleneck at 4B is ANE dispatch overhead (216 round-trips per token), not compute — each matmul finishes in microseconds but dispatch + IOSurface I/O costs ~0.75ms each.

## How It Works

ANE-LM compiles each weight matrix into an ANE convolution kernel with weights baked in at compile time. For a single token:

1. **Embedding lookup** (CPU)
2. **Per-layer** (×36 for Qwen3-4B):
   - RMSNorm (CPU)
   - Fused QKV projection (ANE — single kernel, 3 matmuls)
   - QK-norm + RoPE (CPU)
   - GQA attention + KV cache (CPU)
   - O projection (ANE)
   - RMSNorm (CPU)
   - SwiGLU FFN (ANE — 4 chunked kernels for 4B, 1 fused kernel for 0.8B)
3. **Final norm** (CPU)
4. **LM head** (ANE — 10 chunked kernels for 151K vocab)

Persistent compile cache stores compiled ANE programs on disk, so first-run compilation (~28s for 4B) only happens once.

## ANE Hardware Findings

Through systematic reverse-engineering (25+ targeted test programs), we've documented hardware constraints and working patterns that aren't in any public documentation. Key discoveries:

- **IOSurface W ≥ 32 (SP)** — all runtime inputs must have innermost dim ≥ 32 or eval silently fails
- **`tile` poisons global ANE state** — any program using `tile` corrupts all subsequent evals in the process (status 0x1d). Only recovery is process restart.
- **Conv dynamic weights silently fail** — conv reads weights from a dedicated hardware bus, not IOSurface inputs. Compiles fine, produces garbage.
- **Dynamic matvec works via CPU-side tiling** — same-shape `mul` + `reduce_sum` at 1.65ms for 2560×2560 (5× slower than const-weight conv, but zero recompile on weight swap)
- **Dispatch overhead dominates** — 99.9% of per-token time is CPU↔ANE round-trips, not compute. Fusing ops matters more than quantization.

**📖 Full details: [Apple Neural Engine — Reverse-Engineering Field Guide](https://github.com/skyfallsin/apple-neural-engine)**

## Requirements

- macOS 13.0+
- Apple Silicon (M1/M2/M3/M4/M5)

## Acknowledgments

- [johnmai-dev/ANE-LM](https://github.com/johnmai-dev/ANE-LM) — Original project: ANE runtime, Qwen3/3.5 inference, safetensors loader, tokenizer, chat template engine
- [maderix/ANE](https://github.com/maderix/ANE) — Training neural networks on Apple Neural Engine via reverse-engineered private APIs
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — LLM inference in C/C++
