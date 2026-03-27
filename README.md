# ANE-LM

LLM inference on Apple Neural Engine (ANE) using private `AppleNeuralEngine.framework` APIs.

Every other inference engine on Mac (llama.cpp, MLX, etc.) runs on CPU and GPU. The Neural Engine sits idle. ANE-LM is the first to run full LLM inference directly on the ANE — including 4B+ parameter models.

## What's New: 4B+ Model Support

The original project ran small models (0.6B–0.8B). We've added **chunked FFN compilation** that breaks large feed-forward layers into multiple ANE kernels, enabling models like **Qwen3-4B** (intermediate_size=9728) that previously exceeded ANE single-kernel limits.

**Qwen3-4B** (2560 hidden, 36 layers, 32 Q-heads / 8 KV-heads):
- ~6 tok/s generation on Apple Silicon
- ~8s cached init (216 ANE kernels + 10 LM head chunks)
- First run compiles in ~28s, subsequent runs load from persistent ANE cache

Also includes **dynamic-weight matrix-vector multiply** APIs — the building blocks for weight-swapping without recompilation (see [ANE Hardware Findings](#ane-hardware-findings) below).

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

Through systematic reverse-engineering of the ANE's behavior via private frameworks, we've documented a set of hardware constraints and working patterns that aren't available in any public documentation. These findings were produced through 25+ targeted test programs.

### IOSurface Layout Constraints

The ANE communicates with the host CPU through IOSurface buffers in a 4D `[N, C, H, W]` layout. The hardware imposes a critical constraint:

- **All runtime input IOSurfaces must have W (innermost dimension) ≥ 32 (SP)**. W=16 or W=1 causes silent eval failure. This is the ANE's "spatial" dimension — the minimum granularity of its vector processing unit.
- Data is laid out in FP16 with stride `SP=32` on the W axis, regardless of the logical tensor shape. A tensor with C=2560 and W=SP occupies `2560 × 32 × 2 bytes = 160KB` per surface.
- The `H` dimension is typically 1 for vector operations. `N` and `C` can vary freely.

### Operations That Work on Runtime Tensors

These MIL operations reliably accept runtime (non-constant) IOSurface inputs:

| Operation | Status | Notes |
|-----------|--------|-------|
| `add(a, b)` | ✅ Works | Same-shape and N-broadcast both work |
| `mul(a, b)` | ✅ Works | Same-shape works perfectly. N-broadcast works in isolation but may be unreliable in multi-op programs |
| `reduce_sum(axis)` | ✅ Works | Tested on axis=1 (C dimension), produces correct results |
| `reshape` | ✅ Works | Even changing N dimension works (contradicts early hypothesis) |
| `conv` (dynamic weights) | ❌ Compiles, fails at eval | ANE conv reads weights from a dedicated internal weight bus populated from BLOBFILE, not from runtime IOSurfaces. The weight input compiles but is silently ignored at eval time. |

### The `tile` Poison

**The `tile` operation poisons ANE state.** Any MIL program that uses `tile` — even if it compiles and evaluates correctly itself — causes all subsequent ANE kernel evaluations in the same process to fail with status `0x1d`. This is not a per-kernel failure; it corrupts global ANE state. The only recovery is process restart.

This was discovered through systematic isolation testing:
1. Programs with `tile` + `mul` fail
2. Programs with `tile` + `add` fail
3. Programs with `tile` alone succeed, but...
4. Any program evaluated after a `tile` program fails — even previously-working programs
5. The corruption persists until process exit

**Workaround**: Never use `tile` in any ANE MIL program. Perform tiling on the CPU before writing to IOSurfaces.

### Dynamic-Weight Matrix-Vector Multiply

The key research contribution: a working method for matrix-vector multiply where the weight matrix is a runtime input (not compiled into the kernel). This enables weight-swapping without the 60ms+ recompilation cost per matrix.

**Why it's hard**: ANE's native conv operation reads weights from a dedicated hardware bus, not from IOSurface inputs. You can't just pass weights as a second input to conv — it compiles but produces garbage at eval.

**Working strategy — CPU-side tiling**:

```
Input x:  [1, T, 1, SP]         — the vector (T = ceil(in_dim/32))
Weight W: [out_dim, T, 1, SP]   — the matrix, one row per N slice

1. CPU: tile x → x_tiled [out_dim, T, 1, SP]  (memcpy x into each N slice)
2. ANE: mul(x_tiled, W) → [out_dim, T, 1, SP]  (same-shape, no broadcast needed)
3. ANE: reduce_sum(axis=1) → [out_dim, 1, 1, SP]  (partial sums across T)
4. CPU: sum 32 (SP) values per output channel → final y[out_dim]
```

**Performance at 2560×2560** (hidden_size of Qwen3-4B):

| Component | Time |
|-----------|------|
| W memcpy to IOSurface | 0.27ms |
| x tile + write | 0.39ms |
| ANE eval | 1.00ms |
| **Total dynamic matvec** | **1.65ms** |
| Const-weight conv (for comparison) | 0.32ms |

Dynamic matvec is ~5× slower per eval than const-weight conv, but **zero recompile on weight swap** (vs ~60ms compile per weight set). This makes it viable for scenarios like:
- LoRA adapter hot-swapping
- Mixture-of-experts routing
- Speculative decoding with different draft models

### ANE Compile Budget and Caching

- There's a per-process limit on simultaneously loaded ANE kernels. For Qwen3-4B, 216 layer kernels + 10 LM head kernels = 226 total, which is within budget.
- Freeing kernels (`ane_free`) reclaims budget — you can compile-eval-free in a loop.
- The persistent compile cache (`ane_set_persist_cache(true)`) stores compiled ANE programs on disk. Cache load is ~10× faster than fresh compilation. First run of Qwen3-4B: ~28s. Subsequent runs: ~8s.
- Cache key is derived from the MIL program text + weight data, so different models don't collide.
- LM head compilation is not cached (weights are chunked dynamically), which accounts for most of the remaining 8s cached init time.

### Why the Generation Bottleneck Isn't Compute

At 6.2 tok/s with 216 ANE dispatches per token:

```
Time per token:     ~161ms
Time per dispatch:  ~0.75ms (avg)
Actual compute:     ~0.0004ms per 2560×2560 matmul (at ~15 TOPS FP16)
Dispatch overhead:  ~99.9% of wall time
```

The ANE hardware can do a 2560×2560 matmul in under a microsecond. But each dispatch involves: IOSurface lock → write FP16 input → ANE program dispatch → wait for completion → IOSurface lock → read FP16 output → convert to FP32. This CPU↔ANE round-trip dominates.

**Implications for future optimization**:
- Quantization (INT8/INT4) would reduce memory transfer but not dispatch overhead — minimal speedup expected
- Fusing more operations into single ANE programs (fewer dispatches) is the highest-leverage optimization
- Moving attention to ANE (currently CPU-side) would eliminate the biggest source of mid-layer CPU↔ANE round-trips
- Batch prefill (multiple tokens per dispatch) would amortize dispatch cost over more useful work

### MIL Programming Notes

ANE-LM compiles MIL (Machine Learning Intermediate Language) programs directly via the private `Espresso` and `ANECompiler` frameworks. Useful patterns:

- **Fused multi-output projections**: A single MIL program can compute Q, K, V projections by concatenating weight matrices along the output dimension and using `slice_by_size` to split the result. This turns 3 dispatches into 1.
- **Fused SwiGLU FFN**: `gate_proj`, `up_proj`, `silu`, elementwise multiply, and `down_proj` can all be a single MIL program, turning 3+ dispatches into 1.
- **Chunked FFN**: When intermediate_size is too large for a single kernel (the ANE has finite register/buffer space), split into N chunks along the intermediate dimension. Each chunk computes a partial `down_proj` output, accumulated on the CPU. 4 chunks works for inter=9728.
- **`buildInfo` metadata**: MIL programs require a `buildInfo` dict with a `coremlc-version` key. Use `"3505.4.1"` or similar — the value doesn't appear to affect compilation but its absence causes parse failure.
- **`ios16` target**: Use `func main<ios16>(...)` for broadest compatibility. Higher targets may enable additional ops but aren't needed for matmul/FFN patterns.

## Requirements

- macOS 13.0+
- Apple Silicon (M1/M2/M3/M4/M5)

## Acknowledgments

- [johnmai-dev/ANE-LM](https://github.com/johnmai-dev/ANE-LM) — Original project: ANE runtime, Qwen3/3.5 inference, safetensors loader, tokenizer, chat template engine
- [maderix/ANE](https://github.com/maderix/ANE) — Training neural networks on Apple Neural Engine via reverse-engineered private APIs
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — LLM inference in C/C++
