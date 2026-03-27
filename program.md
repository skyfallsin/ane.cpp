# ANE-LM autoresearch

Autonomous experimentation to maximize tok/s for LLM inference on Apple Neural Engine.

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — same loop (modify → measure → keep/discard), adapted for ANE inference optimization instead of GPU training.

## Setup

To set up a new experiment run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar27`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files** for full context:
   - This file (`program.md`) — experiment instructions and ANE constraints.
   - `models/llm/qwen3.cpp` — the model forward pass. **This is the primary file you modify.**
   - `core/ane_runtime.h` — ANE kernel compilation and eval API.
   - `core/ane_runtime.cpp` — ANE runtime implementation (MIL program generation, IOSurface management, dispatch). **You may modify this.**
   - `core/cpu_ops.h` / `core/cpu_ops.cpp` — CPU operations (RMSNorm, RoPE, attention, softmax). **You may modify this.**
   - `generate.cpp` — generation loop (token streaming, sampling). Generally leave alone.
   - `main.cpp` — CLI entry point. Generally leave alone.
4. **Verify the model exists**: Check that `Qwen3-4B/` (or whatever path) contains safetensors + config.json. If not, tell the human.
5. **Build and verify**: `cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(sysctl -n hw.ncpu)`
6. **Initialize results.tsv** with header row. Run baseline to record starting tok/s.
7. **Confirm and go**.

## The metric

**Maximize generation tok/s.** This is the tokens-per-second reported during autoregressive decoding (not prefill). Lower dispatch overhead, fewer dispatches per token, faster CPU ops — anything that increases this number is a win.

Run the benchmark:

```bash
./build/ane-lm generate --model Qwen3-4B --prompt "Explain the theory of general relativity in detail, covering spacetime curvature, the equivalence principle, gravitational time dilation, and experimental confirmations" --max-tokens 200 2>&1 | tail -1
```

This prints a line like:
```
Generation: 200 tokens, 6.234 tokens-per-sec
```

Extract the metric:
```bash
./build/ane-lm generate --model Qwen3-4B --prompt "Explain the theory of general relativity in detail" --max-tokens 200 2>&1 | grep -oP '[\d.]+(?= tokens-per-sec)' | tail -1
```

**Always generate 200 tokens** for consistency. The prompt should be long enough to produce coherent output but short enough that prefill time is negligible.

## Correctness constraint

The model must still produce coherent text. After any structural change (fused kernels, reordered ops, changed data flow), do a quick sanity check:

```bash
./build/ane-lm generate --model Qwen3-4B --prompt "What is 2+2?" --max-tokens 20 2>&1
```

If the output is garbage, the change broke correctness. Discard it. **Never keep a speed improvement that produces garbage output.**

## What you CAN modify

- `models/llm/qwen3.cpp` — the forward pass. Reorder ops, fuse dispatches, move work between CPU and ANE, change the per-layer dispatch strategy. **This is where most gains will come from.**
- `core/ane_runtime.cpp` — MIL program generation, IOSurface management, dispatch code. Add new fused kernel types, optimize IOSurface read/write paths, reduce dispatch overhead.
- `core/ane_runtime.h` — add new API functions for new kernel types.
- `core/cpu_ops.cpp` / `core/cpu_ops.h` — CPU operations. Optimize with Accelerate/vDSP, fuse operations, reduce memory copies.

## What you CANNOT modify

- `generate.cpp` — the generation/streaming loop. It measures tok/s and we don't want to break measurement.
- `main.cpp` — CLI argument parsing.
- `core/tokenizer.*`, `core/sampling.*`, `core/safetensors.*`, `core/model_loader.*` — tokenizer, sampling, weight loading. These are fixed infrastructure.
- `prepare.py`, `train.py` — these don't exist here; they're from karpathy's repo.
- Do NOT install new dependencies or modify `CMakeLists.txt` (except to add new .cpp files you create).

## ANE hardware constraints

These are hard constraints discovered through systematic testing. Violating them produces silent failures or corrupted state. See the [field guide](https://github.com/skyfallsin/apple-neural-engine-field-guide) for full details.

### Must obey (violations = silent failure or crash)

- **IOSurface W ≥ 32 (SP)**: All runtime input IOSurfaces must have innermost dimension ≥ 32. W=16 or W=1 compiles fine but produces garbage output.
- **Never use `tile` in MIL programs**: The `tile` op corrupts global ANE state. All subsequent kernel evals in the process fail with status 0x1d. Only recovery is process restart. Perform tiling on CPU via memcpy instead.
- **Conv cannot read dynamic weights**: Conv reads weights from a dedicated hardware bus populated at compile time, not from runtime IOSurface inputs. Declaring a conv with runtime weight input compiles but silently ignores the weight data.
- **N-broadcast `mul` is unreliable**: Works in isolation but produces incorrect results in multi-op MIL programs. Use same-shape inputs for `mul`.

### Safe operations on runtime IOSurface inputs

| Operation | Status |
|-----------|--------|
| `add(a, b)` — same-shape or N-broadcast | ✅ |
| `mul(a, b)` — same-shape only | ✅ |
| `reduce_sum(axis)` | ✅ |
| `reshape` | ✅ (even changing N) |
| `slice_by_size` | ✅ |
| `silu`, `sigmoid` | ✅ |
| `conv` with const weights | ✅ (this is the standard path) |

### Untested operations (potential experiments)

These MIL operations have NOT been tested with runtime IOSurface inputs. Testing them is a valid experiment — each could unlock new fusion possibilities:

- `softmax` — if this works on runtime tensors, attention could move to ANE
- `transpose` / `reshape` to higher dims — could enable different data layouts
- `concat` — could avoid separate dispatches for combining outputs
- `sub`, `div`, `exp`, `log`, `sqrt`, `rsqrt` — basic math ops
- `matmul` (the MIL op, not conv) — unclear if it uses the same weight bus as conv
- `gather` / `scatter` — could enable embedding lookups on ANE
- `band_part` / masking ops — needed for causal attention

To test a new op, write a small MIL program, compile it, eval it, and compare against a CPU reference. See `test_mil_variants.cpp` in the field guide repo for the pattern.

## Where the time goes (current bottleneck)

```
Qwen3-4B: ~6.2 tok/s, ~161ms per token
216 ANE dispatches per token (6 per layer × 36 layers)
~0.75ms per dispatch average
ANE compute per 2560×2560 matmul: ~0.0004ms (< 0.1% of dispatch time)
99.9% of time is CPU↔ANE coordination overhead
```

Current per-layer dispatches (6 total):
1. Fused QKV projection (ANE) — `first_proj`: Q+K+V in one conv kernel
2. QK-norm + RoPE (CPU) — rmsnorm per head + rotary embeddings
3. GQA attention + KV cache (CPU) — softmax, matmul with V
4. O projection (ANE) — `o_proj`: output projection
5. RMSNorm (CPU) — post-attention norm
6. SwiGLU FFN (ANE) — `fused_ffn` or `chunked_ffn` (4 chunks for Qwen3-4B)

Note: chunked FFN is 4 dispatches, so actual total is 4 + 2 + CPU work = ~9 dispatches per layer, or 324 per token. The 216 figure counts logical kernel types; actual dispatch count is higher.

## Experiment ideas (ordered by expected impact)

### High impact — reduce dispatch count

1. **Fuse O-projection into attention output**: Instead of a separate `o_proj` ANE dispatch, compute it in the same pass. This eliminates 36 dispatches (one per layer).

2. **Fuse RMSNorm into adjacent ANE kernels**: RMSNorm is currently CPU-only between ANE dispatches. If it can be expressed as MIL ops on runtime tensors (`mul`, `reduce_sum`, `rsqrt`), it could be fused into the QKV or FFN kernel, eliminating the CPU round-trip. RMSNorm is: `x * rsqrt(mean(x²) + eps) * weight`. The `weight` is a const, `x` is runtime. This needs `reduce_sum` (mean), `mul` (squaring, scaling), and `rsqrt` on the reduced value.

3. **Fuse RoPE into QKV kernel**: RoPE applies sin/cos rotations to Q and K after projection. If the trig values can be precomputed and baked into the kernel (they depend on position, which changes per token — so this may require a dynamic input for position-dependent cos/sin tables).

4. **Reduce FFN chunks**: Qwen3-4B uses 4 FFN chunks. If we can fit more intermediate dimensions into fewer chunks (by optimizing MIL program structure or splitting differently), each eliminated chunk saves 36 dispatches.

### Medium impact — faster CPU operations

5. **Vectorize RMSNorm with vDSP**: Current `rmsnorm` uses scalar loops. Accelerate's `vDSP_svesq`, `vDSP_meanv`, `vDSP_vsmul` could speed this up significantly.

6. **Vectorize RoPE with vDSP**: Current RoPE uses scalar sin/cos per element. With precomputed cos/sin tables (already cached in `rope_cos_`/`rope_sin_`), this can be vectorized.

7. **Optimize GQA attention**: Current attention is scalar C. Could use Accelerate's `cblas_sgemv` for Q·K and attention·V matmuls, `vDSP_maxv` + `vDSP_vsub` + `vForce_vexpf` for softmax.

8. **Reduce memory copies**: The forward pass copies data between `x_`, `x_norm_`, `scratch_qkv_`, `scratch_attn_` buffers. Some of these may be eliminable by computing in-place or reusing buffers.

### Exploratory — new ANE capabilities

9. **Test `softmax` on ANE runtime tensors**: If softmax works, the entire attention computation (Q·K softmax, attention·V) could potentially move to ANE.

10. **Test `matmul` MIL op (not conv)**: The MIL language has a `matmul` op separate from `conv`. It's unknown whether `matmul` uses the same dedicated weight bus or can read runtime inputs.

11. **Test `rsqrt` on ANE**: Needed for fusing RMSNorm into ANE kernels.

12. **Batch multiple layers into one dispatch**: If reshape/slice operations work reliably, it might be possible to chain two layers into a single ANE program, halving dispatch count.

## Build and run

```bash
# Build (fast incremental — only recompiles changed files)
cmake --build build -j$(sysctl -n hw.ncpu)

# Run benchmark
./build/ane-lm generate --model Qwen3-4B --prompt "Explain the theory of general relativity in detail, covering spacetime curvature, the equivalence principle, gravitational time dilation, and experimental confirmations" --max-tokens 200 2>&1 | tail -1

# Quick correctness check
./build/ane-lm generate --model Qwen3-4B --prompt "What is 2+2?" --max-tokens 20 2>&1
```

If you add new `.cpp` files, add them to `CMakeLists.txt` in the appropriate source list, then rebuild.

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated).

Header and columns:

```
commit	tok_s	status	description
```

1. git commit hash (short, 7 chars)
2. tok/s achieved (e.g. 6.234) — use 0.0 for crashes
3. status: `keep`, `discard`, or `crash`
4. short text description of what this experiment tried

Example:

```
commit	tok_s	status	description
a1b2c3d	6.234	keep	baseline
b2c3d4e	6.891	keep	vectorize rmsnorm with vDSP
c3d4e5f	5.100	discard	fuse rope into qkv kernel (correctness ok but slower due to dynamic input overhead)
d4e5f6g	0.0	crash	test softmax on ANE runtime tensor (status 0x1d)
```

## The experiment loop

LOOP FOREVER:

1. Look at the current branch state and results so far.
2. Pick an experiment. Prioritize high-impact ideas (dispatch reduction) over micro-optimizations.
3. Make the code change. Commit it.
4. Build: `cmake --build build -j$(sysctl -n hw.ncpu) 2>&1 | tail -5`
5. If build fails, fix and retry. If it can't be fixed in 2 attempts, discard.
6. Run correctness check: `./build/ane-lm generate --model Qwen3-4B --prompt "What is 2+2?" --max-tokens 20 2>&1`
7. If output is garbage, discard and revert.
8. Run benchmark: `./build/ane-lm generate --model Qwen3-4B --prompt "Explain the theory of general relativity in detail, covering spacetime curvature, the equivalence principle, gravitational time dilation, and experimental confirmations" --max-tokens 200 2>&1 | tail -1`
9. Record results in `results.tsv` (do NOT commit results.tsv).
10. If tok/s improved, keep the commit (advance the branch).
11. If tok/s is equal or worse, `git reset --hard HEAD~1` to discard.
12. Go to 1.

**Run the benchmark 2-3 times** when results are close to the previous best (within ~0.3 tok/s). Take the median. ANE dispatch timing has some variance.

**Timeout**: Each benchmark should take under 60 seconds (200 tokens at ~6 tok/s = ~32s + startup). If a run hangs for over 2 minutes, kill it and treat as a crash.

**Crashes**: If ANE eval returns non-zero status or the process segfaults, check if it's a simple bug (wrong dimensions, missing kernel) or a fundamental ANE constraint violation (tile poison, W < 32). Simple bugs = fix and retry. Constraint violations = discard and note in results.

**New ANE op testing**: When testing an untested MIL operation, write a minimal standalone test first (see field guide `tests/` for the pattern). If the op works, then integrate it into the model. If it poisons ANE state, you'll need to restart the process. Note the finding in results.tsv.

**NEVER STOP**: Once the loop begins, do NOT pause to ask the human. The human may be away. If you run out of high-impact ideas, move to medium-impact. If those are exhausted, try exploratory ANE op testing. If you discover a new working ANE op, that opens new fusion possibilities — loop back to high-impact with the new capability. The loop runs until the human interrupts you.
