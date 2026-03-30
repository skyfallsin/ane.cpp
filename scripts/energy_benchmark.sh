#!/bin/bash
# Energy Efficiency Benchmark: ANE vs GPU (Metal) for LLM inference
#
# Measures power consumption (CPU/GPU/ANE rails) during inference using
# macOS powermetrics, then computes tokens/joule for each backend.
#
# Requirements:
#   - sudo access (powermetrics needs root)
#   - ane.cpp built: ./build/ane.cpp
#   - llama.cpp built: llama-cli (or set LLAMA_CLI)
#
# Usage:
#   sudo ./scripts/energy_benchmark.sh [--model-dir /path/to/Qwen3-4B] [--max-tokens 200]

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ANE_LM_DIR="$(dirname "$SCRIPT_DIR")"
ANE_LM_BIN="${ANE_LM_DIR}/build/ane.cpp"

# Defaults
MODEL_DIR="${ANE_LM_DIR}/hf-models/Qwen3.5-4B"
MAX_TOKENS=200
PROMPT="Explain the theory of general relativity and its implications for our understanding of spacetime, gravity, and the universe."
SAMPLE_RATE_MS=200  # powermetrics sample interval
COOLDOWN_SECS=10    # cooldown between runs
OUTPUT_DIR="/tmp/energy_benchmark_$(date +%Y%m%d_%H%M%S)"

LLAMA_SIMPLE="${LLAMA_SIMPLE:-/Users/pradeep/personal/llama.cpp/build/bin/llama-simple}"
LLAMA_BENCH="${LLAMA_BENCH:-/Users/pradeep/personal/llama.cpp/build/bin/llama-bench}"
GGUF_MODEL=""  # set via --gguf or auto-detected as FP16 GGUF beside safetensors

# ── Parse args ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-dir) MODEL_DIR="$2"; shift 2 ;;
        --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
        --prompt) PROMPT="$2"; shift 2 ;;
        --gguf) GGUF_MODEL="$2"; shift 2 ;;
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$OUTPUT_DIR"

# ── Check root ─────────────────────────────────────────────────────────
if [[ $EUID -ne 0 ]]; then
    echo "ERROR: powermetrics requires root. Run with: sudo $0 $*"
    exit 1
fi

# ── Auto-detect FP16 GGUF model (same precision as ane.cpp) ───────────
if [[ -z "$GGUF_MODEL" ]]; then
    MODEL_NAME=$(basename "$MODEL_DIR")
    # Prefer FP16 GGUF sitting next to the safetensors (same model, same precision)
    for candidate in \
        "${MODEL_DIR}/${MODEL_NAME}-F16.gguf" \
        "${MODEL_DIR}/"*"-F16.gguf" \
        ; do
        if [[ -f "$candidate" ]]; then
            GGUF_MODEL="$candidate"
            break
        fi
    done
    if [[ -z "$GGUF_MODEL" ]]; then
        echo ""
        echo "⚠ No FP16 GGUF found. Convert with:"
        echo "  python3 /path/to/llama.cpp/convert_hf_to_gguf.py $MODEL_DIR --outtype f16 --outfile ${MODEL_DIR}/${MODEL_NAME}-F16.gguf"
        echo ""
    fi
fi

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           Energy Efficiency Benchmark: ANE vs GPU           ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║ Model dir:   $(printf '%-46s' "$MODEL_DIR") ║"
echo "║ GGUF model:  $(printf '%-46s' "${GGUF_MODEL:-NOT FOUND}") ║"
echo "║ Max tokens:  $(printf '%-46s' "$MAX_TOKENS") ║"
echo "║ Output:      $(printf '%-46s' "$OUTPUT_DIR") ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Helper: run powermetrics during a command ─────────────────────────
# Usage: measure_power <label> <command...>
# Outputs: $OUTPUT_DIR/<label>_power.plist, <label>_inference.log
measure_power() {
    local label="$1"
    shift
    local power_file="${OUTPUT_DIR}/${label}_power.plist"
    local inference_log="${OUTPUT_DIR}/${label}_inference.log"
    local power_pid

    echo "━━━ Measuring: $label ━━━"
    echo "  Command: $*"
    echo "  Cooling down ${COOLDOWN_SECS}s..."
    sleep "$COOLDOWN_SECS"

    # Start powermetrics in background
    powermetrics \
        --samplers cpu_power,gpu_power,ane_power \
        --format plist \
        -i "$SAMPLE_RATE_MS" \
        -o "$power_file" &
    power_pid=$!

    # Small delay for powermetrics to start
    sleep 1

    # Run inference
    local start_time end_time
    start_time=$(python3 -c "import time; print(time.time())")

    "$@" > "$inference_log" 2>&1 || true

    end_time=$(python3 -c "import time; print(time.time())")

    # Stop powermetrics
    sleep 1  # capture trailing power
    kill "$power_pid" 2>/dev/null || true
    wait "$power_pid" 2>/dev/null || true

    echo "  Duration: $(python3 -c "print(f'{$end_time - $start_time:.1f}s')")"
    echo "  Power log: $power_file"
    echo "  Inference log: $inference_log"
    echo ""
}

# ── Run 1: ane.cpp ─────────────────────────────────────────────────────
if [[ -x "$ANE_LM_BIN" ]]; then
    measure_power "ane_lm" \
        "$ANE_LM_BIN" generate \
        --model "$MODEL_DIR" \
        --prompt "$PROMPT" \
        --max-tokens "$MAX_TOKENS" \
        --temp 0
else
    echo "⚠ ane.cpp binary not found at $ANE_LM_BIN, skipping"
fi

# ── Run 2: llama.cpp (Metal GPU) ──────────────────────────────────────
if [[ -x "$LLAMA_SIMPLE" ]] && [[ -n "$GGUF_MODEL" ]]; then
    measure_power "llama_gpu" \
        "$LLAMA_SIMPLE" \
        -m "$GGUF_MODEL" \
        -n "$MAX_TOKENS" \
        -ngl 99 \
        "$PROMPT"
else
    echo "⚠ llama.cpp or GGUF model not found, skipping GPU benchmark"
    echo "  LLAMA_SIMPLE=$LLAMA_SIMPLE"
    echo "  GGUF_MODEL=$GGUF_MODEL"
fi

# ── Parse results ──────────────────────────────────────────────────────
echo ""
echo "━━━ Parsing power data ━━━"

OUTPUT_DIR="$OUTPUT_DIR" python3 << 'PYEOF'
import plistlib
import sys
import os
import re

output_dir = os.environ.get("OUTPUT_DIR", "/tmp/energy_benchmark")

def parse_power_plist(path):
    """Parse powermetrics plist output, extract per-rail power samples.
    All power keys live under d['processor']: cpu_power, gpu_power, ane_power (mW)."""
    if not os.path.exists(path):
        return None

    with open(path, 'rb') as f:
        data = f.read()

    samples = []
    for chunk in data.split(b'\x00'):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            d = plistlib.loads(chunk)
            proc = d.get('processor', {})
            samples.append({
                'elapsed_ns': d.get('elapsed_ns', 0),
                'cpu_mw': proc.get('cpu_power', 0),
                'gpu_mw': proc.get('gpu_power', 0),
                'ane_mw': proc.get('ane_power', 0),
                'combined_mw': proc.get('combined_power', 0),
            })
        except Exception:
            continue

    return samples


def parse_inference_log(path):
    """Extract token count and timing from inference logs."""
    if not os.path.exists(path):
        return {}

    with open(path) as f:
        text = f.read()

    result = {}

    # ane.cpp: "Generation: 200 tokens, 7.684 tokens-per-sec"
    m = re.search(r'Generation:\s*(\d+)\s*tokens,\s*([\d.]+)\s*tokens-per-sec', text)
    if m:
        result['tokens'] = int(m.group(1))
        result['tok_s'] = float(m.group(2))
        result['time_s'] = result['tokens'] / result['tok_s']
        return result

    # ane.cpp alt: "Prompt: N tokens, X.X tokens-per-sec\nGeneration: ..."
    m = re.search(r'generation:\s*([\d.]+)\s*tok/s', text, re.IGNORECASE)
    if m:
        result['tok_s'] = float(m.group(1))

    # llama-simple: "decoded 200 tokens in 6.53 s, speed: 30.61 t/s"
    m = re.search(r'decoded\s+(\d+)\s+tokens\s+in\s+([\d.]+)\s*s,\s*speed:\s*([\d.]+)\s*t/s', text)
    if m:
        result['tokens'] = int(m.group(1))
        result['time_s'] = float(m.group(2))
        result['tok_s'] = float(m.group(3))
        return result

    # llama.cpp perf: "eval time = 6322.82 ms / 199 runs ( 31.77 ms per token, 31.47 tokens per second)"
    m = re.search(r'eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*runs\s*\(\s*[\d.]+\s*ms per token,\s*([\d.]+)\s*tokens per second\)', text)
    if m:
        result['time_s'] = float(m.group(1)) / 1000
        result['tokens'] = int(m.group(2))
        result['tok_s'] = float(m.group(3))
        return result

    # MLX: "Generation: Y tokens, Z.Z tokens-per-sec"
    m = re.search(r'Generation:\s*(\d+)\s*tokens.*?([\d.]+)\s*tokens-per-sec', text, re.DOTALL)
    if m:
        result['tokens'] = int(m.group(1))
        result['tok_s'] = float(m.group(2))
        result['time_s'] = result['tokens'] / result['tok_s']
        return result

    return result


# ── Process all runs ──────────────────────────────────────────────────
labels = {
    'ane_lm': 'ane.cpp (Neural Engine)',
    'llama_gpu': 'llama.cpp (Metal GPU)',
}

print()
print("=" * 80)
print(f"{'Backend':<28} {'tok/s':>7} {'Tokens':>6} {'Time':>6} {'CPU mW':>8} {'GPU mW':>8} {'ANE mW':>8} {'Total W':>8}")
print("-" * 80)

results = []

for key, label in labels.items():
    power_file = os.path.join(output_dir, f"{key}_power.plist")
    log_file = os.path.join(output_dir, f"{key}_inference.log")

    samples = parse_power_plist(power_file)
    perf = parse_inference_log(log_file)

    if not samples or not perf:
        if os.path.exists(log_file):
            print(f"{label:<28}  (inference failed or output not parsed)")
        continue

    # Skip first sample (captures cooldown tail)
    if len(samples) > 2:
        samples = samples[1:]

    avg_cpu = sum(s['cpu_mw'] for s in samples) / len(samples)
    avg_gpu = sum(s['gpu_mw'] for s in samples) / len(samples)
    avg_ane = sum(s['ane_mw'] for s in samples) / len(samples)
    total_w = (avg_cpu + avg_gpu + avg_ane) / 1000

    tok_s = perf.get('tok_s', 0)
    tokens = perf.get('tokens', 0)
    time_s = perf.get('time_s', 0)

    energy_j = total_w * time_s if time_s else 0
    tok_per_joule = tokens / energy_j if energy_j > 0 else 0

    results.append({
        'label': label, 'key': key, 'tok_s': tok_s, 'tokens': tokens,
        'time_s': time_s, 'cpu_mw': avg_cpu, 'gpu_mw': avg_gpu,
        'ane_mw': avg_ane, 'total_w': total_w, 'energy_j': energy_j,
        'tok_per_joule': tok_per_joule,
    })

    print(f"{label:<28} {tok_s:>7.1f} {tokens:>6d} {time_s:>5.1f}s {avg_cpu:>7.0f} {avg_gpu:>7.0f} {avg_ane:>7.0f} {total_w:>7.2f}")

print("-" * 80)
print()

if results:
    print("=" * 80)
    print(f"{'Backend':<28} {'Total W':>8} {'Energy (J)':>10} {'tok/joule':>10} {'Relative':>10}")
    print("-" * 80)

    results.sort(key=lambda r: r['tok_per_joule'], reverse=True)
    best = results[0]['tok_per_joule'] if results else 1

    for r in results:
        rel = r['tok_per_joule'] / best if best else 0
        print(f"{r['label']:<28} {r['total_w']:>7.2f} {r['energy_j']:>9.1f} {r['tok_per_joule']:>9.1f} {rel:>9.1%}")

    print("-" * 80)
    print()
    print("Notes:")
    print("  tok/joule  = tokens per joule (higher = more efficient)")
    print("  Total W    = avg CPU + GPU + ANE power during generation")
    print("  Energy (J) = total_watts x generation_time")
    print("  powermetrics values are estimates — use for relative comparison, not absolute")
    print()

# Save raw data
summary_path = os.path.join(output_dir, "summary.txt")
with open(summary_path, 'w') as f:
    for r in results:
        f.write(f"{r['key']}: tok/s={r['tok_s']:.1f} cpu={r['cpu_mw']:.0f}mW gpu={r['gpu_mw']:.0f}mW ane={r['ane_mw']:.0f}mW total={r['total_w']:.2f}W energy={r['energy_j']:.1f}J tok/J={r['tok_per_joule']:.1f}\n")
print(f"Raw data saved to: {output_dir}/")

PYEOF

echo ""
echo "Done! Results in: $OUTPUT_DIR"
