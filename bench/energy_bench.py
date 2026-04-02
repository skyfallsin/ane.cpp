#!/usr/bin/env python3
"""
Energy benchmark harness for ane.cpp inference engine.

Measures throughput + power consumption (CPU/GPU/ANE) across model configs
and prompt sizes on Apple Silicon using powermetrics.

Usage:
    python3 bench/energy_bench.py --quick          # 312-token sweep, 4B only
    python3 bench/energy_bench.py --full            # Complete sweep all models
    python3 bench/energy_bench.py --save            # Save JSON results
    python3 bench/energy_bench.py --decode          # Include decode-heavy tests
    python3 bench/energy_bench.py --prompt-sizes 112 612 1412
    python3 bench/energy_bench.py --models 4B-INT8  # Single config
    python3 bench/energy_bench.py --sustained 5     # 5-min multi-turn workload
"""

import argparse
import http.client
import json
import os
import plistlib
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
BINARY = REPO_DIR / "build" / "ane.cpp"
MODELS_DIR = REPO_DIR / "hf-models"
RESULTS_DIR = SCRIPT_DIR / "results"

POWERMETRICS_CMD = [
    "sudo", "powermetrics",
    "--samplers", "cpu_power,gpu_power,ane_power,thermal",
    "-f", "plist",
    "-i", "200",  # 200ms sample interval
]

# Model configurations: (display_name, model_dir, env_vars)
MODEL_CONFIGS = {
    "4B-INT8": ("Qwen3.5-4B INT8", "Qwen3.5-4B", {"ANE_INT8": "1", "ANE_GPU_PREFILL": "1"}),
    "4B-FP16": ("Qwen3.5-4B FP16", "Qwen3.5-4B", {"ANE_INT8": "0", "ANE_GPU_PREFILL": "1"}),
    "9B-INT8": ("Qwen3.5-9B INT8", "Qwen3.5-9B", {"ANE_INT8": "1", "ANE_GPU_PREFILL": "1", "ANE_NO_BLOBS": "1"}),
    "9B-FP16": ("Qwen3.5-9B FP16", "Qwen3.5-9B", {"ANE_INT8": "0", "ANE_GPU_PREFILL": "1", "ANE_NO_BLOBS": "1"}),
}

DEFAULT_PROMPT_SIZES = [112, 312, 612, 1012, 1412]
QUICK_PROMPT_SIZES = [312]
QUICK_MODELS = ["4B-INT8", "4B-FP16"]

# Decode tests: short prompt, longer generation
DECODE_TESTS = [
    {"prompt_words": 5, "max_tokens": 100, "label": "decode-100"},
    {"prompt_words": 5, "max_tokens": 200, "label": "decode-200"},
]

# Thresholds for idle check (mW)
IDLE_WARN_CPU = 15000    # CPU > 15W at idle is suspicious
IDLE_WARN_GPU = 2000     # GPU > 2W at idle is suspicious
IDLE_WARN_ANE = 500      # ANE > 0.5W at idle is suspicious

BENCH_TIMEOUT = 300  # 5 min max per benchmark run

# llama.cpp server path and GGUF model configs
LLAMA_SERVER = Path("/Users/pradeep/personal/llama.cpp/build/bin/llama-server")

# (display_name, gguf_path relative to MODELS_DIR, gpu_layers)
LLAMA_CONFIGS = {
    "4B-F16":    ("Qwen3.5-4B F16 (llama)", "Qwen3.5-4B/Qwen3.5-4B-F16.gguf", 99),
    "4B-Q8_0":   ("Qwen3.5-4B Q8_0 (llama)", "/tmp/Qwen3.5-4B-Q8_0.gguf", 99),
    "4B-Q4_K_M": ("Qwen3.5-4B Q4_K_M (llama)", "Qwen3.5-4B/Qwen3.5-4B-Q4_K_M.gguf", 99),
    "9B-Q4_K_M": ("Qwen3.5-9B Q4_K_M (llama)", None, 99),  # resolved at runtime
}

# Resolve 9B GGUF — check multiple locations
_9b_candidates = [
    MODELS_DIR / "Qwen3.5-9B" / "Qwen3.5-9B-Q4_K_M.gguf",
    Path.home() / ".jo_models/unsloth__Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf",
]
for _p in _9b_candidates:
    if _p.exists():
        LLAMA_CONFIGS["9B-Q4_K_M"] = ("Qwen3.5-9B Q4_K_M (llama)", str(_p), 99)
        break

# Sustained mode defaults
SUSTAINED_PORT = 18088
SUSTAINED_SESSIONS = 1
SERVER_STARTUP_TIMEOUT = 120  # max seconds to wait for model load
POWER_WINDOW_SEC = 30  # aggregate power into 30s windows

# Conversation turn templates — varying complexity to simulate real workload
TURN_PROMPTS = [
    "Explain how a transformer neural network processes a sequence of tokens step by step.",
    "Write a Python function that implements binary search on a sorted list, with error handling.",
    "What are the main differences between TCP and UDP? When would you use each one?",
    "Describe the process of photosynthesis in detail, including the light and dark reactions.",
    "Write a short story about a robot discovering music for the first time.",
    "Explain the CAP theorem in distributed systems and give examples of real systems.",
    "How does garbage collection work in modern programming languages? Compare approaches.",
    "What causes the seasons on Earth? Explain the orbital mechanics involved.",
    "Write a SQL query that finds the top 10 customers by total order value in the last month.",
    "Explain how public key cryptography works, including the role of prime numbers.",
    "Compare and contrast microservices and monolithic architectures with real examples.",
    "How do neural networks learn? Explain backpropagation in simple terms.",
    "Write a bash script that monitors disk usage and sends an alert when it exceeds 90%.",
    "What is the significance of the Turing test? Has any AI passed it convincingly?",
    "Explain how a compiler transforms source code into machine code, covering each phase.",
    "Describe the water cycle and how climate change affects precipitation patterns.",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PowerSample:
    cpu_power: float = 0.0    # mW
    cpu_energy: float = 0.0   # mJ
    gpu_power: float = 0.0    # mW
    gpu_energy: float = 0.0   # mJ
    ane_power: float = 0.0    # mW
    ane_energy: float = 0.0   # mJ
    combined_power: float = 0.0  # mW
    thermal_pressure: str = "Unknown"
    elapsed_ns: int = 0


@dataclass
class PrefillMetrics:
    tokens: int = 0
    total_ms: float = 0.0
    tok_per_sec: float = 0.0
    gemm_ms: float = 0.0
    attn_ms: float = 0.0
    ffn_ms: float = 0.0
    norm_ms: float = 0.0
    cvt_ms: float = 0.0
    is_int8: bool = False
    is_fused: bool = False


@dataclass
class BenchResult:
    config: str = ""
    label: str = ""
    prompt_tokens: int = 0
    gen_tokens: int = 0
    prefill_tok_s: float = 0.0
    decode_tok_s: float = 0.0
    prefill: Optional[dict] = None  # PrefillMetrics as dict
    # Power
    samples: int = 0
    avg_cpu_mw: float = 0.0
    avg_gpu_mw: float = 0.0
    avg_ane_mw: float = 0.0
    avg_combined_mw: float = 0.0
    peak_cpu_mw: float = 0.0
    peak_gpu_mw: float = 0.0
    peak_ane_mw: float = 0.0
    peak_combined_mw: float = 0.0
    total_cpu_mj: float = 0.0
    total_gpu_mj: float = 0.0
    total_ane_mj: float = 0.0
    total_energy_mj: float = 0.0
    # Efficiency
    prefill_tok_per_joule: float = 0.0
    decode_tok_per_joule: float = 0.0
    wall_time_s: float = 0.0
    thermal_pressures: list = field(default_factory=list)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Powermetrics parsing
# ---------------------------------------------------------------------------

def parse_plist_samples(raw: bytes) -> list[PowerSample]:
    """Parse NUL-separated plist samples from powermetrics output."""
    samples = []
    parts = raw.split(b"\x00")
    for part in parts:
        part = part.strip()
        if not part or not part.startswith(b"<?xml"):
            continue
        try:
            d = plistlib.loads(part)
        except Exception:
            continue
        proc = d.get("processor", {})
        s = PowerSample(
            cpu_power=proc.get("cpu_power", 0.0),
            cpu_energy=proc.get("cpu_energy", 0),
            gpu_power=proc.get("gpu_power", 0.0),
            gpu_energy=proc.get("gpu_energy", 0),
            ane_power=proc.get("ane_power", 0.0),
            ane_energy=proc.get("ane_energy", 0),
            combined_power=proc.get("combined_power", 0.0),
            thermal_pressure=d.get("thermal_pressure", "Unknown"),
            elapsed_ns=d.get("elapsed_ns", 0),
        )
        samples.append(s)
    return samples


def summarize_power(samples: list[PowerSample]) -> dict:
    """Compute avg/peak/total power stats from samples."""
    if not samples:
        return {}
    n = len(samples)
    return {
        "samples": n,
        "avg_cpu_mw": sum(s.cpu_power for s in samples) / n,
        "avg_gpu_mw": sum(s.gpu_power for s in samples) / n,
        "avg_ane_mw": sum(s.ane_power for s in samples) / n,
        "avg_combined_mw": sum(s.combined_power for s in samples) / n,
        "peak_cpu_mw": max(s.cpu_power for s in samples),
        "peak_gpu_mw": max(s.gpu_power for s in samples),
        "peak_ane_mw": max(s.ane_power for s in samples),
        "peak_combined_mw": max(s.combined_power for s in samples),
        "total_cpu_mj": sum(s.cpu_energy for s in samples),
        "total_gpu_mj": sum(s.gpu_energy for s in samples),
        "total_ane_mj": sum(s.ane_energy for s in samples),
        "total_energy_mj": (
            sum(s.cpu_energy for s in samples)
            + sum(s.gpu_energy for s in samples)
            + sum(s.ane_energy for s in samples)
        ),
        "thermal_pressures": list(set(s.thermal_pressure for s in samples)),
    }


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def parse_throughput(output: str) -> tuple[int, float, int, float]:
    """Parse Prompt/Generation lines. Returns (prompt_tok, prefill_tps, gen_tok, decode_tps)."""
    prompt_tok, prefill_tps = 0, 0.0
    gen_tok, decode_tps = 0, 0.0

    m = re.search(r"Prompt:\s+(\d+)\s+tokens,\s+([\d.]+)\s+tokens-per-sec", output)
    if m:
        prompt_tok = int(m.group(1))
        prefill_tps = float(m.group(2))

    m = re.search(r"Generation:\s+(\d+)\s+tokens,\s+([\d.]+)\s+tokens-per-sec", output)
    if m:
        gen_tok = int(m.group(1))
        decode_tps = float(m.group(2))

    return prompt_tok, prefill_tps, gen_tok, decode_tps


def parse_gpu_prefill(output: str) -> Optional[PrefillMetrics]:
    """Parse [gpu_prefill] line for timing breakdown."""
    m = re.search(
        r"\[gpu_prefill\]\s+(\d+)\s+tokens\s+in\s+([\d.]+)\s+ms\s+\(([\d.]+)\s+tok/s\)"
        r"(.*?)(?:\|(.*))?$",
        output, re.MULTILINE,
    )
    if not m:
        return None

    tokens = int(m.group(1))
    total_ms = float(m.group(2))
    tok_s = float(m.group(3))
    flags = m.group(4) or ""
    breakdown = m.group(5) or ""

    pm = PrefillMetrics(
        tokens=tokens,
        total_ms=total_ms,
        tok_per_sec=tok_s,
        is_int8="[int8]" in flags,
        is_fused="[fused]" in flags,
    )

    for key in ("gemm", "attn", "ffn", "norm", "cvt"):
        km = re.search(rf"{key}=([\d.]+)ms", breakdown)
        if km:
            setattr(pm, f"{key}_ms", float(km.group(1)))

    return pm


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def generate_prompt(num_tokens: int) -> str:
    """Generate a prompt that tokenizes to approximately num_tokens.
    
    Qwen tokenizers map most common English words to single tokens.
    'word ' repeated N times ≈ N tokens (plus a few for BOS/template).
    We aim for slightly fewer words to hit the target after tokenizer overhead.
    """
    # Based on observation: 100 repetitions of 'word ' -> 112 tokens
    # (includes BOS + chat template overhead of ~12 tokens)
    overhead = 12
    n_words = max(1, num_tokens - overhead)
    return "word " * n_words


def run_benchmark(
    config_key: str,
    prompt_size: int,
    max_tokens: int = 5,
    label: str = "",
    timeout: int = BENCH_TIMEOUT,
) -> BenchResult:
    """Run a single benchmark with power monitoring."""
    display_name, model_dir, env_vars = MODEL_CONFIGS[config_key]
    model_path = MODELS_DIR / model_dir

    if not model_path.exists():
        return BenchResult(config=config_key, label=label, error=f"Model not found: {model_path}")

    prompt = generate_prompt(prompt_size)
    result = BenchResult(config=config_key, label=label or f"prefill-{prompt_size}")

    # --- Start powermetrics ---
    pm_out = tempfile.NamedTemporaryFile(delete=False, suffix=".plist")
    pm_out.close()
    pm_proc = None
    try:
        pm_proc = subprocess.Popen(
            POWERMETRICS_CMD,
            stdout=open(pm_out.name, "wb"),
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setpgrp,  # own process group so we can kill cleanly
        )
        # Give powermetrics a moment to initialize and take first sample
        time.sleep(0.3)

        if pm_proc.poll() is not None:
            result.error = "powermetrics failed to start (check sudoers)"
            return result

    except Exception as e:
        result.error = f"powermetrics launch failed: {e}"
        return result

    # --- Run benchmark ---
    bench_env = {**os.environ, **env_vars}
    bench_cmd = [
        str(BINARY), "generate",
        "--model", str(model_path),
        "--temp", "0",
        "--max-tokens", str(max_tokens),
        "--prompt", prompt,
    ]

    wall_start = time.monotonic()
    try:
        bench_proc = subprocess.run(
            bench_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=bench_env,
            cwd=str(REPO_DIR),
        )
        output = bench_proc.stdout + bench_proc.stderr
    except subprocess.TimeoutExpired:
        output = ""
        result.error = f"Benchmark timed out after {timeout}s"
    except Exception as e:
        output = ""
        result.error = f"Benchmark failed: {e}"

    wall_end = time.monotonic()
    result.wall_time_s = round(wall_end - wall_start, 3)

    # --- Stop powermetrics ---
    if pm_proc and pm_proc.poll() is None:
        try:
            os.killpg(os.getpgid(pm_proc.pid), signal.SIGTERM)
            pm_proc.wait(timeout=3)
        except Exception:
            try:
                os.killpg(os.getpgid(pm_proc.pid), signal.SIGKILL)
                pm_proc.wait(timeout=2)
            except Exception:
                pass

    # Small delay to let file flush
    time.sleep(0.1)

    # --- Parse power data ---
    try:
        with open(pm_out.name, "rb") as f:
            raw = f.read()
        samples = parse_plist_samples(raw)
    except Exception as e:
        samples = []
        if not result.error:
            result.error = f"Power data parse error: {e}"
    finally:
        try:
            os.unlink(pm_out.name)
        except OSError:
            pass

    # --- Parse benchmark output ---
    if output:
        prompt_tok, prefill_tps, gen_tok, decode_tps = parse_throughput(output)
        result.prompt_tokens = prompt_tok
        result.gen_tokens = gen_tok
        result.prefill_tok_s = prefill_tps
        result.decode_tok_s = decode_tps

        pm_metrics = parse_gpu_prefill(output)
        if pm_metrics:
            result.prefill = asdict(pm_metrics)

    # --- Compute power stats ---
    if samples:
        stats = summarize_power(samples)
        for k, v in stats.items():
            if hasattr(result, k):
                setattr(result, k, v)

        # Energy efficiency: tokens / joule
        total_j = stats["total_energy_mj"] / 1000.0 if stats["total_energy_mj"] > 0 else 0
        if total_j > 0:
            if result.prompt_tokens > 0 and result.prefill_tok_s > 0:
                result.prefill_tok_per_joule = round(result.prompt_tokens / total_j, 1)
            if result.gen_tokens > 0 and result.decode_tok_s > 0:
                result.decode_tok_per_joule = round(result.gen_tokens / total_j, 1)

    return result


# ---------------------------------------------------------------------------
# Idle baseline / pre-flight
# ---------------------------------------------------------------------------

def preflight_check(duration_s: float = 2.0) -> tuple[bool, dict]:
    """Sample idle power for duration_s, warn if system isn't quiescent."""
    print("⏳ Pre-flight: sampling idle power...")
    try:
        n_samples = max(3, int(duration_s / 0.2))
        proc = subprocess.run(
            POWERMETRICS_CMD + ["-n", str(n_samples)],
            capture_output=True,
            timeout=duration_s + 5,
        )
        samples = parse_plist_samples(proc.stdout)
    except subprocess.TimeoutExpired:
        print("  ⚠️  powermetrics timed out during pre-flight")
        return False, {}
    except Exception as e:
        print(f"  ⚠️  Pre-flight failed: {e}")
        return False, {}

    if not samples:
        print("  ⚠️  No power samples collected — is sudoers configured?")
        return False, {}

    stats = summarize_power(samples)
    idle = {
        "cpu_mw": round(stats["avg_cpu_mw"], 0),
        "gpu_mw": round(stats["avg_gpu_mw"], 0),
        "ane_mw": round(stats["avg_ane_mw"], 0),
        "combined_mw": round(stats["avg_combined_mw"], 0),
        "thermal": stats["thermal_pressures"],
    }

    warnings = []
    if idle["cpu_mw"] > IDLE_WARN_CPU:
        warnings.append(f"CPU idle {idle['cpu_mw']:.0f} mW > {IDLE_WARN_CPU} mW")
    if idle["gpu_mw"] > IDLE_WARN_GPU:
        warnings.append(f"GPU idle {idle['gpu_mw']:.0f} mW > {IDLE_WARN_GPU} mW")
    if idle["ane_mw"] > IDLE_WARN_ANE:
        warnings.append(f"ANE idle {idle['ane_mw']:.0f} mW > {IDLE_WARN_ANE} mW")

    if warnings:
        print(f"  ⚠️  System not quiescent:")
        for w in warnings:
            print(f"     - {w}")
        print(f"  Idle baseline: CPU={idle['cpu_mw']:.0f} GPU={idle['gpu_mw']:.0f} "
              f"ANE={idle['ane_mw']:.0f} combined={idle['combined_mw']:.0f} mW")
    else:
        print(f"  ✅ Idle baseline: CPU={idle['cpu_mw']:.0f} GPU={idle['gpu_mw']:.0f} "
              f"ANE={idle['ane_mw']:.0f} combined={idle['combined_mw']:.0f} mW "
              f"[{', '.join(idle['thermal'])}]")

    return len(warnings) == 0, idle


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def fmt_power(mw: float) -> str:
    """Format milliwatts as W with 1 decimal."""
    if mw >= 1000:
        return f"{mw / 1000:.1f}W"
    return f"{mw:.0f}mW"


def fmt_energy(mj: float) -> str:
    """Format millijoules as mJ or J."""
    if mj >= 1000:
        return f"{mj / 1000:.2f}J"
    return f"{mj:.0f}mJ"


def print_summary_table(results: list[BenchResult], idle: dict):
    """Print a clean summary table to stdout."""
    if not results:
        print("\nNo results to display.")
        return

    print("\n" + "=" * 120)
    print("ENERGY BENCHMARK RESULTS")
    print("=" * 120)

    if idle:
        print(f"Idle baseline: CPU={idle.get('cpu_mw', 0):.0f}mW  "
              f"GPU={idle.get('gpu_mw', 0):.0f}mW  "
              f"ANE={idle.get('ane_mw', 0):.0f}mW  "
              f"Combined={idle.get('combined_mw', 0):.0f}mW")
        print("-" * 120)

    # Header
    print(f"{'Config':<12} {'Test':<14} {'Prompt':>6} {'Gen':>4} "
          f"{'Prefill':>8} {'Decode':>7} "
          f"{'CPU':>7} {'GPU':>7} {'ANE':>7} {'Total':>7} "
          f"{'Energy':>8} {'tok/J':>7} {'GEMM':>6} {'Attn':>6} "
          f"{'Therm':>8} {'Wall':>5}")
    print(f"{'':12} {'':14} {'tok':>6} {'tok':>4} "
          f"{'tok/s':>8} {'tok/s':>7} "
          f"{'avg':>7} {'avg':>7} {'avg':>7} {'avg':>7} "
          f"{'total':>8} {'pfill':>7} {'ms':>6} {'ms':>6} "
          f"{'':>8} {'s':>5}")
    print("-" * 120)

    for r in results:
        if r.error:
            print(f"{r.config:<12} {r.label:<14} {'ERROR':>6}  {r.error}")
            continue

        thermal = r.thermal_pressures[0] if r.thermal_pressures else "?"
        if len(set(r.thermal_pressures)) > 1:
            thermal = "→".join(sorted(set(r.thermal_pressures)))

        gemm_s = f"{r.prefill['gemm_ms']:.0f}" if r.prefill and r.prefill.get("gemm_ms") else "-"
        attn_s = f"{r.prefill['attn_ms']:.0f}" if r.prefill and r.prefill.get("attn_ms") else "-"

        print(
            f"{r.config:<12} {r.label:<14} "
            f"{r.prompt_tokens:>6} {r.gen_tokens:>4} "
            f"{r.prefill_tok_s:>8.1f} {r.decode_tok_s:>7.1f} "
            f"{fmt_power(r.avg_cpu_mw):>7} {fmt_power(r.avg_gpu_mw):>7} "
            f"{fmt_power(r.avg_ane_mw):>7} {fmt_power(r.avg_combined_mw):>7} "
            f"{fmt_energy(r.total_energy_mj):>8} {r.prefill_tok_per_joule:>7.0f} "
            f"{gemm_s:>6} {attn_s:>6} "
            f"{thermal:>8} {r.wall_time_s:>5.1f}"
        )

    print("=" * 120)

    # Peak power summary
    if any(r.peak_combined_mw > 0 and not r.error for r in results):
        print("\nPeak power (mW):")
        print(f"  {'Config':<12} {'Test':<14} {'CPU':>8} {'GPU':>8} {'ANE':>8} {'Combined':>10}")
        for r in results:
            if r.error or r.peak_combined_mw == 0:
                continue
            print(f"  {r.config:<12} {r.label:<14} "
                  f"{r.peak_cpu_mw:>8.0f} {r.peak_gpu_mw:>8.0f} "
                  f"{r.peak_ane_mw:>8.0f} {r.peak_combined_mw:>10.0f}")


def save_results(results: list[BenchResult], idle: dict):
    """Save detailed JSON results with timestamp."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"energy_{ts}.json"

    data = {
        "timestamp": datetime.now().isoformat(),
        "idle_baseline": idle,
        "results": [asdict(r) for r in results],
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    print(f"\n📁 Results saved to {path}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_test_plan(args) -> list[dict]:
    """Build list of test specs from CLI args."""
    tests = []

    # Determine models and prompt sizes
    if args.models:
        models = args.models
    elif args.quick:
        models = QUICK_MODELS
    else:
        models = list(MODEL_CONFIGS.keys())

    if args.prompt_sizes:
        sizes = args.prompt_sizes
    elif args.quick:
        sizes = QUICK_PROMPT_SIZES
    else:
        sizes = DEFAULT_PROMPT_SIZES

    # Validate model names
    for m in models:
        if m not in MODEL_CONFIGS:
            print(f"❌ Unknown model config: {m}")
            print(f"   Available: {', '.join(MODEL_CONFIGS.keys())}")
            sys.exit(1)

    # Prefill sweep
    for model in models:
        for size in sizes:
            tests.append({
                "config": model,
                "prompt_size": size,
                "max_tokens": 5,
                "label": f"prefill-{size}",
            })

    # Decode tests
    if args.decode:
        decode_models = args.models if args.models else (QUICK_MODELS if args.quick else list(MODEL_CONFIGS.keys()))
        for model in decode_models:
            for dt in DECODE_TESTS:
                tests.append({
                    "config": model,
                    "prompt_size": dt["prompt_words"],
                    "max_tokens": dt["max_tokens"],
                    "label": dt["label"],
                })

    return tests



# ---------------------------------------------------------------------------
# Sustained mode — multi-turn workload via serve mode
# ---------------------------------------------------------------------------

@dataclass
class TurnResult:
    """One conversation turn's metrics."""
    turn: int = 0
    prompt: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    wall_s: float = 0.0
    prefill_tok_s: float = 0.0
    decode_tok_s: float = 0.0
    error: Optional[str] = None


@dataclass
class PowerWindow:
    """Aggregated power over a time window."""
    start_s: float = 0.0
    end_s: float = 0.0
    avg_cpu_mw: float = 0.0
    avg_gpu_mw: float = 0.0
    avg_ane_mw: float = 0.0
    avg_combined_mw: float = 0.0
    peak_combined_mw: float = 0.0
    total_energy_mj: float = 0.0
    thermal_pressure: str = "Unknown"
    samples: int = 0


@dataclass
class SustainedResult:
    """Full sustained benchmark result."""
    config: str = ""
    duration_min: float = 0.0
    wall_time_s: float = 0.0
    turns_completed: int = 0
    turns_failed: int = 0
    # Throughput aggregates
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    avg_prefill_tok_s: float = 0.0
    avg_decode_tok_s: float = 0.0
    # Power aggregates
    avg_cpu_mw: float = 0.0
    avg_gpu_mw: float = 0.0
    avg_ane_mw: float = 0.0
    avg_combined_mw: float = 0.0
    peak_combined_mw: float = 0.0
    total_energy_j: float = 0.0
    # Efficiency
    total_tok_per_joule: float = 0.0
    prompt_tok_per_joule: float = 0.0
    decode_tok_per_joule: float = 0.0
    # Time series
    power_windows: list = field(default_factory=list)
    turns: list = field(default_factory=list)
    thermal_pressures: list = field(default_factory=list)
    error: Optional[str] = None


def _wait_for_server(port: int, timeout: int = SERVER_STARTUP_TIMEOUT) -> bool:
    """Poll /v1/models until server is ready."""
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/v1/models"
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionRefusedError, OSError, http.client.RemoteDisconnected):
            pass
        if attempt % 10 == 0:
            elapsed = time.monotonic() - (deadline - timeout)
            print(f"     still loading... ({elapsed:.0f}s)", flush=True)
        time.sleep(1)
    return False


def _send_turn(port: int, messages: list[dict], max_tokens: int = 200,
               timeout: int = 60) -> TurnResult:
    """Send one chat completion request (non-streaming) and parse the response."""
    result = TurnResult()
    body = json.dumps({
        "model": "bench",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    wall_start = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        result.wall_s = time.monotonic() - wall_start
        result.error = str(e)
        return result

    result.wall_s = time.monotonic() - wall_start

    usage = data.get("usage", {})
    result.prompt_tokens = usage.get("prompt_tokens", 0)
    result.completion_tokens = usage.get("completion_tokens", 0)
    result.total_tokens = usage.get("total_tokens", 0)

    # Use server-reported timings if available (llama.cpp provides these)
    timings = data.get("timings")
    if timings and timings.get("prompt_per_second"):
        result.prefill_tok_s = timings["prompt_per_second"]
        result.decode_tok_s = timings.get("predicted_per_second", 0)
    elif result.wall_s > 0 and result.prompt_tokens > 0:
        # Estimate for ane.cpp (no server-side timings)
        decode_time_est = result.completion_tokens / 13.0 if result.completion_tokens > 0 else 0
        prefill_time_est = max(0.01, result.wall_s - decode_time_est)
        result.prefill_tok_s = result.prompt_tokens / prefill_time_est
        result.decode_tok_s = result.completion_tokens / decode_time_est if decode_time_est > 0 else 0

    # Extract assistant reply for conversation context
    choices = data.get("choices", [])
    if choices:
        msg = choices[0].get("message", {})
        result.prompt = msg.get("content", "")

    return result


def _power_windows_from_samples(samples: list, wall_start: float,
                                 window_sec: float = POWER_WINDOW_SEC) -> list:
    """Bin power samples into time windows."""
    if not samples:
        return []

    windows = []
    # Compute cumulative time from elapsed_ns
    cumulative_s = 0.0
    timed_samples = []
    for s in samples:
        cumulative_s += s.elapsed_ns / 1e9
        timed_samples.append((cumulative_s, s))

    total_duration = cumulative_s
    n_windows = max(1, int(total_duration / window_sec))

    for w in range(n_windows):
        t0 = w * window_sec
        t1 = min((w + 1) * window_sec, total_duration + 0.1)
        bucket = [s for (t, s) in timed_samples if t0 <= t < t1]
        if not bucket:
            continue
        n = len(bucket)
        pw = PowerWindow(
            start_s=round(t0, 1),
            end_s=round(t1, 1),
            avg_cpu_mw=sum(s.cpu_power for s in bucket) / n,
            avg_gpu_mw=sum(s.gpu_power for s in bucket) / n,
            avg_ane_mw=sum(s.ane_power for s in bucket) / n,
            avg_combined_mw=sum(s.combined_power for s in bucket) / n,
            peak_combined_mw=max(s.combined_power for s in bucket),
            total_energy_mj=(sum(s.cpu_energy for s in bucket)
                             + sum(s.gpu_energy for s in bucket)
                             + sum(s.ane_energy for s in bucket)),
            thermal_pressure=bucket[-1].thermal_pressure,
            samples=n,
        )
        windows.append(pw)

    return windows


def _start_ane_server(config_key: str, port: int, n_sessions: int = SUSTAINED_SESSIONS):
    """Start ane.cpp serve mode. Returns (proc, display_name) or raises."""
    display_name, model_dir, env_vars = MODEL_CONFIGS[config_key]
    model_path = MODELS_DIR / model_dir
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    server_env = {**os.environ, **env_vars}
    server_cmd = [
        str(BINARY), "serve",
        "--model", str(model_path),
        "--port", str(port),
        "--sessions", str(n_sessions),
        "--temp", "0.7",
    ]
    log = tempfile.NamedTemporaryFile(delete=False, suffix=".log", mode="w")
    proc = subprocess.Popen(
        server_cmd, stdout=subprocess.DEVNULL, stderr=log,
        env=server_env, cwd=str(REPO_DIR), preexec_fn=os.setpgrp,
    )
    return proc, display_name, log.name


def _start_llama_server(config_key: str, port: int):
    """Start llama-server. Returns (proc, display_name) or raises."""
    if not LLAMA_SERVER.exists():
        raise FileNotFoundError(f"llama-server not found: {LLAMA_SERVER}")

    cfg = LLAMA_CONFIGS[config_key]
    display_name, gguf_rel, ngl = cfg
    if gguf_rel is None:
        raise FileNotFoundError(f"No GGUF found for {config_key}")

    gguf_path = Path(gguf_rel) if Path(gguf_rel).is_absolute() else MODELS_DIR / gguf_rel
    if not gguf_path.exists():
        raise FileNotFoundError(f"GGUF not found: {gguf_path}")

    server_cmd = [
        str(LLAMA_SERVER),
        "-m", str(gguf_path),
        "--port", str(port),
        "-ngl", str(ngl),
        "-c", "8192",
        "--temp", "0.7",
    ]
    log = tempfile.NamedTemporaryFile(delete=False, suffix=".log", mode="w")
    proc = subprocess.Popen(
        server_cmd, stdout=subprocess.DEVNULL, stderr=log,
        preexec_fn=os.setpgrp,
    )
    return proc, display_name, log.name



def _run_conversation_worker(worker_id: int, port: int, deadline: float,
                             max_tokens: int, results_list: list,
                             print_lock: threading.Lock):
    """Worker thread: runs one conversation loop until deadline."""
    conversation = []
    turn_num = 0
    prompt_offset = worker_id * 3  # stagger prompts across workers

    while time.monotonic() < deadline:
        prompt_text = TURN_PROMPTS[(turn_num + prompt_offset) % len(TURN_PROMPTS)]
        messages = conversation[-8:] + [{"role": "user", "content": prompt_text}]

        turn_num += 1
        per_turn_timeout = min(90, max(30, int(deadline - time.monotonic())))
        if per_turn_timeout < 5:
            break

        tr = _send_turn(port, messages, max_tokens=max_tokens, timeout=per_turn_timeout)
        tr.turn = turn_num

        q = "?"
        with print_lock:
            if tr.error:
                print(f"  W{worker_id:>1}:{turn_num:>3} {q:>7} {q:>5} {tr.wall_s:>5.1f}s \u274c {tr.error}")
            else:
                print(f"  W{worker_id:>1}:{turn_num:>3} {tr.prompt_tokens:>5}pt {tr.completion_tokens:>4}gt "
                      f"{tr.wall_s:>5.1f}s \u2705", flush=True)

        if not tr.error:
            conversation.append({"role": "user", "content": prompt_text})
            if tr.prompt:
                conversation.append({"role": "assistant", "content": tr.prompt})

        results_list.append(tr)


def run_sustained(config_key: str, duration_min: float = 5.0,
                  max_tokens: int = 200, port: int = SUSTAINED_PORT,
                  save: bool = False,
                  backend: str = "ane",
                  n_parallel: int = 1) -> SustainedResult:
    """Run a sustained multi-turn workload with continuous power monitoring.
    
    backend: "ane" for ane.cpp serve, "llama" for llama-server
    n_parallel: number of concurrent conversation streams
    """
    par_label = f" \u00d7{n_parallel}" if n_parallel > 1 else ""
    result = SustainedResult(config=f"{config_key} ({backend}{par_label})", duration_min=duration_min)

    # --- Start server ---
    try:
        if backend == "llama":
            server_proc, display_name, log_path = _start_llama_server(config_key, port)
        else:
            n_sessions = max(n_parallel, SUSTAINED_SESSIONS)
            server_proc, display_name, log_path = _start_ane_server(config_key, port, n_sessions=n_sessions)
    except FileNotFoundError as e:
        result.error = str(e)
        return result

    print(f"  🔄 Starting {backend} server ({display_name})...")

    print(f"  ⏳ Waiting for model load...", flush=True)
    if not _wait_for_server(port):
        result.error = "Server failed to start within timeout"
        _kill_proc(server_proc)
        try:
            os.unlink(log_path)
        except OSError:
            pass
        return result
    print(f"  ✅ Server ready on port {port}")

    # --- Start powermetrics ---
    pm_out = tempfile.NamedTemporaryFile(delete=False, suffix=".plist")
    pm_out.close()
    pm_proc = subprocess.Popen(
        POWERMETRICS_CMD,
        stdout=open(pm_out.name, "wb"),
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setpgrp,
    )
    time.sleep(0.3)

    # --- Run conversation loop(s) ---
    duration_s = duration_min * 60
    wall_start = time.monotonic()
    deadline = wall_start + duration_s

    print(f"  \U0001f3cb\ufe0f  Running {duration_min:.1f}min sustained workload "
          f"(max_tokens={max_tokens}, {n_parallel} parallel)...")

    turn_results = []  # thread-safe via GIL for list.append
    print_lock = threading.Lock()

    if n_parallel == 1:
        # Single-threaded path
        print(f"  {'Turn':>5} {'Prompt':>7} {'Gen':>5} {'Wall':>6} {'Status'}", flush=True)
        _run_conversation_worker(0, port, deadline, max_tokens, turn_results, print_lock)
    else:
        print(f"  {'Worker':>7} {'Prompt':>7} {'Gen':>5} {'Wall':>6} {'Status'}", flush=True)
        threads = []
        for w in range(n_parallel):
            t = threading.Thread(
                target=_run_conversation_worker,
                args=(w, port, deadline, max_tokens, turn_results, print_lock),
                daemon=True,
            )
            threads.append(t)
            t.start()
        for t in threads:
            t.join(timeout=duration_s + 30)

    wall_end = time.monotonic()
    result.wall_time_s = round(wall_end - wall_start, 3)

    # --- Stop powermetrics + server ---
    _kill_proc(pm_proc)
    time.sleep(0.1)
    _kill_proc(server_proc)

    # --- Parse power samples ---
    try:
        with open(pm_out.name, "rb") as f:
            raw = f.read()
        power_samples = parse_plist_samples(raw)
    except Exception:
        power_samples = []
    finally:
        try:
            os.unlink(pm_out.name)
        except OSError:
            pass
        try:
            os.unlink(log_path)
        except OSError:
            pass

    # --- Aggregate turn metrics ---
    ok_turns = [t for t in turn_results if not t.error]
    result.turns_completed = len(ok_turns)
    result.turns_failed = len(turn_results) - len(ok_turns)
    result.total_prompt_tokens = sum(t.prompt_tokens for t in ok_turns)
    result.total_completion_tokens = sum(t.completion_tokens for t in ok_turns)
    result.total_tokens = sum(t.total_tokens for t in ok_turns)
    result.turns = [asdict(t) for t in turn_results]

    if ok_turns:
        total_decode_time = sum(
            t.completion_tokens / t.decode_tok_s
            for t in ok_turns if t.decode_tok_s > 0
        )
        total_prefill_time = sum(
            t.prompt_tokens / t.prefill_tok_s
            for t in ok_turns if t.prefill_tok_s > 0
        )
        if total_prefill_time > 0:
            result.avg_prefill_tok_s = round(
                result.total_prompt_tokens / total_prefill_time, 1)
        if total_decode_time > 0:
            result.avg_decode_tok_s = round(
                result.total_completion_tokens / total_decode_time, 1)

    # --- Aggregate power ---
    if power_samples:
        stats = summarize_power(power_samples)
        result.avg_cpu_mw = stats["avg_cpu_mw"]
        result.avg_gpu_mw = stats["avg_gpu_mw"]
        result.avg_ane_mw = stats["avg_ane_mw"]
        result.avg_combined_mw = stats["avg_combined_mw"]
        result.peak_combined_mw = stats["peak_combined_mw"]
        result.total_energy_j = round(stats["total_energy_mj"] / 1000.0, 2)
        result.thermal_pressures = stats["thermal_pressures"]

        # Efficiency
        if result.total_energy_j > 0:
            result.total_tok_per_joule = round(
                result.total_tokens / result.total_energy_j, 1)
            result.prompt_tok_per_joule = round(
                result.total_prompt_tokens / result.total_energy_j, 1)
            result.decode_tok_per_joule = round(
                result.total_completion_tokens / result.total_energy_j, 1)

        # Power windows
        result.power_windows = [
            asdict(w) for w in _power_windows_from_samples(
                power_samples, wall_start)
        ]

    return result


def _kill_proc(proc):
    """Kill a subprocess and its process group."""
    if proc and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=5)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait(timeout=3)
            except Exception:
                pass


def print_sustained_report(r: SustainedResult, idle: dict):
    """Print sustained benchmark report."""
    print("\n" + "=" * 90)
    print(f"SUSTAINED WORKLOAD REPORT — {r.config}")
    print("=" * 90)

    if r.error:
        print(f"❌ {r.error}")
        return

    # Summary
    print(f"Duration: {r.wall_time_s:.0f}s ({r.wall_time_s/60:.1f}min)  "
          f"Turns: {r.turns_completed} ok, {r.turns_failed} failed")
    print(f"Tokens:  {r.total_prompt_tokens} prompt + "
          f"{r.total_completion_tokens} completion = {r.total_tokens} total")
    print(f"Speed:   {r.avg_prefill_tok_s:.0f} prefill tok/s (avg), "
          f"{r.avg_decode_tok_s:.1f} decode tok/s (avg)")
    print()

    # Power summary
    print("Power (averaged over full run):")
    print(f"  CPU:      {fmt_power(r.avg_cpu_mw):>8}  (peak {fmt_power(r.peak_combined_mw)})")
    print(f"  GPU:      {fmt_power(r.avg_gpu_mw):>8}")
    print(f"  ANE:      {fmt_power(r.avg_ane_mw):>8}")
    print(f"  Combined: {fmt_power(r.avg_combined_mw):>8}")
    print(f"  Energy:   {r.total_energy_j:.1f} J")
    print()

    # Efficiency
    print("Efficiency:")
    print(f"  Total:    {r.total_tok_per_joule:.1f} tok/J")
    print(f"  Prefill:  {r.prompt_tok_per_joule:.1f} tok/J")
    print(f"  Decode:   {r.decode_tok_per_joule:.1f} tok/J")
    print()

    # Power timeline
    if r.power_windows:
        print(f"Power timeline ({POWER_WINDOW_SEC}s windows):")
        print(f"  {'Window':>8} {'CPU':>8} {'GPU':>8} {'ANE':>8} "
              f"{'Combined':>10} {'Energy':>8} {'Thermal':>10}")
        print(f"  {'':>8} {'avg':>8} {'avg':>8} {'avg':>8} "
              f"{'avg':>10} {'total':>8} {'':>10}")
        print("  " + "-" * 75)
        for w in r.power_windows:
            t0, t1 = w["start_s"], w["end_s"]
            label = f"{t0:.0f}-{t1:.0f}s"
            print(f"  {label:>8} {fmt_power(w['avg_cpu_mw']):>8} "
                  f"{fmt_power(w['avg_gpu_mw']):>8} "
                  f"{fmt_power(w['avg_ane_mw']):>8} "
                  f"{fmt_power(w['avg_combined_mw']):>10} "
                  f"{fmt_energy(w['total_energy_mj']):>8} "
                  f"{w['thermal_pressure']:>10}")
        print()

    # Thermal
    pressures = set(r.thermal_pressures) if r.thermal_pressures else set()
    if len(pressures) > 1:
        print(f"⚠️  Thermal changed during run: {' → '.join(sorted(pressures))}")
    elif pressures:
        print(f"Thermal: {pressures.pop()}")

    # Idle comparison
    if idle:
        idle_combined = idle.get("combined_mw", 0)
        if idle_combined > 0 and r.avg_combined_mw > 0:
            overhead = r.avg_combined_mw - idle_combined
            print(f"Inference overhead: {fmt_power(overhead)} above idle "
                  f"({fmt_power(idle_combined)})")

    print("=" * 90)


def print_comparison(a: SustainedResult, b: SustainedResult):
    """Side-by-side comparison of two sustained results."""
    print("\n" + "=" * 90)
    print("SIDE-BY-SIDE COMPARISON")
    print("=" * 90)

    def _pct(va, vb):
        if vb and vb > 0:
            diff = ((va / vb) - 1) * 100
            return f"{'+'if diff>=0 else ''}{diff:.0f}%"
        return "—"

    label_a = a.config
    label_b = b.config
    w = max(len(label_a), len(label_b), 22)

    print(f"  {'Metric':<24} {label_a:>{w}} {label_b:>{w}} {'Δ':>8}")
    print("  " + "-" * (24 + w * 2 + 10))

    rows = [
        ("Turns completed",    a.turns_completed,        b.turns_completed, "", True),
        ("Total tokens",       a.total_tokens,           b.total_tokens, "", True),
        ("Prompt tokens",      a.total_prompt_tokens,    b.total_prompt_tokens, "", True),
        ("Gen tokens",         a.total_completion_tokens, b.total_completion_tokens, "", True),
        ("Avg prefill tok/s",  a.avg_prefill_tok_s,      b.avg_prefill_tok_s, "", False),
        ("Avg decode tok/s",   a.avg_decode_tok_s,       b.avg_decode_tok_s, "", False),
        ("Avg combined power", a.avg_combined_mw,        b.avg_combined_mw, "mW", False),
        ("Peak combined power", a.peak_combined_mw,      b.peak_combined_mw, "mW", False),
        ("Total energy",       a.total_energy_j,         b.total_energy_j, "J", False),
        ("Total tok/J",        a.total_tok_per_joule,    b.total_tok_per_joule, "", False),
        ("Decode tok/J",       a.decode_tok_per_joule,   b.decode_tok_per_joule, "", False),
    ]

    for label, va, vb, unit, is_int in rows:
        if is_int:
            sa, sb = f"{int(va)}", f"{int(vb)}"
        elif unit == "mW":
            sa, sb = fmt_power(va), fmt_power(vb)
        elif unit == "J":
            sa, sb = f"{va:.1f}J", f"{vb:.1f}J"
        else:
            sa, sb = f"{va:.1f}", f"{vb:.1f}"

        pct = _pct(va, vb) if vb else "—"
        print(f"  {label:<24} {sa:>{w}} {sb:>{w}} {pct:>8}")

    print("=" * 90)

def main():
    parser = argparse.ArgumentParser(
        description="Energy benchmark harness for ane.cpp",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --quick                    Quick sweep (312 tok, 4B only)
  %(prog)s --full --save              Full sweep, save JSON
  %(prog)s --models 4B-INT8 9B-INT8   Compare INT8 across sizes
  %(prog)s --prompt-sizes 112 1412    Custom prompt sizes
  %(prog)s --decode --models 4B-INT8  Decode-heavy tests

Model configs: 4B-INT8, 4B-FP16, 9B-INT8, 9B-FP16
        """,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true",
                      help="Quick mode: 312 tokens, 4B INT8/FP16 only")
    mode.add_argument("--full", action="store_true",
                      help="Full sweep: all models, all prompt sizes")
    mode.add_argument("--sustained", type=float, metavar="MIN",
                      help="Sustained multi-turn workload for MIN minutes")
    parser.add_argument("--models", nargs="+", metavar="CFG",
                        help="Model configs to test (e.g. 4B-INT8 9B-FP16)")
    parser.add_argument("--prompt-sizes", nargs="+", type=int, metavar="N",
                        help="Prompt sizes in tokens (e.g. 112 312 612)")
    parser.add_argument("--decode", action="store_true",
                        help="Include decode-heavy tests")
    parser.add_argument("--save", action="store_true",
                        help="Save detailed JSON results")
    parser.add_argument("--timeout", type=int, default=BENCH_TIMEOUT,
                        help=f"Per-test timeout in seconds (default: {BENCH_TIMEOUT})")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="Skip idle power pre-flight check")
    parser.add_argument("--warmup", action="store_true",
                        help="Run a throwaway warmup pass before timing")
    parser.add_argument("--port", type=int, default=SUSTAINED_PORT,
                        help=f"Server port for sustained mode (default: {SUSTAINED_PORT})")
    parser.add_argument("--max-gen", type=int, default=200,
                        help="Max generation tokens per turn in sustained mode (default: 200)")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of parallel conversation streams (default: 1)")
    parser.add_argument("--backend", choices=["ane", "llama"], default="ane",
                        help="Server backend for sustained mode (default: ane)")
    parser.add_argument("--llama-model", metavar="CFG",
                        help="llama.cpp model config for --compare (e.g. 4B-Q8_0)")
    parser.add_argument("--compare", action="store_true",
                        help="Run sustained on both ane.cpp and llama.cpp, show side-by-side")
    args = parser.parse_args()

    # Default to --quick if nothing specified
    if (not args.quick and not args.full and not args.models
            and not args.prompt_sizes and not args.sustained):
        args.quick = True

    # Verify binary exists
    if not BINARY.exists():
        print(f"❌ Binary not found: {BINARY}")
        print(f"   Build with: cd {REPO_DIR} && make")
        sys.exit(1)

    # Pre-flight
    idle = {}
    if not args.skip_preflight:
        ok, idle = preflight_check()
        if not ok:
            resp = input("  Continue anyway? [Y/n] ").strip().lower()
            if resp == "n":
                sys.exit(1)
        print()

    # --- Sustained mode ---
    if args.sustained:
        duration = args.sustained
        ane_model = args.models[0] if args.models else "4B-INT8"
        if ane_model not in MODEL_CONFIGS:
            print(f"❌ Unknown ane.cpp config: {ane_model}")
            print(f"   Available: {', '.join(MODEL_CONFIGS.keys())}")
            sys.exit(1)

        if args.compare:
            # Run both backends, then compare
            llama_model = args.llama_model or "4B-Q8_0"
            if llama_model not in LLAMA_CONFIGS:
                print(f"❌ Unknown llama config: {llama_model}")
                print(f"   Available: {', '.join(LLAMA_CONFIGS.keys())}")
                sys.exit(1)

            print(f"⚔️  Compare mode: ane.cpp vs llama.cpp for {duration:.1f} min\n")

            # ane.cpp first
            print(f"{'─' * 40} ane.cpp {'─' * 40}")
            r_ane = run_sustained(
                ane_model, duration_min=duration,
                max_tokens=args.max_gen, port=args.port,
                backend="ane",
                n_parallel=args.parallel,
            )
            print_sustained_report(r_ane, idle)

            # Cool down between runs
            print("\n⏸️  Cooling down 10s between backends...\n")
            time.sleep(10)

            # Re-check idle if not skipped
            if not args.skip_preflight:
                _, idle2 = preflight_check(duration_s=1.5)
                idle = idle2 or idle
                print()

            # llama.cpp second — use next port to avoid conflicts
            print(f"{'─' * 40} llama.cpp {'─' * 39}")
            r_llama = run_sustained(
                llama_model, duration_min=duration,
                max_tokens=args.max_gen, port=args.port + 1,
                backend="llama",
                n_parallel=1,  # llama.cpp baseline always single-stream
            )
            print_sustained_report(r_llama, idle)

            # Side-by-side comparison
            print_comparison(r_ane, r_llama)

            if args.save:
                RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = RESULTS_DIR / f"compare_{ts}.json"
                with open(path, "w") as f:
                    json.dump({
                        "ane": asdict(r_ane),
                        "llama": asdict(r_llama),
                        "idle_baseline": idle,
                    }, f, indent=2, default=str)
                print(f"\n📁 Results saved to {path}")
            sys.exit(1 if (r_ane.error or r_llama.error) else 0)

        else:
            # Single backend
            backend = args.backend
            model = ane_model if backend == "ane" else (args.llama_model or "4B-Q8_0")
            configs = MODEL_CONFIGS if backend == "ane" else LLAMA_CONFIGS
            if model not in configs:
                print(f"❌ Unknown {backend} config: {model}")
                print(f"   Available: {', '.join(configs.keys())}")
                sys.exit(1)

            print(f"🔄 Sustained mode ({backend}): {configs[model][0]} for "
                  f"{duration:.1f} min\n")
            r = run_sustained(
                model, duration_min=duration,
                max_tokens=args.max_gen, port=args.port,
                backend=backend,
                n_parallel=args.parallel,
            )
            print_sustained_report(r, idle)
            if args.save:
                RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = RESULTS_DIR / f"sustained_{backend}_{ts}.json"
                with open(path, "w") as f:
                    json.dump(asdict(r), f, indent=2, default=str)
                print(f"\n📁 Results saved to {path}")
            sys.exit(1 if r.error else 0)

    # Build and display test plan
    tests = build_test_plan(args)
    if not tests:
        print("❌ No tests to run.")
        sys.exit(1)

    print(f"📋 Test plan: {len(tests)} tests")
    for t in tests:
        cfg = MODEL_CONFIGS[t["config"]]
        print(f"   {t['config']:<12} {t['label']:<14} "
              f"prompt≈{t['prompt_size']} max_tokens={t['max_tokens']}")
    print()

    # Optional warmup
    if args.warmup:
        print("🔥 Warmup pass...")
        warmup_cfg = tests[0]["config"]
        run_benchmark(warmup_cfg, prompt_size=112, max_tokens=3, label="warmup", timeout=60)
        print("   Done.\n")

    # Run tests
    results = []
    for i, t in enumerate(tests, 1):
        cfg_display = MODEL_CONFIGS[t["config"]][0]
        print(f"[{i}/{len(tests)}] {cfg_display} — {t['label']}...", end=" ", flush=True)

        r = run_benchmark(
            config_key=t["config"],
            prompt_size=t["prompt_size"],
            max_tokens=t["max_tokens"],
            label=t["label"],
            timeout=args.timeout,
        )
        results.append(r)

        if r.error:
            print(f"❌ {r.error}")
        else:
            print(
                f"✅ {r.prefill_tok_s:.0f} p/s, {r.decode_tok_s:.1f} d/s | "
                f"CPU={fmt_power(r.avg_cpu_mw)} GPU={fmt_power(r.avg_gpu_mw)} "
                f"ANE={fmt_power(r.avg_ane_mw)} | "
                f"{fmt_energy(r.total_energy_mj)} total | "
                f"{r.wall_time_s:.1f}s"
            )

        # Brief pause between tests to let thermals settle
        if i < len(tests):
            time.sleep(1)

    # Summary
    print_summary_table(results, idle)

    # Save
    if args.save:
        save_results(results, idle)

    # Exit code: 1 if any test had errors
    if any(r.error for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
