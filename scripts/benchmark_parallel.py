#!/usr/bin/env python3
import argparse
import asyncio
import json
import math
import os
import re
import statistics
import sys
import time
from itertools import cycle, islice
from pathlib import Path

PROMPT_RE = re.compile(r"Prompt:\s+(\d+)\s+tokens,\s+([0-9.]+)\s+tokens-per-sec")
GEN_RE = re.compile(r"Generation:\s+(\d+)\s+tokens,\s+([0-9.]+)\s+tokens-per-sec")

DEFAULT_PROMPTS = [
    "Explain the theory of general relativity in detail, covering spacetime curvature, the equivalence principle, gravitational time dilation, and experimental confirmations.",
    "Write a concise but technically accurate explanation of how transformers use self-attention during autoregressive decoding.",
    "Summarize the main engineering tradeoffs between CPU, GPU, and Apple Neural Engine inference for local language models.",
    "Explain why batching improves throughput for memory-bandwidth-bound inference workloads.",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to model directory")
    p.add_argument("--requests", type=int, default=8, help="Total requests to run")
    p.add_argument("--concurrency", type=int, default=2, help="Concurrent requests")
    p.add_argument("--max-tokens", type=int, default=100, help="Max tokens per request")
    p.add_argument("--temp", type=float, default=0.6, help="Sampling temperature")
    p.add_argument("--repeat-penalty", type=float, default=1.2, help="Repeat penalty")
    p.add_argument("--prompt", action="append", default=[], help="Prompt to use (repeatable)")
    p.add_argument("--prompts-file", help="Text file with one prompt per line")
    p.add_argument("--prefill-batch", type=int, help="Set ANE_PREFILL_BATCH for child processes")
    p.add_argument("--binary", default="./build/ane.cpp", help="Path to ane.cpp binary")
    p.add_argument("--server-host", default="127.0.0.1", help="Use ane.cpp serve mode at this host")
    p.add_argument("--server-port", type=int, help="Use ane.cpp serve mode at this port")
    p.add_argument("--json-out", help="Write full results JSON to this file")
    p.add_argument("--verbose-errors", action="store_true")
    return p.parse_args()


def load_prompts(args: argparse.Namespace) -> list[str]:
    prompts: list[str] = []
    prompts.extend([p.strip() for p in args.prompt if p.strip()])
    if args.prompts_file:
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            prompts.extend([line.strip() for line in f if line.strip()])
    if not prompts:
        prompts = DEFAULT_PROMPTS.copy()
    return prompts


def parse_metrics(stderr_text: str) -> dict:
    prompt_match = PROMPT_RE.search(stderr_text)
    gen_match = GEN_RE.search(stderr_text)
    return {
        "prompt_tokens": int(prompt_match.group(1)) if prompt_match else None,
        "prompt_tps": float(prompt_match.group(2)) if prompt_match else None,
        "generation_tokens": int(gen_match.group(1)) if gen_match else None,
        "generation_tps": float(gen_match.group(2)) if gen_match else None,
    }


async def run_one(index: int, prompt: str, args: argparse.Namespace, sem: asyncio.Semaphore, env: dict) -> dict:
    async with sem:
        start = time.perf_counter()
        if args.server_port:
            reader, writer = await asyncio.open_connection(args.server_host, args.server_port)
            payload = {
                "id": index,
                "prompt": prompt,
                "max_tokens": args.max_tokens,
                "temp": args.temp,
                "repeat_penalty": args.repeat_penalty,
            }
            writer.write((json.dumps(payload) + "\n").encode("utf-8"))
            await writer.drain()
            response_line = await reader.readline()
            end = time.perf_counter()
            writer.close()
            await writer.wait_closed()

            stdout = ""
            stderr = response_line.decode("utf-8", errors="replace")
            response = json.loads(stderr) if stderr.strip() else {}
            metrics = {
                "prompt_tokens": response.get("prompt_tokens"),
                "prompt_tps": response.get("prompt_tps"),
                "generation_tokens": response.get("generation_tokens"),
                "generation_tps": response.get("generation_tps"),
            }
            result = {
                "index": index,
                "ok": bool(response.get("ok")) and metrics["generation_tps"] is not None,
                "returncode": 0 if response.get("ok") else 1,
                "wall_seconds": end - start,
                "prompt": prompt,
                "stdout": stdout,
                "stderr": stderr,
                **metrics,
            }
        else:
            cmd = [
                args.binary,
                "generate",
                "--model",
                args.model,
                "--prompt",
                prompt,
                "--max-tokens",
                str(args.max_tokens),
                "--temp",
                str(args.temp),
                "--repeat-penalty",
                str(args.repeat_penalty),
            ]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            stdout_b, stderr_b = await proc.communicate()
            end = time.perf_counter()

            stdout = stdout_b.decode("utf-8", errors="replace")
            stderr = stderr_b.decode("utf-8", errors="replace")
            metrics = parse_metrics(stderr)
            result = {
                "index": index,
                "ok": proc.returncode == 0 and metrics["generation_tps"] is not None,
                "returncode": proc.returncode,
                "wall_seconds": end - start,
                "prompt": prompt,
                "stdout": stdout,
                "stderr": stderr,
                **metrics,
            }
    status = "ok" if result["ok"] else "fail"
    prompt_tps = "-" if metrics["prompt_tps"] is None else f"{metrics['prompt_tps']:.3f}"
    gen_tps = "-" if metrics["generation_tps"] is None else f"{metrics['generation_tps']:.3f}"
    gen_tokens = "-" if metrics["generation_tokens"] is None else str(metrics["generation_tokens"])
    print(
        f"[{index:03d}] {status} wall={result['wall_seconds']:.2f}s prompt_tps={prompt_tps} gen_tps={gen_tps} gen_tokens={gen_tokens}",
        flush=True,
    )
    if not result["ok"] and args.verbose_errors:
        print(f"--- stderr for request {index} ---\n{stderr}\n--- end stderr ---", file=sys.stderr, flush=True)
    return result


async def main_async() -> int:
    args = parse_args()
    prompts = load_prompts(args)
    if args.requests < 1:
        raise SystemExit("--requests must be >= 1")
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be >= 1")

    if not args.server_port:
        binary_path = Path(args.binary)
        if not binary_path.exists():
            raise SystemExit(f"binary not found: {binary_path}")
    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"model not found: {model_path}")

    env = os.environ.copy()
    if args.prefill_batch is not None:
        env["ANE_PREFILL_BATCH"] = str(args.prefill_batch)

    chosen_prompts = list(islice(cycle(prompts), args.requests))
    sem = asyncio.Semaphore(args.concurrency)

    started = time.perf_counter()
    results = await asyncio.gather(
        *(run_one(i + 1, prompt, args, sem, env) for i, prompt in enumerate(chosen_prompts))
    )
    finished = time.perf_counter()

    successes = [r for r in results if r["ok"]]
    failures = [r for r in results if not r["ok"]]

    print("\nsummary")
    print(f"  requests: {len(results)}")
    print(f"  concurrency: {args.concurrency}")
    print(f"  successes: {len(successes)}")
    print(f"  failures: {len(failures)}")
    print(f"  wall_time: {finished - started:.2f}s")

    if successes:
        wall_times = [r["wall_seconds"] for r in successes]
        prompt_tps = [r["prompt_tps"] for r in successes if r["prompt_tps"] is not None]
        gen_tps = [r["generation_tps"] for r in successes if r["generation_tps"] is not None]
        prompt_tokens = [r["prompt_tokens"] for r in successes if r["prompt_tokens"] is not None]
        gen_tokens = [r["generation_tokens"] for r in successes if r["generation_tokens"] is not None]

        total_prompt_tokens = sum(prompt_tokens)
        total_gen_tokens = sum(gen_tokens)
        aggregate_prompt_tps = total_prompt_tokens / (finished - started) if total_prompt_tokens else 0.0
        aggregate_gen_tps = total_gen_tokens / (finished - started) if total_gen_tokens else 0.0

        def pct(values: list[float], q: float) -> float:
            if len(values) == 1:
                return values[0]
            idx = (len(values) - 1) * q
            lo = math.floor(idx)
            hi = math.ceil(idx)
            if lo == hi:
                return sorted(values)[lo]
            vals = sorted(values)
            frac = idx - lo
            return vals[lo] * (1.0 - frac) + vals[hi] * frac

        print(f"  avg_wall: {statistics.mean(wall_times):.2f}s")
        print(f"  p50_wall: {statistics.median(wall_times):.2f}s")
        print(f"  p95_wall: {pct(wall_times, 0.95):.2f}s")
        print(f"  avg_prompt_tps: {statistics.mean(prompt_tps):.3f}")
        print(f"  avg_gen_tps: {statistics.mean(gen_tps):.3f}")
        print(f"  aggregate_prompt_tps: {aggregate_prompt_tps:.3f}")
        print(f"  aggregate_gen_tps: {aggregate_gen_tps:.3f}")
        print(f"  total_prompt_tokens: {total_prompt_tokens}")
        print(f"  total_gen_tokens: {total_gen_tokens}")

    if args.json_out:
        payload = {
            "config": {
                "model": args.model,
                "requests": args.requests,
                "concurrency": args.concurrency,
                "max_tokens": args.max_tokens,
                "temp": args.temp,
                "repeat_penalty": args.repeat_penalty,
                "prefill_batch": args.prefill_batch,
                "binary": args.binary,
                "server_host": args.server_host,
                "server_port": args.server_port,
            },
            "results": results,
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"  wrote_json: {args.json_out}")

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async()))
