"""Sustained Apple Silicon unified memory and latency test under continuous load."""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path

import mlx.core as mx

from jev_ultrafast.backends.mlx_direct import MlxDiffusionDirectBackend
from jev_ultrafast.model import action_space
from jev_ultrafast.recorder import DecisionRecorder


def get_process_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # On macOS ru_maxrss is in bytes
    return round(usage.ru_maxrss / (1024 * 1024), 2)


def get_memory_stats() -> dict[str, float]:
    return {
        "active_gb": round(mx.get_active_memory() / (1024**3), 3),
        "peak_gb": round(mx.get_peak_memory() / (1024**3), 3),
        "cache_gb": round(mx.get_cache_memory() / (1024**3), 3),
        "process_rss_mb": get_process_rss_mb(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="fixtures/recorded_decisions.jsonl")
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--canvas-length", type=int, default=32)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--output", default="results/sustained_memory.json")
    args = parser.parse_args()

    print("=" * 60)
    print("Sustained Apple Silicon Unified Memory Test")
    print(f"Total Steps : {args.max_steps}")
    print(f"PID         : {os.getpid()}")
    print(f"Initial Mem : {get_memory_stats()}")
    print("=" * 60)

    dataset = DecisionRecorder.load_dataset(args.dataset)
    if not dataset:
        raise FileNotFoundError(f"No records in {args.dataset}")

    backend = MlxDiffusionDirectBackend(
        canvas_length=args.canvas_length,
        num_passes=args.passes,
        use_self_conditioning=True,
    )

    checkpoints = {1, 10, 25, 50, 100, 250}
    history_records = []
    latencies = []

    for step in range(1, args.max_steps + 1):
        rec = dataset[(step - 1) % len(dataset)]
        goal = rec["goal"]
        page = {
            "url": rec["url"],
            "title": rec["title"],
            "text": rec["text"],
            "actions": rec["actions"],
            "fingerprint": rec["fingerprint"],
        }
        space = action_space(page["actions"])

        t0 = time.perf_counter()
        _ = backend.choose(goal, page, space, rec.get("history", []))
        lat = (time.perf_counter() - t0) * 1000
        latencies.append(lat)

        if step in checkpoints or step == args.max_steps:
            mem = get_memory_stats()
            recent_latencies = latencies[-min(25, len(latencies)) :]
            p50_recent = round(sorted(recent_latencies)[len(recent_latencies) // 2], 1)
            print(
                f"Step {step:3d}: Active={mem['active_gb']:5.2f}GB | Peak={mem['peak_gb']:5.2f}GB | "
                f"Cache={mem['cache_gb']:5.2f}GB | RSS={mem['process_rss_mb']:6.1f}MB | p50_lat={p50_recent:5.1f}ms"
            )
            history_records.append(
                {
                    "step": step,
                    "memory": mem,
                    "recent_p50_latency_ms": p50_recent,
                    "all_p50_latency_ms": round(sorted(latencies)[len(latencies) // 2], 1),
                }
            )

    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps({"checkpoints": history_records}, indent=2))
    print(f"\nSustained memory results saved to {out_file}")


if __name__ == "__main__":
    main()
