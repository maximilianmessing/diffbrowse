"""Real browser benchmark running grounded tasks against local fixtures and live pages."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict

from jev_ultrafast.agent import Agent
from jev_ultrafast.backends import (
    HybridBackend,
    JevBackend,
    MlxDiffusionDirectBackend,
    MlxDiffusionGenerateBackend,
)


def start_demo_server(port: int = 8766):
    """Start local demo HTTP server in background if not already responding."""
    import socket
    import sys

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("127.0.0.1", port)) == 0:
            return None  # Server already running

    proc = subprocess.Popen(
        [sys.executable, "-m", "jev_ultrafast.demo"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(50):
        time.sleep(0.1)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                break
    return proc


def run_travel_fixture_task(backend: Any, port: int = 8766) -> Dict[str, Any]:
    url = f"http://127.0.0.1:{port}/fixture.html?scenario=travel"
    goal = (
        "Use the destination search and filters to find Design stays in Lisbon with Free cancellation, "
        "then open Casa Flora."
    )
    started = time.perf_counter()

    with Agent(url, goal, backend=backend, screenshots=False) as agent:
        try:
            for _ in agent.run():
                pass
        except Exception as e:
            return {"success": False, "error": str(e), "elapsed_ms": (time.perf_counter() - started) * 1000}

        final_state = agent.snapshot()
        verification_text = agent.browser.evaluate("document.body.innerText") or ""
        passed = (
            final_state["status"] == "done"
            and final_state["page"]["url"].endswith("#casa-flora")
            and "Your filters: Design · Free cancellation enabled · Destination Lisbon" in verification_text
        )

        return {
            "task": "travel_fixture",
            "success": passed,
            "status": final_state["status"],
            "elapsed_ms": final_state.get("elapsed_ms", round((time.perf_counter() - started) * 1000)),
            "decisions": len(final_state["decisions"]),
            "actions": len(final_state["history"]),
            "history": final_state["history"],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["jev", "mlx_direct", "mlx_generate", "hybrid"], default="mlx_direct")
    parser.add_argument("--canvas-length", type=int, default=32)
    parser.add_argument("--passes", type=int, default=4)
    parser.add_argument("--output", default="results/browser_benchmark.json")
    args = parser.parse_args()

    server_proc = start_demo_server()
    try:
        if args.backend == "jev":
            backend = JevBackend()
        elif args.backend == "mlx_generate":
            backend = MlxDiffusionGenerateBackend()
        elif args.backend == "hybrid":
            backend = HybridBackend()
        else:
            backend = MlxDiffusionDirectBackend(
                canvas_length=args.canvas_length,
                num_passes=args.passes,
            )

        print(f"Running real browser benchmark with backend: {getattr(backend, 'name', str(backend))}...")
        result = run_travel_fixture_task(backend)
        print("\nBenchmark Result:")
        print(f"  Success   : {result['success']}")
        print(f"  Status    : {result.get('status')}")
        print(f"  Time (ms) : {result.get('elapsed_ms')}")
        print(f"  Actions   : {result.get('actions')}")

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"Saved result to {out_path}")
    finally:
        if server_proc:
            server_proc.terminate()


if __name__ == "__main__":
    main()
