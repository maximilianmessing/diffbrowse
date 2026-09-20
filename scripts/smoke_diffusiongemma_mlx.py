"""Standalone smoke and memory benchmark for DiffusionGemma on Apple Silicon MLX."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import mlx.core as mx
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template

DEFAULT_MODEL = "mlx-community/diffusiongemma-26B-A4B-it-4bit"


def memory_stats() -> dict[str, float]:
    return {
        "active_gb": round(mx.get_active_memory() / (1024**3), 3),
        "peak_gb": round(mx.get_peak_memory() / (1024**3), 3),
        "cache_gb": round(mx.get_cache_memory() / (1024**3), 3),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face repo or local path")
    parser.add_argument("--repeats", type=int, default=3, help="Number of warm inferences to run")
    parser.add_argument("--max-tokens", type=int, default=32, help="Max generation tokens")
    parser.add_argument("--output", default="results/smoke_diffusiongemma.json", help="Output summary file")
    args = parser.parse_args()

    print("=" * 60)
    print("DiffusionGemma MLX Smoke Test")
    print("=" * 60)
    print(f"Target model : {args.model}")
    print(f"Device       : {mx.default_device()}")
    print(f"Initial mem  : {memory_stats()}")
    from jev_ultrafast.backends.mlx_direct import _patch_diffusion_encoder

    _patch_diffusion_encoder()

    # 1. Measure model load time
    t0 = time.perf_counter()
    model, processor = load(args.model)
    load_ms = round((time.perf_counter() - t0) * 1000, 1)
    print(f"Model loaded in {load_ms:.1f} ms")
    print(f"Post-load mem: {memory_stats()}")
    print("-" * 60)

    # 2. Tokenize prompt
    user_prompt = (
        "Select the best browser action: A0: Click 'Search Flights', A1: Click 'Cancel'. Reply with ACTION=<id>."
    )
    formatted_prompt = apply_chat_template(processor, model.config, user_prompt)

    # 3. First inference (Cold)
    mx.reset_peak_memory()
    t_cold = time.perf_counter()
    cold_res = generate(
        model=model,
        processor=processor,
        prompt=formatted_prompt,
        max_tokens=args.max_tokens,
        temperature=0.0,
        generation_mode="diffusion",
    )
    cold_ms = round((time.perf_counter() - t_cold) * 1000, 1)
    cold_mem = memory_stats()
    cold_text = cold_res.text if hasattr(cold_res, "text") else str(cold_res)
    print(f"Cold inference : {cold_ms:.1f} ms")
    print(f"Cold output    : {cold_text.strip()!r}")
    print(f"Cold mem       : {cold_mem}")
    print("-" * 60)

    # 4. Repeated warm inferences
    warm_times = []
    for i in range(args.repeats):
        mx.reset_peak_memory()
        t_warm = time.perf_counter()
        warm_res = generate(
            model=model,
            processor=processor,
            prompt=formatted_prompt,
            max_tokens=args.max_tokens,
            temperature=0.0,
            generation_mode="diffusion",
        )
        w_ms = round((time.perf_counter() - t_warm) * 1000, 1)
        warm_times.append(w_ms)
        w_text = warm_res.text if hasattr(warm_res, "text") else str(warm_res)
        print(f"Warm run #{i + 1}    : {w_ms:.1f} ms | {w_text.strip()!r}")

    warm_p50 = sorted(warm_times)[len(warm_times) // 2]
    warm_mem = memory_stats()
    print("-" * 60)
    print(f"Warm p50       : {warm_p50:.1f} ms")
    print(f"Final mem      : {warm_mem}")
    print("=" * 60)

    results = {
        "model": args.model,
        "model_load_ms": load_ms,
        "first_inference_ms": cold_ms,
        "warm_inference_times_ms": warm_times,
        "warm_inference_p50_ms": warm_p50,
        "cold_output": cold_text.strip(),
        "memory": {
            "post_load": memory_stats(),
            "cold_inference": cold_mem,
            "warm_inference": warm_mem,
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved results to {out_path}")

    # Explicit cleanup
    del model, processor
    gc.collect()
    mx.clear_cache()


if __name__ == "__main__":
    main()
