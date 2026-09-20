"""Master automated experiment suite for DiffusionGemma browser decision model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from jev_ultrafast.backends import (
    MlxAutoregressiveBackend,
    MlxDiffusionDirectBackend,
    MlxDiffusionGenerateBackend,
)
from jev_ultrafast.recorder import DecisionRecorder
from scripts.replay_decisions import run_replay


def run_all_experiments(dataset_path: str, output_dir: str, limit: int = 100):
    dataset = DecisionRecorder.load_dataset(dataset_path)
    if not dataset:
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    results_summary: Dict[str, Any] = {}

    print("=" * 70)
    print(f"RUNNING DIFFUSIONGEMMA MLX EXPERIMENT SUITE ({limit} samples per config)")
    print("=" * 70)

    # Preload DiffusionGemma weights once to avoid reloading 14 GB on each variation
    print("\n[INIT] Preloading DiffusionGemma model & processor into unified memory...")
    from jev_ultrafast.backends.mlx_direct import _patch_diffusion_encoder

    _patch_diffusion_encoder()
    from mlx_vlm import load as mlx_load
    shared_model, shared_processor = mlx_load("mlx-community/diffusiongemma-26B-A4B-it-4bit")
    print("[INIT] DiffusionGemma preloaded successfully!\n")

    # 1. CHECKPOINT 1: PASSES SWEEP
    print("\n>>> [1/7] Running Decoder Passes Sweep (Passes: 1, 2, 4, 8)...")
    pass_results = []
    for p in [1, 2, 4, 8]:
        backend = MlxDiffusionDirectBackend(
            model=shared_model,
            processor=shared_processor,
            canvas_length=32,
            num_passes=p,
            early_stopping=False,
        )
        res = run_replay(backend, dataset, limit=limit)
        pass_results.append({
            "passes": p,
            "accuracy": res["action_accuracy"],
            "operation_accuracy": res["operation_accuracy"],
            "p50_total_ms": res["p50_total_ms"],
            "p50_prefill_ms": res["p50_prefill_ms"],
            "p50_decoder_ms": res["p50_decoder_ms"],
            "entropy": res["mean_entropy"],
            "margin": res["mean_top2_margin"],
        })
        print(
            f"Passes {p:2d}: Acc={res['action_accuracy']:6.1%}, "
            f"p50_dec={res['p50_decoder_ms']}ms, p50_tot={res['p50_total_ms']}ms"
        )

    results_summary["checkpoint_1_passes"] = pass_results

    # 2. CANVAS SIZE SWEEP
    print("\n>>> [2/7] Running Canvas Length Sweep (8, 16, 32, 64, 128, 256)...")
    canvas_results = []
    for c in [8, 16, 32, 64, 128, 256]:
        backend = MlxDiffusionDirectBackend(
            model=shared_model,
            processor=shared_processor,
            canvas_length=c,
            num_passes=2,
            early_stopping=False,
        )
        res = run_replay(backend, dataset, limit=limit)
        canvas_results.append({
            "canvas": c,
            "accuracy": res["action_accuracy"],
            "p50_total_ms": res["p50_total_ms"],
            "p50_decoder_ms": res["p50_decoder_ms"],
        })
        print(f"Canvas {c:3d}: Acc={res['action_accuracy']:6.1%}, p50_total={res['p50_total_ms']}ms")

    results_summary["canvas_sweep"] = canvas_results

    # 3. ACTION SLOT POSITION ABLATION
    print("\n>>> [3/7] Running Action Slot Position Ablation (early=0, mid=16, late=28)...")
    slot_results = []
    for name, s in [("early", 0), ("middle", 16), ("late", 28)]:
        backend = MlxDiffusionDirectBackend(
            model=shared_model,
            processor=shared_processor,
            canvas_length=32,
            num_passes=2,
            action_slot=s,
        )
        res = run_replay(backend, dataset, limit=limit)
        slot_results.append({
            "slot_name": name,
            "slot_idx": s,
            "accuracy": res["action_accuracy"],
            "p50_ms": res["p50_total_ms"],
        })
        print(f"Slot {name:6s} ({s:2d}): Acc={res['action_accuracy']:6.1%}, p50_total={res['p50_total_ms']}ms")

    results_summary["slot_ablation"] = slot_results

    # 4. SELF-CONDITIONING ABLATION
    print("\n>>> [4/7] Running Self-Conditioning Ablation (SC ON vs OFF)...")
    sc_results = []
    for p in [2, 4]:
        for sc in [True, False]:
            backend = MlxDiffusionDirectBackend(
                model=shared_model,
                processor=shared_processor,
                canvas_length=32,
                num_passes=p,
                use_self_conditioning=sc,
            )
            res = run_replay(backend, dataset, limit=limit)
            sc_results.append({
                "passes": p,
                "self_conditioning": sc,
                "accuracy": res["action_accuracy"],
                "p50_ms": res["p50_total_ms"],
            })
            print(f"Passes {p}, SC={sc}: Acc={res['action_accuracy']:6.1%}, p50_total={res['p50_total_ms']}ms")

    results_summary["self_conditioning_ablation"] = sc_results

    # 5. MULTI-SEED STABILITY
    print("\n>>> [5/7] Running Multi-Seed Stability Test (Seeds 1, 2, 3, 4)...")
    seed_results = []
    for s in [1, 2, 3, 4]:
        backend = MlxDiffusionDirectBackend(
            model=shared_model,
            processor=shared_processor,
            canvas_length=32,
            num_passes=2,
            seed=s,
        )
        res = run_replay(backend, dataset, limit=limit)
        seed_results.append({"seed": s, "accuracy": res["action_accuracy"], "entropy": res["mean_entropy"]})
        print(f"Seed {s}: Acc={res['action_accuracy']:6.1%}, Mean Entropy={res['mean_entropy']:.3f}")

    results_summary["seed_stability"] = seed_results

    # 6. AUTOREGRESSIVE CONTROL
    print("\n>>> [6/7] Running Autoregressive Local Control...")
    try:
        ar_backend = MlxAutoregressiveBackend()
        ar_res = run_replay(ar_backend, dataset, limit=limit)
        results_summary["autoregressive_control"] = {
            "accuracy": ar_res["action_accuracy"],
            "p50_total_ms": ar_res["p50_total_ms"],
            "invalid_rate": ar_res["invalid_action_rate"],
        }
        print(
            f"AR Control: Acc={ar_res['action_accuracy']:6.1%}, "
            f"p50={ar_res['p50_total_ms']}ms, Invalid={ar_res['invalid_action_rate']:6.1%}"
        )
    except Exception as e:
        print(f"AR Control skipped or failed: {e}")
        results_summary["autoregressive_control"] = {"error": str(e)}

    # 7. GENERATION BASELINE
    print("\n>>> [7/7] Running Diffusion Generation Sanity Baseline...")
    try:
        gen_backend = MlxDiffusionGenerateBackend(
            model=shared_model,
            processor=shared_processor,
            max_tokens=16,
        )
        gen_res = run_replay(gen_backend, dataset, limit=min(limit, 25))
        results_summary["diffusion_generate_baseline"] = {
            "accuracy": gen_res["action_accuracy"],
            "p50_total_ms": gen_res["p50_total_ms"],
            "invalid_rate": gen_res["invalid_action_rate"],
        }
        print(
            f"DG Generate: Acc={gen_res['action_accuracy']:6.1%}, "
            f"p50={gen_res['p50_total_ms']}ms, Invalid={gen_res['invalid_action_rate']:6.1%}"
        )
    except Exception as e:
        print(f"DG Generate baseline error: {e}")
        results_summary["diffusion_generate_baseline"] = {"error": str(e)}

    # Save complete JSON summary
    summary_file = out_path / "all_experiments_summary.json"
    summary_file.write_text(json.dumps(results_summary, indent=2))
    print(f"\nAll experiments complete! Summary saved to {summary_file}")
    return results_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="fixtures/recorded_decisions.jsonl")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    run_all_experiments(args.dataset, args.output_dir, args.limit)


if __name__ == "__main__":
    main()
