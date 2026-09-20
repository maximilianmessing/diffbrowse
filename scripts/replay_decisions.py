"""Offline decision replay benchmark runner for Jev and MLX backends."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from jev_ultrafast.backends import (
    JevBackend,
    MlxAutoregressiveBackend,
    MlxDiffusionDirectBackend,
    MlxDiffusionGenerateBackend,
)
from jev_ultrafast.model import action_space
from jev_ultrafast.recorder import DecisionRecorder


def percentile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    if len(data) == 1:
        return round(data[0], 2)
    k = (len(data) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(data) - 1)
    if f == c:
        return round(data[f], 2)
    d0 = data[f] * (c - k)
    d1 = data[c] * (k - f)
    return round(d0 + d1, 2)


def run_replay(
    backend: Any,
    dataset: List[Dict[str, Any]],
    output_path: Optional[str] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    records = dataset[:limit] if limit else dataset
    print(f"Replaying {len(records)} decisions on backend: {getattr(backend, 'name', str(backend))}...")

    eval_results = []
    correct_actions = 0
    correct_operations = 0
    agreements_with_jev = 0
    invalid_actions = 0
    false_dones = 0
    false_blockeds = 0

    total_ms_list = []
    prefill_ms_list = []
    decoder_ms_list = []
    top_probs_list = []
    entropies_list = []
    margins_list = []

    # Candidate count buckets
    buckets = {
        "1-10": {"correct": 0, "total": 0, "latencies": []},
        "11-25": {"correct": 0, "total": 0, "latencies": []},
        "26-50": {"correct": 0, "total": 0, "latencies": []},
        "51-100": {"correct": 0, "total": 0, "latencies": []},
        "100+": {"correct": 0, "total": 0, "latencies": []},
    }

    t_start = time.perf_counter()

    for idx, rec in enumerate(records):
        goal = rec["goal"]
        page = {
            "url": rec["url"],
            "title": rec["title"],
            "text": rec["text"],
            "actions": rec["actions"],
            "fingerprint": rec["fingerprint"],
        }
        history = rec.get("history", [])
        expected_act = rec.get("executed_action", rec.get("jev_decision", {}).get("choice"))
        expected_op = rec.get("jev_decision", {}).get("operation")
        jev_choice = rec.get("jev_decision", {}).get("choice")

        # Construct action space
        space = action_space(page["actions"])
        candidate_count = len(rec.get("candidates", page["actions"]))

        # Execute decision
        decision = backend.choose(goal, page, space, history)

        chosen_act = decision.action_id
        chosen_op = decision.operation
        is_correct = chosen_act == expected_act
        agrees_jev = chosen_act == jev_choice
        is_invalid = bool(decision.metadata.get("invalid_action", False))

        is_false_done = chosen_act == "DONE" and expected_act != "DONE"
        is_false_blocked = chosen_act == "BLOCKED" and expected_act != "BLOCKED"

        if is_correct:
            correct_actions += 1
        if chosen_op == expected_op:
            correct_operations += 1
        if agrees_jev:
            agreements_with_jev += 1
        if is_invalid:
            invalid_actions += 1
        if is_false_done:
            false_dones += 1
        if is_false_blocked:
            false_blockeds += 1

        total_ms_list.append(decision.total_decision_ms)
        prefill_ms_list.append(decision.prefill_ms)
        decoder_ms_list.append(decision.decoder_ms)
        top_probs_list.append(decision.raw_top_probability)
        entropies_list.append(decision.entropy)
        margins_list.append(decision.top2_margin)

        # Bucket metrics
        if candidate_count <= 10:
            b_key = "1-10"
        elif candidate_count <= 25:
            b_key = "11-25"
        elif candidate_count <= 50:
            b_key = "26-50"
        elif candidate_count <= 100:
            b_key = "51-100"
        else:
            b_key = "100+"

        buckets[b_key]["total"] += 1
        buckets[b_key]["latencies"].append(decision.total_decision_ms)
        if is_correct:
            buckets[b_key]["correct"] += 1

        eval_item = {
            "decision_id": rec.get("decision_id", f"dec_{idx}"),
            "task_id": rec.get("task_id", "unknown"),
            "expected_action": expected_act,
            "chosen_action": chosen_act,
            "correct": is_correct,
            "agrees_with_jev": agrees_jev,
            "operation": chosen_op,
            "expected_operation": expected_op,
            "candidate_count": candidate_count,
            "total_ms": round(decision.total_decision_ms, 1),
            "prefill_ms": round(decision.prefill_ms, 1),
            "decoder_ms": round(decision.decoder_ms, 1),
            "top1_probability": round(decision.raw_top_probability, 4),
            "entropy": round(decision.entropy, 4),
            "top2_margin": round(decision.top2_margin, 4),
            "stable_steps": decision.stable_steps,
            "backend": getattr(backend, "name", "unknown"),
            "metadata": decision.metadata,
        }
        eval_results.append(eval_item)

        if (idx + 1) % 50 == 0 or idx == len(records) - 1:
            acc_pct = correct_actions / (idx + 1)
            agree_pct = agreements_with_jev / (idx + 1)
            print(f"[{idx + 1}/{len(records)}] Current Acc: {acc_pct:.1%}, Jev Agree: {agree_pct:.1%}")

    total_wall_s = round(time.perf_counter() - t_start, 2)
    n = len(records)
    total_ms_sorted = sorted(total_ms_list)
    prefill_sorted = sorted(prefill_ms_list)
    decoder_sorted = sorted(decoder_ms_list)

    summary = {
        "backend": getattr(backend, "name", "unknown"),
        "total_decisions": n,
        "action_accuracy": round(correct_actions / n, 4) if n else 0.0,
        "operation_accuracy": round(correct_operations / n, 4) if n else 0.0,
        "agreement_with_jev": round(agreements_with_jev / n, 4) if n else 0.0,
        "invalid_action_rate": round(invalid_actions / n, 4) if n else 0.0,
        "false_done_rate": round(false_dones / n, 4) if n else 0.0,
        "false_blocked_rate": round(false_blockeds / n, 4) if n else 0.0,
        "p50_total_ms": percentile(total_ms_sorted, 50),
        "p90_total_ms": percentile(total_ms_sorted, 90),
        "p95_total_ms": percentile(total_ms_sorted, 95),
        "p50_prefill_ms": percentile(prefill_sorted, 50),
        "p50_decoder_ms": percentile(decoder_sorted, 50),
        "p95_decoder_ms": percentile(decoder_sorted, 95),
        "mean_top1_prob": round(statistics.mean(top_probs_list), 4) if top_probs_list else 0.0,
        "mean_entropy": round(statistics.mean(entropies_list), 4) if entropies_list else 0.0,
        "mean_top2_margin": round(statistics.mean(margins_list), 4) if margins_list else 0.0,
        "total_wall_s": total_wall_s,
        "candidate_scaling": {
            k: {
                "count": v["total"],
                "accuracy": round(v["correct"] / v["total"], 4) if v["total"] else 0.0,
                "p50_ms": percentile(sorted(v["latencies"]), 50) if v["latencies"] else 0.0,
            }
            for k, v in buckets.items()
        },
    }

    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            for item in eval_results:
                f.write(json.dumps(item) + "\n")
        print(f"Saved replay evaluations to {out_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="fixtures/recorded_decisions.jsonl")
    parser.add_argument(
        "--backend",
        choices=["jev", "mlx_generate", "mlx_direct", "mlx_autoregressive"],
        default="mlx_direct",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--canvas-length", type=int, default=32)
    parser.add_argument("--passes", type=int, default=4)
    parser.add_argument("--slot", default="0")
    parser.add_argument("--no-self-conditioning", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-strategy", choices=["index", "arbitrary", "semantic"], default="index")
    parser.add_argument("--no-early-stopping", action="store_true")
    parser.add_argument("--output", default="results/replay_eval.jsonl")
    parser.add_argument("--summary-csv", default="results/summary.csv")
    args = parser.parse_args()

    dataset = DecisionRecorder.load_dataset(args.dataset)
    if not dataset:
        raise FileNotFoundError(f"No records found in {args.dataset}. Run scripts/generate_fixtures.py first.")

    # Initialize backend
    if args.backend == "jev":
        backend_obj = JevBackend()
    elif args.backend == "mlx_generate":
        backend_obj = MlxDiffusionGenerateBackend()
    elif args.backend == "mlx_autoregressive":
        backend_obj = MlxAutoregressiveBackend()
    else:
        if args.slot == "0":
            slot_val = 0
        elif args.slot == "middle":
            slot_val = args.canvas_length // 2
        elif args.slot == "late":
            slot_val = max(0, args.canvas_length - 4)
        else:
            slot_val = int(args.slot)

        backend_obj = MlxDiffusionDirectBackend(
            canvas_length=args.canvas_length,
            num_passes=args.passes,
            action_slot=slot_val,
            use_self_conditioning=not args.no_self_conditioning,
            compile_graph=args.compile,
            seed=args.seed,
            label_strategy=args.label_strategy,
            early_stopping=not args.no_early_stopping,
        )

    summary = run_replay(backend_obj, dataset, output_path=args.output, limit=args.limit)

    print("\n" + "=" * 60)
    print("REPLAY SUMMARY RESULTS:")
    print("=" * 60)
    for k, v in summary.items():
        if k != "candidate_scaling":
            print(f"{k:24}: {v}")
    print("\nCandidate Scaling Breakdown:")
    for k, v in summary["candidate_scaling"].items():
        print(f"  Bucket {k:8}: Count={v['count']:3}, Acc={v['accuracy']:6.1%}, p50={v['p50_ms']}ms")
    print("=" * 60)

    # Append to summary CSV
    csv_path = Path(args.summary_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "backend", "canvas", "passes", "compile", "sc", "accuracy", "jev_agreement",
                "p50_total_ms", "p95_total_ms", "p50_prefill_ms", "p50_decoder_ms"
            ])
        writer.writerow([
            summary["backend"],
            args.canvas_length if args.backend == "mlx_direct" else "-",
            args.passes if args.backend == "mlx_direct" else "-",
            "yes" if args.compile else "no",
            "no" if args.no_self_conditioning else "yes",
            summary["action_accuracy"],
            summary["agreement_with_jev"],
            summary["p50_total_ms"],
            summary["p95_total_ms"],
            summary["p50_prefill_ms"],
            summary["p50_decoder_ms"],
        ])


if __name__ == "__main__":
    main()
