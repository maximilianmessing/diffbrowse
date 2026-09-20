"""Export recorded browser decisions to instruction fine-tuning datasets for MLX LoRA."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict

from jev_ultrafast.action_candidates import LabelPool, flatten_actions
from jev_ultrafast.recorder import DecisionRecorder


def format_decision_for_lora(rec: Dict[str, Any], pool: LabelPool) -> Dict[str, str] | None:
    goal = rec.get("goal")
    actions = rec.get("actions")
    executed_action = rec.get("executed_action")
    if not goal or not actions or not executed_action:
        return None

    candidates = flatten_actions(actions)
    if not candidates:
        return None

    cand_token_ids, token_to_cand, cand_to_label, candidate_lines = pool.map_candidates(
        candidates, strategy="index"
    )

    # Find candidate label for the executed action
    target_label = None
    for cand in candidates:
        if cand.action_id == executed_action or cand.candidate_id == executed_action:
            val = cand_to_label.get(cand.candidate_id)
            if val:
                target_label = val[0] if isinstance(val, (tuple, list)) else str(val)
            break

    if not target_label:
        return None

    recent_hist = "\n".join(
        f"- Step {h.get('step', i+1)}: {h.get('action')}"
        for i, h in enumerate(rec.get("history", [])[-5:])
    ) or "None"

    prompt = (
        f"Goal: {goal}\n"
        f"Page Title: {rec.get('title', '')}\n"
        f"Page URL: {rec.get('url', '')}\n"
        f"Visible Page Content:\n{str(rec.get('text', ''))[:2000]}\n\n"
        f"Recent Action History:\n{recent_hist}\n\n"
        f"{candidate_lines}\n\n"
        "Select the single best next action from the list above to achieve the goal.\n"
        "Reply with ACTION=<id>."
    )

    completion = f"ACTION={target_label}"

    return {
        "prompt": prompt,
        "completion": completion,
        "expected_action": executed_action,
        "target_label": target_label,
    }


def export_dataset(
    input_path: str,
    output_dir: str,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Dict[str, int]:
    dataset = DecisionRecorder.load_dataset(input_path)
    if not dataset:
        raise ValueError(f"No records found in {input_path}")

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("mlx-community/diffusiongemma-26B-A4B-it-4bit")
    except Exception:
        from mlx_vlm import load

        _, processor = load("mlx-community/diffusiongemma-26B-A4B-it-4bit")
        tokenizer = processor.tokenizer

    pool = LabelPool(tokenizer)

    formatted_samples = []
    for rec in dataset:
        sample = format_decision_for_lora(rec, pool)
        if sample:
            formatted_samples.append(sample)

    random.seed(seed)
    random.shuffle(formatted_samples)

    split_idx = int(len(formatted_samples) * train_ratio)
    train_samples = formatted_samples[:split_idx]
    valid_samples = formatted_samples[split_idx:]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_file = out_dir / "train.jsonl"
    valid_file = out_dir / "valid.jsonl"

    with open(train_file, "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps({"prompt": s["prompt"], "completion": s["completion"]}) + "\n")

    with open(valid_file, "w", encoding="utf-8") as f:
        for s in valid_samples:
            f.write(json.dumps({"prompt": s["prompt"], "completion": s["completion"]}) + "\n")

    print("=" * 60)
    print("LoRA Fine-Tuning Dataset Export")
    print("=" * 60)
    print(f"Total raw records   : {len(dataset)}")
    print(f"Valid paired samples: {len(formatted_samples)}")
    print(f"Train samples (80%) : {len(train_samples)} -> {train_file}")
    print(f"Valid samples (20%) : {len(valid_samples)} -> {valid_file}")
    print("=" * 60)

    return {
        "raw_records": len(dataset),
        "total_formatted_samples": len(formatted_samples),
        "train_samples": len(train_samples),
        "valid_samples": len(valid_samples),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="fixtures/recorded_decisions.jsonl", help="Input dataset path")
    parser.add_argument("--output-dir", default="data/lora", help="Output directory for train/valid jsonl")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train/Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    args = parser.parse_args()

    export_dataset(args.input, args.output_dir, args.train_ratio, args.seed)


if __name__ == "__main__":
    main()
