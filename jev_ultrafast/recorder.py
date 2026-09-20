"""Decision recorder for offline evaluation and trajectory replay."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from .action_candidates import flatten_actions


class DecisionRecorder:
    """Records browser decision states to JSONL for offline benchmark replay."""

    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.records: List[Dict[str, Any]] = []

    def record_decision(
        self,
        task_id: str,
        goal: str,
        page: Dict[str, Any],
        history: List[Dict[str, Any]],
        decision: Dict[str, Any],
        executed_action: Optional[str] = None,
        execution_result: Optional[Dict[str, Any]] = None,
        page_changed: Optional[bool] = None,
        eventual_task_success: Optional[bool] = None,
        decision_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record a single grounded decision state."""
        decision_id = decision_id or f"dec_{uuid.uuid4().hex[:12]}"
        candidates = flatten_actions(page["actions"])

        record = {
            "task_id": task_id,
            "decision_id": decision_id,
            "goal": goal,
            "url": page.get("url", ""),
            "title": page.get("title", ""),
            "text": page.get("text", "")[:6000],
            "fingerprint": page.get("fingerprint", ""),
            "actions": page.get("actions", []),
            "candidates": [
                {
                    "candidate_id": c.candidate_id,
                    "action_id": c.action_id,
                    "operation": c.operation,
                    "target": c.target,
                    "label": c.label,
                }
                for c in candidates
            ],
            "history": [
                {k: h.get(k) for k in ("step", "action", "kind", "choice", "text", "page_changed", "url")}
                for h in history[-10:]
            ],
            "jev_decision": {
                "choice": decision.get("choice", decision.get("action_id")),
                "operation": decision.get("operation"),
                "target": decision.get("target"),
                "confidence": float(decision.get("confidence", 1.0)),
                "probabilities": decision.get("probabilities", {}),
                "operation_probabilities": decision.get("operation_probabilities", {}),
                "target_probabilities": decision.get("target_probabilities", {}),
                "latency_ms": decision.get("latency_ms", 0),
                "model": decision.get("model", "jev"),
            },
            "executed_action": executed_action or decision.get("choice", decision.get("action_id")),
            "execution_result": execution_result,
            "page_changed": page_changed,
            "eventual_task_success": eventual_task_success,
        }

        self.records.append(record)
        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        return record

    @classmethod
    def load_dataset(cls, dataset_path: str | Path) -> List[Dict[str, Any]]:
        """Load decision records from a JSONL file."""
        records = []
        path = Path(dataset_path)
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
