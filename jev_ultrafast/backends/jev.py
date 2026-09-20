"""Jev (TypeSafe) remote decision backend."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from ..questions import NEXT_ACTION, TARGET
from .base import Decision, calculate_entropy, calculate_top2_margin


class JevBackend:
    """TypeSafe Jev model backend using speculative fan-out."""

    name: str = "jev"

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.environ.get("TYPESAFE_MODEL", "jev-latest")

    def choose(
        self,
        goal: str,
        browser_state: Dict[str, Any],
        action_space: tuple,
        history: List[Dict[str, Any]],
    ) -> Decision:
        from ..model import post_json, validate_choice

        elements, targets, controls = action_space
        labels = {
            "CLICK": "Click an element, button, menu option, autocomplete suggestion, or calendar day.",
            "TYPE_TEXT": "Enter or replace text in an editable field. A small LLM will supply the value from the goal.",
            "SELECT": "Select an observed dropdown value.",
        }
        operations = {key: labels[key] for key in targets}
        operations.update({key: value["label"] for key, value in controls.items()})
        operations.update(
            DONE="Every requirement is visibly satisfied.",
            BLOCKED="No supported operation can progress.",
        )
        questions = {
            "operation": {
                "type": "choice",
                "criteria": operations,
                "instructions": {"goal": goal, "rules": NEXT_ACTION},
            }
        }
        for operation, candidates in targets.items():
            questions[operation.lower() + "_target"] = {
                "type": "choice",
                "criteria": {
                    index: {
                        "element": f"[{index}] {a['label']}",
                        "current_value": a.get("current_value", a.get("value", "")),
                        **{k: a[k] for k in ("role", "checked", "selected", "expanded") if k in a},
                    }
                    for index, a in candidates.items()
                },
                "instructions": {"goal": goal, "operation": operation, "rules": [NEXT_ACTION, TARGET]},
            }
        body = {
            "model": self.model_name,
            "state": {
                "page": {k: browser_state[k] for k in ("url", "title", "text")},
                "elements": elements,
                "recent_actions": [
                    {k: h.get(k) for k in ("action", "kind", "text", "page_changed")} for h in history[-10:]
                ],
            },
            "questions": questions,
        }
        started = time.perf_counter()
        result = post_json("https://api.typesafe.ai/v1/systemone", os.environ["TYPESAFE_API_KEY"], body)
        latency_ms = (time.perf_counter() - started) * 1000

        operation_answer = validate_choice(result["answers"].get("operation", {}), operations)
        operation = operation_answer["choice"]
        target = None
        target_answer = None
        probabilities = {}
        if operation in targets:
            head_key = operation.lower() + "_target"
            target_answer = validate_choice(result["answers"].get(head_key, {}), targets[operation])
            target = target_answer["choice"]
            choice = targets[operation][target]["id"]
            probabilities = {
                a["id"]: target_answer["probabilities"][index] for index, a in targets[operation].items()
            }
        else:
            choice = controls[operation]["id"] if operation in controls else operation
            probabilities[choice] = operation_answer["probabilities"][operation]

        sorted_probs = sorted(probabilities.values(), reverse=True)
        raw_top_prob = sorted_probs[0] if sorted_probs else float(operation_answer["confidence"])
        entropy = calculate_entropy(probabilities)
        margin = calculate_top2_margin(probabilities)

        return Decision(
            action_id=choice,
            operation=operation,
            target=target,
            confidence=float(operation_answer["confidence"]),
            probabilities=probabilities,
            raw_top_probability=raw_top_prob,
            entropy=entropy,
            top2_margin=margin,
            stable_steps=1,
            tokenize_ms=0.0,
            prefill_ms=0.0,
            decoder_ms=0.0,
            total_model_ms=latency_ms,
            total_decision_ms=latency_ms,
            model=result.get("model", self.model_name),
            backend=self.name,
            metadata={
                "raw_answers": result["answers"],
                "request": body,
                "usage": result.get("usage", {}),
                "operation_probabilities": operation_answer["probabilities"],
                "target_probabilities": target_answer["probabilities"] if target_answer else {},
                "target_confidence": target_answer["confidence"] if target_answer else None,
            },
        )
