"""Base protocol and decision container for browser action selection."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


def calculate_entropy(probabilities: Dict[str, float] | List[float]) -> float:
    probs = list(probabilities.values()) if isinstance(probabilities, dict) else list(probabilities)
    if not probs:
        return 0.0
    total = sum(probs)
    if total <= 0:
        return 0.0
    norm_probs = [p / total for p in probs if p > 0]
    return -sum(p * math.log(p) for p in norm_probs)


def calculate_top2_margin(probabilities: Dict[str, float] | List[float]) -> float:
    probs = list(probabilities.values()) if isinstance(probabilities, dict) else list(probabilities)
    if len(probs) < 2:
        return 1.0 if probs else 0.0
    sorted_probs = sorted(probs, reverse=True)
    return sorted_probs[0] - sorted_probs[1]


@dataclass
class Decision(dict):
    """Container for a grounded browser decision with timing and uncertainty metrics.

    Inherits from dict to ensure 100% backward compatibility with existing Jev code
    that accesses decision["choice"], decision["probabilities"], etc.
    """

    action_id: str
    operation: str
    target: Optional[str] = None
    confidence: float = 1.0
    probabilities: Dict[str, float] = field(default_factory=dict)
    raw_top_probability: float = 1.0
    entropy: float = 0.0
    top2_margin: float = 1.0
    stable_steps: int = 1
    tokenize_ms: float = 0.0
    prefill_ms: float = 0.0
    decoder_ms: float = 0.0
    total_model_ms: float = 0.0
    total_decision_ms: float = 0.0
    model: str = "unknown"
    backend: str = "base"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Populate dict keys for legacy code compatibility
        data = {
            "choice": self.action_id,
            "operation": self.operation,
            "target": self.target,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "latency_ms": round(self.total_decision_ms or self.total_model_ms),
            "model": self.model,
            "usage": self.metadata.get("usage", {}),
            "action_id": self.action_id,
            "raw_top_probability": self.raw_top_probability,
            "entropy": self.entropy,
            "top2_margin": self.top2_margin,
            "stable_steps": self.stable_steps,
            "tokenize_ms": self.tokenize_ms,
            "prefill_ms": self.prefill_ms,
            "decoder_ms": self.decoder_ms,
            "total_model_ms": self.total_model_ms,
            "total_decision_ms": self.total_decision_ms,
            "backend": self.backend,
            "metadata": self.metadata,
        }
        # Copy any existing metadata entries like raw_answers, request
        if "raw_answers" in self.metadata:
            data["raw_answers"] = self.metadata["raw_answers"]
        if "request" in self.metadata:
            data["request"] = self.metadata["request"]
        if "operation_probabilities" in self.metadata:
            data["operation_probabilities"] = self.metadata["operation_probabilities"]
        if "target_probabilities" in self.metadata:
            data["target_probabilities"] = self.metadata["target_probabilities"]
        if "target_confidence" in self.metadata:
            data["target_confidence"] = self.metadata["target_confidence"]
        super().__init__(data)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if hasattr(self, key):
            setattr(self, key, value)
        if key == "choice":
            self.action_id = value


@runtime_checkable
class DecisionBackend(Protocol):
    """Protocol for a browser decision backend."""

    name: str

    def choose(
        self,
        goal: str,
        browser_state: Dict[str, Any],
        action_space: tuple,
        history: List[Dict[str, Any]],
    ) -> Decision:
        ...
