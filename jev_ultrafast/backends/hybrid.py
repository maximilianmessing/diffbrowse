"""Hybrid decision backend: Local DiffusionGemma with Jev fallback upon uncertainty."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from .base import Decision, DecisionBackend
from .jev import JevBackend
from .mlx_direct import MlxDiffusionDirectBackend


class HybridBackend:
    """Hybrid decision policy.

    Runs local DiffusionGemma direct backend first.
    If confidence criteria (entropy, margin, stability) are satisfied, executes locally.
    Otherwise, delegates to remote Jev backend.
    """

    name: str = "hybrid"

    def __init__(
        self,
        local_backend: Optional[DecisionBackend] = None,
        remote_backend: Optional[DecisionBackend] = None,
        entropy_threshold: float = 0.35,
        margin_threshold: float = 0.50,
        min_stable_steps: int = 2,
        max_candidate_count: int = 60,
    ):
        self.local = local_backend or MlxDiffusionDirectBackend()
        self.remote = remote_backend or JevBackend()
        self.entropy_threshold = entropy_threshold
        self.margin_threshold = margin_threshold
        self.min_stable_steps = min_stable_steps
        self.max_candidate_count = max_candidate_count

    def choose(
        self,
        goal: str,
        browser_state: Dict[str, Any],
        action_space: tuple,
        history: List[Dict[str, Any]],
    ) -> Decision:
        started = time.perf_counter()
        candidate_count = len(browser_state.get("actions", []))

        # Check candidate space bounds
        if candidate_count > self.max_candidate_count:
            # Fallback immediately for unusually large action spaces
            remote_decision = self.remote.choose(goal, browser_state, action_space, history)
            remote_decision.metadata["hybrid_route"] = "remote_direct"
            remote_decision.metadata["hybrid_reason"] = f"candidate_count_{candidate_count}_exceeds_threshold"
            return remote_decision

        # Run local decision
        local_decision = self.local.choose(goal, browser_state, action_space, history)

        # Evaluate uncertainty signals
        is_confident = (
            (local_decision.entropy <= self.entropy_threshold)
            and (local_decision.top2_margin >= self.margin_threshold)
            and (local_decision.stable_steps >= self.min_stable_steps)
        )

        # Repeated failed action check
        repeated_failure = False
        if history and len(history) >= 2:
            last_two = history[-2:]
            if all(h.get("choice") == local_decision.action_id and h.get("page_changed") is False for h in last_two):
                repeated_failure = True

        if is_confident and not repeated_failure:
            local_decision.metadata["hybrid_route"] = "local"
            return local_decision

        # Fall back to remote Jev, or gracefully keep local if remote fails/offline
        try:
            remote_decision = self.remote.choose(goal, browser_state, action_space, history)
            remote_decision.total_decision_ms = (time.perf_counter() - started) * 1000
            remote_decision.metadata["hybrid_route"] = "fallback_remote"
            remote_decision.metadata["hybrid_reason"] = (
                "repeated_failure"
                if repeated_failure
                else f"uncertain_entropy_{local_decision.entropy:.2f}_margin_{local_decision.top2_margin:.2f}"
            )
            remote_decision.metadata["local_attempt"] = {
                "action_id": local_decision.action_id,
                "entropy": local_decision.entropy,
                "margin": local_decision.top2_margin,
                "stable_steps": local_decision.stable_steps,
            }
            return remote_decision
        except Exception as err:
            local_decision.metadata["hybrid_route"] = "fallback_local_offline"
            local_decision.metadata["hybrid_remote_error"] = str(err)
            return local_decision

    def generate_field_text(self, context: Dict[str, Any], max_tokens: int = 32) -> str:
        if hasattr(self.local, "generate_field_text"):
            return self.local.generate_field_text(context, max_tokens=max_tokens)
        raise NotImplementedError("Local backend cannot generate field text")

