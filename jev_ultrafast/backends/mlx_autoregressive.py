"""Autoregressive MLX local control backend."""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List

from .base import Decision, calculate_entropy, calculate_top2_margin


class MlxAutoregressiveBackend:
    """Control baseline using a lightweight autoregressive model via mlx-lm."""

    name: str = "mlx_autoregressive"

    def __init__(
        self,
        model_name: str = "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        model: Any = None,
        tokenizer: Any = None,
        max_tokens: int = 16,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self._model = model
        self._tokenizer = tokenizer

    def _ensure_loaded(self):
        if self._model is None or self._tokenizer is None:
            from mlx_lm import load

            self._model, self._tokenizer = load(self.model_name)

    def choose(
        self,
        goal: str,
        browser_state: Dict[str, Any],
        action_space: tuple,
        history: List[Dict[str, Any]],
    ) -> Decision:
        from mlx_lm import generate

        from ..action_candidates import LabelPool, flatten_actions

        self._ensure_loaded()
        started = time.perf_counter()

        candidates = flatten_actions(browser_state["actions"])
        pool = LabelPool(self._tokenizer)
        cand_token_ids, token_to_cand, cand_to_label, candidate_lines = pool.map_candidates(
            candidates, strategy="index"
        )

        recent_hist = "\n".join(
            f"- Step {h.get('step', i+1)}: {h.get('action')}"
            for i, h in enumerate(history[-5:])
        ) or "None"

        prompt = (
            f"<|im_start|>system\nYou are a fast browser decision agent.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Goal: {goal}\n"
            f"Page: {browser_state.get('title', '')} ({browser_state.get('url', '')})\n"
            f"Content: {browser_state.get('text', '')[:1500]}\n"
            f"History:\n{recent_hist}\n\n"
            f"{candidate_lines}\n\n"
            "Choose the single best action to advance the goal. Output format: ACTION=<label>\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\nACTION="
        )

        gen_started = time.perf_counter()
        output_text = generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=self.max_tokens,
            verbose=False,
        )
        total_model_ms = (time.perf_counter() - gen_started) * 1000
        total_decision_ms = (time.perf_counter() - started) * 1000

        text = output_text.strip()
        match = re.search(r"\[?([A-Za-z0-9_]+)\]?", text)
        selected_cand = None
        invalid = False

        if match:
            raw_label = match.group(1).strip()
            for cand in candidates:
                cand_label_str, _ = cand_to_label.get(cand.candidate_id, ("", None))
                if raw_label.lower() == cand_label_str.lower() or raw_label.lower() == cand.candidate_id.lower():
                    selected_cand = cand
                    break

        if selected_cand is None:
            invalid = True
            selected_cand = next((c for c in candidates if c.action_id == "wait"), candidates[-1])

        probabilities = {c.action_id: 1.0 if c == selected_cand else 0.0 for c in candidates}
        entropy = calculate_entropy(probabilities)
        margin = calculate_top2_margin(probabilities)

        return Decision(
            action_id=selected_cand.action_id,
            operation=selected_cand.operation,
            target=selected_cand.target,
            confidence=0.0 if invalid else 1.0,
            probabilities=probabilities,
            raw_top_probability=1.0 if not invalid else 0.0,
            entropy=entropy,
            top2_margin=margin,
            stable_steps=1,
            tokenize_ms=0.0,
            prefill_ms=0.0,
            decoder_ms=total_model_ms,
            total_model_ms=total_model_ms,
            total_decision_ms=total_decision_ms,
            model=self.model_name,
            backend=self.name,
            metadata={
                "generated_text": text,
                "invalid_action": invalid,
                "candidate_count": len(candidates),
            },
        )
