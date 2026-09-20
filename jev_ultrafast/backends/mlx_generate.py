"""DiffusionGemma text generation baseline backend (sanity check)."""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List

from .base import Decision, calculate_entropy, calculate_top2_margin


class MlxDiffusionGenerateBackend:
    """Sanity baseline: Prompts DiffusionGemma to generate 'ACTION=<id>'."""

    name: str = "mlx_generate"

    def __init__(
        self,
        model: Any = None,
        processor: Any = None,
        model_name: str = "mlx-community/diffusiongemma-26B-A4B-it-4bit",
        max_tokens: int = 16,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self._model = model
        self._processor = processor

    def _ensure_loaded(self):
        from .mlx_direct import _patch_diffusion_encoder

        _patch_diffusion_encoder()
        if self._model is None or self._processor is None:
            from mlx_vlm import load

            self._model, self._processor = load(self.model_name)

    def choose(
        self,
        goal: str,
        browser_state: Dict[str, Any],
        action_space: tuple,
        history: List[Dict[str, Any]],
    ) -> Decision:
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        from ..action_candidates import LabelPool, flatten_actions

        self._ensure_loaded()
        started = time.perf_counter()

        candidates = flatten_actions(browser_state["actions"])
        pool = LabelPool(self._processor)
        cand_token_ids, token_to_cand, cand_to_label, candidate_lines = pool.map_candidates(
            candidates, strategy="index"
        )

        recent_hist = "\n".join(
            f"- Step {h.get('step', i+1)}: {h.get('action')} -> {'OK' if h.get('page_changed') else 'No change'}"
            for i, h in enumerate(history[-5:])
        ) or "None"

        prompt = (
            f"Goal: {goal}\n"
            f"Page Title: {browser_state.get('title', '')}\n"
            f"Page URL: {browser_state.get('url', '')}\n"
            f"Visible Page Content:\n{browser_state.get('text', '')[:2000]}\n\n"
            f"Recent Action History:\n{recent_hist}\n\n"
            f"{candidate_lines}\n\n"
            "Select the single best next action to advance the goal.\n"
            "Respond strictly with:\nACTION=<label>\n"
        )

        formatted = apply_chat_template(self._processor, self._model.config, prompt)

        gen_started = time.perf_counter()
        result = generate(
            model=self._model,
            processor=self._processor,
            prompt=formatted,
            max_tokens=self.max_tokens,
            temperature=0.0,
            generation_mode="diffusion",
        )
        total_model_ms = (time.perf_counter() - gen_started) * 1000
        total_decision_ms = (time.perf_counter() - started) * 1000

        text = result.text if hasattr(result, "text") else str(result)
        text = text.strip()

        # Parse ACTION=<label>
        match = re.search(r"ACTION\s*=\s*\[?([A-Za-z0-9_]+)\]?", text, re.IGNORECASE)
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
            # Fallback: check if any label appears in output
            for cand in candidates:
                cand_label_str, _ = cand_to_label.get(cand.candidate_id, ("", None))
                if cand_label_str and f"[{cand_label_str}]" in text:
                    selected_cand = cand
                    break

        if selected_cand is None:
            # Mark invalid action: fallback to WAIT or BLOCKED
            invalid = True
            selected_cand = next(
                (c for c in candidates if c.action_id == "wait"),
                candidates[-1],  # BLOCKED
            )

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
