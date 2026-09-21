"""DiffusionGemma direct browser decision backend running on Apple Silicon MLX."""

from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, List, Optional

import mlx.core as mx

from .base import Decision, calculate_entropy, calculate_top2_margin

_ENCODER_PATCHED = False


def _patch_diffusion_encoder():
    global _ENCODER_PATCHED
    if _ENCODER_PATCHED:
        return
    try:
        from mlx_vlm.models.diffusion_gemma.language import EncoderModel

        def layer_eval_call(
            self,
            input_ids,
            attention_mask=None,
            cache=None,
            pixel_values=None,
            mm_token_type_ids=None,
        ):
            h = self._embed_inputs(
                input_ids,
                pixel_values=pixel_values,
                mm_token_type_ids=mm_token_type_ids,
            )
            if cache is None:
                cache = self.make_cache()
            masks = self._make_encoder_masks(
                h, cache, attention_mask, mm_token_type_ids=mm_token_type_ids
            )
            for i, (layer, c, mask) in enumerate(zip(self.decoder.layers, cache, masks)):
                h = layer(
                    h,
                    mask,
                    c,
                    decoder=False,
                    layer_scalar=self.language_model.layers[i].layer_scalar,
                )
                mx.eval(h, c.state)
            return self.decoder.norm(h), cache

        EncoderModel.__call__ = layer_eval_call
        _ENCODER_PATCHED = True
    except Exception:
        pass


class MlxDiffusionDirectBackend:
    """Central Experiment Backend.

    Prefill prompt -> Small diffusion canvas -> Extract ACTION_SLOT logits ->
    Softmax on Metal over valid candidates -> Fast decision with uncertainty metrics.
    """

    name: str = "mlx_direct"

    def __init__(
        self,
        model: Any = None,
        processor: Any = None,
        model_name: str = "mlx-community/diffusiongemma-26B-A4B-it-4bit",
        canvas_length: int = 32,
        num_passes: int = 4,
        action_slot: Optional[int] = None,
        use_self_conditioning: bool = True,
        compile_graph: bool = False,
        seed: Optional[int] = None,
        label_strategy: str = "index",
        canvas_prefix: Optional[str] = "ACTION=",
        early_stopping: bool = True,
        stability_steps: int = 2,
        entropy_threshold: float = 0.05,
        margin_threshold: float = 0.85,
        pass1_margin_threshold: float = 0.65,
        pass1_entropy_threshold: float = 0.35,
        enable_prefix_caching: bool = True,
    ):
        self.model_name = model_name
        self.canvas_length = canvas_length
        self.num_passes = num_passes
        self.action_slot = action_slot  # None means slot 0 (or default)
        self.use_self_conditioning = use_self_conditioning
        self.compile_graph = compile_graph
        self.seed = seed
        self.label_strategy = label_strategy
        self.canvas_prefix = canvas_prefix
        self.early_stopping = early_stopping
        self.stability_steps = stability_steps
        self.entropy_threshold = entropy_threshold
        self.margin_threshold = margin_threshold
        self.pass1_margin_threshold = pass1_margin_threshold
        self.pass1_entropy_threshold = pass1_entropy_threshold
        self.enable_prefix_caching = enable_prefix_caching

        self._model = model
        self._processor = processor
        self._cached_goal: Optional[str] = None
        self._cached_prefix_tokens: Optional[List[int]] = None
        self._cached_prefix_cache: Optional[List[Any]] = None

    def reset_session(self):
        """Clear cached session KV prefix state between distinct tasks."""
        self._cached_goal = None
        self._cached_prefix_tokens = None
        self._cached_prefix_cache = None

    def _ensure_loaded(self):
        _patch_diffusion_encoder()
        try:
            total_ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3)
            if total_ram_gb < 23.5:
                import warnings

                warnings.warn(
                    f"DiffBrowse (DiffusionGemma-26B) strictly requires ≥ 24 GB Unified Memory. "
                    f"Detected {total_ram_gb:.1f} GB. "
                    "You may experience severe swap thrashing or OS memory termination.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        except Exception:
            pass

        if self._model is None or self._processor is None:
            from mlx_vlm import load

            self._model, self._processor = load(self.model_name)

    def _resolve_slot(self, canvas_len: int) -> int:
        if self.action_slot is None:
            return 0
        if self.action_slot < 0:
            return max(0, canvas_len + self.action_slot)
        return min(self.action_slot, canvas_len - 1)

    def choose(
        self,
        goal: str,
        browser_state: Dict[str, Any],
        action_space: tuple,
        history: List[Dict[str, Any]],
    ) -> Decision:
        from mlx_vlm.generate.diffusion import (
            _diffusion_initial_canvas,
            _make_diffusion_decoder_logits_fns,
        )
        from mlx_vlm.prompt_utils import apply_chat_template

        from ..action_candidates import LabelPool, flatten_actions

        self._ensure_loaded()
        total_tic = time.perf_counter()

        # 1. Flatten executable action candidates
        candidates = flatten_actions(browser_state["actions"])
        pool = LabelPool(self._processor)
        cand_token_ids, token_to_cand, cand_to_label, candidate_lines = pool.map_candidates(
            candidates, strategy=self.label_strategy
        )

        # 2. Format compact prompt
        recent_hist = "\n".join(
            f"- Step {h.get('step', i+1)}: {h.get('action')} -> {'OK' if h.get('page_changed') else 'No change'}"
            for i, h in enumerate(history[-5:])
        ) or "None"

        user_content = (
            f"Goal: {goal}\n"
            f"Page Title: {browser_state.get('title', '')}\n"
            f"Page URL: {browser_state.get('url', '')}\n"
            f"Rules:\n"
            f"- Advance the goal from the current page. Do not repeat satisfied steps.\n"
            f"- Do not choose a field that already contains the requested value.\n"
            f"- When typing into a combobox/search field, CLICK the matching autocomplete suggestion next.\n"
            f"- For date pickers, CLICK the field, select the date, then CLICK Done.\n"
            f"- When required fields are ready, CLICK Search immediately.\n\n"
            f"Recent Action History:\n{recent_hist}\n\n"
            f"{candidate_lines}\n\n"
            "Select the single best next action to advance the goal.\n"
            "Respond strictly with:\nACTION=<label>"
        )

        formatted_prompt = apply_chat_template(self._processor, self._model.config, user_content)

        # 3. Tokenize prompt
        tok_tic = time.perf_counter()
        input_ids = self._processor.tokenizer.encode(formatted_prompt, add_special_tokens=False)
        input_ids_mx = mx.array([input_ids], dtype=mx.int32)
        tokenize_ms = (time.perf_counter() - tok_tic) * 1000

        # 4. Prefill KV cache (with session prefix caching if enabled)
        from mlx_vlm.apc import _clone_prompt_cache_for_apc

        prefill_tic = time.perf_counter()
        kv_cache = None
        cache_hit = False

        if (
            self.enable_prefix_caching
            and self._cached_goal == goal
            and self._cached_prefix_tokens is not None
            and self._cached_prefix_cache is not None
        ):
            # Check how many leading tokens match the cached prefix
            prefix_match = 0
            for a, b in zip(input_ids, self._cached_prefix_tokens):
                if a == b:
                    prefix_match += 1
                else:
                    break

            if prefix_match >= len(self._cached_prefix_tokens):
                cloned = _clone_prompt_cache_for_apc(self._cached_prefix_cache)
                if cloned is not None:
                    suffix_ids = input_ids_mx[:, prefix_match:]
                    if suffix_ids.shape[1] > 0:
                        kv_cache = self._model.diffusion_update_cache(suffix_ids, cache=cloned)
                        mx.eval([c.state for c in kv_cache])
                    else:
                        kv_cache = cloned
                    cache_hit = True

        if kv_cache is None:
            if self.enable_prefix_caching and len(input_ids) > 16:
                prefix_cutoff = min(len(input_ids), 256)
                self._cached_goal = goal
                self._cached_prefix_tokens = input_ids[:prefix_cutoff]
                prefix_ids_chunk = input_ids_mx[:, :prefix_cutoff]
                cached_c = self._model.make_cache()
                self._model.diffusion_prefill_cache(prefix_ids_chunk, cache=cached_c)
                mx.eval([c.state for c in cached_c])
                self._cached_prefix_cache = cached_c

                cloned = _clone_prompt_cache_for_apc(self._cached_prefix_cache)
                suffix_ids = input_ids_mx[:, prefix_cutoff:]
                if suffix_ids.shape[1] > 0 and cloned is not None:
                    kv_cache = self._model.diffusion_update_cache(suffix_ids, cache=cloned)
                    mx.eval([c.state for c in kv_cache])
                else:
                    kv_cache = cloned or self._cached_prefix_cache
            else:
                kv_cache = self._model.make_cache()
                self._model.diffusion_prefill_cache(input_ids_mx, cache=kv_cache)
                mx.eval([c.state for c in kv_cache])

        prefill_ms = (time.perf_counter() - prefill_tic) * 1000

        # 5. Initialize Canvas & Slot
        if self.seed is not None:
            mx.random.seed(self.seed)

        vocab_size = int(self._model.config.text_config.vocab_size)
        canvas_len = self.canvas_length

        prefix_ids = None
        prefix_len = 0
        if self.canvas_prefix:
            tokenizer = getattr(self._processor, "tokenizer", self._processor)
            prefix_tokens = tokenizer.encode(self.canvas_prefix, add_special_tokens=False)
            prefix_ids = mx.array([prefix_tokens], dtype=input_ids_mx.dtype)
            prefix_len = len(prefix_tokens)
            slot_idx = prefix_len
        else:
            slot_idx = self._resolve_slot(canvas_len)

        current_canvas = _diffusion_initial_canvas(
            decoder_input_ids=prefix_ids,
            start_index=0,
            batch_size=1,
            canvas_length=canvas_len,
            vocab_size=vocab_size,
            dtype=input_ids_mx.dtype,
        )

        mask_mapping = self._model.diffusion_decoder_masks(current_canvas, kv_cache)
        decoder_without_sc, decoder_with_sc = _make_diffusion_decoder_logits_fns(
            self._model,
            kv_cache,
            mask_mapping,
            compile_graph=self.compile_graph,
        )

        sc_context = self._model.diffusion_prepare_self_conditioning() if self.use_self_conditioning else None
        self_conditioning = None

        cand_token_ids_mx = mx.array(cand_token_ids, dtype=mx.int32)

        # 6. Denoising passes on ACTION_SLOT
        decoder_tic = time.perf_counter()
        pass_trajectories: List[Dict[str, Any]] = []
        final_probs_dict: Dict[str, float] = {}
        top_cand_history: List[int] = []
        selected_candidate = candidates[0]
        actual_passes = 0

        for p in range(1, self.num_passes + 1):
            actual_passes = p
            # Decoder forward
            if self_conditioning is None:
                logits = decoder_without_sc(current_canvas)
            else:
                logits = decoder_with_sc(current_canvas, self_conditioning)

            # RESTRICT TO ACTION_SLOT AND VALID CANDIDATE TOKENS ON METAL
            slot_logits = logits[0, slot_idx, :]  # shape: (vocab_size,)
            cand_logits = slot_logits[cand_token_ids_mx]  # shape: (num_candidates,)
            cand_probs = mx.softmax(cand_logits.astype(mx.float32), axis=-1)

            # Materialize only the small candidate probabilities to CPU
            mx.eval(cand_probs)
            probs_list = cand_probs.tolist()

            # Record pass metrics
            pass_probs = {c.action_id: float(p_val) for c, p_val in zip(candidates, probs_list)}
            pass_entropy = calculate_entropy(probs_list)
            pass_margin = calculate_top2_margin(probs_list)
            top_idx = int(mx.argmax(cand_probs).item())
            top_cand = candidates[top_idx]
            top_prob = float(probs_list[top_idx])

            pass_trajectories.append(
                {
                    "pass": p,
                    "top_action": top_cand.action_id,
                    "top_candidate": top_cand.candidate_id,
                    "probability": round(top_prob, 4),
                    "entropy": round(pass_entropy, 4),
                    "margin": round(pass_margin, 4),
                }
            )

            selected_candidate = top_cand
            final_probs_dict = pass_probs
            top_cand_history.append(top_idx)

            # Early stopping checks
            if self.early_stopping:
                pass1_ready = (
                    pass_margin >= self.pass1_margin_threshold
                    or pass_entropy <= self.pass1_entropy_threshold
                    or top_prob >= 0.60
                )
                if p == 1 and pass1_ready:
                    break
                if p >= self.stability_steps:
                    recent_tops = top_cand_history[-self.stability_steps :]
                    is_stable = len(recent_tops) == self.stability_steps and all(
                        x == recent_tops[0] for x in recent_tops
                    )
                    is_confident = (pass_entropy < self.entropy_threshold) or (pass_margin > self.margin_threshold)
                    if is_stable and is_confident:
                        break

            # Prepare for next pass
            if p < self.num_passes:
                if self.use_self_conditioning and sc_context is not None:
                    self_conditioning = self._model.diffusion_self_conditioning(logits, sc_context)

                # Denoise canvas tokens for next step
                argmax_tokens = mx.argmax(logits, axis=-1).astype(input_ids_mx.dtype)
                if prefix_ids is not None:
                    argmax_tokens[:, :prefix_len] = prefix_ids
                current_canvas = argmax_tokens

        decoder_ms = (time.perf_counter() - decoder_tic) * 1000
        total_model_ms = prefill_ms + decoder_ms
        total_decision_ms = (time.perf_counter() - total_tic) * 1000

        final_entropy = calculate_entropy(final_probs_dict)
        final_margin = calculate_top2_margin(final_probs_dict)
        top_prob = max(final_probs_dict.values()) if final_probs_dict else 1.0

        # Release temporary Metal scratchpad allocations back to system pool
        try:
            mx.metal.clear_cache()
        except Exception:
            pass

        return Decision(
            action_id=selected_candidate.action_id,
            operation=selected_candidate.operation,
            target=selected_candidate.target,
            confidence=top_prob,
            probabilities=final_probs_dict,
            raw_top_probability=top_prob,
            entropy=final_entropy,
            top2_margin=final_margin,
            stable_steps=len([x for x in top_cand_history if x == top_cand_history[-1]]),
            tokenize_ms=tokenize_ms,
            prefill_ms=prefill_ms,
            decoder_ms=decoder_ms,
            total_model_ms=total_model_ms,
            total_decision_ms=total_decision_ms,
            model=self.model_name,
            backend=self.name,
            metadata={
                "canvas_length": self.canvas_length,
                "action_slot": slot_idx,
                "passes_configured": self.num_passes,
                "passes_executed": actual_passes,
                "self_conditioning": self.use_self_conditioning,
                "compiled": self.compile_graph,
                "seed": self.seed,
                "label_strategy": self.label_strategy,
                "cache_hit": cache_hit,
                "trajectory": pass_trajectories,
                "candidate_count": len(candidates),
                "prompt_tokens": len(input_ids),
            },
        )

    def generate_field_text(self, context: Dict[str, Any], max_tokens: int = 32) -> str:
        """Generate text to fill an input field using the local DiffusionGemma weights."""
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        self._ensure_loaded()
        goal = context.get("goal", "")
        field_info = context.get("field", {})
        label = field_info.get("label") or field_info.get("role") or "input field"
        page = context.get("page", {})

        prompt = (
            f"User Goal: {goal}\n"
            f"Page: {page.get('title', '')}\n"
            f"Active Form Field: {label}\n\n"
            f"Task: Infer the exact string to type into this form field to achieve the goal.\n"
            f"Respond strictly in the format:\nVALUE=\"<text>\""
        )
        formatted = apply_chat_template(self._processor, self._model.config, prompt)
        result = generate(
            model=self._model,
            processor=self._processor,
            prompt=formatted,
            max_tokens=max_tokens,
            temperature=0.0,
            generation_mode="diffusion",
        )
        raw = result.text if hasattr(result, "text") else str(result)
        try:
            mx.metal.clear_cache()
        except Exception:
            pass

        match = re.search(r'VALUE\s*[:=]\s*["\']?([^"\'\n]+)["\']?', raw)
        if match:
            return match.group(1).strip().strip('*')
        cleaned = re.sub(r'VALUE\s*[:=]\s*', '', raw).strip().strip('"\'*')
        return cleaned.split("\n")[0].strip() or label


