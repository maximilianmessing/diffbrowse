"""Training-Free Structured Diffusion decision backend on Apple Silicon MLX.

Design references and immutable upstream revisions: docs/structured-diffusion-plan.md.
Combines seeded read-only slots and template pinning with unrestricted answer refinement
and final constrained selection over discrete diffusion logits.
"""

from __future__ import annotations

import glob
import os
import random
import re
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx

from ..decision_request import build_decision_request
from ..structured_canvas import (
    CanvasCompiler,
    CanvasLayout,
    build_cross_head_attention_mask,
)
from .base import Decision, calculate_entropy, calculate_top2_margin

_ENCODER_PATCHED = False


def _patch_diffusion_encoder() -> None:
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


class MlxDiffusionStructuredBackend:
    """Structured Discrete Diffusion Decision Backend on Apple Silicon Metal."""

    name: str = "mlx_structured"

    def __init__(
        self,
        model: Any = None,
        processor: Any = None,
        model_name: str = "mlx-community/diffusiongemma-26B-A4B-it-4bit",
        local_model_path: Optional[str] = None,
        mode: str = "initial_default",
        num_reads: Optional[int] = None,
        num_passes: Optional[int] = None,
        seed: Optional[int] = None,
        max_canvas_width: int = 256,
        offline_only: bool = True,
        enable_prefix_caching: bool = True,
        compile_graph: bool = False,
        adaptive_escalation: bool = True,
        pass1_margin_threshold: float = 0.65,
        pass1_entropy_threshold: float = 0.35,
        use_attention_isolation: bool = True,
        use_temperature_schedule: bool = True,
        hierarchical_partitioning: bool = True,
    ):
        self.model_name = model_name
        self.local_model_path = local_model_path
        self.mode = mode
        self.seed = seed
        self.max_canvas_width = max_canvas_width
        self.offline_only = offline_only
        self.enable_prefix_caching = enable_prefix_caching
        self.compile_graph = compile_graph
        self.pass1_margin_threshold = pass1_margin_threshold
        self.pass1_entropy_threshold = pass1_entropy_threshold
        self.use_attention_isolation = use_attention_isolation
        self.use_temperature_schedule = use_temperature_schedule
        self.hierarchical_partitioning = hierarchical_partitioning

        # Resolve mode-specific defaults if not explicitly overridden
        if mode == "initial_default":
            self.num_reads = num_reads if num_reads is not None else 1
            self.num_passes = num_passes if num_passes is not None else 1
            self.adaptive_escalation = adaptive_escalation
        elif mode == "refinement_comparison":
            self.num_reads = num_reads if num_reads is not None else 1
            self.num_passes = num_passes if num_passes is not None else 2
            self.adaptive_escalation = False
        elif mode == "noise_comparison":
            self.num_reads = num_reads if num_reads is not None else 4
            self.num_passes = num_passes if num_passes is not None else 1
            self.adaptive_escalation = False
        elif mode == "adaptive":
            self.num_reads = num_reads if num_reads is not None else 1
            self.num_passes = num_passes if num_passes is not None else 1
            self.adaptive_escalation = True
        else:
            self.num_reads = num_reads if num_reads is not None else 1
            self.num_passes = num_passes if num_passes is not None else 1
            self.adaptive_escalation = adaptive_escalation

        self._model = model
        self._processor = processor
        self._compiler: Optional[CanvasCompiler] = None
        self._cached_goal: Optional[str] = None
        self._cached_prefix_tokens: Optional[List[int]] = None
        self._cached_prefix_cache: Optional[List[Any]] = None
        self._model_load_ms: Optional[float] = None

    def reset_session(self) -> None:
        """Clear cached session KV prefix state between distinct tasks."""
        self._cached_goal = None
        self._cached_prefix_tokens = None
        self._cached_prefix_cache = None

    def _resolve_local_path(self) -> str:
        if self.local_model_path:
            if os.path.exists(self.local_model_path):
                return self.local_model_path
            raise RuntimeError(
                f"Offline policy violation: local checkpoint directory {self.local_model_path!r} does not exist."
            )
        # Check standard huggingface hub cache
        cache_base = "~/.cache/huggingface/hub/models--mlx-community--diffusiongemma-26B-A4B-it-4bit/snapshots/*"
        pattern = os.path.expanduser(cache_base)
        snapshots = glob.glob(pattern)
        if snapshots and os.path.exists(snapshots[0]):
            return snapshots[0]
        if self.offline_only:
            raise RuntimeError(
                f"Offline policy violation: local checkpoint directory not found for {self.model_name}. "
                "Automatic remote download is disabled under offline_only=True."
            )
        return self.model_name

    def _ensure_loaded(self) -> None:
        _patch_diffusion_encoder()
        try:
            total_ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3)
            if total_ram_gb < 23.5:
                warnings.warn(
                    f"DiffBrowse (DiffusionGemma-26B) strictly requires ≥ 24 GB Unified Memory. "
                    f"Detected {total_ram_gb:.1f} GB. "
                    "You may experience severe swap thrashing or OS memory termination.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        except Exception:
            pass

        try:
            if hasattr(mx, "set_cache_limit"):
                mx.set_cache_limit(512 * 1024 * 1024)
            elif hasattr(mx, "metal") and hasattr(mx.metal, "set_cache_limit"):
                mx.metal.set_cache_limit(512 * 1024 * 1024)
        except Exception:
            pass

        if self._model is None or self._processor is None:
            from mlx_vlm import load

            resolved_path = self._resolve_local_path()
            load_tic = time.perf_counter()
            self._model, self._processor = load(resolved_path)
            self._model_load_ms = (time.perf_counter() - load_tic) * 1000

        if self._compiler is None:
            tok = getattr(self._processor, "tokenizer", self._processor)
            self._compiler = CanvasCompiler(tok, max_width=self.max_canvas_width)

    def _format_prompt(
        self,
        goal: str,
        browser_state: Dict[str, Any],
        questions: Dict[str, Any],
        layout: CanvasLayout,
        history: List[Dict[str, Any]],
    ) -> str:
        lines = [
            "Task: Autonomous Web Navigation",
            f"Goal: {goal}",
            "Supported Operations:",
            "- CLICK: Click an interactive target element.",
            "- TYPE: Type text into an input field or combobox.",
            "- SELECT: Select an option from a dropdown or menu.",
            "- SCROLL: Scroll the viewport to reveal content.",
            "- DONE: Complete task when every requirement is visibly satisfied.",
            "Instructions: Choose the single best operation and target head that directly advances the goal.",
            "",
            "Core Principles:",
            "- Do not repeat satisfied steps or re-select fields that already display the requested value.",
            "- If Search or Submit is visible and required fields are populated, CLICK it immediately.",
            "",
        ]

        if history:
            recent = []
            for i, h in enumerate(history[-6:], 1):
                act = h.get("action") or h.get("kind") or "action"
                txt = h.get("text")
                step_idx = h.get("step", i)
                if txt:
                    recent.append(f"- Step {step_idx}: {act} · {txt}")
                else:
                    recent.append(f"- Step {step_idx}: {act}")
            lines.append("Recent Action History:\n" + "\n".join(recent))
            lines.append("")

        lines.extend([
            f"Page Title: {browser_state.get('title', '')}",
            f"Page URL: {browser_state.get('url', '')}",
        ])
        page_text = browser_state.get("text", "")[:2000]
        if page_text:
            lines.append(f"Visible Page Content:\n{page_text}")

        lines.append("\nQuestions to decide the next action:")
        for slot in layout.slots:
            if slot.position is None:
                lines.append(f"Question: {slot.name} (Resolved: {slot.choices[0]})")
                continue
            q_data = questions.get(slot.name, {})
            criteria = q_data.get("criteria", {})
            instr = q_data.get("instructions", {})
            head_desc = f"Question: {slot.name}"
            if "operation" in instr:
                head_desc += f" (Which target if {instr['operation']} is selected?)"
            elif "goal" in instr:
                head_desc += f" ({instr['goal']})"
            lines.append(head_desc)
            for choice, label in zip(slot.choices, slot.labels):
                crit_val = criteria.get(choice, "")
                if isinstance(crit_val, dict):
                    elem_str = crit_val.get("element", "")
                    role = crit_val.get("role", "")
                    val_str = crit_val.get("current_value", "")
                    tags = []
                    if role:
                        tags.append(role)
                    if crit_val.get("checked") == "true":
                        tags.append("checked")
                    elif crit_val.get("selected") == "true":
                        tags.append("selected")
                    if crit_val.get("expanded") == "true":
                        tags.append("expanded")

                    annot = f" ({', '.join(tags)})" if tags else ""
                    if val_str:
                        crit_text = f"{elem_str}{annot} [value: \"{val_str}\"]"
                    else:
                        crit_text = f"{elem_str}{annot}"
                else:
                    crit_text = str(crit_val)
                lines.append(f"  [{label}] {choice}: {crit_text}")

        lines.append("\nFill in each answer slot with the label that advances the goal.")
        return "\n".join(lines)

    def _compute_invariant_prefix_tokens(self, goal: str) -> List[int]:
        from mlx_vlm.prompt_utils import apply_chat_template

        preamble = (
            "Task: Autonomous Web Navigation\n"
            f"Goal: {goal}\n"
            "Supported Operations:\n"
            "- CLICK: Click an interactive target element.\n"
            "- TYPE: Type text into an input field or combobox.\n"
            "- SELECT: Select an option from a dropdown or menu.\n"
            "- SCROLL: Scroll the viewport to reveal content.\n"
            "- DONE: Complete task when every requirement is visibly satisfied.\n"
            "Instructions: Choose the single best operation and target head that directly advances the goal.\n\n"
            "Core Principles:\n"
            "- Do not repeat satisfied steps or re-select fields that already display the requested value.\n"
            "- If Search or Submit is visible and required fields are populated, CLICK it immediately.\n\n"
        )
        t1 = apply_chat_template(self._processor, self._model.config, preamble + "A\n")
        t2 = apply_chat_template(self._processor, self._model.config, preamble + "B\n")
        tok = getattr(self._processor, "tokenizer", self._processor)
        ids1 = tok.encode(t1, add_special_tokens=False)
        ids2 = tok.encode(t2, add_special_tokens=False)
        common_len = 0
        for a, b in zip(ids1, ids2):
            if a == b:
                common_len += 1
            else:
                break
        return ids1[:common_len]

    def choose(
        self,
        goal: str,
        browser_state: Dict[str, Any],
        action_space: Tuple[Any, ...],
        history: List[Dict[str, Any]],
    ) -> Decision:
        from mlx_vlm.apc import _clone_prompt_cache_for_apc
        from mlx_vlm.generate.diffusion import (
            _diffusion_entropy_transfer_mask,
            _diffusion_initialize_canvas,
            _diffusion_token_entropy,
            _make_diffusion_decoder_logits_fns,
        )
        from mlx_vlm.prompt_utils import apply_chat_template

        self._ensure_loaded()
        total_tic = time.perf_counter()

        elements, targets, controls = action_space
        body = build_decision_request(self.model_name, goal, browser_state, action_space, history)
        questions = body["questions"]

        # 1. Compile structured canvas layout
        layout = self._compiler.compile(questions, hierarchical=self.hierarchical_partitioning)
        active_questions = layout.questions if layout.questions is not None else questions
        tokenizer = getattr(self._processor, "tokenizer", self._processor)

        # 2. Format observation prompt
        user_content = self._format_prompt(goal, browser_state, active_questions, layout, history)
        formatted_prompt = apply_chat_template(self._processor, self._model.config, user_content)

        tokenize_tic = time.perf_counter()
        input_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)
        input_ids_mx = mx.array([input_ids], dtype=mx.int32)
        tokenize_ms = (time.perf_counter() - tokenize_tic) * 1000

        # 3. Prefill observation into KV cache (with multi-turn prefix caching)
        prefill_tic = time.perf_counter()
        cache_hit = False
        match_len = 0
        kv_cache = None

        if (
            self.enable_prefix_caching
            and self._cached_goal == goal
            and self._cached_prefix_tokens is not None
            and self._cached_prefix_cache is not None
        ):
            p_len = len(self._cached_prefix_tokens)
            if len(input_ids) >= p_len and input_ids[:p_len] == self._cached_prefix_tokens:
                try:
                    cloned = _clone_prompt_cache_for_apc(self._cached_prefix_cache)
                except Exception:
                    cloned = None

                if cloned is not None:
                    suffix_ids = input_ids_mx[:, p_len:]
                    if suffix_ids.shape[1] > 0:
                        kv_cache = self._model.diffusion_update_cache(suffix_ids, cache=cloned)
                        mx.eval([c.state for c in kv_cache])
                    else:
                        kv_cache = cloned
                    cache_hit = True
                    match_len = p_len

        if kv_cache is None:
            if self.enable_prefix_caching and len(input_ids) >= 16:
                p_len = 0
                try:
                    p_toks = self._compute_invariant_prefix_tokens(goal)
                    if len(input_ids) >= len(p_toks) and input_ids[:len(p_toks)] == p_toks:
                        p_len = len(p_toks)
                except Exception:
                    p_len = 0

                if p_len >= 16:
                    prefix_mx = input_ids_mx[:, :p_len]
                    cached_c = self._model.make_cache()
                    self._model.diffusion_prefill_cache(prefix_mx, cache=cached_c)
                    mx.eval([c.state for c in cached_c])
                    self._cached_goal = goal
                    self._cached_prefix_tokens = input_ids[:p_len]
                    self._cached_prefix_cache = cached_c

                    try:
                        cloned = _clone_prompt_cache_for_apc(cached_c)
                    except Exception:
                        cloned = None
                    suffix_ids = input_ids_mx[:, p_len:]
                    if suffix_ids.shape[1] > 0 and cloned is not None:
                        kv_cache = self._model.diffusion_update_cache(suffix_ids, cache=cloned)
                        mx.eval([c.state for c in kv_cache])
                    else:
                        kv_cache = cloned or cached_c
                else:
                    kv_cache = self._model.make_cache()
                    self._model.diffusion_prefill_cache(input_ids_mx, cache=kv_cache)
                    mx.eval([c.state for c in kv_cache])
                    self._cached_goal = goal
                    self._cached_prefix_tokens = input_ids
                    self._cached_prefix_cache = kv_cache
            else:
                kv_cache = self._model.make_cache()
                self._model.diffusion_prefill_cache(input_ids_mx, cache=kv_cache)
                mx.eval([c.state for c in kv_cache])

        prefill_ms = (time.perf_counter() - prefill_tic) * 1000

        # 4. Run independent reads
        if hasattr(self._model.config, "text_config"):
            tc = self._model.config.text_config
            vocab_size = int(getattr(tc, "vocab_size", getattr(tokenizer, "vocab_size", 256)))
        elif isinstance(self._model.config, dict):
            tc = self._model.config.get("text_config", {})
            if isinstance(tc, dict):
                vocab_size = int(tc.get("vocab_size", getattr(tokenizer, "vocab_size", 256)))
            else:
                vocab_size = int(getattr(tc, "vocab_size", getattr(tokenizer, "vocab_size", 256)))
        else:
            vocab_size = int(getattr(self._model.config, "vocab_size", getattr(tokenizer, "vocab_size", 256)))
        decoder_tic = time.perf_counter()
        actual_forward_passes = 0

        read_distributions: List[Dict[str, Dict[str, float]]] = []
        read_top_choices: List[Dict[str, str]] = []
        read_allowed_mass: List[Dict[str, float]] = []
        read_entropy: List[Dict[str, float]] = []
        read_margin: List[Dict[str, float]] = []
        read_validity: List[Dict[str, bool]] = []

        template_tokens = list(layout.tokens)
        template_mx = mx.array([layout.tokens], dtype=input_ids_mx.dtype)
        fixed_mask_arr = mx.array(layout.fixed)
        fixed_mask_1d = fixed_mask_arr[None, :]
        fixed_mask_sc = fixed_mask_arr[None, :, None]

        # Cross-head attention mask isolation
        mask_mapping = self._model.diffusion_decoder_masks(template_mx, kv_cache)
        if self.use_attention_isolation:
            if isinstance(mask_mapping, dict):
                updated_masks = {}
                for k, v in mask_mapping.items():
                    if v is not None:
                        enc_len = v.shape[-1] - layout.width
                        if enc_len >= 0:
                            b_mask = build_cross_head_attention_mask(layout, enc_len)
                            updated_masks[k] = v & b_mask
                        else:
                            updated_masks[k] = v
                    else:
                        enc_len = input_ids_mx.shape[1]
                        updated_masks[k] = build_cross_head_attention_mask(layout, enc_len)
                mask_mapping = updated_masks

        decoder_without_sc, decoder_with_sc = _make_diffusion_decoder_logits_fns(
            self._model,
            kv_cache,
            mask_mapping,
            compile_graph=self.compile_graph,
        )

        def _extract_slot_metrics(target_logits: mx.array, temp_scale: Optional[float] = None) -> Tuple[
            Dict[str, Dict[str, float]],
            Dict[str, str],
            Dict[str, float],
            Dict[str, float],
            Dict[str, float],
            Dict[str, bool],
        ]:
            if temp_scale is not None and temp_scale > 0:
                eval_logits = target_logits / temp_scale
            else:
                eval_logits = target_logits

            head_probs: Dict[str, Dict[str, float]] = {}
            head_tops: Dict[str, str] = {}
            head_mass: Dict[str, float] = {}
            head_ent: Dict[str, float] = {}
            head_mar: Dict[str, float] = {}
            head_val: Dict[str, bool] = {}

            tensors_to_eval = []
            slot_eval_meta = []

            for s in layout.slots:
                if s.position is None:
                    choice = s.choices[0]
                    head_probs[s.name] = {choice: 1.0}
                    head_tops[s.name] = choice
                    head_mass[s.name] = 1.0
                    head_ent[s.name] = 0.0
                    head_mar[s.name] = 1.0
                    head_val[s.name] = True
                    continue

                pos_logits = eval_logits[0, s.position, :]
                tok_ids_mx = (
                    s.token_ids_mx
                    if getattr(s, "token_ids_mx", None) is not None
                    else mx.array(s.token_ids, dtype=mx.int32)
                )
                allowed_logits = pos_logits[tok_ids_mx]
                cond_probs = mx.softmax(allowed_logits.astype(mx.float32), axis=-1)
                full_lse = mx.logsumexp(pos_logits.astype(mx.float32))
                allowed_mass = mx.sum(mx.exp(allowed_logits.astype(mx.float32) - full_lse))
                cond_entropy = -mx.sum(cond_probs * mx.log(mx.maximum(cond_probs, 1e-12)))

                if len(s.token_ids) > 1:
                    sorted_probs = mx.sort(cond_probs)
                    margin = sorted_probs[-1] - sorted_probs[-2]
                else:
                    margin = mx.array(1.0, dtype=mx.float32)

                unrestricted_argmax = mx.argmax(pos_logits)
                argmax_valid = mx.any(tok_ids_mx == unrestricted_argmax)

                tensors_to_eval.extend([cond_probs, allowed_mass, cond_entropy, margin, argmax_valid])
                slot_eval_meta.append(s)

            if tensors_to_eval:
                mx.eval(*tensors_to_eval)

            t_idx = 0
            for s in slot_eval_meta:
                c_probs = tensors_to_eval[t_idx].tolist()
                a_mass = float(tensors_to_eval[t_idx + 1].item())
                f_ent = float(tensors_to_eval[t_idx + 2].item())
                mar = float(tensors_to_eval[t_idx + 3].item())
                a_val = bool(tensors_to_eval[t_idx + 4].item())
                t_idx += 5

                prob_dict = {choice: float(p) for choice, p in zip(s.choices, c_probs)}
                top_choice = max(prob_dict.items(), key=lambda x: x[1])[0]
                head_probs[s.name] = prob_dict
                head_tops[s.name] = top_choice
                head_mass[s.name] = a_mass
                head_ent[s.name] = f_ent
                head_mar[s.name] = mar
                head_val[s.name] = a_val

            return head_probs, head_tops, head_mass, head_ent, head_mar, head_val

        def _build_noisy_canvas(draw_seed: Optional[int]) -> mx.array:
            if draw_seed is not None:
                mx.random.seed(draw_seed)
            canvas_tokens = list(template_tokens)
            for s in layout.slots:
                if s.position is not None:
                    canvas_tokens[s.position] = random.randint(0, vocab_size - 1)
            return mx.array([canvas_tokens], dtype=input_ids_mx.dtype)

        def _refine_canvas(logits_tensor: mx.array, curr_canvas: mx.array) -> Tuple[mx.array, Any]:
            sc_context = self._model.diffusion_prepare_self_conditioning()
            next_sc = self._model.diffusion_self_conditioning(logits_tensor, sc_context)
            self_conditioning = mx.where(fixed_mask_sc, 0.0, next_sc)

            argmax_canvas = mx.argmax(logits_tensor, axis=-1).astype(input_ids_mx.dtype)
            token_entropy = _diffusion_token_entropy(logits_tensor)
            acceptance_mask = _diffusion_entropy_transfer_mask(token_entropy, 0.1)
            refined_canvas = mx.where(
                acceptance_mask,
                argmax_canvas,
                _diffusion_initialize_canvas(1, layout.width, vocab_size, input_ids_mx.dtype),
            )
            restored_canvas = mx.where(fixed_mask_1d, template_mx, refined_canvas)
            return restored_canvas, self_conditioning

        escalation_stage = "deterministic_fixed"
        t_pass1 = 0.8 if self.use_temperature_schedule else None
        t_pass2 = 0.4 if self.use_temperature_schedule else None

        if not self.adaptive_escalation:
            for r in range(self.num_reads):
                draw_seed = (self.seed + r * 1000 + 7) if self.seed is not None else None
                curr_canvas = _build_noisy_canvas(draw_seed)
                actual_forward_passes += 1
                logits = decoder_without_sc(curr_canvas)

                if self.num_passes >= 2:
                    actual_forward_passes += 1
                    restored_canvas, self_cond = _refine_canvas(logits, curr_canvas)
                    logits = decoder_with_sc(restored_canvas, self_cond)
                    scale_temp = t_pass2
                else:
                    scale_temp = t_pass1

                is_finite = mx.all(mx.isfinite(logits))
                mx.eval(is_finite)
                if not bool(is_finite.item()):
                    raise ValueError("Non-finite logits detected; no action executed.")

                h_probs, h_tops, h_mass, h_ent, h_mar, h_val = _extract_slot_metrics(logits, scale_temp)
                read_distributions.append(h_probs)
                read_top_choices.append(h_tops)
                read_allowed_mass.append(h_mass)
                read_entropy.append(h_ent)
                read_margin.append(h_mar)
                read_validity.append(h_val)
        else:
            # Adaptive escalation: 1x1 -> 1x2 -> 4x1
            draw_seed = (self.seed + 7) if self.seed is not None else None
            curr_canvas = _build_noisy_canvas(draw_seed)
            actual_forward_passes += 1
            logits = decoder_without_sc(curr_canvas)

            is_finite = mx.all(mx.isfinite(logits))
            mx.eval(is_finite)
            if not bool(is_finite.item()):
                raise ValueError("Non-finite logits detected; no action executed.")

            p1_probs, p1_tops, p1_mass, p1_ent, p1_mar, p1_val = _extract_slot_metrics(logits, t_pass1)

            # Exit on Pass 1 if operation is DONE or margin M = p1 - p2 >= 0.65 or entropy H <= 0.35
            op_mar = p1_mar.get("operation", 1.0)
            op_ent = p1_ent.get("operation", 0.0)
            is_pass1_confident = (
                p1_tops.get("operation") == "DONE"
                or op_mar >= self.pass1_margin_threshold
                or op_ent <= self.pass1_entropy_threshold
            )

            if is_pass1_confident:
                escalation_stage = "pass1_exit"
                read_distributions.append(p1_probs)
                read_top_choices.append(p1_tops)
                read_allowed_mass.append(p1_mass)
                read_entropy.append(p1_ent)
                read_margin.append(p1_mar)
                read_validity.append(p1_val)
            else:
                # Escalate to Pass 2 refinement with pinned-token restoration
                actual_forward_passes += 1
                restored_canvas, self_cond = _refine_canvas(logits, curr_canvas)
                p2_logits = decoder_with_sc(restored_canvas, self_cond)

                is_finite_2 = mx.all(mx.isfinite(p2_logits))
                mx.eval(is_finite_2)
                if not bool(is_finite_2.item()):
                    raise ValueError("Non-finite logits detected; no action executed.")

                p2_probs, p2_tops, p2_mass, p2_ent, p2_mar, p2_val = _extract_slot_metrics(p2_logits, t_pass2)
                op_mar_2 = p2_mar.get("operation", 1.0)
                op_ent_2 = p2_ent.get("operation", 0.0)
                is_pass2_confident = (
                    op_mar_2 >= self.pass1_margin_threshold or op_ent_2 <= self.pass1_entropy_threshold
                )

                if is_pass2_confident:
                    escalation_stage = "pass2_refinement"
                    read_distributions.append(p2_probs)
                    read_top_choices.append(p2_tops)
                    read_allowed_mass.append(p2_mass)
                    read_entropy.append(p2_ent)
                    read_margin.append(p2_mar)
                    read_validity.append(p2_val)
                else:
                    # Uncertainty persists after Pass 2 -> Fall back to averaging 4 independent noise draws
                    escalation_stage = "multi_read_draws"
                    read_distributions.append(p2_probs)
                    read_top_choices.append(p2_tops)
                    read_allowed_mass.append(p2_mass)
                    read_entropy.append(p2_ent)
                    read_margin.append(p2_mar)
                    read_validity.append(p2_val)

                    # Draw 2, 3, 4
                    for add_r in range(1, 4):
                        add_seed = (self.seed + add_r * 1000 + 7) if self.seed is not None else None
                        add_canvas = _build_noisy_canvas(add_seed)
                        actual_forward_passes += 1
                        add_l1 = decoder_without_sc(add_canvas)

                        actual_forward_passes += 1
                        add_restored, add_cond = _refine_canvas(add_l1, add_canvas)
                        add_l2 = decoder_with_sc(add_restored, add_cond)

                        is_finite_add = mx.all(mx.isfinite(add_l2))
                        mx.eval(is_finite_add)
                        if not bool(is_finite_add.item()):
                            raise ValueError("Non-finite logits detected; no action executed.")

                        add_p, add_t, add_m, add_e, add_mr, add_v = _extract_slot_metrics(add_l2, t_pass2)
                        read_distributions.append(add_p)
                        read_top_choices.append(add_t)
                        read_allowed_mass.append(add_m)
                        read_entropy.append(add_e)
                        read_margin.append(add_mr)
                        read_validity.append(add_v)

        decoder_ms = (time.perf_counter() - decoder_tic) * 1000

        # 5. Aggregate metrics across independent reads
        averaged_head_distributions: Dict[str, Dict[str, float]] = {}
        for s in layout.slots:
            head_name = s.name
            choice_keys = s.choices
            avg_probs = {
                c: sum(rd[head_name][c] for rd in read_distributions) / len(read_distributions)
                for c in choice_keys
            }
            averaged_head_distributions[head_name] = avg_probs

        selected_operation = max(
            averaged_head_distributions["operation"].items(), key=lambda x: x[1]
        )[0]

        # 6. Consume target only for the selected operation (with hierarchical candidate routing)
        selected_target: Optional[str] = None
        action_id: Optional[str] = None
        selected_group: Optional[str] = None
        group_prob: float = 1.0
        sub_target_prob: float = 1.0

        if selected_operation in ("CLICK", "TYPE_TEXT", "SELECT"):
            target_head = f"{selected_operation.lower()}_target"
            if layout.hierarchical_map and target_head in layout.hierarchical_map:
                h_info = layout.hierarchical_map[target_head]
                group_head = h_info["group_head"]
                if group_head in averaged_head_distributions:
                    selected_group = max(
                        averaged_head_distributions[group_head].items(), key=lambda x: x[1]
                    )[0]
                    group_prob = averaged_head_distributions[group_head][selected_group]
                    sub_head = h_info["sub_heads"].get(selected_group)
                    if sub_head and sub_head in averaged_head_distributions:
                        sub_dist = dict(averaged_head_distributions[sub_head])
                        if selected_operation == "CLICK" and history:
                            last_choice = history[-1].get("choice")
                            for tgt, act_dict in targets.get("CLICK", {}).items():
                                is_already_active = (
                                    act_dict.get("checked") == "true"
                                    or act_dict.get("selected") == "true"
                                    or act_dict.get("label", "").startswith("Open ")
                                )
                                was_recently_clicked = (
                                    act_dict.get("id") == last_choice
                                    or any(h.get("choice") == act_dict.get("id") for h in history[-2:])
                                )
                                if is_already_active and was_recently_clicked and tgt in sub_dist:
                                    sub_dist[tgt] = 0.0

                        selected_target = max(
                            sub_dist.items(), key=lambda x: x[1]
                        )[0]
                        sub_target_prob = sub_dist[selected_target]
                        action_dict = targets.get(selected_operation, {}).get(selected_target)
                        if action_dict:
                            action_id = action_dict["id"]
            elif target_head in averaged_head_distributions:
                target_dist = dict(averaged_head_distributions[target_head])
                if selected_operation == "CLICK" and history:
                    last_choice = history[-1].get("choice")
                    for tgt, act_dict in targets.get("CLICK", {}).items():
                        is_already_active = (
                            act_dict.get("checked") == "true"
                            or act_dict.get("selected") == "true"
                            or act_dict.get("label", "").startswith("Open ")
                        )
                        was_recently_clicked = (
                            act_dict.get("id") == last_choice
                            or any(h.get("choice") == act_dict.get("id") for h in history[-2:])
                        )
                        if is_already_active and was_recently_clicked and tgt in target_dist:
                            target_dist[tgt] = 0.0

                selected_target = max(
                    target_dist.items(), key=lambda x: x[1]
                )[0]
                sub_target_prob = target_dist[selected_target]
                action_dict = targets.get(selected_operation, {}).get(selected_target)
                if action_dict:
                    action_id = action_dict["id"]
        elif selected_operation in controls:
            action_id = controls[selected_operation]["id"]
            selected_target = None
        elif selected_operation in ("DONE", "BLOCKED"):
            action_id = selected_operation.lower()
            selected_target = None

        if not action_id:
            raise ValueError(
                f"Selected operation {selected_operation!r} target {selected_target!r} "
                "could not be grounded to an observed action."
            )

        # 7. Compute agreement across independent reads (supporting hierarchical heads)
        agreement_count = 0
        target_head = f"{selected_operation.lower()}_target"
        is_hier = bool(layout.hierarchical_map and target_head in layout.hierarchical_map)
        h_info = layout.hierarchical_map.get(target_head) if is_hier else None

        for tops in read_top_choices:
            if tops.get("operation") != selected_operation:
                continue
            if selected_target is None:
                agreement_count += 1
                continue
            if is_hier and h_info and selected_group:
                grp_head = h_info["group_head"]
                sub_head = h_info["sub_heads"].get(selected_group)
                if tops.get(grp_head) == selected_group and tops.get(sub_head) == selected_target:
                    agreement_count += 1
            else:
                if tops.get(target_head) == selected_target:
                    agreement_count += 1

        agreement_fraction = agreement_count / len(read_top_choices) if read_top_choices else 1.0

        # 8. Decision summary & telemetry
        top_prob = averaged_head_distributions["operation"][selected_operation]
        if selected_target:
            target_prob = group_prob * sub_target_prob
            combined_prob = round(top_prob * target_prob, 4)
        else:
            combined_prob = round(top_prob, 4)

        op_probs_dict = averaged_head_distributions["operation"]
        final_entropy = calculate_entropy(list(op_probs_dict.values()))
        final_margin = calculate_top2_margin(op_probs_dict)
        total_ms = (time.perf_counter() - total_tic) * 1000

        try:
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
            elif hasattr(mx.metal, "clear_cache"):
                mx.metal.clear_cache()
        except Exception:
            pass

        return Decision(
            action_id=action_id,
            operation=selected_operation,
            target=selected_target,
            confidence=combined_prob,
            probabilities=op_probs_dict,
            raw_top_probability=top_prob,
            entropy=final_entropy,
            top2_margin=final_margin,
            stable_steps=actual_forward_passes,
            tokenize_ms=tokenize_ms,
            prefill_ms=prefill_ms,
            decoder_ms=decoder_ms,
            total_decision_ms=total_ms,
            model=self.model_name,
            backend=self.name,
            metadata={
                "representation": "structured_canvas",
                "backend": self.name,
                "mode": self.mode,
                "num_reads": self.num_reads,
                "num_passes": self.num_passes,
                "canvas_width": layout.width,
                "prompt_tokens": len(input_ids),
                "actual_forward_passes": actual_forward_passes,
                "escalation_stage": escalation_stage,
                "adaptive_escalation": self.adaptive_escalation,
                "temperature_schedule": self.use_temperature_schedule,
                "attention_isolation": self.use_attention_isolation,
                "hierarchical_partitioning": self.hierarchical_partitioning,
                "hierarchical_group": selected_group,
                "per_head_distributions": averaged_head_distributions,
                "allowed_label_mass": read_allowed_mass[0] if read_allowed_mass else {},
                "head_entropy": read_entropy[0] if read_entropy else {},
                "head_margin": read_margin[0] if read_margin else {},
                "argmax_validity": read_validity[0] if read_validity else {},
                "agreement": agreement_fraction,
                "cache_hit": cache_hit,
                "cached_prefix_tokens": match_len if cache_hit else 0,
                "timing": {
                    "tokenize_ms": round(tokenize_ms, 2),
                    "prefill_ms": round(prefill_ms, 2),
                    "decoder_ms": round(decoder_ms, 2),
                    "total_ms": round(total_ms, 2),
                },
                "model_load_ms": round(self._model_load_ms, 2) if self._model_load_ms else None,
            },
        )

    def generate_field_text(self, context: Dict[str, Any], max_tokens: int = 16) -> str:
        """Synthesize form text using resident model weights without external fallback."""
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        self._ensure_loaded()
        goal = context.get("goal", "")
        field_info = context.get("field", {})
        label = field_info.get("label") or field_info.get("role") or "input field"
        role = field_info.get("role", "")
        page = context.get("page", {})
        recent_actions = context.get("recent_actions", [])

        lines = [
            f"User Goal: {goal}",
            f"Page: {page.get('title', '')}",
        ]
        if recent_actions:
            acts = [
                f"- {a.get('action')}: {a.get('text')}" if a.get('text') else f"- {a.get('action')}"
                for a in recent_actions[-4:]
            ]
            lines.append("Recent Actions Completed:\n" + "\n".join(acts))
        role_str = f" (role: {role})" if role else ""
        lines.append(f"Active Form Field: {label}{role_str}\n")
        lines.append("Task: Infer the exact value to enter into this form field to advance the user goal.")
        lines.append("Respond strictly in the format:\nVALUE=\"<text>\"")
        prompt = "\n".join(lines)
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
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
            elif hasattr(mx.metal, "clear_cache"):
                mx.metal.clear_cache()
        except Exception:
            pass

        # Strict extraction: preserve unicode, punctuation, apostrophes; never fall back to field label
        match = re.search(r'VALUE\s*[:=]\s*["\']?([^"\n]+)["\']?', raw)
        if match:
            return match.group(1).strip().strip("*")
        match2 = re.search(r'VALUE\s*[:=]\s*(.+)', raw)
        if match2:
            return match2.group(1).strip().strip('"\'*')
        lines = [line.strip().strip('"\'*') for line in raw.split("\n") if line.strip()]
        if lines:
            return lines[0]
        return ""
