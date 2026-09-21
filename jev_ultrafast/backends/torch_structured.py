"""Training-Free Structured Diffusion decision backend on PyTorch (CUDA and ROCm).

Supports:
- NVIDIA DGX / Hopper / Blackwell / RTX (CUDA 12.4+ with FlashAttention-2)
- AMD Strix Halo / Ryzen AI Max 395 (ROCm 6.2+ / HIP on unified LPDDR5X)

Implements pinned answer layouts, 2D cross-head attention mask isolation,
append-only invariant prefix KV-caching, and adaptive escalation.
"""

from __future__ import annotations

import re
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

from ..decision_request import build_decision_request
from ..structured_canvas import (
    CanvasCompiler,
    CanvasLayout,
    build_cross_head_attention_mask,
)
from .base import Decision


def _clone_past_key_values(past_key_values: Any) -> Any:
    """Clone past_key_values across turns without in-place mutation side-effects."""
    if past_key_values is None:
        return None
    if hasattr(past_key_values, "copy"):
        return past_key_values.copy()
    if isinstance(past_key_values, (tuple, list)):
        return tuple(
            tuple(t.clone() if hasattr(t, "clone") else t for t in layer_kv)
            if isinstance(layer_kv, (tuple, list))
            else layer_kv.clone()
            if hasattr(layer_kv, "clone")
            else layer_kv
            for layer_kv in past_key_values
        )
    return past_key_values


class TorchDiffusionStructuredBackend:
    """Structured Discrete Diffusion Decision Backend on NVIDIA CUDA and AMD ROCm."""

    name: str = "torch_structured"

    def __init__(
        self,
        model: Any = None,
        tokenizer: Any = None,
        model_name: str = "google/diffusiongemma-26B-A4B-it",
        device: Optional[str] = None,
        torch_dtype: str = "bfloat16",
        load_in_4bit: bool = True,
        mode: str = "initial_default",
        num_reads: Optional[int] = None,
        num_passes: Optional[int] = None,
        seed: Optional[int] = None,
        max_canvas_width: int = 256,
        enable_prefix_caching: bool = True,
        adaptive_escalation: bool = True,
        pass1_margin_threshold: float = 0.65,
        pass1_entropy_threshold: float = 0.35,
        use_attention_isolation: bool = True,
        use_temperature_schedule: bool = True,
        hierarchical_partitioning: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        self.load_in_4bit = load_in_4bit
        self.mode = mode
        self.seed = seed
        self.max_canvas_width = max_canvas_width
        self.enable_prefix_caching = enable_prefix_caching
        self.pass1_margin_threshold = pass1_margin_threshold
        self.pass1_entropy_threshold = pass1_entropy_threshold
        self.use_attention_isolation = use_attention_isolation
        self.use_temperature_schedule = use_temperature_schedule
        self.hierarchical_partitioning = hierarchical_partitioning

        # Resolve mode-specific defaults
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
        self._tokenizer = tokenizer
        self._compiler: Optional[CanvasCompiler] = None
        self._device_type = "unknown"
        self._target_device: Optional[str] = None
        self._cached_goal: Optional[str] = None
        self._cached_prefix_tokens: Optional[List[int]] = None
        self._cached_prefix_cache: Optional[Any] = None
        self._model_load_ms: Optional[float] = None

    def reset_session(self) -> None:
        """Clear cached session KV prefix state between distinct tasks."""
        self._cached_goal = None
        self._cached_prefix_tokens = None
        self._cached_prefix_cache = None

    def _resolve_device(self) -> str:
        if self.device:
            return self.device
        try:
            import torch

            if torch.cuda.is_available():
                is_rocm = getattr(torch.version, "hip", None) is not None
                self._device_type = "rocm (AMD Strix Halo)" if is_rocm else "cuda (NVIDIA DGX)"
                return "cuda"
            return "cpu"
        except ImportError:
            return "cpu"

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            if self._compiler is None:
                self._compiler = CanvasCompiler(self._tokenizer, max_width=self.max_canvas_width)
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as err:
            raise ImportError(
                "TorchDiffusionStructuredBackend requires torch and transformers.\n"
                "- NVIDIA DGX/RTX: uv sync --extra cuda\n"
                "- AMD Strix Halo: pip install torch --index-url https://download.pytorch.org/whl/rocm6.2 "
                "&& uv sync --extra rocm"
            ) from err

        target_dev = self._resolve_device()
        self._target_device = target_dev
        dtype = getattr(torch, self.torch_dtype, torch.bfloat16)

        # Check Strix Halo / NVIDIA hardware prerequisites
        if target_dev != "cpu" and torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if vram_gb < 23.5 and not self.load_in_4bit:
                warnings.warn(
                    f"VRAM / Unified Memory: {vram_gb:.1f} GB. Running unquantized 26B model requires ≥ 48 GB.\n"
                    "Enabling 4-bit quantization automatically.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self.load_in_4bit = True

        load_tic = time.perf_counter()
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        load_kwargs: Dict[str, Any] = {"torch_dtype": dtype}
        if self.load_in_4bit and target_dev != "cpu":
            try:
                from transformers import BitsAndBytesConfig

                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_quant_type="nf4",
                )
            except ImportError:
                warnings.warn(
                    "BitsAndBytesConfig unavailable; falling back to native dtype.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        if target_dev != "cpu":
            load_kwargs["device_map"] = "auto"

        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs)
        self._model.eval()
        self._model_load_ms = (time.perf_counter() - load_tic) * 1000

        if self._compiler is None:
            self._compiler = CanvasCompiler(self._tokenizer, max_width=self.max_canvas_width)

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
        try:
            t1 = self._tokenizer.apply_chat_template(
                [{"role": "user", "content": preamble + "A\n"}],
                tokenize=False,
                add_generation_prompt=True,
            )
            t2 = self._tokenizer.apply_chat_template(
                [{"role": "user", "content": preamble + "B\n"}],
                tokenize=False,
                add_generation_prompt=True,
            )
            ids1 = self._tokenizer.encode(t1, add_special_tokens=False)
            ids2 = self._tokenizer.encode(t2, add_special_tokens=False)
            common_len = 0
            for a, b in zip(ids1, ids2):
                if a == b:
                    common_len += 1
                else:
                    break
            return ids1[:common_len]
        except Exception:
            return []

    def choose(
        self,
        goal: str,
        browser_state: Dict[str, Any],
        action_space: Tuple[Any, ...],
        history: List[Dict[str, Any]],
    ) -> Decision:
        import torch

        self._ensure_loaded()
        total_tic = time.perf_counter()

        elements, targets, controls = action_space
        body = build_decision_request(self.model_name, goal, browser_state, action_space, history)
        questions = body["questions"]

        # 1. Compile structured canvas layout
        layout = self._compiler.compile(questions, hierarchical=self.hierarchical_partitioning)
        active_questions = layout.questions if layout.questions is not None else questions

        # 2. Format observation prompt
        user_content = self._format_prompt(goal, browser_state, active_questions, layout, history)
        try:
            formatted_prompt = self._tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            formatted_prompt = user_content

        target_device = next(self._model.parameters()).device
        tokenize_tic = time.perf_counter()
        inputs = self._tokenizer(formatted_prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(target_device)
        input_ids_list = input_ids[0].tolist()
        tokenize_ms = (time.perf_counter() - tokenize_tic) * 1000

        # 3. Invariant Prefix Caching in PyTorch
        prefill_tic = time.perf_counter()
        cache_hit = False
        kv_cache = None

        if (
            self.enable_prefix_caching
            and self._cached_goal == goal
            and self._cached_prefix_tokens is not None
            and self._cached_prefix_cache is not None
        ):
            p_len = len(self._cached_prefix_tokens)
            if len(input_ids_list) >= p_len and input_ids_list[:p_len] == self._cached_prefix_tokens:
                cloned = _clone_past_key_values(self._cached_prefix_cache)
                suffix_ids = input_ids[:, p_len:]
                if suffix_ids.shape[1] > 0:
                    with torch.no_grad():
                        out = self._model(input_ids=suffix_ids, past_key_values=cloned, use_cache=True)
                        kv_cache = out.past_key_values
                else:
                    kv_cache = cloned
                cache_hit = True

        if kv_cache is None:
            if self.enable_prefix_caching and len(input_ids_list) >= 16:
                p_toks = self._compute_invariant_prefix_tokens(goal)
                if p_toks and len(input_ids_list) >= len(p_toks) and input_ids_list[:len(p_toks)] == p_toks:
                    p_len = len(p_toks)
                    prefix_ids = input_ids[:, :p_len]
                    with torch.no_grad():
                        out = self._model(input_ids=prefix_ids, use_cache=True)
                        cached_c = out.past_key_values
                    self._cached_goal = goal
                    self._cached_prefix_tokens = p_toks
                    self._cached_prefix_cache = cached_c

                    cloned = _clone_past_key_values(cached_c)
                    suffix_ids = input_ids[:, p_len:]
                    if suffix_ids.shape[1] > 0:
                        with torch.no_grad():
                            out = self._model(input_ids=suffix_ids, past_key_values=cloned, use_cache=True)
                            kv_cache = out.past_key_values
                    else:
                        kv_cache = cloned
                else:
                    with torch.no_grad():
                        out = self._model(input_ids=input_ids, use_cache=True)
                        kv_cache = out.past_key_values
                    self._cached_goal = goal
                    self._cached_prefix_tokens = input_ids_list
                    self._cached_prefix_cache = kv_cache
            else:
                with torch.no_grad():
                    out = self._model(input_ids=input_ids, use_cache=True)
                    kv_cache = out.past_key_values

        prefill_ms = (time.perf_counter() - prefill_tic) * 1000

        # 4. Canvas Evaluation & Attention Masking
        decoder_tic = time.perf_counter()
        template_tensor = torch.tensor([layout.tokens], dtype=torch.long, device=target_device)
        enc_len = input_ids.shape[1]

        # 4D attention mask: (1, 1, width, total_len)
        additive_mask = None
        if self.use_attention_isolation:
            b_mask = build_cross_head_attention_mask(layout, enc_len, framework="torch").to(target_device)
            min_val = torch.finfo(next(self._model.parameters()).dtype).min
            additive_mask = torch.where(b_mask, 0.0, min_val)

        def _extract_slot_metrics(target_logits: torch.Tensor, temp_scale: Optional[float] = None) -> Tuple[
            Dict[str, Dict[str, float]],
            Dict[str, str],
            Dict[str, float],
            Dict[str, float],
            Dict[str, float],
            Dict[str, bool],
        ]:
            eval_logits = target_logits / temp_scale if temp_scale and temp_scale > 0 else target_logits

            head_probs: Dict[str, Dict[str, float]] = {}
            head_tops: Dict[str, str] = {}
            head_mass: Dict[str, float] = {}
            head_ent: Dict[str, float] = {}
            head_mar: Dict[str, float] = {}
            head_val: Dict[str, bool] = {}

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
                tok_ids = torch.tensor(s.token_ids, dtype=torch.long, device=target_device)
                allowed_logits = pos_logits[tok_ids]
                cond_probs = torch.softmax(allowed_logits.float(), dim=-1)
                full_lse = torch.logsumexp(pos_logits.float(), dim=-1)
                allowed_mass = torch.sum(torch.exp(allowed_logits.float() - full_lse)).item()
                cond_entropy = -torch.sum(cond_probs * torch.log(cond_probs.clamp(min=1e-12))).item()

                if len(s.token_ids) > 1:
                    sorted_probs, _ = torch.sort(cond_probs)
                    margin = (sorted_probs[-1] - sorted_probs[-2]).item()
                else:
                    margin = 1.0

                unrestricted_argmax = int(torch.argmax(pos_logits).item())
                argmax_valid = unrestricted_argmax in s.token_ids

                prob_dict = {choice: float(p) for choice, p in zip(s.choices, cond_probs.tolist())}
                top_choice = max(prob_dict.items(), key=lambda x: x[1])[0]
                head_probs[s.name] = prob_dict
                head_tops[s.name] = top_choice
                head_mass[s.name] = float(allowed_mass)
                head_ent[s.name] = float(cond_entropy)
                head_mar[s.name] = float(margin)
                head_val[s.name] = bool(argmax_valid)

            return head_probs, head_tops, head_mass, head_ent, head_mar, head_val

        # Pass 1 forward
        cloned_kv = _clone_past_key_values(kv_cache)
        with torch.no_grad():
            out = self._model(
                input_ids=template_tensor,
                past_key_values=cloned_kv,
                attention_mask=additive_mask,
                use_cache=False,
            )
            p1_logits = out.logits

        p1_probs, p1_tops, p1_mass, p1_ent, p1_mar, p1_val = _extract_slot_metrics(p1_logits)
        actual_forward_passes = 1
        escalated = False

        final_probs = p1_probs
        final_tops = p1_tops
        final_mass = p1_mass
        final_ent = p1_ent
        final_mar = p1_mar
        final_val = p1_val

        # Non-targeted short-circuiting: if DONE is chosen, exit immediately
        op_choice = p1_tops.get("operation")
        if op_choice == "DONE":
            pass  # Terminate early without running Pass 2
        elif self.adaptive_escalation:
            # Check confidence metrics
            p1_op_mar = p1_mar.get("operation", 1.0)
            p1_op_ent = p1_ent.get("operation", 0.0)
            ambiguous = (p1_op_mar < self.pass1_margin_threshold) or (p1_op_ent > self.pass1_entropy_threshold)

            if ambiguous or self.num_passes > 1:
                # Pass 2 refinement with temperature schedule
                escalated = True
                actual_forward_passes += 1
                t2_tensor = template_tensor.clone()
                # Pin high-confidence Pass 1 choices
                for s in layout.slots:
                    if s.position is not None and p1_mar.get(s.name, 0.0) >= 0.85:
                        top_label = s.choice_to_label.get(p1_tops[s.name])
                        if top_label:
                            label_idx = s.labels.index(top_label)
                            t2_tensor[0, s.position] = s.token_ids[label_idx]

                temp = 0.7 if self.use_temperature_schedule else 1.0
                cloned_kv2 = _clone_past_key_values(kv_cache)
                with torch.no_grad():
                    out2 = self._model(
                        input_ids=t2_tensor,
                        past_key_values=cloned_kv2,
                        attention_mask=additive_mask,
                        use_cache=False,
                    )
                    p2_logits = out2.logits

                final_probs, final_tops, final_mass, final_ent, final_mar, final_val = _extract_slot_metrics(
                    p2_logits, temp_scale=temp
                )

        decoder_ms = (time.perf_counter() - decoder_tic) * 1000

        # 5. Map selected choices to decision
        selected_operation = final_tops.get("operation", "CLICK")
        target_head_map = {
            "CLICK": "click_target",
            "TYPE": "type_target",
            "TYPE_TEXT": "type_target",
            "SELECT": "select_target",
            "SCROLL": "scroll_target",
            "WAIT": "wait_target",
        }
        target_head_name = target_head_map.get(selected_operation)
        selected_target = final_tops.get(target_head_name) if target_head_name else None

        # Reconstruct action ID
        if selected_operation == "DONE":
            action_id = "DONE"
            top_prob = final_probs.get("operation", {}).get("DONE", 0.99)
            margin = final_mar.get("operation", 1.0)
            entropy = final_ent.get("operation", 0.0)
        elif selected_operation == "SCROLL":
            action_id = "scroll_down" if selected_target == "down" else "scroll_up"
            top_prob = final_probs.get("operation", {}).get("SCROLL", 0.95)
            margin = final_mar.get("operation", 1.0)
            entropy = final_ent.get("operation", 0.0)
        else:
            action_id = selected_target or "unknown"
            target_probs = final_probs.get(target_head_name, {}) if target_head_name else {}
            top_prob = target_probs.get(selected_target, 0.90)
            margin = final_mar.get(target_head_name, 1.0) if target_head_name else 1.0
            entropy = final_ent.get(target_head_name, 0.0) if target_head_name else 0.0

        total_decision_ms = (time.perf_counter() - total_tic) * 1000

        return Decision(
            action_id=action_id,
            operation=selected_operation,
            target=selected_target,
            confidence=float(top_prob),
            probabilities=final_probs.get(target_head_name, final_probs.get("operation", {})),
            raw_top_probability=float(top_prob),
            entropy=float(entropy),
            top2_margin=float(margin),
            stable_steps=actual_forward_passes,
            tokenize_ms=tokenize_ms,
            prefill_ms=prefill_ms,
            decoder_ms=decoder_ms,
            total_model_ms=prefill_ms + decoder_ms,
            total_decision_ms=total_decision_ms,
            model=self.model_name,
            backend=f"{self.name}_{self._device_type.split()[0]}",
            metadata={
                "device_type": self._device_type,
                "hardware": "NVIDIA DGX" if "cuda" in self._device_type else "AMD Strix Halo",
                "cache_hit": cache_hit,
                "escalated": escalated,
                "forward_passes": actual_forward_passes,
                "prompt_tokens": input_ids.shape[-1],
                "canvas_width": layout.width,
            },
        )

    def generate_field_text(self, context: Dict[str, Any], max_tokens: int = 16) -> str:
        """Locally synthesize field string with bounded canvas."""
        import torch

        self._ensure_loaded()
        goal = context.get("goal", "")
        field_info = context.get("field", {})
        label = field_info.get("label") or field_info.get("role") or "input field"
        page = context.get("page", {})

        prompt = (
            f"User Goal: {goal}\n"
            f"Page: {page.get('title', '')}\n"
            f"Active Field: {label}\n\n"
            f"Task: Infer the exact string to type into this field.\n"
            f"VALUE=\""
        )
        target_device = next(self._model.parameters()).device
        inputs = self._tokenizer(prompt, return_tensors="pt").to(target_device)
        with torch.no_grad():
            output_tokens = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        raw = self._tokenizer.decode(output_tokens[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        match = re.search(r'VALUE\s*[:=]\s*["\']?([^"\'\n]+)["\']?', raw)
        if match:
            return match.group(1).strip().strip('*')
        cleaned = re.sub(r'VALUE\s*[:=]\s*', '', raw).strip().strip('"\'*')
        return cleaned.split("\n")[0].strip() or label
