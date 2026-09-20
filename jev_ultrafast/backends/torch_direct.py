"""PyTorch backend for NVIDIA DGX (CUDA) and AMD Strix Halo (ROCm).

Supports:
- NVIDIA DGX Spark / Hopper / Blackwell (CUDA with FlashAttention-2)
- AMD Strix Halo / Ryzen AI Max 395 (ROCm 6.2+ / HIP on unified LPDDR5X)
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional

from .base import Decision, calculate_entropy, calculate_top2_margin


class TorchDiffusionDirectBackend:
    """Diffusion decision backend for NVIDIA CUDA and AMD ROCm systems."""

    name: str = "torch_direct"

    def __init__(
        self,
        model_name: str = "google/diffusiongemma-26b-it",
        device: Optional[str] = None,
        torch_dtype: str = "bfloat16",
        canvas_length: int = 32,
        num_passes: int = 2,
        load_in_4bit: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        self.canvas_length = canvas_length
        self.num_passes = num_passes
        self.load_in_4bit = load_in_4bit

        self._model = None
        self._tokenizer = None
        self._device_type = "unknown"

    def _resolve_device(self) -> str:
        if self.device:
            return self.device
        try:
            import torch

            if torch.cuda.is_available():
                # Detect whether running under AMD ROCm (HIP) or NVIDIA CUDA
                is_rocm = getattr(torch.version, "hip", None) is not None
                self._device_type = "rocm (AMD Strix Halo)" if is_rocm else "cuda (NVIDIA DGX)"
                return "cuda"
            return "cpu"
        except ImportError:
            return "cpu"

    def _ensure_loaded(self):
        if self._model is not None and self._tokenizer is not None:
            return

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as err:
            raise ImportError(
                "TorchDiffusionDirectBackend requires torch and transformers. "
                "For NVIDIA DGX: install torch with CUDA.\n"
                "For AMD Strix Halo: install torch with ROCm (e.g. pip install torch --index-url https://download.pytorch.org/whl/rocm6.2)"
            ) from err

        target_device = self._resolve_device()
        dtype = getattr(torch, self.torch_dtype, torch.bfloat16)

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        load_kwargs: Dict[str, Any] = {"torch_dtype": dtype}
        if self.load_in_4bit and target_device != "cpu":
            try:
                from transformers import BitsAndBytesConfig

                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_quant_type="nf4",
                )
            except ImportError:
                pass  # Fall back to native dtype

        if target_device != "cpu":
            load_kwargs["device_map"] = "auto"

        self._model = AutoModel.from_pretrained(self.model_name, **load_kwargs)
        self._model.eval()

    def choose(
        self,
        goal: str,
        browser_state: Dict[str, Any],
        action_space: tuple,
        history: List[Dict[str, Any]],
    ) -> Decision:
        import torch

        from ..action_candidates import LabelPool, flatten_actions

        self._ensure_loaded()
        started = time.perf_counter()

        candidates = flatten_actions(browser_state["actions"])
        pool = LabelPool(self._tokenizer)
        cand_token_ids, _, cand_to_label, candidate_lines = pool.map_candidates(
            candidates, strategy="index"
        )

        recent_hist = "\n".join(
            f"- Step {h.get('step', i+1)}: {h.get('action')}"
            for i, h in enumerate(history[-5:])
        ) or "None"

        prompt = (
            f"Goal: {goal}\n"
            f"Page Title: {browser_state.get('title', '')}\n"
            f"Page URL: {browser_state.get('url', '')}\n"
            f"Visible Content:\n{browser_state.get('text', '')[:2000]}\n\n"
            f"Recent History:\n{recent_hist}\n\n"
            f"{candidate_lines}\n\n"
            "Select the single best next action.\n"
            "ACTION="
        )

        tokenize_tic = time.perf_counter()
        inputs = self._tokenizer(prompt, return_tensors="pt")
        target_device = next(self._model.parameters()).device
        input_ids = inputs["input_ids"].to(target_device)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(target_device)
        tokenize_ms = (time.perf_counter() - tokenize_tic) * 1000

        prefill_tic = time.perf_counter()
        with torch.no_grad():
            outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :]  # Take last token logits
        prefill_ms = (time.perf_counter() - prefill_tic) * 1000

        cand_indices = torch.tensor(cand_token_ids, dtype=torch.long, device=target_device)
        cand_logits = logits[0, cand_indices]
        probs = torch.softmax(cand_logits, dim=-1).cpu().tolist()

        pass_probs = {c.action_id: float(p) for c, p in zip(candidates, probs)}
        entropy = calculate_entropy(pass_probs)
        margin = calculate_top2_margin(pass_probs)

        best_idx = int(torch.argmax(cand_logits).item())
        selected_cand = candidates[best_idx]
        top_prob = float(probs[best_idx])
        total_ms = (time.perf_counter() - started) * 1000

        return Decision(
            action_id=selected_cand.action_id,
            operation=selected_cand.operation,
            target=selected_cand.target,
            confidence=top_prob,
            probabilities=pass_probs,
            raw_top_probability=top_prob,
            entropy=entropy,
            top2_margin=margin,
            stable_steps=1,
            tokenize_ms=tokenize_ms,
            prefill_ms=prefill_ms,
            decoder_ms=0.0,
            total_model_ms=prefill_ms,
            total_decision_ms=total_ms,
            model=self.model_name,
            backend=f"{self.name}_{self._device_type.split()[0]}",
            metadata={
                "device_type": self._device_type,
                "hardware": "NVIDIA DGX" if "cuda" in self._device_type else "AMD Strix Halo",
                "candidate_count": len(candidates),
                "prompt_tokens": input_ids.shape[-1],
            },
        )

    def generate_field_text(self, context: Dict[str, Any], max_tokens: int = 32) -> str:
        """Generate field text on NVIDIA CUDA or AMD ROCm."""
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
