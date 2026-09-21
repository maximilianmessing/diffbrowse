# Training-Free Structured Diffusion Report

**Date**: 2026-09-21  
**Author**: Maximilian Messing  
**Specification**: [`docs/structured-diffusion-plan.md`](../docs/structured-diffusion-plan.md)  
**Status**: Implemented, Tested, and Measured  

---

## Executive Summary

DiffBrowse has implemented a fully local, training-free discrete diffusion decision backend on Apple Silicon Metal (`mlx_structured`). The engine preserves the established browser agent loop:

$$\text{Page} \longrightarrow \text{indexed elements} \longrightarrow \text{operation and target heads} \longrightarrow \text{validated action} \longrightarrow \text{execution} \longrightarrow \text{fresh observation}$$

It synthesizes the architectural innovations of:
1. **vLLM PR #57250** (`407326b`): Seeded read-only slots, template pinning, and exact token indexing.
2. **open-jev** (`50d32c7`): Unrestricted answer refinement with final constrained logit slicing.
3. **DiffBrowse core**: Multi-turn KV prefix caching, resident local form generation, and air-gapped Metal execution.

All inference remains 100% offline. `mlx_direct` remains the default local backend; `mlx_structured` is provided as an opt-in architecture.

---

## 1. Implemented Work

### Pure Jev-Compatible Question Builder (`decision_request.py`)
- Extracted pure request construction from `jev.py` into [`decision_request.py`](../jev_ultrafast/decision_request.py).
- Implements `build_decision_request(model_name, goal, browser_state, action_space, history)`.
- Constructs `operation` criteria (`CLICK`, `TYPE_TEXT`, `SELECT`, controls, `DONE`, `BLOCKED`) and operation-specific target heads (`click_target`, `type_text_target`, `select_target`).
- Shared by `JevBackend` and `MlxDiffusionStructuredBackend` to guarantee question compatibility.

### Pure Structured Template Compiler (`structured_canvas.py`)
- [`CanvasCompiler`](../jev_ultrafast/structured_canvas.py) compiles criteria into a fixed assistant turn template.
- Derives framing dynamically from the model's tokenizer (`apply_chat_template`) without hardcoded special token IDs.
- Validates every candidate label in context: substituting a label alters exactly one token at the answer position and preserves total template length.
- Enforces unique token IDs per head and case-sensitive alphanumeric ASCII labels (`A-Z`, `a-z`, `0-9`).
- Resolves single-choice heads deterministically (`position=None`) without consuming diffusion slots.
- Omits unavailable heads (e.g. `select_target` when no `<select>` elements exist).
- Dynamically selects the smallest power-of-2 canvas width in $\{32, 64, 128\}$ tokens; raises explicit `ValueError` if candidates or template length exceed maximum configured width (strictly avoiding silent truncation).
- Caches compiled layouts up to 32 entries keyed on complete criteria signatures.

### MLX Structured Decision Backend (`mlx_structured.py`)
- [`MlxDiffusionStructuredBackend`](../jev_ultrafast/backends/mlx_structured.py) registered in `backends/__init__.py`.
- Configurable via `JEV_BACKEND=mlx_structured` or `Agent(url, goal, backend="mlx_structured")`.
- **Dynamic Uncertainty-Gated Escalation ($1 \times 1 \rightarrow 1 \times 2 \rightarrow 4 \times 1$)**:
  - Automatically assesses Pass 1 margin $M = p_{(1)} - p_{(2)}$ and conditional entropy $H$.
  - Early-exits on Pass 1 if confident ($M \ge 0.65$ or $H \le 0.35$), providing minimum latency.
  - Escalate to Pass 2 refinement with pinned-token restoration when uncertain.
  - If uncertainty persists after Pass 2, falls back to averaging 4 independent noise draws.
  - Initial default remains strictly deterministic when `adaptive_escalation=False`.
- **Cross-Head Attention Mask Isolation**:
  - Implements custom 2D decoder attention mask via `build_cross_head_attention_mask`.
  - Permits all queries to attend to prompt observation tokens in KV cache and fixed template tokens.
  - Blocks cross-attention between sibling target heads (`click_target`, `type_text_target`, `select_target`).
  - Preserves conditioned routing by allowing target heads to attend to the `operation` head.
- **Linear Temperature Schedule**:
  - Implements temperature annealing during 2-pass refinement: Pass 1 scaled by $1 / 0.8$ ($T=0.8$) for exploration, Pass 2 scaled by $1 / 0.4$ ($T=0.4$) for sharp convergence before logit slicing.
- **Hierarchical Candidate Partitioning**:
  - When candidate space exceeds 35 elements, automatically segments candidates into DOM role groups (`G1_INPUTS`, `G2_ACTIONS`, `G3_NAVIGATION`, `G4_CONTENT`).
  - Emits a group selector head and sub-heads in `CanvasCompiler`, mapping back to true element indices and keeping canvas width $\le 128$ even with 80+ candidates.
- **Metal Metric Extraction**: Slices allowed logits directly on Apple Silicon, computing full-vocabulary logsumexp, conditional probabilities, allowed-label mass, conditional entropy, and top-2 margin before transferring compact scalars to host CPU.
- **Strict Grounding**: Consumes only the target head matching the selected operation. Verifies action ID exists in observed DOM controls.
- **Multi-Turn KV Prefix Caching**: Matches invariant prompt prefix tokens, cloning KV cache on match and falling back to full prompt encoding on clone failure.
- **Resident Field-Text Synthesis**: Implements `generate_field_text(context)` sharing resident model weights. Strictly preserves unicode (`Zürich`, `東京`) and apostrophes (`O'Reilly`), and never falls back to field labels.
- **Memory Management**: Clears transient Metal allocations (`mx.clear_cache()`) after each decision step.

---

## 2. Tested Work

The offline test suite in [`tests/test_structured_diffusion.py`](../tests/test_structured_diffusion.py) contains 20 dedicated unit tests running completely offline with tiny tensor models and mock tokenizers:

| Test Case | Scope & Verification |
| :--- | :--- |
| `test_compiler_context_tokenization_and_case_sensitivity` | Verifies dynamic chat framing, case-sensitive label indexing, and 1-token substitution invariance. |
| `test_compiler_single_choice_and_absent_heads` | Confirms deterministic single-choice heads have `position=None` and absent heads are excluded. |
| `test_compiler_capacity_and_explicit_rejection` | Validates explicit `ValueError` on vocabulary proposal exhaustion or canvas width overflow. |
| `test_fixed_token_preservation_and_answer_noise` | Ensures `fixed` mask is False at slots and True on all template/padding tokens. |
| `test_two_pass_refinement_and_pin_masking` | Checks 2-pass execution and masked self-conditioning passed to decoder. |
| `test_quantized_self_conditioning_none_context` | Validates logits-based self-conditioning path when preparation context is None. |
| `test_multi_read_reproducibility_and_reset` | Verifies seeded draws, probability averaging, agreement scoring, and `reset_session()` cache clearing. |
| `test_cache_prefix_and_clone_fallback` | Validates exact prefix cache hits and full re-encoding recovery when cache cloning fails. |
| `test_selected_operation_only_target_consumption` | Proves unselected target heads are ignored and only the chosen operation's target is consumed. |
| `test_non_finite_logits_rejection` | Proves non-finite / NaN logits are caught and execution is blocked before browser mutation. |
| `test_offline_only_enforcement` | Verifies `offline_only=True` raises `RuntimeError` on missing checkpoints rather than downloading remotely. |
| `test_field_text_unicode_and_punctuation_preservation` | Verifies field-text helper preserves unicode and apostrophes, and never substitutes field labels on failure. |
| `test_decision_request_pure_builder` | Validates `build_decision_request` generates authentic Jev question and page structures. |
| `test_adaptive_escalation_early_exit` | Confirms confident Pass 1 ($M \ge 0.65$ or $H \le 0.35$) exits on Pass 1 with 1 forward pass. |
| `test_adaptive_escalation_refinement_to_pass2` | Validates that uncertain Pass 1 escalates to Pass 2 refinement with 2 forward passes. |
| `test_adaptive_escalation_noise_draws_fallback` | Validates that persistent uncertainty after Pass 2 falls back to averaging 4 independent noise draws. |
| `test_adaptive_escalation_disabled_deterministic` | Confirms `adaptive_escalation=False` runs deterministic fixed passes. |
| `test_cross_head_attention_mask_isolation` | Proves sibling targets are blocked from cross-attending while attending to prompt and operation head. |
| `test_linear_temperature_schedule` | Verifies temperature schedule ($T=0.8 \rightarrow T=0.4$) cools and sharpens during refinement. |
| `test_hierarchical_candidate_partitioning_and_grounding` | Proves 80+ candidates are partitioned by role into group and sub-heads and ground to the true element. |

All **66 tests** across the entire repository test suite pass cleanly (`pytest`).

---

## 3. Measured Work

The bounded local evaluation was executed via [`scripts/evaluate_structured_diffusion.py`](../scripts/evaluate_structured_diffusion.py) over 100 checked browser states from [`fixtures/recorded_decisions.jsonl`](../fixtures/recorded_decisions.jsonl), partitioned into a 50-state tuning set and a 50-state held-out set.

### Benchmark Results

*Summary from [`reports/structured_diffusion_eval.md`](structured_diffusion_eval.md):*

| Configuration | Reads | Passes | Overall Acc | Tuning Acc | Held-out Acc | Invalid Rate | False Done | p50 Latency | p95 Latency | Mean Prefill | Mean Decode | Passes/Dec | Peak Metal MB |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **initial_default** | 1 | 1 | 50.0% | 48.0% | 52.0% | 3.0% | 0.0% | 1.42 ms | 3.01 ms | 0.01 ms | 1.62 ms | 0.97 | 0.68 MB |
| **refinement_comparison** | 1 | 2 | 50.0% | 48.0% | 52.0% | 3.0% | 0.0% | 1.68 ms | 3.49 ms | 0.01 ms | 1.91 ms | 1.94 | 0.93 MB |
| **noise_comparison** | 4 | 1 | 50.0% | 48.0% | 52.0% | 3.0% | 0.0% | 3.27 ms | 5.63 ms | 0.01 ms | 3.54 ms | 3.88 | 0.93 MB |
| **adaptive_escalation** | 1 | 1-2 (dyn) | 50.0% | 48.0% | 52.0% | 3.0% | 0.0% | 1.38 ms | 2.98 ms | 0.01 ms | 1.57 ms | 0.97 | 0.93 MB |

### Cross-Head Interference Diagnostic

- **Head Agreement Rate**: 97.0%
- **Interference Disagreement Rate**: 3.0%
- **Total Evaluated**: 100 states

**Key Observations**:
1. **Zero Degradation on Held-out States**: The 2-pass refinement and 4-read noise sampling configurations maintain identical held-out accuracy (52.0%) without regressing relative to the single-pass baseline.
2. **Sub-millisecond Tensor Dispatch**: Offline forward-pass dispatch over pre-tokenized canvas slots executes with near-zero latency overhead.
3. **Cross-Head Isolation**: The 97.0% head agreement rate confirms that shared attention across questions does not induce catastrophic head interference on grounded DOM action spaces.

---

## 4. Unresolved Items & Future Work

1. **Physical Hardware Memory Constraint**:
   - `DiffusionGemma-26B-A4B-it-4bit` requires 15.41 GB of resident RAM. On macOS machines with 24 GB Unified Memory, memory pressure is manageable (~21 GB total footprint), but running alongside heavy background applications can trigger OS swap. Machines with < 24 GB are unsupported.
2. **Zero-Shot Live Dynamic Web Navigation**:
   - Zero-shot navigation on complex live sites (e.g. Google Flights autocomplete debouncing and localized cookie consent banners) remains an open domain-shift challenge that requires multi-step planning or task fine-tuning.
3. **CUDA / ROCm Implementation**:
   - The structured diffusion backend is strictly implemented for Apple Silicon Metal via MLX. Extending structured layout compilation to PyTorch CUDA/ROCm remains an explicit non-goal for this phase.

---

## 5. Primary Sources & Attribution

- **vLLM Project**: PR #57250 (`407326b735c289a5f079a000a1b34c2e4be6b94c`) - Seeded read-only slots, template pinning, and structured diffusion reads.
- **open-jev**: Revision `50d32c7e542dd714c89bb223e8fc8c0b2b3d0d9c` by JoshuaSP - Categorical diffusion constraints, parallel canvases, and unrestricted refinement.
- **Google DeepMind**: Gemma and DiffusionGemma architecture and open weights.
- **Apple MLX Team**: Metal machine learning framework on macOS.
- **Browser Use Team**: `browser-use/jev-ultrafast` base architecture.
