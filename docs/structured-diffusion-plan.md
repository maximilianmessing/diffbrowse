# Training-Free Structured Diffusion for DiffBrowse

Status: approved implementation plan. This document is a specification, not evidence that the backend or evaluation is complete.

## Objective

Implement a simple, fully local, training-free DiffusionGemma decision backend on Apple Silicon.

Preserve the existing browser loop:

**Page → indexed elements → operation and operation-specific target heads → validated action → execution → fresh observation.**

Use a fixed answer template with noisy answer slots. Read answer distributions directly from diffusion logits. Do not generate selectors, executable code, or multi-action plans.

All inference and field-text generation stay local. Website traffic is allowed: “air-gapped” means offline AI inference, not an internet-disconnected browser. This supersedes earlier training and free-form action generation proposals.

## Constraints and non-goals

- No training, LoRA, distillation, or custom learned heads.
- No Jev benchmark reproduction or claims of parity.
- No CUDA/ROCm implementation, vLLM migration, Modal deployment, or cloud fallback.
- No general JSON Schema engine, multi-token candidate trie, or new inference server.
- Preserve public `Agent` and `Decision` compatibility and the separate local TYPE_TEXT helper.
- Never retry browser mutations. Log execution before observing its result.
- Verify final outcomes independently; DONE is not proof.
- Tests must not call paid APIs.
- Preserve existing working-tree edits. Do not commit or push unless requested.

Read [README.md](../README.md) and [AGENTS.md](../AGENTS.md) before editing.

## Primary sources and pinned revisions

### vLLM structured reads

Inspected revision: `407326b735c289a5f079a000a1b34c2e4be6b94c`. The PR was open and unmerged when reviewed.

- [PR #57250](https://github.com/vllm-project/vllm/pull/57250)
- [Changes](https://github.com/vllm-project/vllm/pull/57250/files)
- [Structured-read documentation](https://github.com/vllm-project/vllm/blob/407326b735c289a5f079a000a1b34c2e4be6b94c/examples/features/diffusion_reads/README.md)
- [Schema compiler and server](https://github.com/vllm-project/vllm/blob/407326b735c289a5f079a000a1b34c2e4be6b94c/examples/features/diffusion_reads/structured_server.py)
- [Model and sampler](https://github.com/vllm-project/vllm/blob/407326b735c289a5f079a000a1b34c2e4be6b94c/vllm/model_executor/models/diffusion_gemma.py)
- [Sampler tests](https://github.com/vllm-project/vllm/blob/407326b735c289a5f079a000a1b34c2e4be6b94c/tests/v1/sample/test_diffusion_gemma_reads.py)

### Open-jev

Resolved `main` on 2026-09-21: `50d32c7e542dd714c89bb223e8fc8c0b2b3d0d9c`.

- [Repository at pinned revision](https://github.com/JoshuaSP/open-jev/tree/50d32c7e542dd714c89bb223e8fc8c0b2b3d0d9c)
- [Inference harness](https://github.com/JoshuaSP/open-jev/blob/50d32c7e542dd714c89bb223e8fc8c0b2b3d0d9c/inference.py)
- [Final-readout canvases](https://github.com/JoshuaSP/open-jev/blob/50d32c7e542dd714c89bb223e8fc8c0b2b3d0d9c/json_canvas.py)
- [Field compiler](https://github.com/JoshuaSP/open-jev/blob/50d32c7e542dd714c89bb223e8fc8c0b2b3d0d9c/parallel_canvas.py)
- [Categorical constraints](https://github.com/JoshuaSP/open-jev/blob/50d32c7e542dd714c89bb223e8fc8c0b2b3d0d9c/constraints.py)
- [Grouping evaluation](https://github.com/JoshuaSP/open-jev/blob/50d32c7e542dd714c89bb223e8fc8c0b2b3d0d9c/PARALLEL_EVALS.md)
- [License](https://github.com/JoshuaSP/open-jev/blob/50d32c7e542dd714c89bb223e8fc8c0b2b3d0d9c/LICENSE)
- [Third-party attribution](https://github.com/JoshuaSP/open-jev/blob/50d32c7e542dd714c89bb223e8fc8c0b2b3d0d9c/THIRD_PARTY.md)

### Model and runtime

- [Google architecture explanation](https://ai.google.dev/gemma/docs/diffusiongemma/explained)
- [Model card](https://ai.google.dev/gemma/docs/diffusiongemma/model_card)
- [Developer guide](https://developers.googleblog.com/diffusiongemma-the-developer-guide/)
- [MLX model documentation](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/diffusion_gemma/README.md)
- [MLX diffusion generation](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/generate/diffusion.py)
- [MLX model implementation](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/diffusion_gemma/language.py)

Use Context7 for library API questions and inspect installed source before using private APIs. Installed versions at planning: MLX 0.32.2 and mlx-vlm 0.7.1. Record the actual runtime used for measurements.

## Architecture

Combine the vLLM PR's seeded read-only slots and template pinning with open-jev's unrestricted answer refinement and final constrained selection.

Conceptual canvas:

```text
operation: @
click_target: @
type_text_target: @
select_target: @
```

Each answer position has a per-head allowed-label mapping. The prompt explains each mapped operation/target. All heads share an encoded observation and canvas. Consume only the selected operation's target. Target questions mean “which target if this operation is selected?”

Shared attention does not imply question isolation; measure interference explicitly. Restricted probabilities are not calibrated truth probabilities.

## Repository integration

- [Jev backend/question construction](../jev_ultrafast/backends/jev.py)
- [Instructions](../jev_ultrafast/questions.py)
- [Current MLX backend](../jev_ultrafast/backends/mlx_direct.py)
- [Candidates](../jev_ultrafast/action_candidates.py)
- [Decision container](../jev_ultrafast/backends/base.py)
- [Agent](../jev_ultrafast/agent.py)
- [Validation/text helper](../jev_ultrafast/model.py)
- [Registration](../jev_ultrafast/backends/__init__.py)

Add opt-in `mlx_structured`. Retain `mlx_direct`; do not silently change the default. Extract Jev question/request construction into a shared pure function without changing its request semantics.

## Stage 1: Structured layout compiler

Create a pure, independently testable compiler returning complete template token IDs, answer positions, per-head allowed IDs, semantic mappings, fixed mask, and width.

- Derive chat/channel framing from the checkpoint; no foreign hardcoded token IDs.
- Validate every label in the complete template: substituting it changes exactly one token at the same position and preserves template length.
- Require unique IDs per head and case-sensitive labels.
- Resolve single-choice heads deterministically; omit unavailable operation heads.
- Compile heads independently, not Cartesian combinations.
- Cache by complete structural/tokenization inputs.
- Choose the smallest width in 32/64/128 that fits closing tokens too.
- Explicitly reject impossible layouts; never silently truncate candidates.

Construct the final answer object in host code, not by parsing generated JSON.

## Stage 2: One-pass MLX read

- Load the quantized checkpoint from an explicit local directory.
- Encode the complete observation once.
- Seed the template and randomize answer slots with full-vocabulary uniform noise.
- Run one decoder forward pass and gather exact allowed-label logits at each answer slot.
- Compute temperature-1 full-vocabulary normalization, allowed-label mass, conditional label probabilities, full-vocabulary entropy, top-label margin, and unrestricted-argmax validity.
- Transfer only compact head statistics to CPU.
- Skip output encoder commit and unused self-conditioning.
- Select operation, consume its target only, and validate against observed supported actions before returning `Decision`.

## Stage 3: Explicit refinement configurations

| Mode | Independent reads | Passes per read |
| --- | ---: | ---: |
| Initial default | 1 | 1 |
| Refinement comparison | 1 | 2 |
| Noise comparison | 4 | 1 |

For two passes, keep answers unrestricted until final selection, use compatible upstream refinement, restore fixed tokens, and zero self-conditioning contributions at pinned positions while retaining answer-slot contributions. A `None` preparation context can legitimately mean logits-based self-conditioning for quantized embeddings; do not skip that path.

For independent reads, use distinct reproducible seeds, share immutable prompt cache, reset canvas/self-conditioning each draw, average conditional probabilities, and report agreement/variation separately.

No automatic escalation in this delivery.

## Stage 4: Safe reuse and local execution

- Keep model/tokenizer resident; share weights with the local field-text helper.
- Cache only exact invariant token prefixes; reuse only on exact match.
- On cache-clone failure, encode the entire prompt, never a prefix-only context.
- Bound cache lifetime and reset between tasks.
- Avoid unnecessary CPU synchronization and unrelated kernel/quantization work.
- Reject implicit downloads, external inference, and external text-helper fallback.
- Preserve stale-state checks and mutation consumption.
- Reject invalid layouts, non-finite logits, and unsupported targets before browser execution.
- Preserve apostrophes/Unicode in field text; never substitute a label on failure.
- Screenshots remain optional and are not model inputs.

## Interfaces and diagnostics

Preserve `Agent` and `Decision`. Add backend settings for local path, read count, passes, seed, maximum canvas width, and offline policy.

Metadata records representation/backend, canvas width, prompt tokens, actual forward passes, per-head distributions, allowed-label mass, entropy/margin/argmax validity, agreement, cache hits, and tokenization/prefill/decoding/readout/total timing. Document timer boundaries and report model loading separately.

## Validation

### Offline tests

Use fakes/tiny tensors; no downloads or paid APIs. Cover:

- Context-dependent tokenization, case sensitivity, single-choice/absent heads.
- Large candidate sets, explicit capacity failure, and complete scoring beyond top-k.
- Fixed-token preservation, answer noise, and final-only constraints.
- Quantized self-conditioning and pin masking.
- Fixed-seed reproducibility and independent-read reset.
- Cache equivalence and clone-failure fallback.
- Selected-operation-only target consumption.
- Invalid/non-finite output preventing execution.
- Offline loading, remote fallback rejection, and text punctuation/Unicode.

### Bounded local evaluation

Use 100 checked browser states and 10 local scenarios covering forms, autocomplete, selection, waiting, recovery, and completion. Accept multiple valid next actions. Split tuning/held-out by task/template.

Compare the three configurations on identical observations/seeds. Measure action correctness, invalid actions, false completion, independently verified task success, warm p50/p95 complete inference latency, forward passes, peak memory, and sustained pressure.

Add separate-head reads as a diagnostic for interference; do not build automatic question-isolation machinery. Select the fastest configuration with no additional held-out errors or scenario regressions relative to the strongest tested configuration. State sample-size limits; do not claim general reliability or parity.

Aim for sub-second warm decisions and 500 ms p50. These are objectives, not promised acceptance results.

### Required checks

```bash
uv run ruff check .
uv run pytest
node --check jev_ultrafast/static/app.js
node --check jev_ultrafast/snapshot.js
uv build
```

## Deliverables and completion criteria

1. Shared question builder preserving Jev behavior.
2. Pure structured-template compiler.
3. Opt-in MLX backend with three explicit configurations.
4. Offline tests and bounded local evaluation results.
5. Setup/configuration/limitations documentation and source attribution.
6. Final report separating implemented, tested, measured, and unresolved work.

Preserve applicable license notices for adapted code. Record upstream revisions. Completion requires grounded decisions, tested cache/slot correctness, no external AI dependency, required checks passing, and measured local quality/latency. No commit, push, training, or default-backend switch is included.
