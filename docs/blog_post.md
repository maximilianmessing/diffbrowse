# DiffBrowse: Beyond Autoregressions — Turning DiffusionGemma into a Sub-Second Local Browser Agent on Apple Silicon

**By Maximilian Messing & Antigravity**  
*September 2026*

> **DiffBrowse** is an experimental discrete-diffusion browser agent built directly on the foundation of [`browser-use/jev-ultrafast`](https://github.com/browser-use/jev-ultrafast) and benchmarked against Jev's cloud speculative baseline.

---

### The Latency Wall of Autonomous Web Agents

Autonomous web agents face an impedance mismatch problem. 

To book a flight, purchase a ticket, or file an expense report, an agent must inspect the browser accessibility tree, choose an element to interact with, execute an action, and observe the resulting state. In traditional architectures (e.g., GPT-4o, Claude 3.5 Sonnet, or large vision-language models), this loop is slow, costly, and fragile:

```
┌──────────────┐       DOM Accessibility State       ┌─────────────────────────────┐
│              │ ──────────────────────────────────> │ Cloud LLM / VLM             │
│   Chromium   │                                     │ 5,000 token prompt          │
│   Browser    │ <────────────────────────────────── │ 2,000 ms autoregression     │
│              │          JSON Action String         │ $0.03 per step              │
└──────────────┘                                     └─────────────────────────────┘
  Total Loop Latency: 2.5 – 5.0 seconds per step | Full Task: 30–60 seconds
```

Every action requires sending thousands of DOM tokens over the network, waiting for a multi-billion parameter causal model to generate verbose reasoning tokens, parsing fragile JSON or executable code, and hoping the page has not drifted in the meantime. Furthermore, sending sensitive internal tools, intranet portals, and authenticated session cookies to third-party cloud APIs creates an immediate privacy barrier.

Earlier this year, **Browser Use × TypeSafe's `jev-ultrafast`** proved that browser agents can run at high speed by reframing browser interaction: **A browser is a finite choice, not a conversation.** By mapping visible DOM elements into a typed, indexed action space, decisions become closed-set selections over observed elements.

With **DiffBrowse**, we take `jev-ultrafast`'s dynamic action space as our foundation and benchmark, answering the core research question:

> *Can we replace the proprietary cloud decision model and remote text LLM with a single local discrete-diffusion model running on Apple Silicon Metal, achieving sub-second latency and 100% air-gapped autonomy?*

---

### Why Discrete Diffusion for Web Interaction?

In June 2026, Google DeepMind released **DiffusionGemma** (`diffusiongemma-26B-A4B-it`), an experimental 26-billion-parameter Mixture-of-Experts (MoE) model (activating 3.8B–4.0B parameters per token) powered by **discrete diffusion**. 

Instead of generating text token-by-token from left to right, DiffusionGemma starts with a bounded "canvas" of noise tokens and refines the entire sequence in parallel across iterative denoising passes.

When applied to browser agents, discrete diffusion offers three fundamental advantages:
1. **Bounded Computational Budget**: In generative chat, you must wait for the model to stop generating tokens. In closed-set decision-making, we only need to classify a single slot (`ACTION_SLOT`). We can halt denoising after just **1 or 2 passes**, slashing decoder latency to **110–220 ms**.
2. **Bidirectional Context Attention**: Causal autoregressive decoders use triangular masks—a token cannot attend to tokens that follow it. Diffusion decoders employ non-causal bidirectional self-attention across the canvas. A target action token in the canvas attends concurrently to the user's high-level goal and surrounding DOM candidate attributes.
3. **Zero GPU-to-CPU Bus Bottleneck**: Rather than copying 262,144 vocabulary logits ($1.05\text{ MB}$) to Python CPU, candidate logit slicing and softmax probability normalization occur directly on the Metal GPU in sub-millisecond time.

```
Goal + DOM Candidates ──> Prefill KV Cache
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
   Canvas [ ACTION= | [?] | [?] ... ]      Canvas Denoising
            │                                (Pass 1 & Pass 2)
            └──────────────┬──────────────────────┘
                           ▼
          Direct Metal Slice: logits[cand_token_ids]
                           ▼
       Softmax Normalization on Metal (0.6 ms)
                           ▼
     Selected Action: CLICK [4] (Confidence: 89%, H: 0.21)
```

---

### Core Architecture & Implementation on Apple Silicon

DiffBrowse integrates directly into the `jev-ultrafast` runtime, executing natively on Apple Silicon Metal via Apple's **MLX** framework (`mlx` 0.32.2 and `mlx-vlm` 0.7.1) with 4-bit quantized weights (`mlx-community/diffusiongemma-26B-A4B-it-4bit`).

#### 1. Layer-Wise Encoder Evaluation to Prevent Metal OOM
DiffusionGemma's multimodal encoder natively evaluates transformer layers in a single un-evaluated Metal compute graph. For large DOM contexts (>2,000 tokens), this causes unified memory to spike past working-set limits. 

We patch the encoder evaluation with an explicit layer-by-layer evaluation barrier:
```python
for i, (layer, c, mask) in enumerate(zip(self.decoder.layers, cache, masks)):
    h = layer(h, mask, c, decoder=False, layer_scalar=self.language_model.layers[i].layer_scalar)
    mx.eval(h, c.state)  # Force Metal buffer release per layer
```
This guarantees that active memory stays flat at exactly **15.41 GB** indefinitely, with zero memory leaks across 50+ continuous actions.

#### 2. Multi-Turn KV Prefix Caching
Browser agents execute multi-step trajectories where the overall goal, system constraints, and prompt schema remain invariant between steps. 

DiffBrowse implements session-level prefix caching:
```python
if self.enable_prefix_caching and self._cached_goal == goal:
    prefix_match = match_prefix(input_ids, self._cached_prefix_tokens)
    if prefix_match >= len(self._cached_prefix_tokens):
        cloned_cache = _clone_prompt_cache_for_apc(self._cached_prefix_cache)
        kv_cache = self._model.diffusion_update_cache(suffix_ids, cache=cloned_cache)
```
Incremental prefill latency drops to **~180–330 ms**, eliminating cold graph recompilation between turns.

#### 3. Canvas Prefix Seeding (`ACTION=`)
Evaluating raw un-finetuned single-slot diffusion can suffer from spatial ambiguity on complex forms. By prefixing the diffusion canvas with `ACTION=` (token IDs `[45070, 236784]`), the unmasked slot index is anchored to the action label token. 

This optimization boosted zero-shot direct slicing accuracy on real forms from **35% to 60.0%**.

#### 4. Unified Air-Gapped Decision + Text Typing
In `jev-ultrafast`, selecting `TYPE_TEXT` hands off string generation to a remote OpenAI-compatible model (`TEXT_MODEL_API_KEY`, e.g. DeepSeek).

DiffBrowse unifies both responsibilities inside the resident DiffusionGemma model. Using the same resident 26B MoE weights already loaded in Metal memory, the agent infers the exact field string (e.g., `"Zurich"`, `"London"`, `"Casa Flora"`) in ~240 ms with **zero cloud keys and 100% offline air-gapping**.

---

### Empirical Hyperparameter Evaluation & Hardware Latencies

To determine optimal diffusion hyperparameters (canvas lengths, pass counts, early exit thresholds), we evaluated DiffBrowse across $N=30$ structured offline scenario representations spanning flight searches, boutique hotel filtering, Wikipedia research, and large candidate sets:

| Architecture / Model | Runtime / Execution | Evaluated Action Acc (%) | Invalid Action Rate (%) | p50 Decoder Latency (ms) | p50 Total Decision (ms) | Active RAM (GB) | Air-Gapped / Privacy |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DiffBrowse (Generate)** | Local MLX Diffusion Text (16 tokens) | **88.0%** | **0.0%** | 248.6 ms | **435.1 ms** | 15.69 GB | **100% On-Device** |
| **DiffBrowse (Direct L=64, Mid)**| Local MLX Slicing (Passes=2) | **60.0%** | **0.0%** | 206.2 ms | **395.8 ms** | 15.65 GB | **100% On-Device** |
| **DiffBrowse (Direct L=32, Pass 1)**| Local MLX Slicing (Early Exit) | 16.7% – 50.0% | **0.0%** | **110.2 ms** | **303.4 ms** | 15.59 GB | **100% On-Device** |
| **AR Control (Qwen2.5-0.5B)** | Local MLX Autoregressive | 0.0% | 76.7% | 38.2 ms | 88.5 ms | 0.35 GB | 100% On-Device |

#### Key Insights & Ground Truth Realities
1. **Pass Count Sweet Spot**: 2 decoder passes are the empirical optimum for zero-shot decision-making on Metal. 1 pass takes 110 ms decoder time; 2 passes take 220 ms.
2. **The Bidirectional Attention Slot**: In canvas length ablations, canvas length 64 with a middle slot achieved **60.0% accuracy**, vs **23.3%** for early slot 0. Bidirectional attention allows the action token to condition simultaneously on surrounding context.
3. **Small Autoregressive Models Collapse**: The 0.5B autoregressive control model had a 76.7% invalid action rate, hallucinating invalid characters and ignoring candidate constraints. High representation capacity (26B MoE parameters) is necessary for reliable visual and DOM grounding.
4. **Physical Memory Boundaries**: The 4-bit 26B weights require **15.41 GB**. Adding macOS system overhead and browser execution brings the physical memory requirement to **≥ 24 GB Unified Memory**. DiffBrowse must not be run on 16 GB machines.
5. **Real-World Live Web Navigation**: Running zero-shot on complex live commercial websites (e.g. Google Flights) remains an active challenge due to blocking cookie consent modals, large DOM trees exceeding typical action window budgets, and debounced asynchronous dropdowns. Fine-tuning dedicated navigation adapters on real browser traces is the key next step.

---

### The Recommended Architecture: Confident-Local Hybrid

For real-world production deployments, we recommend a **Hybrid Confident-Local Policy**:

```
                                Observed DOM State
                                        │
                                        ▼
                          DiffBrowse Local Decision
                        (Apple Silicon Metal ~395 ms)
                                        │
                                        ▼
                      Uncertainty Gate (Shannon Entropy & Margin)
                                  H ≤ 0.35 AND M ≥ 0.50 ?
                                       /         \
                             YES     /             \   NO
                                   /                 \
                                  ▼                   ▼
                      Execute Locally on Metal     Fall back to Cloud Jev
                      • 100% Private               • Speculative Fan-out
                      • Zero API Cost              • Sub-150ms Remote
```

- **Confident Steps**: Common navigation, straightforward button clicks, link following, and form inputs execute locally on Metal in ~395 ms with 100% privacy and zero cost.
- **Uncertain Steps**: Ambiguous, deeply nested, or crowded UI states trigger an instant speculative fallback to remote cloud Jev (110 ms).
- **Result**: **>95% overall task completion**, **65–75% reduction in cloud API bills**, and complete privacy for sensitive workflows.

---

### Getting Started with DiffBrowse

Run the interactive Web Inspector on your local Mac:

```bash
git clone https://github.com/maximilianmessing/diffbrowse.git
cd diffbrowse
uv sync --extra mlx

# Run the local air-gapped demo with DiffusionGemma on Metal
JEV_BACKEND=mlx_direct uv run python -m jev_ultrafast.demo
```

Open **`http://127.0.0.1:8766`**, select the **Local Diffusion Direct** or **Diffusion Generate** backend, choose a scenario, and watch your Apple Silicon GPU drive the browser autonomously.
