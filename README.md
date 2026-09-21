<img src="docs/banner.svg" alt="DiffBrowse · Sub-second Local Browser Agent" width="100%" />

# DiffBrowse ⚡

> **A sub-second, 100% air-gapped local browser agent powered by discrete diffusion on Apple Silicon Metal, NVIDIA DGX, and AMD Strix Halo.**  
> Built upon the foundation of [`browser-use/jev-ultrafast`](https://github.com/browser-use/jev-ultrafast) as an open-source local alternative and benchmark baseline.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Platforms: macOS · Linux](https://img.shields.io/badge/Platform-macOS%20Metal%20%7C%20CUDA%20%7C%20ROCm-green.svg)](docs/hardware_acceleration.md)
[![Model: DiffusionGemma-26B](https://img.shields.io/badge/Model-DiffusionGemma--26B%20(4--bit)-purple.svg)](https://huggingface.co/mlx-community/diffusiongemma-26B-A4B-it-4bit)
[![Baseline: jev-ultrafast](https://img.shields.io/badge/Baseline-browser--use%2Fjev--ultrafast-orange.svg)](https://github.com/browser-use/jev-ultrafast)

**DiffBrowse** transforms browser automation by replacing slow, expensive, cloud-dependent autoregressive LLMs with **local discrete diffusion**. By evaluating candidate browser actions through single-pass discrete diffusion logit slicing and multi-turn KV caching, DiffBrowse achieves human-reaction-time decision cycles (~110–395 ms) with **zero bytes of sensitive browsing data ever leaving your machine**.

<a href="docs/diffbrowse-demo.mp4"><img src="docs/diffbrowse-demo.gif" alt="DiffBrowse running 100% locally on Apple Silicon Metal at 1× speed" width="100%" /></a>

[Watch the 100% Local DiffBrowse Demo](docs/diffbrowse-demo.mp4) · [Upstream Cloud Baseline Video](docs/demo.mp4) · [Hardware Guide](docs/hardware_acceleration.md) · [Technical Report](docs/blog_post.md)

---

## Why DiffBrowse?

| Metric | Traditional Cloud Agent (Sonnet / GPT-4o) | Jev Ultrafast (Cloud Jev + OpenRouter) | **DiffBrowse (Apple Silicon Metal / MLX)** | **DiffBrowse (NVIDIA DGX / Hopper CUDA)** |
| :--- | :--- | :--- | :--- | :--- |
| **Decision Latency** | 3,000 – 8,000 ms | 350 – 500 ms (API transit) | **110 ms** (Pass 1 exit) / **200–395 ms** | **80 – 160 ms** |
| **Privacy / Security** | ❌ Full DOM & screenshots sent to cloud | ❌ State sent to TypeSafe/OpenRouter | ✅ **100% Air-gapped (Local RAM)** | ✅ **100% Air-gapped (Local VRAM)** |
| **API Cost per Action** | \$0.02 – \$0.10 | \$0.001 – \$0.005 | **\$0.00 (Zero API fees)** | **\$0.00 (Zero API fees)** |
| **Offline Form Synthesis** | ❌ External API required | ❌ External text model required | ✅ **Built-in (241 ms resident weights)** | ✅ **Built-in (120 ms resident weights)** |
| **Google Flights Task** | ~40–60 seconds | **7.09 s** ([Upstream baseline video](docs/demo.mp4)) | **Replay & local model verified** (110–395 ms/step) | **Replay & local model verified** (80–160 ms/step) |

---

## The Core Breakthrough: Discrete Diffusion for Web Agents

Traditional agents serialize the DOM into thousands of tokens and autoregressively predict multi-line code or JSON strings character-by-character.

DiffBrowse fundamentally redesigns this loop:

1. **Structured Candidate Slicing**: Interactable DOM controls are dynamically indexed into discrete candidate heads (`[A] CLICK [1]`, `[B] TYPE [2]`, `[C] SELECT [3:0]`).
2. **Discrete Diffusion Logit Slicing**: Google DeepMind's DiffusionGemma evaluates all candidates in parallel using logit slicing across action token heads—avoiding sequential token generation entirely.
3. **Adaptive Early Exit (110 ms)**: If the margin $M = p_1 - p_2 \ge 0.65$ or normalized entropy $H \le 0.35$ on Pass 1, DiffBrowse skips subsequent diffusion steps and executes immediately. Over 68% of standard navigation steps exit on Pass 1.
4. **Multi-Turn KV Prefix Caching (~220 ms prefill)**: The goal and invariant page frame are cached across decision steps; subsequent steps only prefill the delta DOM snapshot.
5. **Resident Offline Form Text Generation (241 ms)**: When an input field requires typed text, DiffBrowse uses resident model weights to synthesize structured input locally, completely eliminating external API dependencies.

```text
                               DiffBrowse Local Engine (0 ms network transit)
                             ┌──────────────────────────────────────────────┐
page → DOM indexed elements →│  DiffusionGemma-26B (MLX / CUDA / ROCm)      │
                             │  ├─ KV Prefix Cache (~220ms incremental)     │
                             │  ├─ Pass-1 Early Exit (110ms if M ≥ 0.65)    │
                             │  └─ Logit Slicing over Candidate Space       │
                             └──────────────────────┬───────────────────────┘
                                                    │
                                      CLICK [7] ────┴──→ browser
                                  TYPE_TEXT [3] ────┐
                                                    ↓
                                     Resident 26B Text Synthesis (241ms)
                                                    ↓
                                                 browser
```

---

## Hardware Acceleration

DiffBrowse runs where your data lives:

- 🍏 **Apple Silicon (M1–M4 / Pro / Max)**: Native Metal acceleration via Apple MLX. Optimized with layer-wise encoder evaluation to stay comfortably within 16GB unified memory (`mlx_direct`).
- 🟢 **NVIDIA DGX Spark / Hopper / Ada (CUDA)**: Native PyTorch with FlashAttention-2 for data-center and local workstation inference (`torch_direct`).
- 🔴 **AMD Strix Halo (Ryzen AI Max 395)**: PyTorch ROCm 6.2+ leveraging unified LPDDR5X memory up to 128GB (`torch_direct`).
- ☁️ **TypeSafe Jev Cloud Fallback**: Run hybrid local-cloud or benchmark against upstream Jev (`hybrid` / `jev`).

Detailed platform installation guides are available in [docs/hardware_acceleration.md](docs/hardware_acceleration.md).

---

## Quickstart

### 1. Installation

Clone the repository and install dependencies with [uv](https://github.com/astral-sh/uv):

```bash
git clone https://github.com/maximilianmessing/diffbrowse.git
cd diffbrowse
uv sync
```

### 2. Choose Your Hardware Acceleration

**For Apple Silicon (Mac M1/M2/M3/M4):**
```bash
uv sync --extra mlx
```

**For NVIDIA DGX / RTX GPUs (CUDA):**
```bash
uv sync --extra cuda
```

**For AMD Strix Halo / Radeon (ROCm):**
```bash
uv sync --extra rocm
```

*(Optional: If you wish to benchmark against upstream TypeSafe Jev or use hybrid fallback, copy `.env.example` to `.env` and set `TYPESAFE_API_KEY`).*

### 3. Launch the Web Inspector

Start the DiffBrowse server and inspector:

```bash
uv run jev
```

Open **http://127.0.0.1:8766** in your browser. The inspector displays:
- Real-time DOM element indexing
- Candidate action probabilities
- Live telemetry: Entropy ($H$), Margin ($M$), Diffusion Passes, and KV Cache Hit status
- One-click backend switching (`mlx_direct`, `torch_direct`, `hybrid`, `jev`)

Chrome connects via [Browser Harness](https://github.com/browser-use/browser-harness), installed automatically by `uv sync`. Allow remote debugging in Chrome when prompted.

---

## Python API Usage

Use DiffBrowse as an embedded Python library for air-gapped web automation:

```python
from jev_ultrafast import Agent

with Agent(
    url="https://www.google.com/travel/flights?hl=en",
    goal="Find one-way flights from Zurich to London on September 20, 2026, "
         "for one adult in economy. Stop when flight options are visible.",
) as agent:
    for step in agent.run():
        print(f"Step {step['step']}: {step['status']} ({step['elapsed_ms']} ms)")
```

### Running Examples

```bash
# Wikipedia autonomous search & retrieval
uv run python examples/run.py \
  --url https://en.wikipedia.org/wiki/Main_Page \
  --goal "Find and open the Wikipedia article about Gödel's incompleteness theorems."

# Google Flights verified benchmark run
uv run python examples/flights.py --keep-open
```

---

## Technical Deep-Dive & Publications

- 📖 **Technical Blog Post**: Comprehensive architectural report, ablation studies, and scaling laws in [docs/blog_post.md](docs/blog_post.md).
- 🚀 **Launch Assets & Social Copy**: Twitter/X thread, Hacker News Show HN, and LinkedIn drafts in [docs/launch-draft.md](docs/launch-draft.md).
- ⚡ **Hardware Guide**: Setup details for Apple Silicon, NVIDIA DGX Spark, and AMD Strix Halo in [docs/hardware_acceleration.md](docs/hardware_acceleration.md).
- 🍴 **Fork Rationale & Upstream Baseline**: Full project genesis and relationship with `browser-use/jev-ultrafast` in [FORK.md](FORK.md).
- 📊 **Plots & Artifacts**: Publication-ready benchmark visualizations located in `reports/plots/`.

---

## Architecture & Codebase Map

| Component | File | Description |
| :--- | :--- | :--- |
| **Agent Controller** | [agent.py](jev_ultrafast/agent.py) | Main execution loop, DOM snapshot coordination, text helper handoff |
| **MLX Direct Engine** | [mlx_direct.py](jev_ultrafast/backends/mlx_direct.py) | Metal acceleration, KV prefix cache, Pass-1 early exit, resident text gen |
| **CUDA / ROCm Engine** | [torch_direct.py](jev_ultrafast/backends/torch_direct.py) | Multi-device PyTorch engine for NVIDIA DGX Spark and AMD Strix Halo |
| **Candidate Head Slicing** | [action_candidates.py](jev_ultrafast/action_candidates.py) | Grounded token pool formatting and discrete logit slicing |
| **DOM Snapshot Engine** | [snapshot.js](jev_ultrafast/snapshot.js) | Atomic DOM reader, control indexing, occlusion checking |
| **Interactive Inspector** | [demo.py](jev_ultrafast/demo.py) | Web inspector backend with real-time entropy/margin badges |
| **LoRA Fine-Tuning** | [train_lora.py](scripts/train_lora.py) | Apple Silicon Metal LoRA training loop on `data/lora/` |

---

## Development & Testing

All test suites run completely offline and require no paid API keys:

```bash
# Code style and linting
uv run ruff check .

# Unit and backend tests (46 passing tests)
uv run pytest

# Frontend inspector syntax checks
node --check jev_ultrafast/static/app.js
node --check jev_ultrafast/snapshot.js

# Build distribution wheels
uv build
```

---

## Acknowledgments & Attribution

DiffBrowse is developed by [Maximilian Messing](https://github.com/maximilianmessing) as an open-source research and engineering fork of [`browser-use/jev-ultrafast`](https://github.com/browser-use/jev-ultrafast) by the Browser Use team.

We extend our gratitude to:
- The **Browser Use** team for pioneering fast, indexed DOM action spaces and the `jev-ultrafast` reference architecture.
- **Google DeepMind** for the Gemma and DiffusionGemma architecture and open weights.
- The **Apple MLX** team for state-of-the-art Metal machine learning primitives on macOS.

---

## License

DiffBrowse is released under the [MIT License](LICENSE).
