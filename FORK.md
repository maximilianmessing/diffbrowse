# DiffBrowse

> **DiffBrowse: Sub-Second Local Browser Agent on Apple Silicon Metal**  
> Maintained by [@maximilianmessing](https://github.com/maximilianmessing) · Forked from and built on the foundation of [`browser-use/jev-ultrafast`](https://github.com/browser-use/jev-ultrafast).

---

## Overview

**DiffBrowse** is an experimental, 100% air-gapped browser agent that replaces or augments cloud decision models with local **DiffusionGemma-26B-A4B** running natively on Apple Silicon unified memory via MLX.

### Built on Jev Ultrafast
DiffBrowse takes the dynamic, indexed action-space philosophy pioneered by [Browser Use × TypeSafe's Jev Ultrafast](https://github.com/browser-use/jev-ultrafast) as its foundation:
- **Foundational Architecture**: We build directly on `jev-ultrafast`'s atomic DOM accessibility snapshotting, node-anchored actions, and freshness guards.
- **Benchmark Baseline**: We evaluate DiffBrowse against Jev's speculative cloud multi-head classifier across exact action accuracy, end-to-end latency, and grounding reliability.
- **The Diffusion Evolution**: Where Jev relies on a proprietary cloud classifier and third-party LLM for text typing, DiffBrowse uses a single resident discrete-diffusion model on Metal to perform **both** bounded action candidate slicing and form text generation.

---

## 1. Quickstart by Hardware Platform

### A. Apple Silicon (M-Series via MLX Metal)
```bash
uv sync --extra mlx
JEV_BACKEND=mlx_direct uv run python -m jev_ultrafast.demo
```

### B. NVIDIA DGX Spark / Hopper / Blackwell (CUDA)
```bash
uv sync --extra cuda
JEV_BACKEND=torch_direct uv run python -m jev_ultrafast.demo
```

### C. AMD Strix Halo / Ryzen AI Max 395 (ROCm 6.2+ Unified LPDDR5X)
```bash
# Set RDNA 3.5 architecture override
export HSA_OVERRIDE_GFX_VERSION=11.5.0
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
uv sync --extra rocm

JEV_BACKEND=torch_direct uv run python -m jev_ultrafast.demo
```

Open **`http://127.0.0.1:8766`** in Chrome:
- Use the **Backend selector** to toggle between **Apple Silicon Metal**, **NVIDIA DGX / AMD Strix Halo**, **Hybrid**, and **Cloud Jev**.
- Watch live telemetry badges: Entropy ($H$), Top-2 Margin ($M$), Denoising Passes ($p$), and Prefix Cache Hits (`⚡ Cache HIT`).
- See [docs/hardware_acceleration.md](docs/hardware_acceleration.md) for full hardware guides and optimization flags.

---


## 2. Git Remotes Configuration for `maximilianmessing/diffbrowse`

To manage this repository alongside upstream `browser-use/jev-ultrafast`:

```bash
# 1. Verify current remotes
git remote -v

# 2. Add upstream tracking
git remote add upstream https://github.com/browser-use/jev-ultrafast.git

# 3. Set your fork as origin
git remote set-url origin https://github.com/maximilianmessing/diffbrowse.git

# 4. Fetch upstream updates
git fetch upstream

# 5. Push your feature branch or main
git push -u origin main
```

---

## 3. Upstream Contribution Plan

When contributing these improvements back to [`browser-use/jev-ultrafast`](https://github.com/browser-use/jev-ultrafast):

### Scope of Upstream Pull Request
The core repository prioritizes minimalism. All Apple Silicon Metal additions are isolated cleanly:
1. **Optional Dependency**: `mlx` dependencies are isolated under `[project.optional-dependencies]` in `pyproject.toml` (`uv sync --extra mlx`). Systems without MLX run `JevBackend` without issues.
2. **Modular Backends**: Kept inside `jev_ultrafast/backends/`:
   - `base.py`: Protocol and backwards-compatible `Decision` object.
   - `mlx_direct.py`: Direct Metal logit slicing, KV prefix caching, adaptive early exit.
   - `mlx_generate.py`: Text generation baseline.
   - `hybrid.py`: Uncertainty-gated local execution with Jev fallback.
3. **Local Text Helper Fallback**: In `jev_ultrafast/model.py:field_text`, falls back to local `backend.generate_field_text` when `TEXT_MODEL_API_KEY` is not provided.
4. **Web Inspector UI**: Adds backend `<select id="backend">` and live telemetry badges to `index.html` and `app.js`.

### Proposed PR Title & Description
- **Title**: `feat: Local Apple Silicon Metal decision backend via DiffusionGemma (MLX)`
- **Summary**: Adds an optional, 100% air-gapped local backend running on Apple Silicon Metal via MLX. Allows users without cloud API keys to run autonomous browser steps in ~200–400 ms on M-series Macs.

---

## 4. Key Documentation & Artifacts

- **[Technical Blog Post](docs/blog_post.md)**: Full write-up with benchmarks comparing DiffBrowse vs. Cloud Jev, diagrams, and architectural insights.
- **[Comprehensive Research Report](reports/mlx_diffusiongemma.md)**: Detailed ablation study answering all 16 research questions.
- **[Benchmark Plots](reports/plots/)**: 9 publication-grade visualization plots covering pass count, canvas length, slot position, latency distribution, calibration, and sustained memory stability.
- **[LoRA Training Module](scripts/train_lora.py)**: Metal-native LoRA adapter training for attention projections (`q_proj`, `v_proj`).
