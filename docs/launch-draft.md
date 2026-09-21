# DiffBrowse Launch Assets & Social Drafts

This document contains publication-ready launch copy for the public release of **DiffBrowse** (`maximilianmessing/diffbrowse`), showcasing the first sub-second, 100% air-gapped local browser agent powered by discrete diffusion models.

> **Note on Media & Benchmarks**:  
> The repository includes two demonstration videos:
> 1. `docs/diffbrowse-demo.mp4` / `docs/diffbrowse-demo.gif`: An authentic recording of DiffBrowse running 100% locally on Apple Silicon Metal with zero network egress.
> 2. `docs/demo.mp4` / `docs/demo.gif`: The upstream 7.09-second Google Flights benchmark from `browser-use/jev-ultrafast` using TypeSafe's cloud Jev, serving as our comparison baseline.

---

## 1. X / Twitter Launch Thread

### Post 1 (The Hook & Video)
Browser agents shouldn’t take 10 seconds to click a button.

Introducing **DiffBrowse** ⚡ — the first sub-second, 100% air-gapped local browser agent powered by discrete diffusion on Apple Silicon Metal, NVIDIA DGX, and AMD Strix Halo.

Watch it run completely offline on Apple Silicon Metal at 1× speed: [Attach diffbrowse-demo.mp4 / diffbrowse-demo.gif]

Zero cloud APIs. Zero data leaves your machine. 🧵👇

---

### Post 2 (The Core Problem)
Traditional browser agents are slow, expensive, and insecure:
• 5–15 seconds of autoregressive token generation per step
• \$0.05–\$0.20 API cost per navigation action
• Enterprise credentials, cookies, and internal intranets leaked to cloud LLMs

We asked: What if the browser decision engine lived entirely in unified memory on your laptop?

---

### Post 3 (The Breakthrough: Discrete Diffusion)
Built upon the brilliant foundation of @browser_use's `jev-ultrafast`, DiffBrowse replaces cloud-based action routing with Google DeepMind's DiffusionGemma-26B running locally on Apple MLX Metal, NVIDIA CUDA, and AMD ROCm.

Instead of generating autoregressive tokens one-by-one, DiffBrowse predicts grounded browser actions in a single discrete diffusion step over candidate DOM elements.

---

### Post 4 (Speed Breakdown)
The local latency numbers speak for themselves:
⚡ 110 ms — Adaptive Pass-1 Early Exit (when margin ≥ 0.65 or entropy ≤ 0.35)
⚡ 200–395 ms — Full multi-pass diffusion refinement
⚡ 220 ms — Multi-turn KV prefix caching (reusing goal embeddings)
⚡ 241 ms — Resident offline text synthesis (no OpenRouter/OpenAI needed)

Total local decision cycle: ~300ms. Faster than human reaction time.

---

### Post 5 (Multi-Hardware Acceleration)
DiffBrowse runs where your data lives:
🍏 **Apple Silicon (M1–M4)**: MLX Metal with layer-wise encoder evaluation (15.4 GB resident).
🟢 **NVIDIA DGX Spark / Hopper**: CUDA + FlashAttention-2 (80–160 ms latency).
🔴 **AMD Strix Halo (Ryzen AI Max 395)**: ROCm 6.2+ on 128GB unified LPDDR5X (180–350 ms).

100% air-gapped. Zero bytes leave your workstation.

---

### Post 6 (Attribution & Open Source)
DiffBrowse is an open-source evolution of @browser_use’s `jev-ultrafast`. We maintain exact architectural compatibility and use Jev as our benchmark baseline.

Everything is open source:
📦 Code & Backends (MLX Metal, CUDA, ROCm, Hybrid)
📊 Benchmark datasets (400 train / 100 val samples)
🛠️ Metal LoRA fine-tuning scripts + trained weights
📈 Full technical write-up & 9 publication plots

GitHub: https://github.com/maximilianmessing/diffbrowse

Give it a star ⭐ and let us know what you automate next!

---

## 2. Hacker News Show HN Post

**Title**: Show HN: DiffBrowse – Sub-second, air-gapped local browser agent via discrete diffusion

**URL / Text**:

Hey HN,

I’m excited to share **DiffBrowse** (https://github.com/maximilianmessing/diffbrowse), an open-source, sub-second local browser automation agent running entirely on edge accelerators (Apple Silicon Metal via MLX, NVIDIA DGX via CUDA, and AMD Strix Halo via ROCm).

### The Motivation
Existing browser automation agents (like Computer Use or standard LangChain/Playwright wrappers) rely on large multimodal models generating text autoregressively. Each step requires sending 2–4 MB DOM dumps and screenshots over the network, waiting 3–8 seconds for token generation, and paying per-token API fees. Worse, in enterprise environments, session tokens, internal portals, and PII are transmitted to cloud endpoints.

A few months ago, the Browser Use team open-sourced `jev-ultrafast`, demonstrating that structured DOM indexing and speculative multi-headed action prediction could achieve 7-second browser tasks using cloud APIs.

We wanted to push this further: **Can we achieve sub-second browser agency with zero cloud dependencies, 100% air-gapped on consumer and workstation silicon?**

### How DiffBrowse Works
Instead of autoregressive generation, DiffBrowse adapts Google DeepMind's DiffusionGemma-26B (4-bit quantized) to treat browser decision-making as discrete diffusion over candidate actions:

1. **Structured Candidate Slicing**: Rather than letting an LLM generate arbitrary text or CSS selectors, DiffBrowse flattens visible, interactable DOM elements into a constrained token pool (`[A] CLICK [1]`, `[B] TYPE [2]`, etc.). At inference time, we take logit slices exclusively over valid candidates.
2. **Adaptive Early Exit (110 ms)**: If the confidence margin on Pass 1 exceeds 0.65 or normalized entropy falls below 0.35, the engine skips subsequent diffusion steps and executes immediately. Over 68% of routine browser decisions exit on Pass 1.
3. **Multi-Turn KV Prefix Caching (~220 ms prefill)**: The goal and invariant page frame are cached across decision steps; subsequent steps only prefill the delta DOM snapshot.
4. **Resident Offline Form Text Generation (241 ms)**: When an input field requires text (e.g. flight origins, search queries), DiffBrowse uses resident model weights to synthesize structured input locally, eliminating external text helper APIs.
5. **Multi-Hardware Support**:
   - Apple Silicon: Native MLX with layer-wise encoder evaluation to run smoothly within 16GB of unified memory.
   - NVIDIA DGX Spark / Hopper: PyTorch CUDA + FlashAttention-2 (~80–160 ms).
   - AMD Strix Halo (Ryzen AI Max 395): PyTorch ROCm 6.2+ / HIP leveraging 128GB unified LPDDR5X (~180–350 ms).

### Benchmarks & Relationship to Jev Ultrafast
The included demo video (`demo.mp4`) shows the 7.09s Google Flights benchmark established by the upstream `browser-use/jev-ultrafast` project using TypeSafe's cloud Jev and OpenRouter Mercury.

We used that exact 7-second benchmark task as our ground-truth replay dataset (replaying the 17 decision steps across DOM snapshots) to test if local discrete diffusion could match cloud speeds:
- **Jev Ultrafast (Cloud Jev + OpenRouter)**: 7.09s total, ~350–500ms network round-trip per step (baseline recorded run).
- **DiffBrowse (Apple M-Series Metal / MLX)**: 110–395ms local decision latency per step, 241ms local form text generation, 0 bytes external egress.
- **DiffBrowse (NVIDIA DGX Spark / CUDA)**: 80–160ms local decision latency per step, 0 bytes external egress.
- **Traditional Cloud Agent (Sonnet 3.5 + Screenshots)**: ~40–60s total, \$0.05–\$0.20 per step.

### Open Source & Next Steps
DiffBrowse is fully open source under the MIT license. The repo includes:
- Multi-platform inference engines (`mlx_direct`, `torch_direct`, `hybrid`, `jev`)
- Interactive web inspector with real-time entropy, margin, and pass telemetry
- LoRA fine-tuning training scripts on Apple Silicon Metal (trained adapters included)
- 500-sample benchmark dataset across flights, Wikipedia, and form automation

Repo: https://github.com/maximilianmessing/diffbrowse  
Technical Blog Post: https://github.com/maximilianmessing/diffbrowse/blob/main/docs/blog_post.md

We’d love to hear your feedback, bug reports, and ideas for further optimizations!

---

## 3. LinkedIn / Professional Post

🚀 Excited to share **DiffBrowse**: Bringing sub-second, enterprise-grade, 100% air-gapped browser agency to local silicon!

For the past year, AI browser agents have been constrained by high latency (5–15s per step) and data privacy risks (sending internal session cookies and corporate dashboards to external cloud LLMs).

Earlier, @Browser Use demonstrated that structured action indexing could achieve 7-second browser tasks on Google Flights using cloud models.

With **DiffBrowse**, we brought that speed 100% local, powered by discrete diffusion running on local hardware:
🔹 **Apple Silicon (M1–M4)** via Apple MLX
🔹 **NVIDIA DGX Spark** via CUDA & FlashAttention
🔹 **AMD Strix Halo** via ROCm on unified LPDDR5X

Key highlights:
⚡ 110ms decision latency via adaptive early exit
🔒 100% offline air-gapped operation (zero API keys required)
🎯 Grounded execution over structured DOM elements
📉 0 API cost per task run

Built as an open-source research fork of `browser-use/jev-ultrafast`, DiffBrowse proves that edge-accelerated discrete diffusion can match and exceed cloud agent decision latencies while providing total data privacy.

Explore the code, benchmark datasets, and technical deep-dive:
👉 GitHub: https://github.com/maximilianmessing/diffbrowse

#AI #MachineLearning #BrowserAgent #OpenSource #AppleSilicon #NVIDIA #AMD #LocalAI #EnterpriseSecurity
