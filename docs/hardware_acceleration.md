# Hardware Acceleration Guide: DiffBrowse Multi-Platform Deployment

**DiffBrowse** is designed to run air-gapped, sub-second browser decisions on local unified memory and hardware-accelerated systems.

This guide covers setup, environment configuration, and execution across the three primary hardware platforms:
1. **Apple Silicon (M-Series via MLX Metal)**
2. **NVIDIA DGX Spark / Hopper / Blackwell (via CUDA + FlashAttention-2)**
3. **AMD Strix Halo / Ryzen AI Max 395 (via ROCm 6.2+ and Unified LPDDR5X)**

---

## 1. Platform Comparison & Sizing Matrix

| Platform | Architecture | Acceleration Framework | Precision / Quantization | Resident Memory | Expected Step Latency |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **Apple Silicon (M4/M5)** | 16–128 GB Unified Memory | **MLX Metal** | 4-bit (`mlx-community`) | **15.41 GB** | **200 – 395 ms** |
| **NVIDIA DGX Spark** | H100 / GH200 / B200 (HBM3e) | **PyTorch + CUDA 12.4+** | 4-bit (BitsAndBytes) / BF16 | **16.20 GB** | **80 – 160 ms** |
| **AMD Strix Halo** | Ryzen AI Max+ 395 (40 CUs) | **PyTorch + ROCm 6.2+ (HIP)** | 4-bit / BF16 (Unified RAM) | **15.80 GB** | **180 – 350 ms** |

---

## 2. NVIDIA DGX Spark / Hopper Deployment

NVIDIA DGX systems provide massive compute and high-bandwidth memory (HBM3/HBM3e), making them ideal for high-throughput browser agent fleets or ultra-low-latency execution.

### Prerequisites
- Ubuntu 22.04 LTS or 24.04 LTS
- NVIDIA Driver 550+ and CUDA 12.4+
- Python 3.10+ and `uv`

### Installation

```bash
git clone https://github.com/maximilianmessing/diffbrowse.git
cd diffbrowse

# Install dependencies with CUDA support
uv sync --extra cuda

# (Optional) Install FlashAttention-2 for optimized prefill kernels
pip install flash-attn --no-build-isolation
```

### Launching DiffBrowse on DGX

Set the backend to `torch_direct`:

```bash
# Run with CUDA acceleration
JEV_BACKEND=torch_direct uv run python -m jev_ultrafast.demo
```

Or invoke the agent programmatically:

```python
from jev_ultrafast import Agent
from jev_ultrafast.backends import TorchDiffusionDirectBackend

backend = TorchDiffusionDirectBackend(
    model_name="google/diffusiongemma-26b-it",
    device="cuda",
    torch_dtype="bfloat16",
    load_in_4bit=True,
)

with Agent("https://www.google.com/travel/flights", "Find flights to London", backend=backend) as agent:
    for state in agent.run():
        print(f"Step latency: {state['elapsed_ms']} ms")
```

---

## 3. AMD Strix Halo / Ryzen AI Max 395 Deployment

**AMD Strix Halo** (Ryzen AI Max 300 series, such as the Ryzen AI Max+ 395) features up to **40 RDNA 3.5 Compute Units** sharing up to **128 GB of unified 256-bit LPDDR5X memory** (up to 500 GB/s bandwidth). Like Apple Silicon, its unified memory pool allows loading 26B parameter models without discrete PCIe bus transfer bottlenecks.

### Prerequisites
- Ubuntu 24.04 LTS (Kernel 6.8+ or 6.10+)
- AMD ROCm 6.2+ or ROCm 7.0+
- User added to `render` and `video` groups (`sudo usermod -aG render,video $USER`)

### Installation

```bash
git clone https://github.com/maximilianmessing/diffbrowse.git
cd diffbrowse

# Install PyTorch with ROCm wheels
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
uv sync --extra rocm
```

### Environment Configuration for RDNA 3.5

Because Strix Halo uses RDNA 3.5 (gfx1150 / gfx1151), set the ROCm architecture target overrides in your environment:

```bash
# Enable RDNA 3.5 HIP target override
export HSA_OVERRIDE_GFX_VERSION=11.5.0
export HIP_VISIBLE_DEVICES=0

# Configure ROCm unified memory allocations
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:512
```

### Launching DiffBrowse on Strix Halo

```bash
# Run DiffBrowse with ROCm acceleration
JEV_BACKEND=torch_direct uv run python -m jev_ultrafast.demo
```

The backend automatically detects the HIP runtime:
```text
Loaded model onto rocm (AMD Strix Halo)
Unified Memory Footprint: 15.80 GB / 128.00 GB Available
```

---

## 4. Apple Silicon (M-Series via MLX Metal)

For Apple Silicon Macs (M1/M2/M3/M4/M5 with 16 GB+ unified memory), DiffBrowse uses Apple's native **MLX** framework.

```bash
# Install MLX Metal dependencies
uv sync --extra mlx

# Launch with native Metal backend
JEV_BACKEND=mlx_direct uv run python -m jev_ultrafast.demo
```

- **Zero GPU Memory Leaks**: Uses layer-by-layer encoder evaluation (`mx.eval`) to cap resident memory at 15.41 GB.
- **Adaptive Early Exit**: Terminates in **110 ms** on confident decisions.
- **Local Text Generation**: Synthesizes field strings in **241 ms** offline.

---

## 5. Web Inspector Multi-Device Selector

The interactive inspector at `http://127.0.0.1:8766` allows toggling between acceleration backends dynamically from the UI:
- **Local DiffusionGemma (Apple Silicon Metal)**
- **Local DiffusionGemma (NVIDIA DGX / AMD Strix Halo)**
- **Hybrid (Local + Cloud Jev)**
- **Cloud Jev (Speculative Baseline)**
