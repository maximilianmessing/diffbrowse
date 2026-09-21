# Hardware Acceleration Guide: DiffBrowse Multi-Platform Deployment

**DiffBrowse** is designed to run air-gapped, sub-second browser decisions on local unified memory and hardware-accelerated systems.

This guide covers setup, environment configuration, and execution across the three primary hardware platforms:
1. **Apple Silicon (M-Series via MLX Metal)**
2. **NVIDIA DGX / Hopper / Blackwell / RTX (via CUDA 12.4+ & FlashAttention-2)**
3. **AMD Strix Halo / Ryzen AI Max 395 (via ROCm 6.2+ and Unified LPDDR5X)**

---

## 1. Platform Comparison & Sizing Matrix

| Platform | Architecture | Acceleration Framework | Precision / Quantization | Memory Requirement | Supported Backend | Latency Profile |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: |
| **Apple Silicon (M-Series)** | Pro / Max / Ultra (Unified RAM) | **MLX Metal** | 4-bit (`mlx-community`) | **Strictly ≥ 24 GB** (15.41 GB weights) | `mlx_structured` | **110–117 ms** (Pass 1) / **395–415 ms** (warm multi-pass) |
| **AMD ROCm** | Strix Halo (RDNA 3.5 APU) | **PyTorch + ROCm 6.2+ (HIP)** | 4-bit / BF16 (Unified LPDDR5X) | **≥ 24 GB Unified RAM** | `torch_structured` | Sub-second structured discrete diffusion |
| **NVIDIA CUDA** | RTX 3090/4090, A100/H100 | **PyTorch + CUDA 12.4+** | 4-bit (BitsAndBytes NF4) / BF16 | **≥ 24 GB VRAM** | `torch_structured` | Sub-second structured discrete diffusion |

---

## 2. AMD Strix Halo / Ryzen AI Max 395 Deployment

**AMD Strix Halo** (Ryzen AI Max 300 series, such as the Ryzen AI Max+ 395) features up to **40 RDNA 3.5 Compute Units** sharing up to **128 GB of unified 256-bit LPDDR5X memory** (up to 500 GB/s bandwidth). Like Apple Silicon, its unified memory pool allows loading 26B parameter models without discrete PCIe bus transfer bottlenecks.

### Turnkey Setup Script
Run the automated Strix Halo setup script:
```bash
chmod +x scripts/setup_strix_halo.sh
./scripts/setup_strix_halo.sh
```

### Manual Setup
```bash
# 1. Add user to render and video groups
sudo usermod -aG render,video $USER

# 2. Install PyTorch with ROCm 6.2 wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
uv sync --extra rocm

# 3. Configure RDNA 3.5 HIP target overrides
export HSA_OVERRIDE_GFX_VERSION=11.5.0
export HIP_VISIBLE_DEVICES=0
export PYTORCH_HIP_ALLOC_CONF="garbage_collection_threshold:0.8,max_split_size_mb:512"
```

### Launching DiffBrowse on Strix Halo
```bash
# Launch with ROCm structured diffusion engine
JEV_BACKEND=torch_structured uv run python -m jev_ultrafast.demo
```

### Zero-Setup Container (Docker)
```bash
docker build -t diffbrowse:rocm -f docker/Dockerfile.rocm .
docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  -p 8766:8766 \
  diffbrowse:rocm
```

---

## 3. NVIDIA DGX / Hopper / Blackwell / RTX Deployment

NVIDIA systems provide high-throughput Tensor Cores and FlashAttention-2 support for ultra-low latency structured canvas evaluation.

### Turnkey Setup Script
Run the automated NVIDIA setup script:
```bash
chmod +x scripts/setup_nvidia_cuda.sh
./scripts/setup_nvidia_cuda.sh
```

### Manual Setup
```bash
# 1. Install PyTorch with CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
uv sync --extra cuda
pip install bitsandbytes>=0.43.0

# 2. (Optional) Install FlashAttention-2
pip install flash-attn --no-build-isolation
```

### Launching DiffBrowse on NVIDIA
```bash
# Launch with CUDA structured diffusion engine
JEV_BACKEND=torch_structured uv run python -m jev_ultrafast.demo
```

### Zero-Setup Container (Docker)
```bash
docker build -t diffbrowse:cuda -f docker/Dockerfile.cuda .
docker run --rm -it --gpus all -p 8766:8766 diffbrowse:cuda
```

---

## 4. Apple Silicon (M-Series via MLX Metal)

For Apple Silicon Macs (M-series Pro / Max / Ultra), DiffBrowse uses Apple's native **MLX** framework.

> [!IMPORTANT]
> **Strict Memory Requirement**: DiffBrowse strictly requires **≥ 24 GB Unified Memory**. The 4-bit model weights occupy 15.41 GB. Combined with macOS WindowServer (~3 GB), Chromium (~1 GB), and transient Metal buffers (~1.5 GB), total footprint is ~21 GB. Running on 16 GB machines will trigger heavy swap thrashing and OS process termination.

```bash
# Install MLX Metal dependencies
uv sync --extra mlx

# Launch with native Metal structured backend
JEV_BACKEND=mlx_structured uv run python -m jev_ultrafast.demo
```

- **Invariant Prefix Caching**: Prefills static task preamble once; multi-turn turns execute in **3.97 ms** prefill.
- **Adaptive Escalation**: Terminates in **110–117 ms** on confident decisions ($M \ge 0.65$ or $H \le 0.35$).
- **Local Text Generation**: Synthesizes form field strings offline without external APIs.

---

## 5. Hardware Diagnostics CLI

DiffBrowse includes a multi-platform hardware diagnostic utility that inspects memory, runtime drivers, and environment overrides across all three architectures:

```bash
uv run python scripts/check_hardware.py
```

Output highlights missing overrides (e.g. `HSA_OVERRIDE_GFX_VERSION=11.5.0` on Strix Halo) or memory constraints.
