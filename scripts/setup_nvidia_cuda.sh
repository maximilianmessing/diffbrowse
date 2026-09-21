#!/usr/bin/env bash
# Turnkey setup and environment configuration for NVIDIA DGX / Hopper / Blackwell / RTX
# Target: Ubuntu 22.04 / 24.04 LTS (CUDA 12.4+ with FlashAttention-2)
set -euo pipefail

echo "======================================================================"
echo "DiffBrowse Setup: NVIDIA CUDA Acceleration"
echo "======================================================================"

# 1. Verify nvidia-smi visibility
if ! command -v nvidia-smi &>/dev/null; then
    echo "[!] nvidia-smi not found. Please install NVIDIA drivers (>= 550) first."
    exit 1
fi

echo "[OK] NVIDIA GPU detected:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# 2. Configure environment profile
ENV_CONFIG_FILE="$HOME/.diffbrowse_nvidia.env"
cat << 'EOF' > "$ENV_CONFIG_FILE"
# NVIDIA DGX / RTX CUDA Overrides
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export JEV_BACKEND=torch_structured
EOF

echo "[OK] Generated environment profile at $ENV_CONFIG_FILE"
echo "     Source it before running: source $ENV_CONFIG_FILE"

# Export into current shell
export JEV_BACKEND=torch_structured

# 3. Check / install PyTorch with CUDA 12.4
if ! python3 -c "import torch; assert torch.cuda.is_available() and getattr(torch.version, 'hip', None) is None" 2>/dev/null; then
    echo "[+] Installing PyTorch CUDA wheel..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
else
    echo "[OK] PyTorch with CUDA is already installed."
fi

# 4. Install DiffBrowse CUDA dependencies via uv
if command -v uv &>/dev/null; then
    echo "[+] Synchronizing DiffBrowse dependencies..."
    uv sync --extra cuda
else
    echo "[+] Installing packages via pip..."
    pip install "transformers>=4.44.0" "accelerate>=0.33.0" "browser-harness==0.1.13" "httpx[http2]>=0.28,<1"
fi

# 5. Install BitsAndBytes for 4-bit quantization
echo "[+] Installing BitsAndBytes..."
pip install bitsandbytes>=0.43.0

# 6. Optional FlashAttention-2
echo "[+] Checking FlashAttention-2..."
if ! python3 -c "import flash_attn" 2>/dev/null; then
    echo "[+] Installing FlashAttention-2 (optimized prefill kernels)..."
    pip install flash-attn --no-build-isolation || echo "[!] flash-attn compilation skipped; native PyTorch SDPA will be used."
else
    echo "[OK] FlashAttention-2 kernel is already installed."
fi

# 7. Run hardware validation inspector
echo "----------------------------------------------------------------------"
echo "[+] Validating NVIDIA hardware prerequisites..."
python3 scripts/check_hardware.py

echo "======================================================================"
echo "NVIDIA CUDA setup complete!"
echo "To run DiffBrowse with CUDA structured diffusion:"
echo "    source ~/.diffbrowse_nvidia.env"
echo "    uv run python -m jev_ultrafast.demo"
echo "======================================================================"
