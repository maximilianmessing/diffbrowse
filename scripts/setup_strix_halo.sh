#!/usr/bin/env bash
# Turnkey setup and environment configuration for AMD Strix Halo / Ryzen AI Max 395
# Target: Ubuntu 24.04 LTS (ROCm 6.2+ / 7.0+ on RDNA 3.5 gfx1150/gfx1151)
set -euo pipefail

echo "======================================================================"
echo "DiffBrowse Setup: AMD Strix Halo / Ryzen AI Max 395 Acceleration"
echo "======================================================================"

# 1. Check user group permissions for DRM / render nodes
CURRENT_USER="${USER:-$(whoami)}"
if ! id -nG "$CURRENT_USER" | grep -qw "render"; then
    echo "[!] User '$CURRENT_USER' is not in the 'render' group."
    echo "    Run: sudo usermod -aG render,video $CURRENT_USER"
    echo "    Then log out and log back in for permissions to take effect."
else
    echo "[OK] User '$CURRENT_USER' has render/video group permissions."
fi

# 2. Configure RDNA 3.5 APU environment overrides
ENV_CONFIG_FILE="$HOME/.diffbrowse_strix_halo.env"
cat << 'EOF' > "$ENV_CONFIG_FILE"
# AMD Strix Halo (RDNA 3.5 / gfx1150) ROCm Overrides
export HSA_OVERRIDE_GFX_VERSION=11.5.0
export HIP_VISIBLE_DEVICES=0
export PYTORCH_HIP_ALLOC_CONF="garbage_collection_threshold:0.8,max_split_size_mb:512"
export JEV_BACKEND=torch_structured
EOF

echo "[OK] Generated environment profile at $ENV_CONFIG_FILE"
echo "     Source it before running: source $ENV_CONFIG_FILE"

# Export into current shell
export HSA_OVERRIDE_GFX_VERSION=11.5.0
export HIP_VISIBLE_DEVICES=0
export PYTORCH_HIP_ALLOC_CONF="garbage_collection_threshold:0.8,max_split_size_mb:512"
export JEV_BACKEND=torch_structured

# 3. Check / install PyTorch with ROCm
if ! python3 -c "import torch; assert getattr(torch.version, 'hip', None) is not None" 2>/dev/null; then
    echo "[+] Installing PyTorch ROCm 6.2 wheel..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
else
    echo "[OK] PyTorch with ROCm is already installed."
fi

# 4. Install DiffBrowse ROCm dependencies via uv
if command -v uv &>/dev/null; then
    echo "[+] Synchronizing DiffBrowse dependencies..."
    uv sync --extra rocm
else
    echo "[+] Installing packages via pip..."
    pip install "transformers>=4.44.0" "accelerate>=0.33.0" "browser-harness==0.1.13" "httpx[http2]>=0.28,<1"
fi

# 5. Run hardware validation inspector
echo "----------------------------------------------------------------------"
echo "[+] Validating Strix Halo hardware prerequisites..."
python3 scripts/check_hardware.py

echo "======================================================================"
echo "Strix Halo setup complete!"
echo "To run DiffBrowse with ROCm structured diffusion:"
echo "    source ~/.diffbrowse_strix_halo.env"
echo "    uv run python -m jev_ultrafast.demo"
echo "======================================================================"
