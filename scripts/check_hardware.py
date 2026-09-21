#!/usr/bin/env python3
"""Hardware acceleration inspector and platform validator for DiffBrowse.

Validates environment, memory, and acceleration prerequisites across:
1. Apple Silicon (MLX Metal)
2. AMD Strix Halo / Ryzen AI Max 395 (ROCm 6.2+ HIP)
3. NVIDIA DGX / Hopper / Blackwell / RTX (CUDA + FlashAttention-2)
"""

from __future__ import annotations

import os
import platform
import sys
from typing import Any, Dict


def check_apple_silicon() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "platform": "Apple Silicon (Metal)",
        "supported": False,
        "details": [],
        "warnings": [],
        "recommended_backend": "mlx_structured",
    }
    if platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"):
        info["details"].append("Not running on macOS ARM64.")
        return info

    try:
        total_ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3)
        info["total_memory_gb"] = round(total_ram_gb, 1)
        if total_ram_gb < 23.5:
            info["warnings"].append(
                f"Unified Memory: {total_ram_gb:.1f} GB detected. "
                "DiffBrowse 26B strictly requires ≥ 24 GB Unified Memory."
            )
        else:
            info["details"].append(f"Unified Memory: {total_ram_gb:.1f} GB (Requirement satisfied: ≥ 24 GB).")
    except Exception:
        pass

    try:
        import mlx.core as mx

        info["mlx_version"] = getattr(mx, "__version__", "installed")
        info["details"].append(f"MLX Metal framework installed (v{info['mlx_version']}).")
        info["supported"] = True
    except ImportError:
        info["warnings"].append("MLX is not installed. Run: uv sync --extra mlx")

    return info


def check_amd_strix_halo() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "platform": "AMD Strix Halo (ROCm / HIP)",
        "supported": False,
        "details": [],
        "warnings": [],
        "recommended_backend": "torch_structured",
    }

    hsa_override = os.environ.get("HSA_OVERRIDE_GFX_VERSION")
    if hsa_override:
        info["details"].append(f"HSA_OVERRIDE_GFX_VERSION={hsa_override}")
    else:
        info["warnings"].append(
            "HSA_OVERRIDE_GFX_VERSION is not set. For Strix Halo (RDNA 3.5), set:\n"
            "export HSA_OVERRIDE_GFX_VERSION=11.5.0"
        )

    try:
        import torch

        if getattr(torch.version, "hip", None):
            hip_ver = torch.version.hip
            info["hip_version"] = hip_ver
            info["details"].append(f"PyTorch ROCm/HIP detected (v{hip_ver}).")
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                info["device_name"] = device_name
                info["details"].append(f"HIP Device: {device_name}")
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                info["memory_gb"] = round(vram_gb, 1)
                info["details"].append(f"Available Unified Memory / VRAM: {vram_gb:.1f} GB")
                if vram_gb < 23.5:
                    info["warnings"].append(f"Memory {vram_gb:.1f} GB < 24 GB minimum required for 26B model.")
                else:
                    info["supported"] = True
            else:
                info["warnings"].append("ROCm installed, but torch.cuda.is_available() returned False.")
        else:
            info["details"].append("PyTorch is not built with ROCm/HIP support.")
    except ImportError:
        info["warnings"].append(
            "PyTorch not installed. Install ROCm PyTorch via:\n"
            "pip install torch --index-url https://download.pytorch.org/whl/rocm6.2"
        )

    return info


def check_nvidia_cuda() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "platform": "NVIDIA CUDA (DGX / RTX / Hopper)",
        "supported": False,
        "details": [],
        "warnings": [],
        "recommended_backend": "torch_structured",
    }

    try:
        import torch

        if torch.cuda.is_available() and not getattr(torch.version, "hip", None):
            cuda_ver = torch.version.cuda
            info["cuda_version"] = cuda_ver
            info["details"].append(f"PyTorch CUDA detected (v{cuda_ver}).")
            device_name = torch.cuda.get_device_name(0)
            info["device_name"] = device_name
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            info["vram_gb"] = round(vram_gb, 1)
            info["details"].append(f"Device: {device_name} ({vram_gb:.1f} GB VRAM)")

            if vram_gb < 23.5:
                info["warnings"].append(
                    f"VRAM: {vram_gb:.1f} GB. Loading 26B 4-bit model requires ≥ 24 GB VRAM.\n"
                    "Consider multi-GPU or CPU offload if running on smaller GPUs."
                )
            else:
                info["supported"] = True

            try:
                import flash_attn  # noqa: F401

                info["details"].append("FlashAttention-2 kernel detected.")
            except ImportError:
                info["details"].append("FlashAttention-2 optional kernel not installed. Using PyTorch SDPA.")
        else:
            info["details"].append("NVIDIA CUDA device not available in current environment.")
    except ImportError:
        info["warnings"].append("PyTorch not installed. Install with CUDA via: uv sync --extra cuda")

    return info


def main():
    print("=" * 70)
    print("DiffBrowse Multi-Platform Hardware Acceleration Inspector")
    print("=" * 70)
    print(f"OS: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python: {sys.version.split()[0]}")
    print("-" * 70)

    checks = [
        check_apple_silicon(),
        check_amd_strix_halo(),
        check_nvidia_cuda(),
    ]

    active_target = None
    for c in checks:
        print(f"\n[{c['platform']}]")
        for d in c["details"]:
            print(f"  [OK] {d}")
        for w in c["warnings"]:
            print(f"  [!]  {w}")
        if c["supported"]:
            active_target = c

    print("\n" + "=" * 70)
    if active_target:
        print(f"Primary Active Hardware Target: {active_target['platform']}")
        print(f"Recommended Backend: JEV_BACKEND={active_target['recommended_backend']}")
        print("=" * 70)
    else:
        print("No fully validated 24GB+ hardware target detected in this environment.")
        print("Check the warnings above to configure Apple Silicon, AMD ROCm, or NVIDIA CUDA.")
        print("=" * 70)


if __name__ == "__main__":
    main()
