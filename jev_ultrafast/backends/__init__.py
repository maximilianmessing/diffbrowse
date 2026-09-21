"""Decision backends for jev-ultrafast."""

from .base import Decision, DecisionBackend, calculate_entropy, calculate_top2_margin
from .jev import JevBackend


def __getattr__(name: str):
    if name == "MlxDiffusionDirectBackend":
        from .mlx_direct import MlxDiffusionDirectBackend

        return MlxDiffusionDirectBackend
    if name == "MlxDiffusionGenerateBackend":
        from .mlx_generate import MlxDiffusionGenerateBackend

        return MlxDiffusionGenerateBackend
    if name == "MlxAutoregressiveBackend":
        from .mlx_autoregressive import MlxAutoregressiveBackend

        return MlxAutoregressiveBackend
    if name == "HybridBackend":
        from .hybrid import HybridBackend

        return HybridBackend
    if name == "MlxDiffusionStructuredBackend":
        from .mlx_structured import MlxDiffusionStructuredBackend

        return MlxDiffusionStructuredBackend
    if name == "TorchDiffusionDirectBackend":
        from .torch_direct import TorchDiffusionDirectBackend

        return TorchDiffusionDirectBackend
    if name == "TorchDiffusionStructuredBackend":
        from .torch_structured import TorchDiffusionStructuredBackend

        return TorchDiffusionStructuredBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Decision",
    "DecisionBackend",
    "JevBackend",
    "HybridBackend",
    "MlxDiffusionDirectBackend",
    "MlxDiffusionStructuredBackend",
    "MlxDiffusionGenerateBackend",
    "MlxAutoregressiveBackend",
    "TorchDiffusionDirectBackend",
    "TorchDiffusionStructuredBackend",
    "calculate_entropy",
    "calculate_top2_margin",
]
