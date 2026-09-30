"""Fail-closed model loading for the maintained package (not a sandbox)."""

from pathlib import Path

from packaging.version import Version

MIN_TORCH = "2.14.0"


def load_weights(path, *, map_location="cpu"):
    """Never fall back to unrestricted pickle, including for legacy checkpoints."""
    import torch

    if Version(torch.__version__) < Version(MIN_TORCH):
        raise RuntimeError(f"Model loading requires torch>={MIN_TORCH}; upgrade the torch extra")
    return torch.load(path, map_location=map_location, weights_only=True)


def model_path(directory, filename):
    """A manifest may reference only files inside its own model directory."""
    root = Path(directory).resolve()
    relative = Path(filename)
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root):
        raise ValueError("Model manifest path escapes its model directory")
    return path
