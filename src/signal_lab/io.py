from __future__ import annotations

from pathlib import Path
import numpy as np


def save_iq_npy(path: str | Path, iq: np.ndarray) -> Path:
    """
    Save complex 1D IQ array to a .npy file.
    returns: Path of the saved file
    """
    p = Path(path)

    # if user gave "file" without suffix -> add .npy
    if p.suffix == "":
        p = p.with_suffix(".npy")

    if p.suffix.lower() != ".npy":
        raise ValueError("path must end with .npy (or have no suffix)")

    iq = np.asarray(iq)

    if not np.issubdtype(iq.dtype, np.complexfloating):
        raise ValueError("iq must be complex array")

    if iq.ndim != 1:
        raise ValueError("iq must be a 1D array")

    if iq.size == 0:
        raise ValueError("iq must not be empty")

    if not np.all(np.isfinite(iq)):
        raise ValueError("iq contains INF or NaN values")

    # Create parent directories if needed
    if p.parent != Path("."):
        p.parent.mkdir(parents=True, exist_ok=True)

    np.save(p, iq)  # stores dtype + shape
    return p


def load_iq_npy(path: str | Path) -> np.ndarray:
    """
    Load complex 1D IQ array from a .npy file.
    """
    p = Path(path)

    if p.suffix.lower() != ".npy":
        raise ValueError("path must end with .npy")

    if not p.exists():
        raise FileNotFoundError(f"file not found: {p}")

    iq = np.load(p, allow_pickle=False)

    if not isinstance(iq, np.ndarray):
        raise ValueError("loaded object is not a numpy array")

    if iq.ndim != 1:
        raise ValueError("loaded iq must be a 1D array")

    if not np.issubdtype(iq.dtype, np.complexfloating):
        raise ValueError("loaded iq must be complex array")

    if iq.size == 0:
        raise ValueError("loaded iq must not be empty")

    if not np.all(np.isfinite(iq)):
        raise ValueError("loaded iq contains INF or NaN values")

    return iq