import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def constellation_points(iq: np.ndarray, max_points: int = 5000) -> tuple[np.ndarray, np.ndarray]:

    iq = np.asarray(iq)

    if not np.issubdtype(iq.dtype, np.complexfloating):
        raise ValueError("iq must be a complex array")

    if iq.ndim != 1:
        raise ValueError("iq must be a 1D array")

    if iq.size == 0:
        raise ValueError("iq must not be empty")

    if not np.all(np.isfinite(iq)):
        raise ValueError("iq contains INF or NaN values")

    max_points = int(max_points)
    if max_points <= 0:
        raise ValueError("max_points must be positive")

    if iq.size > max_points:
        step = int(np.ceil(iq.size / max_points))
        iq = iq[::step]

    I = iq.real
    Q = iq.imag

    return I, Q


def plot_constellation(iq: np.ndarray, *, max_points: int = 5000, title: str = "Constellation",
    out_path: str | Path | None = None, show: bool = False, equal_aspect: bool = True) -> Path | None:

    I, Q = constellation_points(iq, max_points=max_points)
    plt.figure()
    plt.scatter(I, Q, s=6)

    plt.axhline(0.0)
    plt.axvline(0.0)
    plt.grid(True, which="both")
    plt.xlabel("I (real)")
    plt.ylabel("Q (imag)")
    plt.title(title)

    m = 0.1
    xmax = np.max(np.abs(I)) * (1 + m)
    ymax = np.max(np.abs(Q)) * (1 + m)
    lim = max(xmax, ymax, 1e-6)
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)

    if equal_aspect:
        plt.gca().set_aspect("equal", adjustable="box")

    saved_path: Path | None = None
    if out_path is not None:
        p = Path(out_path)
        if p.parent != Path("."):
            p.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(p, dpi=150)
        saved_path = p

    if show:
        plt.show()
    plt.close()
    return saved_path






