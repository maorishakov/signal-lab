from __future__ import annotations

from typing import Tuple
import numpy as np


def downsample_iq(iq: np.ndarray, max_points: int = 5000) -> np.ndarray:
    iq = np.asarray(iq)
    if iq.size <= max_points:
        return iq
    step = int(np.ceil(iq.size / max_points))
    return iq[::step]


def plot_constellation(ax, iq: np.ndarray, title: str, max_points: int = 5000) -> None:
    iq = downsample_iq(iq, max_points=max_points)
    I = iq.real
    Q = iq.imag

    ax.clear()
    ax.scatter(I, Q, s=6)
    ax.axhline(0.0)
    ax.axvline(0.0)
    ax.grid(True, which="both")
    ax.set_xlabel("I (real)")
    ax.set_ylabel("Q (imag)")
    ax.set_title(title)

    # auto bounds but keep sane
    lim = 0.0
    if I.size > 0:
        lim = float(max(np.max(np.abs(I)), np.max(np.abs(Q))))
    lim = max(lim, 1.5)
    lim = min(lim * 1.2, 10.0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    ax.figure.tight_layout()


def plot_psd(ax, f: np.ndarray, psd_db: np.ndarray, *, label: str, title: str) -> None:
    ax.plot(f, psd_db, label=label)
    ax.set_xlabel("Frequency (normalized)")
    ax.set_ylabel("PSD [dB]")
    ax.set_title(title)
    ax.grid(True, which="both")


def plot_spectrogram(
    ax,
    t: np.ndarray,
    f: np.ndarray,
    S_db: np.ndarray,
    *,
    title: str,
    vmin: float,
    vmax: float,
):
    S_plot = S_db - np.max(S_db)
    extent = [t[0], t[-1], f[0], f[-1]]

    ax.clear()
    im = ax.imshow(
        S_plot.T,
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("Time (frame index)")
    ax.set_ylabel("Frequency (normalized)")
    ax.set_title(title)

    return im


