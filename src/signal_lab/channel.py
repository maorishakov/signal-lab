import math

import numpy as np

"""
Noise:
I ~ N(0, σ²)
Q ~ N(0, σ²)

sigma = sqrt(power_noise/2)
"""


def awgn_channel(symbols: np.ndarray, snr_db: float, rng: np.random.Generator = None) -> np.ndarray:

    symbols = np.asarray(symbols)
    snr_db = float(snr_db)

    if not np.isfinite(snr_db):
        raise ValueError("snr_db must be finite")

    if rng is None:
        rng = np.random.default_rng()

    if not np.issubdtype(symbols.dtype, np.complexfloating):
        raise ValueError("symbols must be complex number")

    if symbols.ndim != 1:
        raise ValueError("symbols must be a 1D array")

    if not np.all(np.isfinite(symbols)):
        raise ValueError("symbols contains INF or NaN values")

    signal_power_linear = np.mean(np.abs(symbols)**2)
    # signal_power_log = 10 * np.log10(signal_power_linear)

    snr_linear = 10 ** (snr_db / 10)

    noise_power = signal_power_linear / snr_linear

    sigma = math.sqrt(noise_power / 2)

    I = rng.normal(0, sigma, size=symbols.shape)
    Q = rng.normal(0, sigma, size=symbols.shape)

    noise = I + 1j * Q

    y = symbols + noise

    return y.astype(np.complex128, copy=False)

"""
CFO - Carrier Frequency Offset
"""

def cfo_channel(iq: np.ndarray, f_cfo: float, phase0: float = 0.0) -> np.ndarray:

    iq = np.asarray(iq)

    if not np.issubdtype(iq.dtype, np.complexfloating):
        raise ValueError("iq must be complex array")

    if iq.ndim != 1:
        raise ValueError("iq must be a 1D array")

    if iq.size == 0:
        raise ValueError("iq must not be empty")

    if not np.all(np.isfinite(iq)):
        raise ValueError("iq contains INF or NaN values")

    f_cfo = float(f_cfo)
    phase0 = float(phase0)

    if not np.isfinite(f_cfo):
        raise ValueError("f0 must be finite")

    if f_cfo <= -0.5 or f_cfo >= 0.5:
        raise ValueError("f0 must be between -0.5 to 0.5")

    if not np.isfinite(phase0):
        raise ValueError("phase0 cant be inf")


    n = np.arange(len(iq))
    phase = 2 * np.pi * f_cfo * n + phase0
    phase_rotation_vector = np.exp(1j * phase)

    iq_cfo = iq * phase_rotation_vector

    return iq_cfo

"""
Phase Noise

y[n]=x[n]⋅ejϕ[n]
ϕ[n]=ϕ[n−1]+Δϕ[n]
Δϕ[n]∼N(0,σ^2)

sigma - σ

"""


def phase_noise_channel(iq: np.ndarray, sigma: float, seed: int | None = None, rng: np.random.Generator | None = None) -> np.ndarray:

    iq = np.asarray(iq)

    if not np.issubdtype(iq.dtype, np.complexfloating):
        raise ValueError("iq must be complex array")

    if iq.ndim != 1:
        raise ValueError("iq must be a 1D array")

    if iq.size == 0:
        raise ValueError("iq must not be empty")

    if not np.all(np.isfinite(iq)):
        raise ValueError("iq contains INF or NaN values")

    sigma = float(sigma)

    if sigma < 0:
        raise ValueError("sigma must be >= 0")

    if not np.isfinite(sigma):
        raise ValueError("sigma must be finite")

    if rng is None:
        rng = np.random.default_rng(seed)

    delta_phase = rng.normal(loc=0.0, scale=sigma, size=iq.size)
    total_phase = np.cumsum(delta_phase)
    rotation_vector = np.exp(1j * total_phase)

    iq_phase_noise = iq * rotation_vector

    return iq_phase_noise


def timing_offset_channel(iq: np.ndarray, offset: int, fill: str = "zeros") -> np.ndarray:
    """
    Apply integer timing offset (sample delay/advance) to a complex IQ signal.

    Parameters
    ----------
    offset : int
        Timing offset in samples.
        offset > 0 : delay (shift right)
        offset < 0 : advance (shift left)
    fill : str
        How to fill missing samples. Supported:
        - "zeros" : fill with zeros (default)
        - "wrap"  : circular shift (non-physical, optional)
    """

    iq = np.asarray(iq)

    if not np.issubdtype(iq.dtype, np.complexfloating):
        raise ValueError("iq must be a complex array")

    if iq.ndim != 1:
        raise ValueError("iq must be a 1D array")

    if iq.size == 0:
        raise ValueError("iq must not be empty")

    if not np.all(np.isfinite(iq)):
        raise ValueError("iq contains INF or NaN values")

    if not isinstance(offset, (int, np.integer)):
        raise ValueError("offset must be an integer")

    fill = str(fill).lower()
    if fill not in {"zeros", "wrap"}:
        raise ValueError('fill must be either "zeros" or "wrap"')

    if offset == 0:
        return iq.astype(np.complex128, copy=False)

    N = iq.size

    if fill == "wrap":
        return np.roll(iq, shift=offset).astype(np.complex128, copy=False)

    y = np.zeros_like(iq, dtype=np.complex128)

    if offset > 0:
        # delay: shift right
        if offset < N:
            y[offset:] = iq[:-offset]
        # else: all zeros

    else:
        # advance: shift left
        k = -offset
        if k < N:
            y[:-k] = iq[k:]
        # else: all zeros

    return y
