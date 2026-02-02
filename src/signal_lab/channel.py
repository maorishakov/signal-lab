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
