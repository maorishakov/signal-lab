import numpy as np

"""
Generate complex IQ signal
"""


def tone_iq(n: int, f0: float, phase0: float = 0.0, amplitude: float = 1.0) -> np.ndarray:

    # n - number of samples
    if not isinstance(n, (int, np.integer)):
        raise ValueError("n must be an int")

    if n <= 0:
        raise ValueError("n must be positive")

    f0 = float(f0)
    phase0 = float(phase0)
    amplitude = float(amplitude)

    if f0 < -0.5 or f0 >= 0.5:
        raise ValueError("f0 must be between -0.5 to 0.5")

    if not np.isfinite(f0):
        raise ValueError("f0 cant be inf")

    if not np.isfinite(phase0):
        raise ValueError("phase0 cant be inf")

    if not np.isfinite(amplitude):
        raise ValueError("famplitude cant be inf")

    n_vec = np.arange(n)
    phase = 2 * np.pi * f0 * n_vec + phase0
    iq = amplitude * np.exp(1j * phase)

    return iq

