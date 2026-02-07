import numpy as np

from signal_lab.analysis import power

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


def multi_tone_iq(n, f0_list, phase0_list=None, amplitude_list=None, normalize=True) -> np.ndarray:

    # n - number of samples
    if not isinstance(n, (int, np.integer)):
        raise ValueError("n must be an int")

    if n <= 0:
        raise ValueError("n must be positive")

    if f0_list is None or phase0_list is None or amplitude_list is None:
        raise ValueError("one of the lists is none")

    if len(f0_list) == 0 or len(phase0_list) == 0 or len(amplitude_list) == 0:
        raise ValueError("one of the lists is empty")

    if not len(f0_list) == len(phase0_list) == len(amplitude_list):
        raise ValueError("f0_list, phase0_list and amplitude_list must be the same length")

    for f0 in f0_list:
        if f0 < -0.5 or f0 >= 0.5:
            raise ValueError("f0 must be between -0.5 to 0.5")

        if not np.isfinite(f0):
            raise ValueError("f0 must be finite")

    for phase0 in phase0_list:
        if not np.isfinite(phase0):
            raise ValueError("phase0 must be finite")

    for amplitude in amplitude_list:
        if not np.isfinite(amplitude):
            raise ValueError("Amplitude must be finite")

    iq = np.zeros(n, dtype=np.complex128)
    n_vec = np.arange(n)
    for i in range(len(f0_list)):
        phase = 2 * np.pi * f0_list[i] * n_vec + phase0_list[i]
        iq += amplitude_list[i] * np.exp(1j * phase)

    iq_power = power(iq)
    if not normalize or iq_power == 0:
        return iq

    iq_normalize = iq / np.sqrt(iq_power)

    return iq_normalize
