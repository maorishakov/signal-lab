import numpy as np

"""
bits → modulation → AWGN → demodulation → bits_hat
"""

def ber(bits: np.ndarray, bits_hat: np.ndarray) -> float:

    bits = np.asarray(bits)
    bits_hat = np.asarray(bits_hat)

    if bits.shape != bits_hat.shape:
        raise ValueError("bits and bits hat not the same shape")

    if bits.ndim != 1 or bits_hat.ndim != 1:
        raise ValueError("bits and bits hat must be 1D array")

    # errors = bits ^ bits_hat
    # errors_count = np.sum(errors == 1)

    errors_count = np.count_nonzero(bits != bits_hat)

    return float (errors_count / len(bits))
