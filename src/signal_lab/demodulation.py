import numpy as np


"""
I >= 0, Q >= 0 --> (0,0)
I < 0, Q >= 0 --> (0,1)
I < 0, Q < 0 --> (1,1)
I >= 0, Q < 0 --> (1,0)
"""


def qpsk_demodulation(symbols: np.ndarray) -> np.ndarray:

    symbols = np.asarray(symbols)

    if not np.issubdtype(symbols.dtype, np.complexfloating):
        raise ValueError("symbols must be complex number")

    if symbols.ndim != 1:
        raise ValueError("symbols must be a 1D array")

    if not np.all(np.isfinite(symbols)):
        raise ValueError("symbols contains INF or NaN values")

    Inphase = symbols.real
    Quadrature = symbols.imag

    bits_list = []

    for I, Q in zip(Inphase, Quadrature):
        if I >= 0 and Q >= 0:
            bits_list.extend([0,0])
        elif I < 0 and Q >= 0:
            bits_list.extend([0,1])
        elif I < 0 and Q < 0:
            bits_list.extend([1,1])
        else:
            bits_list.extend([1,0])

    bits = np.array(bits_list, dtype=np.uint8)
    return bits




