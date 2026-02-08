import numpy as np

"""
QPSK hard decison:
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
            bits_list.extend([0, 0])
        elif I < 0 and Q >= 0:
            bits_list.extend([0, 1])
        elif I < 0 and Q < 0:
            bits_list.extend([1, 1])
        else:
            bits_list.extend([1, 0])

    bits = np.array(bits_list, dtype=np.uint8)
    return bits


"""
BPSK hard decison:
I >= 0 --> 0
I < 0 --> 1
"""


def bpsk_demodulation(symbols: np.ndarray) -> np.ndarray:
    symbols = np.asarray(symbols)

    if not np.issubdtype(symbols.dtype, np.complexfloating):
        raise ValueError("symbols must be complex number")

    if symbols.ndim != 1:
        raise ValueError("symbols must be a 1D array")

    if not np.all(np.isfinite(symbols)):
        raise ValueError("symbols contains INF or NaN values")

    Inphase = symbols.real

    bits_list = []

    for I in Inphase:
        if I >= 0:
            bits_list.append(0)
        else:
            bits_list.append(1)

    bits = np.array(bits_list, dtype=np.uint8)
    return bits


def qam16_demodulation(symbols: np.ndarray) -> np.ndarray:
    symbols = np.asarray(symbols)

    if not np.issubdtype(symbols.dtype, np.complexfloating):
        raise ValueError("symbols must be complex number")

    if symbols.ndim != 1:
        raise ValueError("symbols must be a 1D array")

    if not np.all(np.isfinite(symbols)):
        raise ValueError("symbols contains INF or NaN values")

    # symbols scaled
    symbols = symbols * np.sqrt(10)

    levels = np.array([-3, -1, 1, 3], dtype=np.int64)

    Inphase = symbols.real
    Quadrature = symbols.imag

    inverse_mapping = {-3: (0, 0), -1: (0, 1), 1: (1, 1), 3: (1, 0)}

    bits_list = []

    for I, Q in zip(Inphase, Quadrature):

        index_I = np.argmin(np.abs(I - levels))
        I_hat = levels[index_I]
        b0_b1 = inverse_mapping[I_hat]

        index_Q = np.argmin(np.abs(Q - levels))
        Q_hat = levels[index_Q]
        b2_b3 = inverse_mapping[Q_hat]

        bits_list.extend([b0_b1[0], b0_b1[1], b2_b3[0], b2_b3[1]])

    bits_hat = np.array(bits_list, dtype=np.uint8)

    return bits_hat
