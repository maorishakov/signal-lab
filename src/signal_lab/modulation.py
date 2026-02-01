import math

import numpy as np

"""
QPSK Gray mapping:

00 -> (1,1)
01 -> (-1,1)
11 -> (-1,-1)
10 -> (1,-1)

Noramlization - > (I ּּ+ jQ)/sqrt(2)
"""

def qpsk_modulation(bits: np.ndarray) -> np.ndarray:

    bits = np.asarray(bits)

    if not np.issubdtype(bits.dtype, np.number):
        raise ValueError("bits must be numeric 0 or 1")

    if bits.ndim != 1:
        raise ValueError("bits must be a 1D array")

    if len(bits) % 2 != 0:
        raise ValueError("bits length must be even")

    bits_set = set(bits)
    for bit in bits_set:
        if bit != 0 and bit != 1:
            raise ValueError("bits must be just 0 or 1!")

    bit_pairs = bits.reshape(len(bits) // 2, 2)
    mapping_dict = {(0, 0): 1 + 1j, (0, 1): -1 + 1j, (1, 1): -1 - 1j, (1, 0): 1 - 1j}

    symbols_list = []
    for pair in bit_pairs:
        key = tuple(pair)
        symbols_list.append(mapping_dict[key] / math.sqrt(2))

    symbols = np.array(symbols_list, dtype=np.complex128)
    return symbols

