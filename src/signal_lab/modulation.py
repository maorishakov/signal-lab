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

    if not np.all(np.isfinite(bits)):
        raise ValueError("bits INF or NaN values")

    if bits.size % 2 != 0:
        raise ValueError("bits length must be even")

    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError("bits must be just 0 or 1!")

    bit_pairs = bits.reshape(len(bits) // 2, 2)
    mapping_dict = {(0, 0): 1 + 1j, (0, 1): -1 + 1j, (1, 1): -1 - 1j, (1, 0): 1 - 1j}

    symbols_list = []
    for pair in bit_pairs:
        key = tuple(pair)
        symbols_list.append(mapping_dict[key] / math.sqrt(2))

    symbols = np.array(symbols_list, dtype=np.complex128)
    return symbols


"""
BPSK mapping:
0 -> 1
1 -> -1
"""

def bpsk_modulation(bits: np.ndarray) -> np.ndarray:

    bits = np.asarray(bits)

    if not np.issubdtype(bits.dtype, np.number):
        raise ValueError("bits must be numeric 0 or 1")

    if bits.ndim != 1:
        raise ValueError("bits must be a 1D array")

    if not np.all(np.isfinite(bits)):
        raise ValueError("bits INF or NaN values")

    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError("bits must be just 0 or 1!")

    mapping_dict = {0: 1 + 0j, 1: -1 + 0j}

    symbols_list = []
    for bit in bits:
        symbols_list.append(mapping_dict[bit])

    symbols = np.array(symbols_list, dtype=np.complex128)
    return symbols


"""
16-QAM Gray mapping:

I/Q:
00 -> -3
01 -> -1
10 -> 1
11 -> 3

symbole - b0 b1 b2 b3
I - b0 b1
Q - b2 b3

Noramlization:

E[I^2] = (1^2 + 1^2 + 3^2 + 3^2)/4 = (1+1+9+9)/4 = 20/4 = 5
E[Q^2] = (1^2 + 1^2 + 3^2 + 3^2)/4 = (1+1+9+9)/4 = 20/4 = 5

Es = E[|s|^2] = E[I^2 + Q^2] = 5 + 5 = 10

norm = (I + jQ) / sqrt(10)
"""


def qam16_modulation(bits: np.ndarray) -> np.ndarray:

    bits = np.asarray(bits)

    if not np.issubdtype(bits.dtype, np.number):
        raise ValueError("bits must be numeric 0 or 1")

    if bits.ndim != 1:
        raise ValueError("bits must be a 1D array")

    if not np.all(np.isfinite(bits)):
        raise ValueError("bits INF or NaN values")

    if bits.size % 4 != 0:
        raise ValueError("bits length must be a multiple of 4")

    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError("bits must be just 0 or 1!")

    bits_blocks = bits.reshape(len(bits) // 4, 4)
    mapping_dict = {(0, 0): -3, (0, 1): -1, (1, 1): 1, (1, 0): 3}

    symbols_list = []
    for bits_block in bits_blocks:
        I = mapping_dict[(bits_block[0], bits_block[1])]
        Q = mapping_dict[(bits_block[2], bits_block[3])]
        symbols_list.append((I + 1j * Q) / math.sqrt(10))

    symbols = np.array(symbols_list, dtype=np.complex128)
    return symbols