import numpy as np
import pytest
import math

from signal_lab.modulation import qpsk_modulation, bpsk_modulation, qam16_modulation

# QPSK TESTS:
def test_qpsk_accepts_python_list():
    bits = [0, 0, 0, 1]
    symbols = qpsk_modulation(bits)
    assert symbols.shape == (2,)
    assert np.iscomplexobj(symbols)


def test_qpsk_odd_length_raises():
    bits = [0, 1, 0]
    with pytest.raises(ValueError):
        qpsk_modulation(bits)


def test_qpsk_non_binary_raises():
    bits = [0, 2, 1, 0]
    with pytest.raises(ValueError):
        qpsk_modulation(bits)


def test_qpsk_non_1d_raises():
    bits = np.array([[0, 1], [1, 0]])
    with pytest.raises(ValueError):
        qpsk_modulation(bits)


def test_qpsk_non_numeric_raises():
    bits = ["0", "1", "0", "1"]
    with pytest.raises(ValueError):
        qpsk_modulation(bits)


def test_qpsk_known_mapping():
    # Bits arranged as: 00, 01, 11, 10
    bits = np.array([0, 0, 0, 1, 1, 1, 1, 0], dtype=int)

    symbols = qpsk_modulation(bits)

    expected = np.array([
        1 + 1j,   # 00
        -1 + 1j,  # 01
        -1 - 1j,  # 11
        1 - 1j,   # 10
    ], dtype=np.complex128) / np.sqrt(2)

    assert np.allclose(symbols, expected)

#-----------------------------------------------------------------------------------------------------------------------
#BPSK TESTS:

def test_bpsk_modulation_mapping_basic():
    bits = np.array([0, 1, 0, 1], dtype=np.uint8)
    sym = bpsk_modulation(bits)

    assert sym.dtype == np.complexfloating or np.issubdtype(sym.dtype, np.complexfloating)
    assert sym.shape == (4,)

    # Expect mapping: 0 -> +1, 1 -> -1  (as real axis)
    expected = np.array([1+0j, -1+0j, 1+0j, -1+0j], dtype=np.complex128)
    assert np.all(sym == expected)


def test_bpsk_modulation_output_is_complex128():
    bits = np.array([0, 1, 1, 0], dtype=np.uint8)
    sym = bpsk_modulation(bits)
    assert sym.dtype == np.complex128


def test_bpsk_modulation_non_numeric_dtype_raises():
    bits = np.array(["0", "1", "0"], dtype=object)
    with pytest.raises(ValueError):
        bpsk_modulation(bits)


def test_bpsk_modulation_non_1d_raises():
    bits = np.zeros((2, 2), dtype=np.uint8)
    with pytest.raises(ValueError):
        bpsk_modulation(bits)


def test_bpsk_modulation_non_finite_raises():
    bits = np.array([0.0, np.nan, 1.0], dtype=float)
    with pytest.raises(ValueError):
        bpsk_modulation(bits)


@pytest.mark.parametrize("bad_bits", [
    np.array([2, 0, 1], dtype=np.int64),
    np.array([-1, 0, 1], dtype=np.int64),
    np.array([0, 1, 0.5], dtype=float),
])
def test_bpsk_modulation_values_not_0_or_1_raise(bad_bits):
    with pytest.raises(ValueError):
        bpsk_modulation(bad_bits)


#-----------------------------------------------------------------------------------------------------------------------
#16-QAM TESTS:

def test_qam16_modulation_output_dtype_and_shape():
    bits = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.uint8)  # 2 symbols
    symbols = qam16_modulation(bits)

    assert symbols.dtype == np.complex128
    assert symbols.ndim == 1
    assert symbols.shape == (2,)


def test_qam16_modulation_known_mapping_basic():
    # Using Gray per-axis mapping:
    # 00 -> -3, 01 -> -1, 11 -> +1, 10 -> +3
    # bits [b0 b1 b2 b3] => I from b0b1, Q from b2b3

    bits = np.array(
        [
            0, 0, 0, 0,   # I=-3, Q=-3 => -3-3j
            0, 1, 1, 1,   # I=-1, Q=+1 => -1+1j
            1, 0, 0, 1,   # I=+3, Q=-1 =>  3-1j
            1, 1, 1, 0,   # I=+1, Q=+3 =>  1+3j
        ],
        dtype=np.uint8,
    )

    symbols = qam16_modulation(bits)

    expected = np.array(
        [
            (-3 - 3j) / math.sqrt(10),
            (-1 + 1j) / math.sqrt(10),
            ( 3 - 1j) / math.sqrt(10),
            ( 1 + 3j) / math.sqrt(10),
        ],
        dtype=np.complex128,
    )

    assert np.allclose(symbols, expected, rtol=0, atol=1e-12)


def test_qam16_modulation_normalized_average_power_is_one():
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=40_000, dtype=np.uint8)  # multiple of 4? 40k is multiple of 4
    symbols = qam16_modulation(bits)

    p = np.mean(np.abs(symbols) ** 2)
    assert p == pytest.approx(1.0, rel=0.01, abs=0.0)  # 1% tolerance


def test_qam16_modulation_non_numeric_raises():
    bits = np.array(["0", "1", "0", "1"], dtype=object)
    with pytest.raises(ValueError):
        qam16_modulation(bits)


def test_qam16_modulation_not_1d_raises():
    bits = np.zeros((2, 4), dtype=np.uint8)
    with pytest.raises(ValueError):
        qam16_modulation(bits)


def test_qam16_modulation_non_finite_raises():
    bits = np.array([0.0, 1.0, np.nan, 0.0], dtype=float)
    with pytest.raises(ValueError):
        qam16_modulation(bits)


def test_qam16_modulation_length_not_multiple_of_4_raises():
    bits = np.array([0, 1, 0, 1, 1, 0], dtype=np.uint8)  # length 6
    with pytest.raises(ValueError):
        qam16_modulation(bits)


def test_qam16_modulation_bits_not_0_1_raises():
    bits = np.array([0, 1, 2, 0], dtype=np.int64)
    with pytest.raises(ValueError):
        qam16_modulation(bits)


