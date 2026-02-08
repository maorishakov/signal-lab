import pytest
import numpy as np
import math

from signal_lab.demodulation import qpsk_demodulation, bpsk_demodulation, qam16_demodulation
from signal_lab.modulation import qam16_modulation


#-----------------------------------------------------------------------------------------------------------------------
# QPSK TESTS:

def test_qpsk_demodulation_quadrants_basic():
    symbols = np.array([1+1j, -1+1j, -1-1j, 1-1j], dtype=np.complex128)
    bits = qpsk_demodulation(symbols)
    expected = np.array([0,0, 0,1, 1,1, 1,0], dtype=np.uint8)
    assert np.array_equal(bits, expected)


def test_qpsk_demodulation_zero_boundaries():
    symbols = np.array([0+0j, 0+2j, 2+0j, 0-2j, -2+0j], dtype=np.complex128)
    bits = qpsk_demodulation(symbols)
    # לפי ההגדרה שלך: I>=0/Q>=0 => 00 ; I>=0/Q<0 => 10 ; I<0/Q>=0 => 01
    expected = np.array([
        0,0,  # 0+0j
        0,0,  # 0+2j
        0,0,  # 2+0j
        1,0,  # 0-2j (I>=0, Q<0)
        0,1   # -2+0j (I<0, Q>=0)
    ], dtype=np.uint8)
    assert np.array_equal(bits, expected)


def test_qpsk_demodulation_output_length_is_2n():
    n = 17
    symbols = (np.random.randn(n) + 1j*np.random.randn(n)).astype(np.complex128)
    bits = qpsk_demodulation(symbols)
    assert bits.shape == (2*n,)


def test_qpsk_demodulation_output_dtype_uint8():
    symbols = np.array([1+1j], dtype=np.complex64)
    bits = qpsk_demodulation(symbols)
    assert bits.dtype == np.uint8


def test_qpsk_demodulation_rejects_real_input():
    symbols = np.array([1.0, -1.0], dtype=np.float64)
    with pytest.raises(ValueError, match="must be complex"):
        qpsk_demodulation(symbols)


def test_qpsk_demodulation_rejects_non_1d():
    symbols = np.array([[1+1j, -1+1j]], dtype=np.complex128)
    with pytest.raises(ValueError, match="1D"):
        qpsk_demodulation(symbols)


def test_qpsk_demodulation_rejects_nan_inf():
    symbols_nan = np.array([1+1j, np.nan+1j], dtype=np.complex128)
    with pytest.raises(ValueError, match="INF or NaN"):
        qpsk_demodulation(symbols_nan)

    symbols_inf = np.array([1+1j, 1+1j*np.inf], dtype=np.complex128)
    with pytest.raises(ValueError, match="INF or NaN"):
        qpsk_demodulation(symbols_inf)


def test_qpsk_demodulation_empty_input():
    symbols = np.array([], dtype=np.complex128)
    bits = qpsk_demodulation(symbols)
    assert bits.size == 0
    assert bits.dtype == np.uint8


def test_qpsk_known_demapping():
    symbols = np.array([
        1 + 1j,
        -1 + 1j,
        -1 - 1j,
        1 - 1j], dtype=np.complex128) / np.sqrt(2)

    bits = qpsk_demodulation(symbols)

    expected = np.array([0,0, 0,1, 1,1, 1,0], dtype=int)
    assert np.array_equal(bits, expected)

#-----------------------------------------------------------------------------------------------------------------------
# BPSK TESTS:

def test_bpsk_demodulation_basic_mapping():
    # Expect decision on real axis:
    # real >= 0 -> 0, real < 0 -> 1
    symbols = np.array([1+0j, -1+0j, 2+0j, -0.1+0j], dtype=np.complex128)
    bits = bpsk_demodulation(symbols)

    expected = np.array([0, 1, 0, 1], dtype=np.uint8)
    assert bits.dtype == np.uint8
    assert bits.shape == (4,)
    assert np.all(bits == expected)


def test_bpsk_demodulation_ignores_imag_part():
    # Imag part should not affect decision (still based on real sign)
    symbols = np.array([1 + 5j, 1 - 9j, -1 + 3j, -2 - 0.5j], dtype=np.complex128)
    bits = bpsk_demodulation(symbols)
    expected = np.array([0, 0, 1, 1], dtype=np.uint8)
    assert np.all(bits == expected)


def test_bpsk_round_trip_with_modulation():
    # Round-trip sanity: bits -> BPSK -> demod -> same bits
    from signal_lab.modulation import bpsk_modulation

    bits = np.array([0, 1, 1, 0, 0, 1, 0, 1], dtype=np.uint8)
    symbols = bpsk_modulation(bits)
    bits_hat = bpsk_demodulation(symbols)

    assert np.array_equal(bits_hat, bits)


def test_bpsk_demodulation_non_complex_raises():
    x = np.array([1.0, -1.0, 0.5], dtype=float)
    with pytest.raises(ValueError):
        bpsk_demodulation(x)


def test_bpsk_demodulation_non_1d_raises():
    x = np.ones((10, 2), dtype=np.complex128)
    with pytest.raises(ValueError):
        bpsk_demodulation(x)


def test_bpsk_demodulation_non_finite_raises():
    x = np.array([1+0j, np.nan + 0j], dtype=np.complex128)
    with pytest.raises(ValueError):
        bpsk_demodulation(x)


#-----------------------------------------------------------------------------------------------------------------------
# 16-QAM TESTS:


def test_qam16_demodulation_known_points():
    # Build symbols from exact constellation points (already normalized /sqrt(10))
    symbols = np.array(
        [
            (-3 - 3j) / math.sqrt(10),
            (-1 + 1j) / math.sqrt(10),
            ( 3 - 1j) / math.sqrt(10),
            ( 1 + 3j) / math.sqrt(10),
        ],
        dtype=np.complex128,
    )

    bits_hat = qam16_demodulation(symbols)

    expected_bits = np.array(
        [
            0, 0, 0, 0,   # -3, -3
            0, 1, 1, 1,   # -1, +1
            1, 0, 0, 1,   # +3, -1
            1, 1, 1, 0,   # +1, +3
        ],
        dtype=np.uint8,
    )

    assert np.array_equal(bits_hat, expected_bits)


def test_qam16_demodulation_round_trip_no_noise():
    rng = np.random.default_rng(0)
    num_bits = 40_000  # must be multiple of 4
    bits_tx = rng.integers(0, 2, size=num_bits, dtype=np.uint8)

    symbols = qam16_modulation(bits_tx)
    bits_rx = qam16_demodulation(symbols)

    assert np.array_equal(bits_rx, bits_tx)


def test_qam16_demodulation_non_1d_raises():
    x = np.ones((10, 2), dtype=np.complex128)
    with pytest.raises(ValueError):
        qam16_demodulation(x)


def test_qam16_demodulation_non_complex_raises():
    x = np.ones(10, dtype=np.float64)
    with pytest.raises(ValueError):
        qam16_demodulation(x)


def test_qam16_demodulation_non_finite_symbols_raises():
    x = np.array([1 + 1j, np.nan + 1j], dtype=np.complex128)
    with pytest.raises(ValueError):
        qam16_demodulation(x)
