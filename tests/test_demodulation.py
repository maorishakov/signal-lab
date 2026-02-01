import pytest
import numpy as np

from signal_lab.demodulation import qpsk_demodulation


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
        1 - 1j,
    ], dtype=np.complex128) / np.sqrt(2)

    bits = qpsk_demodulation(symbols)

    expected = np.array([0,0, 0,1, 1,1, 1,0], dtype=int)
    assert np.array_equal(bits, expected)
