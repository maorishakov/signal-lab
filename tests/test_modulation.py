import numpy as np
import pytest

from signal_lab.modulation import qpsk_modulation

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

