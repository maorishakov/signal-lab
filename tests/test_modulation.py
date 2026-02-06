import numpy as np
import pytest

from signal_lab.modulation import qpsk_modulation, bpsk_modulation

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
#QPSK TESTS:

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