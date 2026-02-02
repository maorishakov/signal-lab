import numpy as np

from signal_lab.metrics import ber

def test_ber_zero_when_identical():
    bits = np.array([0, 1, 1, 0, 0, 0], dtype=np.uint8)
    bits_hat = bits.copy()
    assert ber(bits, bits_hat) == 0.0


def test_ber_half_errors():
    bits = np.array([0, 1, 1, 0], dtype=np.uint8)
    bits_hat = np.array([1, 1, 0, 0], dtype=np.uint8)
    assert ber(bits, bits_hat) == 0.5
