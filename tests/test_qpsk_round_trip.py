import numpy as np
import pytest

from signal_lab.modulation import qpsk_modulation
from signal_lab.demodulation import qpsk_demodulation


def test_qpsk_round_trip_no_noise():
    rng = np.random.default_rng(seed=42)

    num_bits = 1000
    bits_tx = rng.integers(0, 2, size=num_bits, dtype=np.uint8)

    symbols = qpsk_modulation(bits_tx)
    bits_rx = qpsk_demodulation(symbols)

    assert np.array_equal(bits_rx, bits_tx)
