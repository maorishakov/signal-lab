import numpy as np
import pytest

from signal_lab.channel import awgn_channel


def test_awgn_measured_snr_close_to_target():
    rng = np.random.default_rng(123)

    x = (rng.choice([-1.0, 1.0], size=200000) + 1j * rng.choice([-1.0, 1.0], size=200000)) / np.sqrt(2)

    snr_db_target = 10.0
    y = awgn_channel(x, snr_db_target, rng=rng)

    signal_power = np.mean(np.abs(x) ** 2)
    noise_power = np.mean(np.abs(y - x) ** 2)

    snr_db_measured = 10.0 * np.log10(signal_power / noise_power)

    assert abs(snr_db_measured - snr_db_target) < 0.4


def test_awgn_high_snr_small_perturbation():
    rng = np.random.default_rng(0)
    x = np.ones(50_000, dtype=np.complex128)

    y = awgn_channel(x, 60.0, rng=rng)

    assert np.mean(np.abs(y - x)) < 1e-2


def test_awgn_non_1d_raises():
    rng = np.random.default_rng(0)
    x = np.ones((10, 2), dtype=np.complex128)

    with pytest.raises(ValueError):
        awgn_channel(x, 10.0, rng=rng)


def test_awgn_non_finite_symbols_raises():
    rng = np.random.default_rng(0)
    x = np.array([1 + 1j, np.nan + 1j], dtype=np.complex128)

    with pytest.raises(ValueError):
        awgn_channel(x, 10.0, rng=rng)


def test_awgn_non_finite_snr_raises():
    rng = np.random.default_rng(0)
    x = np.ones(10, dtype=np.complex128)

    with pytest.raises(ValueError):
        awgn_channel(x, float("nan"), rng=rng)
