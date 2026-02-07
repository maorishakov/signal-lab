import numpy as np
import pytest

from signal_lab.channel import awgn_channel, cfo_channel
from signal_lab.analysis import power

# AWGN TESTS:
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


#-----------------------------------------------------------------------------------------------------------------------
# CFO TESTS:

def test_cfo_channel_zero_cfo_is_identity():
    rng = np.random.default_rng(0)
    iq = rng.normal(size=1024) + 1j * rng.normal(size=1024)

    out = cfo_channel(iq, f_cfo=0.0, phase0=0.0)

    assert np.allclose(out, iq, atol=0.0, rtol=0.0)


def test_cfo_channel_preserves_power():
    rng = np.random.default_rng(1)
    iq = rng.normal(size=4096) + 1j * rng.normal(size=4096)

    p_in = power(iq)
    out = cfo_channel(iq, f_cfo=0.123, phase0=0.7)
    p_out = power(out)

    assert p_out == pytest.approx(p_in, rel=0.0, abs=1e-12)


# def test_cfo_channel_constant_phase_rotation_when_cfo_zero():
#     # if f_cfo=0, then it is just a constant phase rotation exp(1j*phase0)
#     rng = np.random.default_rng(2)
#     iq = rng.normal(size=256) + 1j * rng.normal(size=256)
#
#     phase0 = 1.234
#     out = cfo_channel(iq, f_cfo=0.0, phase0=phase0)
#     expected = iq * np.exp(1j * phase0)
#
#     assert np.allclose(out, expected, atol=0.0, rtol=0.0)


def test_cfo_channel_known_rotation_matches_formula():
    # Use an all-ones input so output should equal exp(j*(2*pi*f*n + phase0))
    n = 64
    iq = np.ones(n, dtype=np.complex128)

    f_cfo = 0.2
    phase0 = -0.3
    out = cfo_channel(iq, f_cfo=f_cfo, phase0=phase0)

    k = np.arange(n)
    expected = np.exp(1j * (2 * np.pi * f_cfo * k + phase0))

    assert np.allclose(out, expected, atol=0.0, rtol=0.0)


def test_cfo_channel_input_validation_non_complex_raises():
    x = np.ones(16, dtype=np.float64)
    with pytest.raises(ValueError):
        cfo_channel(x, f_cfo=0.1)


def test_cfo_channel_input_validation_not_1d_raises():
    x = np.ones((4, 4), dtype=np.complex128)
    with pytest.raises(ValueError):
        cfo_channel(x, f_cfo=0.1)


def test_cfo_channel_input_validation_empty_raises():
    x = np.array([], dtype=np.complex128)
    with pytest.raises(ValueError):
        cfo_channel(x, f_cfo=0.1)


def test_cfo_channel_input_validation_non_finite_raises():
    x = np.ones(16, dtype=np.complex128)
    x[3] = np.nan + 1j
    with pytest.raises(ValueError):
        cfo_channel(x, f_cfo=0.1)


@pytest.mark.parametrize("bad_f", [-0.5, 0.5, 1.0, -1.0])
def test_cfo_channel_f_cfo_out_of_range_raises(bad_f):
    x = np.ones(16, dtype=np.complex128)
    with pytest.raises(ValueError):
        cfo_channel(x, f_cfo=bad_f)


def test_cfo_channel_phase0_must_be_finite():
    x = np.ones(16, dtype=np.complex128)
    with pytest.raises(ValueError):
        cfo_channel(x, f_cfo=0.1, phase0=np.inf)