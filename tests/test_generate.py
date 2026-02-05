import numpy as np
import pytest

from signal_lab.generate import tone_iq
from signal_lab.analysis import power


def test_tone_iq_length_and_dtype():
    iq = tone_iq(n=1000, f0=0.1)
    assert iq.shape == (1000,)
    assert np.issubdtype(iq.dtype, np.complexfloating)


def test_tone_iq_f0_zero_is_constant():
    n = 256
    phase0 = 0.7
    amp = 2.0
    iq = tone_iq(n=n, f0=0.0, phase0=phase0, amplitude=amp)

    expected = amp * np.exp(1j * phase0)
    assert np.allclose(iq, expected)


def test_tone_iq_power_matches_amplitude_squared():
    n = 4096
    amp = 3.0
    iq = tone_iq(n=n, f0=0.12, amplitude=amp)

    # For a pure tone with constant amplitude, mean(|x|^2) == amp^2
    assert power(iq) == pytest.approx(amp * amp, rel=0, abs=1e-12)


def test_tone_iq_invalid_n_raises():
    with pytest.raises(ValueError):
        tone_iq(n=0, f0=0.1)

    with pytest.raises(ValueError):
        tone_iq(n=-5, f0=0.1)


def test_tone_iq_non_finite_inputs_raise():
    with pytest.raises(ValueError):
        tone_iq(n=10, f0=float("nan"))

    with pytest.raises(ValueError):
        tone_iq(n=10, f0=0.1, phase0=float("inf"))

    with pytest.raises(ValueError):
        tone_iq(n=10, f0=0.1, amplitude=float("nan"))


def test_tone_iq_out_of_range_frequency_raises():
    with pytest.raises(ValueError):
        tone_iq(n=10, f0=0.6)

    with pytest.raises(ValueError):
        tone_iq(n=10, f0=-0.6)
