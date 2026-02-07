import numpy as np
import pytest

from signal_lab.generate import tone_iq, multi_tone_iq
from signal_lab.analysis import power, spectrum, dominant_freq


# single tone tests:
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

#-----------------------------------------------------------------------------------------------------------------------
# multi tone tests:


def test_multi_tone_shape_and_dtype():
    iq = multi_tone_iq(
        n=1024,
        f0_list=[0.12, -0.22, 0.31],
        phase0_list=[0.0, 0.1, -0.2],
        amplitude_list=[1.0, 0.5, 2.0],
        normalize=True,
    )
    assert iq.shape == (1024,)
    assert np.issubdtype(iq.dtype, np.complexfloating)


def test_multi_tone_rejects_n_non_positive():
    with pytest.raises(ValueError):
        multi_tone_iq(
            n=0,
            f0_list=[0.1],
            phase0_list=[0.0],
            amplitude_list=[1.0],
        )
    with pytest.raises(ValueError):
        multi_tone_iq(
            n=-5,
            f0_list=[0.1],
            phase0_list=[0.0],
            amplitude_list=[1.0],
        )


def test_multi_tone_requires_all_lists_not_none():
    with pytest.raises(ValueError):
        multi_tone_iq(n=256, f0_list=[0.1], phase0_list=[0.0], amplitude_list=None)
    with pytest.raises(ValueError):
        multi_tone_iq(n=256, f0_list=[0.1], phase0_list=None, amplitude_list=[1.0])
    with pytest.raises(ValueError):
        multi_tone_iq(n=256, f0_list=None, phase0_list=[0.0], amplitude_list=[1.0])


def test_multi_tone_rejects_empty_lists():
    with pytest.raises(ValueError):
        multi_tone_iq(n=256, f0_list=[], phase0_list=[], amplitude_list=[])


def test_multi_tone_requires_same_length_lists():
    with pytest.raises(ValueError):
        multi_tone_iq(n=256, f0_list=[0.1, 0.2], phase0_list=[0.0], amplitude_list=[1.0])
    with pytest.raises(ValueError):
        multi_tone_iq(n=256, f0_list=[0.1], phase0_list=[0.0, 0.0], amplitude_list=[1.0])
    with pytest.raises(ValueError):
        multi_tone_iq(n=256, f0_list=[0.1], phase0_list=[0.0], amplitude_list=[1.0, 1.0])


def test_multi_tone_rejects_f0_out_of_range():
    with pytest.raises(ValueError):
        multi_tone_iq(n=256, f0_list=[0.5], phase0_list=[0.0], amplitude_list=[1.0])   # 0.5 not allowed
    with pytest.raises(ValueError):
        multi_tone_iq(n=256, f0_list=[-0.5001], phase0_list=[0.0], amplitude_list=[1.0])
    with pytest.raises(ValueError):
        multi_tone_iq(n=256, f0_list=[0.5001], phase0_list=[0.0], amplitude_list=[1.0])


def test_multi_tone_rejects_non_finite_inputs():
    with pytest.raises(ValueError):
        multi_tone_iq(n=256, f0_list=[np.inf], phase0_list=[0.0], amplitude_list=[1.0])
    with pytest.raises(ValueError):
        multi_tone_iq(n=256, f0_list=[0.1], phase0_list=[np.nan], amplitude_list=[1.0])
    with pytest.raises(ValueError):
        multi_tone_iq(n=256, f0_list=[0.1], phase0_list=[0.0], amplitude_list=[np.inf])


def test_multi_tone_normalize_power_is_one():
    iq = multi_tone_iq(
        n=4096,
        f0_list=[0.12, -0.22, 0.31],
        phase0_list=[0.0, 0.1, -0.2],
        amplitude_list=[1.0, 0.5, 2.0],
        normalize=True,
    )
    assert power(iq) == pytest.approx(1.0, rel=0.0, abs=1e-10)


def test_multi_tone_without_normalize_has_higher_power_when_more_tones():
    iq1 = multi_tone_iq(
        n=4096,
        f0_list=[0.12],
        phase0_list=[0.0],
        amplitude_list=[1.0],
        normalize=False,
    )
    iq3 = multi_tone_iq(
        n=4096,
        f0_list=[0.12, -0.22, 0.31],
        phase0_list=[0.0, 0.1, -0.2],
        amplitude_list=[1.0, 0.5, 2.0],
        normalize=False,
    )
    assert power(iq3) > power(iq1)


def test_multi_tone_spectrum_peak_is_one_of_the_tones():
    nfft = 4096
    tones = [0.12, -0.22, 0.31]
    iq = multi_tone_iq(
        n=nfft,
        f0_list=tones,
        phase0_list=[0.0, 0.1, -0.2],
        amplitude_list=[1.0, 1.0, 1.0],
        normalize=True,
    )

    f, psd_db = spectrum(iq, nfft=nfft, window="hann")
    f_peak = dominant_freq(f, psd_db)

    df = 1.0 / nfft
    assert any(abs(f_peak - t) <= df for t in tones)