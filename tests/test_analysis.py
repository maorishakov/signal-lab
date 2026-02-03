import numpy as np
import pytest

from signal_lab.analysis import power, spectrum, dominant_freq


def test_power_ones_is_one():
    iq = np.ones(1024, dtype=np.complex128)
    p = power(iq)
    assert p == pytest.approx(1.0, rel=0, abs=1e-12)


def test_spectrum_shapes_and_finite():
    rng = np.random.default_rng(0)
    iq = (rng.normal(size=2048) + 1j * rng.normal(size=2048)).astype(np.complex128)

    nfft = 4096
    f, psd_db = spectrum(iq, nfft=nfft, window="hann")

    assert f.shape == (nfft,)
    assert psd_db.shape == (nfft,)
    assert np.all(np.isfinite(f))
    assert np.all(np.isfinite(psd_db))


def test_dominant_freq_tone_close_to_f0():
    f0 = 0.125
    n = 2048
    nfft = 4096

    n_idx = np.arange(n)
    iq = np.exp(1j * 2.0 * np.pi * f0 * n_idx).astype(np.complex128)

    f, psd_db = spectrum(iq, nfft=nfft, window="rect")
    f_peak = dominant_freq(f, psd_db)

    bin_width = 1.0 / nfft
    assert abs(f_peak - f0) <= 2 * bin_width


def test_spectrum_non_complex_raises():
    iq = np.ones(10, dtype=np.float64)
    with pytest.raises(ValueError):
        spectrum(iq)


def test_spectrum_non_1d_raises():
    iq = np.ones((10, 2), dtype=np.complex128)
    with pytest.raises(ValueError):
        spectrum(iq)


def test_dominant_freq_shape_mismatch_raises():
    f = np.linspace(-0.5, 0.5, 8, endpoint=False)
    psd_db = np.zeros(7)
    with pytest.raises(ValueError):
        dominant_freq(f, psd_db)
