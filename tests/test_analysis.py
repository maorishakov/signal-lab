import numpy as np
import pytest

from signal_lab.analysis import power, spectrum, dominant_freq, spectrogram


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


def test_spectrogram_shapes_and_axes():
    rng = np.random.default_rng(0)
    N = 4096
    nfft = 256
    hop = 128

    iq = rng.normal(size=N) + 1j * rng.normal(size=N)

    t, f, S_db = spectrogram(iq, nfft=nfft, hop=hop, window="hann")

    num_frames = 1 + (N - nfft) // hop

    assert t.shape == (num_frames,)
    assert f.shape == (nfft,)
    assert S_db.shape == (num_frames, nfft)

    # frequency axis should be normalized [-0.5, 0.5)
    assert np.isclose(f[0], -0.5)
    assert np.isclose(f[-1], 0.5 - 1 / nfft)


def test_spectrogram_tone_peak_frequency_is_correct_across_frames():
    N = 4096
    nfft = 256
    hop = 128
    f0 = 0.12  # normalized cycles/sample

    n = np.arange(N)
    iq = np.exp(1j * 2 * np.pi * f0 * n).astype(np.complex128)

    t, f, S_db = spectrogram(iq, nfft=nfft, hop=hop, window="hann")

    # For each frame, find the frequency bin of the maximum
    peak_bins = np.argmax(S_db, axis=1)
    peak_freqs = f[peak_bins]

    # FFT bin spacing
    df = 1.0 / nfft

    # Most frames should identify a peak close to f0 (within ~1 bin)
    assert np.mean(np.abs(peak_freqs - f0) <= df) > 0.9


def test_spectrogram_invalid_non_complex_raises():
    iq = np.ones(1024, dtype=np.float64)
    with pytest.raises(ValueError):
        spectrogram(iq, nfft=256, hop=128, window="hann")


def test_spectrogram_invalid_non_1d_raises():
    iq = np.ones((256, 4), dtype=np.complex128)
    with pytest.raises(ValueError):
        spectrogram(iq, nfft=256, hop=128, window="hann")


def test_spectrogram_non_finite_raises():
    iq = np.array([1 + 1j, np.nan + 1j], dtype=np.complex128)
    with pytest.raises(ValueError):
        spectrogram(iq, nfft=4, hop=2, window="hann")


def test_spectrogram_nfft_too_large_raises():
    iq = np.ones(100, dtype=np.complex128)
    with pytest.raises(ValueError):
        spectrogram(iq, nfft=256, hop=128, window="hann")


def test_spectrogram_invalid_hop_raises():
    iq = np.ones(1024, dtype=np.complex128)

    with pytest.raises(ValueError):
        spectrogram(iq, nfft=256, hop=0, window="hann")

    with pytest.raises(ValueError):
        spectrogram(iq, nfft=256, hop=300, window="hann")  # hop > nfft


def test_spectrogram_invalid_window_raises():
    iq = np.ones(1024, dtype=np.complex128)
    with pytest.raises(ValueError):
        spectrogram(iq, nfft=256, hop=128, window="blackman")