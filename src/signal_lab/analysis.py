import numpy as np

"""
IQ baseband analysis
"""


def power(iq: np.ndarray) -> float:
    iq = np.asarray(iq)

    if not np.issubdtype(iq.dtype, np.complexfloating):
        raise ValueError("iq must be complex number")

    if iq.ndim != 1:
        raise ValueError("iq must be a 1D array")

    if not np.all(np.isfinite(iq)):
        raise ValueError("iq contains INF or NaN values")

    power = np.mean(np.abs(iq) ** 2)

    return float(power)


"""
return (f, psd_db)
f normalized [-0.5, 0.5]
psd_db length: nfft
window support: "rect", "hann"
"""


def spectrum(iq: np.ndarray, nfft: int = 4096, window: str = "hann") -> tuple[np.ndarray, np.ndarray]:
    iq = np.asarray(iq)

    if not np.issubdtype(iq.dtype, np.complexfloating):
        raise ValueError("iq must be complex number")

    if iq.ndim != 1:
        raise ValueError("iq must be a 1D array")

    if not np.all(np.isfinite(iq)):
        raise ValueError("iq contains INF or NaN values")

    n = iq.size
    window = window.replace(" ", "").lower()

    if window == "rect":
        w = np.ones(n)

    elif window == "hann":
        w = np.hanning(n)

    else:
        raise ValueError("window must be rect or hann")

    windowed_iq = iq * w

    # window power normalization (mean of w^2)
    window_power = np.mean(w ** 2)

    if not isinstance(nfft, int) or nfft <= 2:
        raise ValueError("nfft must be an int > 2")

    X = np.fft.fft(windowed_iq, n=nfft)
    X_shifted = np.fft.fftshift(X)

    f = np.linspace(-0.5, 0.5, num=nfft, endpoint=False)

    # normalize PSD by (nfft * window_power)
    psd_linear = np.abs(X_shifted) ** 2 / (nfft * window_power)

    # add eps to prevent a situation of psd_linear = 0 lead to psd_db = -inf
    eps = 1e-12
    psd_db = 10 * np.log10(psd_linear + eps)

    return f, psd_db


def dominant_freq(f: np.ndarray, psd_db: np.ndarray) -> float:
    f = np.asarray(f)
    psd_db = np.asarray(psd_db)

    if f.ndim != 1:
        raise ValueError("f must be a 1D array")

    if psd_db.ndim != 1:
        raise ValueError("psd_db must be a 1D array")

    if not f.shape == psd_db.shape:
        raise ValueError("psd_db and f must be the same shape")

    if f.size == 0:
        raise ValueError("f must not be empty")

    if not np.all(np.isfinite(f)):
        raise ValueError("f contains INF or NaN values")

    if not np.all(np.isfinite(psd_db)):
        raise ValueError("psd_db contains INF or NaN values")

    index_peak_psd = np.argmax(psd_db)

    return float(f[index_peak_psd])


def spectrogram(iq: np.ndarray, nfft: int = 256, hop: int = 128, window: str = "hann")  -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    """
    Compute spectrogram (time–frequency representation) of a complex IQ signal.
    The signal is divided into overlapping frames. Each frame is windowed,
    transformed using FFT, and converted to power spectral density (PSD).
    Spectrogram is computed as a sequence of short-time FFTs.

    I will cut iq to many frames at length nfft

    hop - Hop size between consecutive frames (in samples).

    return (t, f, S_db)
    t - Time axis (frame index or normalized time).
    f - Normalized frequency axis in range [-0.5, 0.5).
    S_db - Spectrogram matrix in dB, shape (num_frames, nfft).

    """

    iq = np.asarray(iq)

    if not np.issubdtype(iq.dtype, np.complexfloating):
        raise ValueError("iq must be complex number")

    if iq.ndim != 1:
        raise ValueError("iq must be a 1D array")

    if not np.all(np.isfinite(iq)):
        raise ValueError("iq contains INF or NaN values")

    if not isinstance(nfft, int) or nfft <= 2:
        raise ValueError("nfft must be an int > 2")

    if not isinstance(hop, int) or hop <= 0 or hop > nfft:
        raise ValueError("hop must be an int in [1, nfft]")

    window = window.replace(" ", "").lower()

    if window == "rect":
        w = np.ones(nfft)
    elif window == "hann":
        w = np.hanning(nfft)
    else:
        raise ValueError("window must be rect or hann")

    N = iq.size
    if N < nfft:
        raise ValueError("iq length must be >= nfft")

    num_frames = 1 + (N - nfft) // hop

    f = np.linspace(-0.5, 0.5, num=nfft, endpoint=False)

    t = np.arange(num_frames)

    # S_db is a matrix with shape (num_frames, nfft)
    S_db = np.empty((num_frames, nfft), dtype=np.float64)

    eps = 1e-12
    window_power = np.mean(w ** 2)

    for num in range(num_frames):
        start = num * hop
        frame = iq[start: start + nfft]

        windowed_frame = frame * w

        X = np.fft.fft(windowed_frame, n=nfft)
        X_shifted = np.fft.fftshift(X)

        psd_linear = (np.abs(X_shifted) ** 2) / (nfft * window_power)

        S_db[num, :] = 10 * np.log10(psd_linear + eps)

    return t, f, S_db

