import numpy as np

"""
IQ baseband analysis
"""

def power (iq: np.ndarray) -> float:

    iq = np.asarray(iq)

    if not np.issubdtype(iq.dtype, np.complexfloating):
        raise ValueError("iq must be complex number")

    if iq.ndim != 1:
        raise ValueError("iq must be a 1D array")

    if not np.all(np.isfinite(iq)):
        raise ValueError("iq contains INF or NaN values")

    power = np.mean(np.abs(iq)**2)

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

    if not isinstance(nfft, int) or nfft <= 2:
        raise ValueError ("nfft must be an int > 2")

    X = np.fft.fft(windowed_iq, n=nfft)
    X_shifted = np.fft.fftshift(X)

    f = np.linspace(-0.5, 0.5, num=nfft, endpoint=False)

    psd_linear = np.abs(X_shifted)**2

    # add eps to prevent a situation of psd_linear = 0 lead to psd_db = -inf
    eps = 1e-12
    psd_db = 10 * np.log10(psd_linear + eps)

    return (f, psd_db)


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


