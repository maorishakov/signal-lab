import numpy as np
import matplotlib.pyplot as plt

from signal_lab.generate import tone_iq
from signal_lab.channel import awgn_channel, cfo_channel
from signal_lab.analysis import power, spectrum, dominant_freq, spectrogram

"""
Demo: Tone -> CFO -> (optional AWGN) -> Analysis + plots

We expect:
dominant_freq(after CFO) ≈ dominant_freq(before) + f_cfo   (wrapped to [-0.5, 0.5))
"""

SEED = 968

# tone
N = 4096
F0 = 0.12
PHASE0 = 0.0
AMPLITUDE = 1.0

# channel
F_CFO = 0.07
CFO_PHASE0 = 0.3

ADD_AWGN = True
SNR_DB = 10.0

# analysis / plots
NFFT = 4096
WINDOW = "hann"

SPEC_NFFT = 256
SPEC_HOP = 128


def _wrap_freq(f: float) -> float:
    """Wrap normalized frequency into [-0.5, 0.5)."""
    return ((f + 0.5) % 1.0) - 0.5


def main():
    rng = np.random.default_rng(SEED)

    # --- generate clean tone ---
    iq_clean = tone_iq(n=N, f0=F0, phase0=PHASE0, amplitude=AMPLITUDE)

    # --- apply CFO ---
    iq_cfo = cfo_channel(iq_clean, f_cfo=F_CFO, phase0=CFO_PHASE0)

    # --- optional AWGN (usually after impairments) ---
    if ADD_AWGN:
        iq_clean_ch = awgn_channel(iq_clean, SNR_DB, rng=rng)
        iq_cfo_ch = awgn_channel(iq_cfo, SNR_DB, rng=rng)
        tag = f"+ AWGN (SNR={SNR_DB:.1f} dB)"
    else:
        iq_clean_ch = iq_clean
        iq_cfo_ch = iq_cfo
        tag = "(no AWGN)"

    # --- analysis ---
    p_clean = power(iq_clean_ch)
    p_cfo = power(iq_cfo_ch)

    f1, psd1 = spectrum(iq_clean_ch, nfft=NFFT, window=WINDOW)
    f2, psd2 = spectrum(iq_cfo_ch, nfft=NFFT, window=WINDOW)

    f_peak_clean = dominant_freq(f1, psd1)
    f_peak_cfo = dominant_freq(f2, psd2)

    expected = _wrap_freq(F0 + F_CFO)
    delta = _wrap_freq(f_peak_cfo - f_peak_clean)

    print("=== CFO demo ===")
    print(f"Tone f0 = {F0}")
    print(f"CFO f_cfo = {F_CFO} (phase0={CFO_PHASE0})")
    print(f"Channel: {tag}")
    print()
    print(f"Power clean : {p_clean:.6f}")
    print(f"Power CFO   : {p_cfo:.6f}")
    print()
    print(f"Dominant freq clean: {f_peak_clean:.6f}")
    print(f"Dominant freq CFO  : {f_peak_cfo:.6f}")
    print(f"Delta (CFO-clean)  : {delta:.6f}   (expected ~ {F_CFO:.6f})")
    print(f"Expected absolute  : {expected:.6f}")

    # --- PSD plot ---
    plt.figure()
    plt.plot(f1, psd1, label="clean")
    plt.plot(f2, psd2, label=f"with CFO (f_cfo={F_CFO})")
    plt.xlabel("Frequency (normalized)")
    plt.ylabel("PSD [dB]")
    plt.title(f"PSD vs Frequency (Tone) {tag}")
    plt.grid(True, which="both")
    plt.legend()
    plt.show()

    # --- Spectrograms (optional but nice) ---
    t_clean, f_clean, S_clean = spectrogram(iq_clean_ch, nfft=SPEC_NFFT, hop=SPEC_HOP, window=WINDOW)
    t_cfo, f_cfo_ax, S_cfo = spectrogram(iq_cfo_ch, nfft=SPEC_NFFT, hop=SPEC_HOP, window=WINDOW)

    # visual-only normalize
    S_clean_plot = S_clean - np.max(S_clean)
    S_cfo_plot = S_cfo - np.max(S_cfo)

    vmin, vmax = -60.0, 0.0
    extent = [t_clean[0], t_clean[-1], f_clean[0], f_clean[-1]]

    plt.figure()
    plt.imshow(S_clean_plot.T, origin="lower", aspect="auto", extent=extent, vmin=vmin, vmax=vmax)
    plt.title(f"Spectrogram (clean tone) {tag}")
    plt.xlabel("Time (frame index)")
    plt.ylabel("Frequency (normalized)")
    plt.colorbar(label="Power [dB] (relative to peak)")
    plt.grid(False)
    plt.show()

    plt.figure()
    plt.imshow(S_cfo_plot.T, origin="lower", aspect="auto", extent=extent, vmin=vmin, vmax=vmax)
    plt.title(f"Spectrogram (tone + CFO f_cfo={F_CFO}) {tag}")
    plt.xlabel("Time (frame index)")
    plt.ylabel("Frequency (normalized)")
    plt.colorbar(label="Power [dB] (relative to peak)")
    plt.grid(False)
    plt.show()


if __name__ == "__main__":
    main()
