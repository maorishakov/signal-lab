import numpy as np
import matplotlib.pyplot as plt

from signal_lab.analysis import power, spectrum, dominant_freq, spectrogram
from signal_lab.channel import awgn_channel
from signal_lab.generate import multi_tone_iq

"""
Multi-Tone IQ (clean) → AWGN Channel → Analysis (power, spectrum, dominant_freq, spectrogram)
→ Plot PSD (clean vs noisy) + Spectrograms
"""

SEED = 968
N = 4096
NFFT = 4096
WINDOW = "hann"

# Multi-tone parameters (normalized frequencies)
F0_LIST = [0.12, -0.22, 0.31]
PHASE0_LIST = [0.0, 0.4, -0.2]
AMPLITUDE_LIST = [1.0, 0.7, 0.5]

SNR_DB = 5
SPEC_NFFT = 256
SPEC_HOP = 128


def _plot_spectrogram(title: str, t: np.ndarray, f: np.ndarray, S_db: np.ndarray) -> None:
    # visual-only normalization to peak = 0 dB
    S_plot = S_db - np.max(S_db)

    vmin, vmax = -60.0, 0.0
    extent = [t[0], t[-1], f[0], f[-1]]

    plt.figure()
    plt.imshow(S_plot.T, origin="lower", aspect="auto", extent=extent, vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.xlabel("Time (frame index)")
    plt.ylabel("Frequency (normalized)")
    plt.colorbar(label="Power [dB] (relative to peak)")
    plt.grid(False)
    plt.show()


def main():
    rng = np.random.default_rng(SEED)

    # Generate clean multi-tone IQ
    iq_clean = multi_tone_iq(n=N, f0_list=F0_LIST, phase0_list=PHASE0_LIST, amplitude_list=AMPLITUDE_LIST, normalize=True)

    # Add AWGN
    iq_noisy = awgn_channel(iq_clean, SNR_DB, rng=rng)

    # Analysis for iq_clean
    power_iq_clean = power(iq_clean)
    f_iq_clean, psd_iq_clean = spectrum(iq_clean, NFFT, WINDOW)
    dominant_freq_iq_clean = dominant_freq(f_iq_clean, psd_iq_clean)

    # Analysis for iq_noisy
    power_iq_noisy = power(iq_noisy)
    f_iq_noisy, psd_iq_noisy = spectrum(iq_noisy, NFFT, WINDOW)
    dominant_freq_iq_noisy = dominant_freq(f_iq_noisy, psd_iq_noisy)

    print(f"dominant freq iq clean: {dominant_freq_iq_clean}")
    print(f"dominant freq iq noisy: {dominant_freq_iq_noisy}")
    print(f"power clean: {power_iq_clean}")
    print(f"power noisy: {power_iq_noisy}")
    print(f"power noise: {power(iq_noisy - iq_clean)}")

    # PSD plot (clean vs noisy)
    plt.figure()
    plt.plot(f_iq_clean, psd_iq_clean, label="clean (multi-tone)")
    plt.plot(f_iq_noisy, psd_iq_noisy, label=f"noisy (SNR={SNR_DB} dB)")
    plt.xlabel("Frequency (normalized)")
    plt.ylabel("PSD [dB]")
    plt.title("PSD vs Frequency (Multi-Tone)")
    plt.grid(True, which="both")
    plt.legend()
    plt.show()

    # Spectrograms
    t_clean, f_clean, S_db_clean = spectrogram(iq_clean, SPEC_NFFT, SPEC_HOP, WINDOW)
    t_noisy, f_noisy, S_db_noisy = spectrogram(iq_noisy, SPEC_NFFT, SPEC_HOP, WINDOW)

    _plot_spectrogram("Spectrogram (clean multi-tone)", t_clean, f_clean, S_db_clean)
    _plot_spectrogram(f"Spectrogram (multi-tone + AWGN, SNR={SNR_DB:.1f} dB)", t_noisy, f_noisy, S_db_noisy)


if __name__ == "__main__":
    main()
