import numpy as np
import matplotlib.pyplot as plt

from signal_lab.modulation import qpsk_modulation
from signal_lab.analysis import power
from signal_lab.analysis import spectrum
from signal_lab.analysis import spectrogram
from signal_lab.channel import awgn_channel

"""
bits → QPSK → AWGN → Analysis (power / spectrum / spectrogram) → plots + prints
"""

SEED = 968
N_BITS = 200000
NFFT = 4096
WINDOW = "hann"
# Tone frequency
SNR_DB = 5
SPEC_NFFT = 256
SPEC_HOP = 128


def main():
    rng = np.random.default_rng(SEED)
    bits = rng.integers(0, 2, N_BITS, dtype=np.uint8)
    iq_clean = qpsk_modulation(bits)
    iq_noisy = awgn_channel(iq_clean, SNR_DB, rng=rng)

    p_clean = power(iq_clean)
    p_noisy = power(iq_noisy)
    p_noise = power(iq_noisy - iq_clean)
    snr_measured_db = 10 * np.log10(p_clean / p_noise)
    print(f"power clean: {p_clean:.6f}")
    print(f"power noisy: {p_noisy:.6f}")
    print(f"power noise: {p_noise:.6f}")
    print(f"measured SNR: {snr_measured_db:.2f} dB")

    f_clean, psd_clean = spectrum(iq_clean, nfft=NFFT, window=WINDOW)
    f_noisy, psd_noisy = spectrum(iq_noisy, nfft=NFFT, window=WINDOW)

    # plot
    plt.figure()
    plt.plot(f_clean, psd_clean, label="clean")
    plt.plot(f_noisy, psd_noisy, label=f"noisy (SNR={SNR_DB} dB)")
    plt.xlabel("Frequency")
    plt.ylabel("PSD [dB]")
    plt.title("PSD vs Frequency (QPSK)")
    plt.grid(True, which="both")
    plt.legend()
    plt.show()

    # spectrogram

    t_clean, f_clean, S_db_clean = spectrogram(iq_clean, nfft=SPEC_NFFT, hop=SPEC_HOP, window=WINDOW)
    t_noisy, f_noisy, S_db_noisy = spectrogram(iq_noisy, nfft=SPEC_NFFT, hop=SPEC_HOP, window=WINDOW)

    S_clean_plot = S_db_clean - np.max(S_db_clean)
    S_noisy_plot = S_db_noisy - np.max(S_db_noisy)

    # Choose display dynamic range (in dB below peak)
    vmin, vmax = -60.0, 0.0

    # X axis is frame index (time), Y axis is normalized frequency
    extent = [t_clean[0], t_clean[-1], f_clean[0], f_clean[-1]]

    # Clean spectrogram
    plt.figure()
    plt.imshow(
        S_clean_plot.T,
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
    )
    plt.title("Spectrogram (clean QPSK)")
    plt.xlabel("Time (frame index)")
    plt.ylabel("Frequency (normalized)")
    plt.colorbar(label="Power [dB] (relative to peak)")
    plt.grid(False)
    plt.show()

    # Noisy spectrogram
    plt.figure()
    plt.imshow(
        S_noisy_plot.T,
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
    )
    plt.title(f"Spectrogram (QPSK + AWGN, SNR={SNR_DB:.1f} dB)")
    plt.xlabel("Time (frame index)")
    plt.ylabel("Frequency (normalized)")
    plt.colorbar(label="Power [dB] (relative to peak)")
    plt.grid(False)
    plt.show()


if __name__ == "__main__":
    main()

