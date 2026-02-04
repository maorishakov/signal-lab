import math

import numpy as np
import matplotlib.pyplot as plt

from signal_lab.analysis import power
from signal_lab.analysis import spectrum
from signal_lab.analysis import dominant_freq
from signal_lab.channel import awgn_channel

"""
“Tone IQ (clean) → AWGN Channel → Analysis (power, spectrum, dominant_freq) → Plot PSD (clean vs noisy)”

n = samples_vector
ϕ[n]= 2π * f0 * n
x[n]= e ^ jϕ[n]

"""

SEED = 968
N = 4096
NFFT = 4096
WINDOW = "hann"
# Tone frequency
F0 = 0.12
SNR_DB = 5


def main():
    rng = np.random.default_rng(SEED)
    samples_vector = np.arange(N)
    phase = 2 * np.pi * F0 * samples_vector
    iq_clean = np.exp(1j * phase)

    iq_noisy = awgn_channel(iq_clean, SNR_DB, rng=rng)

    # analysis for iq_clean:
    power_iq_clean = power(iq_clean)
    f_iq_clean, psd_iq_clean = spectrum(iq_clean, NFFT, WINDOW)
    dominant_freq_iq_clean = dominant_freq(f_iq_clean, psd_iq_clean)

    # analysis for iq_noisy:
    power_iq_noisy = power(iq_noisy)
    f_iq_noisy, psd_iq_noisy = spectrum(iq_noisy, NFFT, WINDOW)
    dominant_freq_iq_noisy = dominant_freq(f_iq_noisy, psd_iq_noisy)

    print("dominant freq iq clean: " + f"{dominant_freq_iq_clean}")
    print("dominant freq iq noisy: " + f"{dominant_freq_iq_noisy}")
    print(f"power clean: {power_iq_clean}")
    print(f"power noisy: {power_iq_noisy}")
    print(f"power noise: {power(iq_noisy - iq_clean)}")

    # plot

    plt.plot(f_iq_clean, psd_iq_clean, label="clean")
    plt.plot(f_iq_noisy, psd_iq_noisy, label="noisy")
    plt.xlabel("Frequency")
    plt.ylabel("PSD [dB]")
    plt.title("PSD vs Frequency single Tone")
    plt.grid(True, which="both")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
