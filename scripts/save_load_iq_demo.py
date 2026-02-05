import numpy as np
import matplotlib.pyplot as plt

from signal_lab.generate import tone_iq
from signal_lab.io import save_iq_npy, load_iq_npy
from signal_lab.analysis import power, spectrum


"""
Demo: Generate -> Save -> Load -> Analyze

tone_iq -> save_iq_npy -> load_iq_npy -> power/spectrum sanity checks
"""


# Parameters
N = 4096
F0 = 0.12
AMP = 1.0
PHASE0 = 0.0

NFFT = 4096
WINDOW = "hann"

OUT_PATH = "outputs/demo/tone_iq.npy"


def main():
    #Generate
    iq = tone_iq(n=N, f0=F0, phase0=PHASE0, amplitude=AMP)

    #Save
    saved_path = save_iq_npy(OUT_PATH, iq)
    print(f"Saved IQ to: {saved_path}")

    #Load
    iq2 = load_iq_npy(saved_path)
    print(f"Loaded IQ from: {saved_path}")

    #Sanity checks
    max_abs_err = float(np.max(np.abs(iq2 - iq)))
    p1 = power(iq)
    p2 = power(iq2)

    print("\n--- Sanity ---")
    print(f"Max |iq_loaded - iq|: {max_abs_err:.3e}")
    print(f"Power (original):     {p1:.12f}")
    print(f"Power (loaded):       {p2:.12f}")
    print(f"Power diff:           {abs(p2 - p1):.3e}")

    # Spectrum sanity
    f1, psd1 = spectrum(iq, nfft=NFFT, window=WINDOW)
    f2, psd2 = spectrum(iq2, nfft=NFFT, window=WINDOW)
    max_psd_err = float(np.max(np.abs(psd2 - psd1)))

    print(f"Max |PSD_loaded - PSD|: {max_psd_err:.3e}")

    # Plot
    plt.figure()
    plt.plot(f1, psd1, label="original", alpha=0.8)
    plt.plot(f2, psd2, label="loaded", alpha=0.6, linestyle="--")
    plt.xlabel("Frequency (normalized)")
    plt.ylabel("PSD [dB]")
    plt.title("PSD sanity: original vs loaded IQ")
    plt.grid(True, which="both")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
