import numpy as np
import matplotlib.pyplot as plt

from signal_lab.generate import tone_iq
from signal_lab.channel import awgn_channel, phase_noise_channel
from signal_lab.analysis import power, spectrum, dominant_freq, spectrogram

"""
Demo: Tone -> (optional AWGN) -> Phase Noise -> Analysis + Plots

We compare:
1) clean tone
2) clean + AWGN
3) clean + phase noise (+ optional AWGN)

Expected:
- Power should stay ~same (phase rotation magnitude=1)
- Dominant freq should stay ~same (phase noise spreads spectrum, not shift center)
- PSD peak becomes "wider"/raised skirt with phase noise
"""

SEED = 968

N = 4096
F0 = 0.12
PHASE0 = 0.0
AMP = 1.0

# Channel params
SNR_DB = 15.0          # AWGN SNR
SIGMA = 0.03           # phase noise step std [rad/sample]
APPLY_AWGN_BEFORE_PN = True  # order experiment

# Analysis params
NFFT = 4096
WINDOW = "hann"
SPEC_NFFT = 256
SPEC_HOP = 128


def plot_psd_overlay(f, psd_list, labels, title):
    plt.figure()
    for psd_db, lab in zip(psd_list, labels):
        plt.plot(f, psd_db, label=lab)
    plt.xlabel("Frequency (normalized)")
    plt.ylabel("PSD [dB]")
    plt.title(title)
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_spectrogram(iq, title):
    t, f, S_db = spectrogram(iq, nfft=SPEC_NFFT, hop=SPEC_HOP, window=WINDOW)

    # Visual-only normalize to 0 dB peak
    S_plot = S_db - np.max(S_db)

    vmin, vmax = -60.0, 0.0
    extent = [t[0], t[-1], f[0], f[-1]]

    plt.figure()
    plt.imshow(
        S_plot.T,
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
    )
    plt.title(title)
    plt.xlabel("Time (frame index)")
    plt.ylabel("Frequency (normalized)")
    plt.colorbar(label="Power [dB] (relative to peak)")
    plt.tight_layout()
    plt.show()


def main():
    rng = np.random.default_rng(SEED)

    # 1) generate clean tone
    iq_clean = tone_iq(n=N, f0=F0, phase0=PHASE0, amplitude=AMP)

    # 2) AWGN (optional)
    if APPLY_AWGN_BEFORE_PN:
        iq_awgn = awgn_channel(iq_clean, SNR_DB, rng=rng)
        iq_pn = phase_noise_channel(iq_awgn, sigma=SIGMA, rng=rng)
        chain_desc = "AWGN -> PhaseNoise"
    else:
        iq_pn_only = phase_noise_channel(iq_clean, sigma=SIGMA, rng=rng)
        iq_pn = awgn_channel(iq_pn_only, SNR_DB, rng=rng)
        iq_awgn = awgn_channel(iq_clean, SNR_DB, rng=rng)
        chain_desc = "PhaseNoise -> AWGN"

    # ---- Analysis ----
    p_clean = power(iq_clean)
    p_awgn = power(iq_awgn)
    p_pn = power(iq_pn)

    f_c, psd_clean = spectrum(iq_clean, nfft=NFFT, window=WINDOW)
    f_a, psd_awgn = spectrum(iq_awgn, nfft=NFFT, window=WINDOW)
    f_p, psd_pn = spectrum(iq_pn, nfft=NFFT, window=WINDOW)

    df_clean = dominant_freq(f_c, psd_clean)
    df_awgn = dominant_freq(f_a, psd_awgn)
    df_pn = dominant_freq(f_p, psd_pn)

    print("=== Phase Noise demo ===")
    print(f"Tone f0 = {F0}")
    print(f"Phase noise sigma = {SIGMA} rad/sample")
    print(f"AWGN SNR = {SNR_DB} dB")
    print(f"Order: {chain_desc}")
    print()
    print(f"Power clean : {p_clean:.6f}")
    print(f"Power awgn  : {p_awgn:.6f}")
    print(f"Power pn    : {p_pn:.6f}")
    print()
    print(f"Dominant freq clean: {df_clean:.6f}")
    print(f"Dominant freq awgn : {df_awgn:.6f}")
    print(f"Dominant freq pn   : {df_pn:.6f}")
    print()

    # ---- Plots ----
    plot_psd_overlay(
        f_c,
        [psd_clean, psd_awgn, psd_pn],
        ["clean", f"AWGN (SNR={SNR_DB} dB)", f"PhaseNoise (sigma={SIGMA}) [{chain_desc}]"],
        title="PSD vs Frequency (Tone) + AWGN + Phase Noise",
    )

    # spectrograms
    plot_spectrogram(iq_clean, "Spectrogram (clean tone)")
    plot_spectrogram(iq_awgn, f"Spectrogram (tone + AWGN, SNR={SNR_DB} dB)")
    plot_spectrogram(iq_pn, f"Spectrogram (tone + phase noise sigma={SIGMA}) [{chain_desc}]")


if __name__ == "__main__":
    main()
