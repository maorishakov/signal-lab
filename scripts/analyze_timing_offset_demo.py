from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from signal_lab.generate import tone_iq
from signal_lab.modulation import qpsk_modulation
from signal_lab.channel import awgn_channel, timing_offset_channel
from signal_lab.analysis import power, spectrum, spectrogram
from signal_lab.visualization import plot_constellation


"""
Timing Offset Demo

Part A (Tone):
tone_iq(clean) -> AWGN -> timing_offset -> analysis + plots

Part B (QPSK):
bits -> qpsk_modulation(clean) -> AWGN -> timing_offset -> constellation + plots
"""


# ---------- Common ----------
SEED = 968
OUT_DIR = "outputs/timing_offset_demo"
SNR_DB = 12.0
OFFSET = 200            # integer sample offset (delay if >0, advance if <0)
WINDOW = "hann"

# ---------- Spectrum ----------
NFFT = 4096

# ---------- Spectrogram ----------
SPEC_NFFT = 256
SPEC_HOP = 128

# ---------- Tone ----------
TONE_N = 4096
TONE_F0 = 0.12

# ---------- QPSK ----------
N_BITS = 200_000
MAX_POINTS = 10_000


def _plot_psd(f1, psd1, f2, psd2, title: str, label1: str, label2: str) -> None:
    plt.figure()
    plt.plot(f1, psd1, label=label1)
    plt.plot(f2, psd2, label=label2)
    plt.xlabel("Frequency (normalized)")
    plt.ylabel("PSD [dB]")
    plt.title(title)
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.show()


def _plot_spectrogram(t, f, S_db, title: str) -> None:
    # Visual-only normalize to peak = 0 dB
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


def main() -> None:
    rng = np.random.default_rng(SEED)

    print("=== Timing Offset demo ===")
    print(f"SEED={SEED}")
    print(f"SNR_DB={SNR_DB}")
    print(f"OFFSET={OFFSET} samples (zeros fill)")
    print()

    # =========================================================
    # Part A: Tone
    # =========================================================
    print("=== Part A: Tone ===")
    iq_clean = tone_iq(n=TONE_N, f0=TONE_F0, phase0=0.0, amplitude=1.0)
    iq_awgn = awgn_channel(iq_clean, SNR_DB, rng=rng)
    iq_shift = timing_offset_channel(iq_awgn, offset=OFFSET, fill="zeros")

    p_clean = power(iq_clean)
    p_awgn = power(iq_awgn)
    p_shift = power(iq_shift)

    print(f"Tone f0={TONE_F0}")
    print(f"Power clean : {p_clean:.6f}")
    print(f"Power awgn  : {p_awgn:.6f}")
    print(f"Power shift : {p_shift:.6f}")
    print("Note: with zeros-fill, power often drops a bit because we injected zeros.\n")

    f_awgn, psd_awgn = spectrum(iq_awgn, nfft=NFFT, window=WINDOW)
    f_shift, psd_shift = spectrum(iq_shift, nfft=NFFT, window=WINDOW)

    _plot_psd(
        f_awgn, psd_awgn,
        f_shift, psd_shift,
        title=f"Tone PSD: AWGN vs AWGN+TimingOffset (offset={OFFSET})",
        label1="AWGN",
        label2="AWGN + timing offset",
    )

    t1, f1, S1 = spectrogram(iq_awgn, nfft=SPEC_NFFT, hop=SPEC_HOP, window=WINDOW)
    t2, f2, S2 = spectrogram(iq_shift, nfft=SPEC_NFFT, hop=SPEC_HOP, window=WINDOW)

    _plot_spectrogram(t1, f1, S1, title="Tone Spectrogram (AWGN)")
    _plot_spectrogram(t2, f2, S2, title=f"Tone Spectrogram (AWGN + TimingOffset={OFFSET})")

    # =========================================================
    # Part B: QPSK
    # =========================================================
    print("=== Part B: QPSK ===")
    if N_BITS % 2 != 0:
        raise ValueError("N_BITS must be even for QPSK")

    bits = rng.integers(0, 2, size=N_BITS, dtype=np.uint8)
    qpsk_clean = qpsk_modulation(bits)
    qpsk_awgn = awgn_channel(qpsk_clean, SNR_DB, rng=rng)
    qpsk_shift = timing_offset_channel(qpsk_awgn, offset=OFFSET, fill="zeros")

    print(f"N_BITS={N_BITS}")
    print(f"QPSK Power clean : {power(qpsk_clean):.6f}")
    print(f"QPSK Power awgn  : {power(qpsk_awgn):.6f}")
    print(f"QPSK Power shift : {power(qpsk_shift):.6f}")
    print()

    # Constellation plots (saved to files)
    plot_constellation(
        qpsk_clean,
        max_points=MAX_POINTS,
        title="QPSK Constellation (clean)",
        out_path=f"{OUT_DIR}/qpsk_clean.png",
        show=False,
    )

    plot_constellation(
        qpsk_awgn,
        max_points=MAX_POINTS,
        title=f"QPSK + AWGN (SNR={SNR_DB:.1f} dB)",
        out_path=f"{OUT_DIR}/qpsk_awgn.png",
        show=False,
    )

    plot_constellation(
        qpsk_shift,
        max_points=MAX_POINTS,
        title=f"QPSK + AWGN + TimingOffset (offset={OFFSET})",
        out_path=f"{OUT_DIR}/qpsk_awgn_timing_offset.png",
        show=False,
    )

    print(f"Saved constellation PNGs under: {OUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
