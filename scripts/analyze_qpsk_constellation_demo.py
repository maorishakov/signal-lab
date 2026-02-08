from __future__ import annotations

import numpy as np

from signal_lab.modulation import qpsk_modulation
from signal_lab.channel import awgn_channel, cfo_channel, phase_noise_channel
from signal_lab.visualization import plot_constellation


"""
QPSK Constellation Demo

Flow (variants):
1) Clean
2) AWGN
3) AWGN -> CFO
4) AWGN -> Phase Noise
5) AWGN -> CFO -> Phase Noise

Saves PNGs under outputs/constellation/
"""


SEED = 968
N_BITS = 200_000  # must be even for QPSK
SNR_DB = 12.0

F_CFO = 0.02      # normalized cycles/sample, in (-0.5, 0.5)
PHASE0 = 0.3      # radians

SIGMA_PN = 0.02   # rad/sample (random walk std per sample)

MAX_POINTS = 10_000


def main() -> None:
    if N_BITS % 2 != 0:
        raise ValueError("N_BITS must be even for QPSK")

    rng = np.random.default_rng(SEED)

    # --- Generate clean QPSK IQ ---
    bits = rng.integers(0, 2, size=N_BITS, dtype=np.uint8)
    iq_clean = qpsk_modulation(bits)

    # --- Apply channels ---
    iq_awgn = awgn_channel(iq_clean, SNR_DB, rng=rng)

    iq_awgn_cfo = cfo_channel(iq_awgn, f_cfo=F_CFO, phase0=PHASE0)

    iq_awgn_pn = phase_noise_channel(iq_awgn, sigma=SIGMA_PN, rng=rng)

    iq_awgn_cfo_pn = phase_noise_channel(iq_awgn_cfo, sigma=SIGMA_PN, rng=rng)

    # --- Print configuration ---
    print("=== Constellation demo (QPSK) ===")
    print(f"SEED={SEED}, N_BITS={N_BITS}")
    print(f"SNR_DB={SNR_DB}")
    print(f"CFO: f_cfo={F_CFO}, phase0={PHASE0}")
    print(f"Phase noise: sigma={SIGMA_PN} rad/sample")
    print("Order used: AWGN -> (CFO) -> (PhaseNoise)")
    print("Saving plots to outputs/constellation/ ...")

    # --- Save plots ---
    plot_constellation(
        iq_clean,
        max_points=MAX_POINTS,
        title="QPSK Constellation (clean)",
        out_path="outputs/constellation/qpsk_clean.png",
        show=False,
    )

    plot_constellation(
        iq_awgn,
        max_points=MAX_POINTS,
        title=f"QPSK + AWGN (SNR={SNR_DB:.1f} dB)",
        out_path="outputs/constellation/qpsk_awgn.png",
        show=False,
    )

    plot_constellation(
        iq_awgn_cfo,
        max_points=MAX_POINTS,
        title=f"QPSK + AWGN + CFO (SNR={SNR_DB:.1f} dB, f_cfo={F_CFO:.3f})",
        out_path="outputs/constellation/qpsk_awgn_cfo.png",
        show=False,
    )

    plot_constellation(
        iq_awgn_pn,
        max_points=MAX_POINTS,
        title=f"QPSK + AWGN + PhaseNoise (SNR={SNR_DB:.1f} dB, sigma={SIGMA_PN:.3f})",
        out_path="outputs/constellation/qpsk_awgn_pn.png",
        show=False,
    )

    plot_constellation(
        iq_awgn_cfo_pn,
        max_points=MAX_POINTS,
        title=f"QPSK + AWGN + CFO + PhaseNoise (SNR={SNR_DB:.1f} dB)",
        out_path="outputs/constellation/qpsk_awgn_cfo_pn.png",
        show=False,
    )

    print("Done.")


if __name__ == "__main__":
    main()
