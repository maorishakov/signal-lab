import math

import numpy as np
import matplotlib.pyplot as plt

from signal_lab.modulation import qpsk_modulation
from signal_lab.demodulation import qpsk_demodulation
from signal_lab.channel import awgn_channel
from signal_lab.metrics import ber

"""
simulate QPSK over AWGN and compute BER vs SNR

bits → qpsk_modulation → awgn_channel → qpsk_demodulation → ber

BERqpsk = 0.5 * erfc (sqrt(Es / 2N0))
"""

# choose random number for seed
SEED = 968
N_BITS = 200000
SNR_DB_LIST = [0, 2, 4, 6, 8, 10, 12]


def main():
    # create random number generator
    rng = np.random.default_rng(SEED)

    if N_BITS % 2 != 0:
        raise ValueError("N BITS must be even for QPSK")

    bits = rng.integers(0, 2, N_BITS)
    symbols = qpsk_modulation(bits)

    # results = [(snr_db, ber_val)]
    results = []
    ber_val_simulated = []
    ber_theoretical = []

    print("  SNR(dB)   BER")
    print("-" * 18)

    for snr_db in SNR_DB_LIST:
        signal_and_noise = awgn_channel(symbols, snr_db, rng=rng)
        bits_hat = qpsk_demodulation(signal_and_noise)
        ber_val = ber(bits, bits_hat)
        ber_val_simulated.append(ber_val)
        snr_linear = 10 ** (snr_db / 10)
        # two bits in one symbol
        ber_theoretical.append(0.5 * math.erfc(math.sqrt(snr_linear / 2)))
        results.append((snr_db, ber_val))
        print(f"{snr_db:>6.1f}  {ber_val:>10.3e}", flush=True)

    plt.semilogy(SNR_DB_LIST, ber_val_simulated, label="Simulation")
    plt.semilogy(SNR_DB_LIST, ber_theoretical, label="Theory")
    plt.xlabel("SNR [dB]")
    plt.ylabel("BER")
    plt.title("QPSK BER vs SNR")
    plt.grid(True, which="both")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
