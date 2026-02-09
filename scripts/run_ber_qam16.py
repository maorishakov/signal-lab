import math

import numpy as np
import matplotlib.pyplot as plt

from signal_lab.modulation import qam16_modulation
from signal_lab.demodulation import qam16_demodulation
from signal_lab.channel import awgn_channel
from signal_lab.metrics import ber

"""
simulate 16-QAM over AWGN and compute BER vs SNR

bits → qam16_modulation → awgn_channel → qam16_demodulation → ber

BERqam16 = (3 / 8) * erfc (sqrt(0.4 * (Eb / N0)))
"""

# choose random number for seed
SEED = 968
N_BITS = 200000
SNR_DB_LIST = [0, 2, 4, 6, 8, 10, 12]


def main():
    # create random number generator
    rng = np.random.default_rng(SEED)

    if N_BITS % 4 != 0:
        raise ValueError("bits length must be a multiple of 4")

    bits = rng.integers(0, 2, N_BITS, dtype=np.uint8)
    symbols = qam16_modulation(bits)

    # results = [(snr_db, ber_val)]
    results = []
    ber_val_simulated = []
    ber_theoretical = []

    print("  Eb/N0(dB)   BER")
    print("-" * 18)

    #snr_db = Eb/N0

    for snr_db in SNR_DB_LIST:
        EsN0_db = snr_db + 10 * np.log10(4)
        signal_and_noise = awgn_channel(symbols, EsN0_db, rng=rng)
        bits_hat = qam16_demodulation(signal_and_noise)
        ber_val = ber(bits, bits_hat)
        ber_val_simulated.append(ber_val)
        snr_linear = 10 ** (snr_db / 10)
        # 4 bits in one symbol
        ber_theoretical.append((3 / 8) * math.erfc(math.sqrt(0.4 * snr_linear)))
        results.append((snr_db, ber_val))
        print(f"{snr_db:>6.1f}  {ber_val:>10.3e}", flush=True)

    plt.semilogy(SNR_DB_LIST, ber_val_simulated, label="Simulation")
    plt.semilogy(SNR_DB_LIST, ber_theoretical, label="Theory")
    plt.xlabel("Eb/N0 [dB]")
    plt.ylabel("BER")
    plt.title("16-QAM BER vs Eb/N0")
    plt.grid(True, which="both")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
