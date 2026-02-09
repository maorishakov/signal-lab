from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from signal_lab.modulation import bpsk_modulation, qpsk_modulation, qam16_modulation
from signal_lab.demodulation import bpsk_demodulation, qpsk_demodulation, qam16_demodulation
from signal_lab.channel import awgn_channel, cfo_channel, phase_noise_channel


def _ber(bits_tx: np.ndarray, bits_rx: np.ndarray) -> float:
    bits_tx = np.asarray(bits_tx, dtype=np.uint8)
    bits_rx = np.asarray(bits_rx, dtype=np.uint8)
    if bits_tx.shape != bits_rx.shape:
        raise ValueError("bits_tx and bits_rx must have same shape")
    if bits_tx.size == 0:
        raise ValueError("bits_tx must not be empty")
    return float(np.mean(bits_tx != bits_rx))


def timing_offset_channel(iq: np.ndarray, offset: int) -> np.ndarray:
    """
    Simple "timing offset" in discrete-time: shift the sequence by offset samples.
    Positive offset => delay (prepend zeros, drop tail)
    Negative offset => advance (drop head, append zeros)

    Note: This does NOT model fractional timing / ISI. It's a simple sample shift.
    """
    iq = np.asarray(iq)
    if iq.ndim != 1:
        raise ValueError("iq must be 1D")
    if iq.size == 0:
        raise ValueError("iq must not be empty")
    if not np.issubdtype(iq.dtype, np.complexfloating):
        raise ValueError("iq must be complex array")
    if not np.all(np.isfinite(iq)):
        raise ValueError("iq contains INF or NaN values")

    offset = int(offset)
    if offset == 0:
        return iq.copy()

    out = np.zeros_like(iq)
    n = iq.size

    if offset > 0:
        # delay: out[offset:] = iq[:n-offset]
        if offset >= n:
            return out
        out[offset:] = iq[: n - offset]
    else:
        # advance: out[:n-k] = iq[k:]
        k = -offset
        if k >= n:
            return out
        out[: n - k] = iq[k:]
    return out


ModName = Literal["BPSK", "QPSK", "16QAM"]


@dataclass
class ChannelConfig:
    # AWGN
    enable_awgn: bool = True
    snr_db: float = 12.0  # interpreted as SNR wrt signal power in awgn_channel

    # CFO
    enable_cfo: bool = False
    f_cfo: float = 0.0
    phase0: float = 0.0

    # Phase Noise
    enable_phase_noise: bool = False
    sigma: float = 0.0
    pn_seed: int = 968

    # Timing Offset
    enable_timing_offset: bool = False
    offset: int = 0

    # Order
    # Recommended for your current blocks:
    # CFO -> PhaseNoise -> AWGN -> TimingOffset
    # (timing offset is a sample shift; AWGN anywhere is OK, but this order is intuitive)
    order: tuple[str, ...] = ("cfo", "phase_noise", "awgn", "timing_offset")


def _modulate(mod: ModName, bits: np.ndarray) -> np.ndarray:
    if mod == "BPSK":
        return bpsk_modulation(bits)
    if mod == "QPSK":
        # QPSK needs even bits
        if bits.size % 2 != 0:
            bits = bits[:-1]
        return qpsk_modulation(bits)
    if mod == "16QAM":
        # 16QAM needs multiple-of-4 bits
        r = bits.size % 4
        if r != 0:
            bits = bits[: bits.size - r]
        return qam16_modulation(bits)
    raise ValueError(f"Unknown modulation: {mod}")


def _demodulate(mod: ModName, symbols: np.ndarray) -> np.ndarray:
    if mod == "BPSK":
        return bpsk_demodulation(symbols)
    if mod == "QPSK":
        return qpsk_demodulation(symbols)
    if mod == "16QAM":
        return qam16_demodulation(symbols)
    raise ValueError(f"Unknown modulation: {mod}")


def run_chain(
    *,
    modulation: ModName,
    n_bits: int,
    seed: int,
    cfg: ChannelConfig,
    compute_ber: bool = True,
) -> dict:
    """
    Returns:
      {
        "bits_tx": ...,
        "symbols_tx": ...,
        "symbols_rx": ...,
        "bits_rx": ... (optional),
        "ber": ... (optional),
      }
    """
    rng = np.random.default_rng(int(seed))
    n_bits = int(n_bits)
    if n_bits <= 0:
        raise ValueError("n_bits must be positive")

    bits_tx = rng.integers(0, 2, size=n_bits, dtype=np.uint8)
    symbols_tx = _modulate(modulation, bits_tx)

    # run through channel impairments
    x = symbols_tx.copy()

    for stage in cfg.order:
        if stage == "cfo" and cfg.enable_cfo:
            x = cfo_channel(x, f_cfo=float(cfg.f_cfo), phase0=float(cfg.phase0))
        elif stage == "phase_noise" and cfg.enable_phase_noise:
            # Use deterministic seed so GUI is repeatable
            x = phase_noise_channel(x, sigma=float(cfg.sigma), seed=int(cfg.pn_seed))
        elif stage == "awgn" and cfg.enable_awgn:
            x = awgn_channel(x, snr_db=float(cfg.snr_db), rng=rng)
        elif stage == "timing_offset" and cfg.enable_timing_offset:
            x = timing_offset_channel(x, offset=int(cfg.offset))

    symbols_rx = x

    out = {
        "bits_tx": bits_tx,
        "symbols_tx": symbols_tx,
        "symbols_rx": symbols_rx,
    }

    if compute_ber:
        bits_rx = _demodulate(modulation, symbols_rx)

        # align lengths (because QPSK/16QAM may have truncated bits in modulation)
        min_len = min(bits_tx.size, bits_rx.size)
        bits_tx2 = bits_tx[:min_len]
        bits_rx2 = bits_rx[:min_len]

        out["bits_rx"] = bits_rx2

        # Try using user's metrics.ber if exists; else fallback
        try:
            from signal_lab.metrics import ber as ber_fn  # type: ignore

            out["ber"] = float(ber_fn(bits_tx2, bits_rx2))
        except Exception:
            out["ber"] = _ber(bits_tx2, bits_rx2)

    return out
