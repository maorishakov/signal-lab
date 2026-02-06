from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from signal_lab.generate import tone_iq
from signal_lab.io import save_iq_npy, load_iq_npy
from signal_lab.analysis import power, spectrum, dominant_freq, spectrogram
from signal_lab.channel import awgn_channel


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def cmd_generate_tone(args: argparse.Namespace) -> int:
    out_path = Path(args.out)

    iq = tone_iq(n=args.n, f0=args.f0, phase0=args.phase0, amplitude=args.amplitude)

    saved_path = save_iq_npy(out_path, iq)

    print(f"Saved tone IQ to: {saved_path}")
    print(f"n={args.n}, f0={args.f0}, phase0={args.phase0}, amplitude={args.amplitude}")
    return 0


def _save_metrics_json(out_dir: Path, metrics: dict) -> None:
    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to: {metrics_path}")


def _save_psd_png(out_dir: Path, f: np.ndarray, psd_db: np.ndarray, title: str) -> None:
    png_path = out_dir / "psd.png"
    plt.figure()
    plt.plot(f, psd_db)
    plt.xlabel("Frequency (normalized)")
    plt.ylabel("PSD [dB]")
    plt.title(title)
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"Saved PSD plot to: {png_path}")


def _save_spectrogram_png(out_dir: Path, t: np.ndarray, f: np.ndarray, S_db: np.ndarray, title: str, vmin: float = -60.0,
    vmax: float = 0.0) -> None:
    png_path = out_dir / "spectrogram.png"

    # visual-only normalization to 0 dB peak
    S_plot = S_db - np.max(S_db)

    extent = [t[0], t[-1], f[0], f[-1]]

    plt.figure()
    plt.imshow(S_plot.T, origin="lower", aspect="auto", extent=extent, vmin=vmin, vmax=vmax)
    plt.xlabel("Time (frame index)")
    plt.ylabel("Frequency (normalized)")
    plt.title(title)
    plt.colorbar(label="Power [dB] (relative to peak)")
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"Saved spectrogram plot to: {png_path}")


def cmd_analyze(args: argparse.Namespace) -> int:
    in_path = Path(args.input)
    out_dir = Path(args.out)

    _ensure_dir(out_dir)

    iq = load_iq_npy(in_path)

    p = power(iq)
    f, psd_db = spectrum(iq, nfft=args.nfft, window=args.window)
    f_peak = dominant_freq(f, psd_db)

    # Metrics JSON
    metrics = {
        "input_path": str(in_path),
        "n_samples": int(iq.size),
        "dtype": str(iq.dtype),
        "power": float(p),
        "dominant_freq_normalized": float(f_peak),
        "spectrum": {
            "nfft": int(args.nfft),
            "window": str(args.window),
        },
    }

    _save_metrics_json(out_dir, metrics)

    # PSD plot
    _save_psd_png(out_dir, f=f, psd_db=psd_db, title=f"PSD (nfft={args.nfft}, window={args.window})")

    # Optional spectrogram
    if args.spectrogram:
        t, f2, S_db = spectrogram(iq, nfft=args.spec_nfft, hop=args.spec_hop, window=args.window)
        _save_spectrogram_png(out_dir, t=t, f=f2, S_db=S_db,
            title=f"Spectrogram (nfft={args.spec_nfft}, hop={args.spec_hop}, window={args.window})",
            vmin=args.spec_vmin, vmax=args.spec_vmax)

    print("Analyze complete.")
    return 0


# ---------------- Pipeline (generate + analyze) ----------------

def cmd_pipeline_tone(args: argparse.Namespace) -> int:
    """
    Pipeline:
    1) Generate tone IQ
    2) (Optional) add AWGN
    3) Save IQ files
    4) Analyze (power/spectrum/dominant_freq) + save psd.png + metrics.json
    5) (Optional) spectrogram + spectrogram.png
    """
    out_dir = Path(args.out)
    _ensure_dir(out_dir)

    rng = np.random.default_rng(args.seed)

    # 1) generate clean tone
    iq_clean = tone_iq(n=args.n, f0=args.f0, phase0=args.phase0, amplitude=args.amplitude)
    save_iq_npy(out_dir / "iq_clean.npy", iq_clean)

    # 2) optional AWGN
    iq_used = iq_clean
    if args.snr_db is not None:
        iq_noisy = awgn_channel(iq_clean, snr_db=args.snr_db, rng=rng)
        save_iq_npy(out_dir / "iq_noisy.npy", iq_noisy)
        iq_used = iq_noisy

    # 3) analysis on iq_used
    p = power(iq_used)
    f, psd_db = spectrum(iq_used, nfft=args.nfft, window=args.window)
    f_peak = dominant_freq(f, psd_db)

    metrics = {
        "pipeline": "tone",
        "out_dir": str(out_dir),
        "seed": int(args.seed),
        "generated": {
            "n": int(args.n),
            "f0": float(args.f0),
            "phase0": float(args.phase0),
            "amplitude": float(args.amplitude),
        },
        "channel": {
            "snr_db": None if args.snr_db is None else float(args.snr_db),
        },
        "analyze_on": "iq_noisy.npy" if args.snr_db is not None else "iq_clean.npy",
        "n_samples": int(iq_used.size),
        "dtype": str(iq_used.dtype),
        "power": float(p),
        "dominant_freq_normalized": float(f_peak),
        "spectrum": {
            "nfft": int(args.nfft),
            "window": str(args.window),
        },
    }

    _save_metrics_json(out_dir, metrics)

    _save_psd_png(out_dir, f=f, psd_db=psd_db, title=f"PSD (nfft={args.nfft}, window={args.window})")

    if args.spectrogram:
        t, f2, S_db = spectrogram(iq_used, nfft=args.spec_nfft, hop=args.spec_hop, window=args.window)
        _save_spectrogram_png(
            out_dir,
            t=t,
            f=f2,
            S_db=S_db,
            title=f"Spectrogram (nfft={args.spec_nfft}, hop={args.spec_hop}, window={args.window})",
            vmin=args.spec_vmin,
            vmax=args.spec_vmax,
        )

    print(f"Pipeline complete. Output directory: {out_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="signal-lab", description="Signal Lab CLI (v0)")

    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # ---- generate ----
    p_gen = subparsers.add_parser("generate", help="Generate IQ signals")
    gen_sub = p_gen.add_subparsers(dest="gen_cmd", required=True)

    # generate tone
    p_tone = gen_sub.add_parser("tone", help="Generate a complex tone IQ signal")
    p_tone.add_argument("--n", type=int, required=True, help="Number of samples")
    p_tone.add_argument("--f0", type=float, required=True, help="Normalized frequency in [-0.5, 0.5)")
    p_tone.add_argument("--phase0", type=float, default=0.0, help="Initial phase (radians)")
    p_tone.add_argument("--amplitude", type=float, default=1.0, help="Amplitude (>= 0 recommended)")
    p_tone.add_argument("--out", type=str, required=True, help="Output .npy path (e.g., outputs/tone.npy)")
    p_tone.set_defaults(func=cmd_generate_tone)

    # ---- analyze ----
    p_an = subparsers.add_parser("analyze", help="Analyze an IQ .npy file and save outputs")
    p_an.add_argument("--input", type=str, required=True, help="Input .npy path")
    p_an.add_argument("--out", type=str, required=True, help="Output directory (e.g., outputs/run1)")
    p_an.add_argument("--nfft", type=int, default=4096, help="FFT size for spectrum")
    p_an.add_argument("--window", type=str, default="hann", help='Window: "rect" or "hann"')

    p_an.add_argument("--spectrogram", action="store_true", help="Also compute and save spectrogram.png")
    p_an.add_argument("--spec-nfft", dest="spec_nfft", type=int, default=256, help="Spectrogram nfft")
    p_an.add_argument("--spec-hop", dest="spec_hop", type=int, default=128, help="Spectrogram hop")
    p_an.add_argument("--spec-vmin", dest="spec_vmin", type=float, default=-60.0, help="Spectrogram vmin (dB)")
    p_an.add_argument("--spec-vmax", dest="spec_vmax", type=float, default=0.0, help="Spectrogram vmax (dB)")
    p_an.set_defaults(func=cmd_analyze)

    # ---- pipeline ----
    p_pipe = subparsers.add_parser("pipeline", help="Generate + analyze in one command")
    pipe_sub = p_pipe.add_subparsers(dest="pipe_cmd", required=True)

    # pipeline tone
    p_pipe_tone = pipe_sub.add_parser("tone", help="Pipeline: tone -> (optional AWGN) -> analyze")
    p_pipe_tone.add_argument("--n", type=int, required=True, help="Number of samples")
    p_pipe_tone.add_argument("--f0", type=float, required=True, help="Normalized frequency in [-0.5, 0.5)")
    p_pipe_tone.add_argument("--phase0", type=float, default=0.0, help="Initial phase (radians)")
    p_pipe_tone.add_argument("--amplitude", type=float, default=1.0, help="Amplitude")
    p_pipe_tone.add_argument("--snr-db", dest="snr_db", type=float, default=None, help="If set, add AWGN at SNR [dB]")
    p_pipe_tone.add_argument("--seed", type=int, default=968, help="RNG seed for noise")

    p_pipe_tone.add_argument("--out", type=str, required=True, help="Output directory (e.g., outputs/run_pipeline)")

    # analysis params
    p_pipe_tone.add_argument("--nfft", type=int, default=4096, help="FFT size for spectrum")
    p_pipe_tone.add_argument("--window", type=str, default="hann", help='Window: "rect" or "hann"')

    p_pipe_tone.add_argument("--spectrogram", action="store_true", help="Also compute and save spectrogram.png")
    p_pipe_tone.add_argument("--spec-nfft", dest="spec_nfft", type=int, default=256, help="Spectrogram nfft")
    p_pipe_tone.add_argument("--spec-hop", dest="spec_hop", type=int, default=128, help="Spectrogram hop")
    p_pipe_tone.add_argument("--spec-vmin", dest="spec_vmin", type=float, default=-60.0, help="Spectrogram vmin (dB)")
    p_pipe_tone.add_argument("--spec-vmax", dest="spec_vmax", type=float, default=0.0, help="Spectrogram vmax (dB)")

    p_pipe_tone.set_defaults(func=cmd_pipeline_tone)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
