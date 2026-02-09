from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

import numpy as np

from signal_lab.pipeline import ChannelConfig, run_chain
from signal_lab.analysis import spectrum, spectrogram
from signal_lab.generate import tone_iq, multi_tone_iq
from signal_lab.channel import awgn_channel, cfo_channel, phase_noise_channel, timing_offset_channel

from .mpl_canvas import MplCanvas
from .plots import plot_constellation, plot_psd, plot_spectrogram


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Signal Lab GUI")

        root = QWidget()
        self.setCentralWidget(root)

        layout = QHBoxLayout(root)

        # Left control panel
        left = QFrame()
        left.setFrameShape(QFrame.StyledPanel)
        left.setFixedWidth(420)

        left.setStyleSheet("""
        QWidget {
            font-size: 8pt;
        }
        """)

        layout.addWidget(left)

        left_layout = QVBoxLayout(left)
        left_layout.setAlignment(Qt.AlignTop)

        title = QLabel("<b>Controls</b>")
        left_layout.addWidget(title)

        # --- Signal Type controls ---
        sig_box = QGroupBox("Signal Type")
        left_layout.addWidget(sig_box)
        sig_form = QFormLayout(sig_box)

        self.cmb_signal = QComboBox()
        self.cmb_signal.addItems(["Modulated", "Single Tone", "Multi Tone"])
        sig_form.addRow("Type:", self.cmb_signal)

        # --- Modulation controls ---
        self.mod_box = QGroupBox("Modulation")
        left_layout.addWidget(self.mod_box)
        mod_form = QFormLayout(self.mod_box)

        self.cmb_mod = QComboBox()
        self.cmb_mod.addItems(["BPSK", "QPSK", "16QAM"])
        mod_form.addRow("Type:", self.cmb_mod)

        self.spin_nbits = QSpinBox()
        self.spin_nbits.setRange(100, 5_000_000)
        self.spin_nbits.setValue(2000)
        mod_form.addRow("N_BITS:", self.spin_nbits)

        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 2_000_000_000)
        self.spin_seed.setValue(42)
        mod_form.addRow("Seed:", self.spin_seed)

        self.chk_compute_ber = QCheckBox("Compute BER")
        self.chk_compute_ber.setChecked(True)
        mod_form.addRow(self.chk_compute_ber)

        # --- Single Tone controls ---
        self.tone_box = QGroupBox("Single Tone")
        left_layout.addWidget(self.tone_box)
        tone_form = QFormLayout(self.tone_box)

        self.spin_tone_n = QSpinBox()
        self.spin_tone_n.setRange(128, 10_000_000)
        self.spin_tone_n.setValue(4096)
        tone_form.addRow("N samples:", self.spin_tone_n)

        self.spin_tone_f0 = QDoubleSpinBox()
        self.spin_tone_f0.setDecimals(6)
        self.spin_tone_f0.setRange(-0.49, 0.49)
        self.spin_tone_f0.setSingleStep(0.001)
        self.spin_tone_f0.setValue(0.12)
        tone_form.addRow("f0:", self.spin_tone_f0)

        self.spin_tone_phase0 = QDoubleSpinBox()
        self.spin_tone_phase0.setDecimals(6)
        self.spin_tone_phase0.setRange(-1000.0, 1000.0)
        self.spin_tone_phase0.setSingleStep(0.1)
        self.spin_tone_phase0.setValue(0.0)
        tone_form.addRow("phase0:", self.spin_tone_phase0)

        self.spin_tone_amp = QDoubleSpinBox()
        self.spin_tone_amp.setDecimals(6)
        self.spin_tone_amp.setRange(0.0, 1000.0)
        self.spin_tone_amp.setSingleStep(0.1)
        self.spin_tone_amp.setValue(1.0)
        tone_form.addRow("amplitude:", self.spin_tone_amp)

        self.chk_tone_norm = QCheckBox("Normalize")
        self.chk_tone_norm.setChecked(False)
        tone_form.addRow(self.chk_tone_norm)

        # --- Multi Tone controls ---
        self.mt_box = QGroupBox("Multi Tone")
        left_layout.addWidget(self.mt_box)
        mt_form = QFormLayout(self.mt_box)

        self.spin_mt_n = QSpinBox()
        self.spin_mt_n.setRange(128, 10_000_000)
        self.spin_mt_n.setValue(4096)
        mt_form.addRow("N samples:", self.spin_mt_n)

        self.chk_mt_norm = QCheckBox("Normalize")
        self.chk_mt_norm.setChecked(True)
        mt_form.addRow(self.chk_mt_norm)

        # 3 tones (enable + f0 + phase + amp)
        self.mt_enable = []
        self.mt_f0 = []
        self.mt_phase0 = []
        self.mt_amp = []

        default_tones = [
            (True,  0.12, 0.0, 1.0),
            (True, -0.18, 0.3, 0.8),
            (False, 0.25, 0.0, 0.5),
        ]

        for idx, (en, f0, ph, amp) in enumerate(default_tones, start=1):
            chk = QCheckBox(f"Tone {idx}")
            chk.setChecked(en)
            self.mt_enable.append(chk)

            f0_box = QDoubleSpinBox()
            f0_box.setDecimals(6)
            f0_box.setRange(-0.49, 0.49)
            f0_box.setSingleStep(0.001)
            f0_box.setValue(f0)
            self.mt_f0.append(f0_box)

            ph_box = QDoubleSpinBox()
            ph_box.setDecimals(6)
            ph_box.setRange(-1000.0, 1000.0)
            ph_box.setSingleStep(0.1)
            ph_box.setValue(ph)
            self.mt_phase0.append(ph_box)

            amp_box = QDoubleSpinBox()
            amp_box.setDecimals(6)
            amp_box.setRange(0.0, 1000.0)
            amp_box.setSingleStep(0.1)
            amp_box.setValue(amp)
            self.mt_amp.append(amp_box)

            row = QHBoxLayout()
            row.addWidget(chk)
            row.addWidget(QLabel("f0"))
            row.addWidget(f0_box)
            row.addWidget(QLabel("ph"))
            row.addWidget(ph_box)
            row.addWidget(QLabel("amp"))
            row.addWidget(amp_box)

            wrap = QWidget()
            wrap.setLayout(row)
            mt_form.addRow(wrap)

        # --- Channel controls ---
        ch_box = QGroupBox("Channel")
        left_layout.addWidget(ch_box)
        ch_form = QFormLayout(ch_box)

        # AWGN
        self.chk_awgn = QCheckBox("Enable AWGN")
        self.chk_awgn.setChecked(True)
        ch_form.addRow(self.chk_awgn)

        self.spin_snr = QDoubleSpinBox()
        self.spin_snr.setDecimals(2)
        self.spin_snr.setRange(-10.0, 60.0)
        self.spin_snr.setValue(12.0)
        ch_form.addRow("SNR [dB]:", self.spin_snr)

        # CFO
        self.chk_cfo = QCheckBox("Enable CFO")
        self.chk_cfo.setChecked(False)
        ch_form.addRow(self.chk_cfo)

        self.spin_fcfo = QDoubleSpinBox()
        self.spin_fcfo.setDecimals(6)
        self.spin_fcfo.setRange(-0.49, 0.49)
        self.spin_fcfo.setSingleStep(0.001)
        self.spin_fcfo.setValue(0.0)
        ch_form.addRow("f_cfo:", self.spin_fcfo)

        self.spin_phase0 = QDoubleSpinBox()
        self.spin_phase0.setDecimals(6)
        self.spin_phase0.setRange(-1000.0, 1000.0)
        self.spin_phase0.setSingleStep(0.1)
        self.spin_phase0.setValue(0.0)
        ch_form.addRow("phase0:", self.spin_phase0)

        # Phase noise
        self.chk_pn = QCheckBox("Enable Phase Noise")
        self.chk_pn.setChecked(False)
        ch_form.addRow(self.chk_pn)

        self.spin_sigma = QDoubleSpinBox()
        self.spin_sigma.setDecimals(6)
        self.spin_sigma.setRange(0.0, 2.0)
        self.spin_sigma.setSingleStep(0.001)
        self.spin_sigma.setValue(0.03)
        ch_form.addRow("sigma:", self.spin_sigma)

        self.spin_pn_seed = QSpinBox()
        self.spin_pn_seed.setRange(0, 2_000_000_000)
        self.spin_pn_seed.setValue(968)
        ch_form.addRow("pn_seed:", self.spin_pn_seed)

        # Timing offset
        self.chk_to = QCheckBox("Enable Timing Offset")
        self.chk_to.setChecked(False)
        ch_form.addRow(self.chk_to)

        self.spin_offset = QSpinBox()
        self.spin_offset.setRange(-1_000_000, 1_000_000)
        self.spin_offset.setValue(0)
        ch_form.addRow("offset:", self.spin_offset)

        # --- Analysis controls ---
        an_box = QGroupBox("Analysis")
        left_layout.addWidget(an_box)
        an_form = QFormLayout(an_box)

        self.spin_nfft = QSpinBox()
        self.spin_nfft.setRange(128, 262144)
        self.spin_nfft.setValue(4096)
        an_form.addRow("PSD NFFT:", self.spin_nfft)

        self.cmb_window = QComboBox()
        self.cmb_window.addItems(["hann", "rect"])
        an_form.addRow("Window:", self.cmb_window)

        self.spin_spec_nfft = QSpinBox()
        self.spin_spec_nfft.setRange(32, 8192)
        self.spin_spec_nfft.setValue(256)
        an_form.addRow("Spec NFFT:", self.spin_spec_nfft)

        self.spin_spec_hop = QSpinBox()
        self.spin_spec_hop.setRange(1, 8192)
        self.spin_spec_hop.setValue(128)
        an_form.addRow("Spec HOP:", self.spin_spec_hop)

        self.spin_spec_vmin = QDoubleSpinBox()
        self.spin_spec_vmin.setDecimals(1)
        self.spin_spec_vmin.setRange(-200.0, 0.0)
        self.spin_spec_vmin.setValue(-60.0)
        an_form.addRow("Spec vmin:", self.spin_spec_vmin)

        self.spin_spec_vmax = QDoubleSpinBox()
        self.spin_spec_vmax.setDecimals(1)
        self.spin_spec_vmax.setRange(-200.0, 50.0)
        self.spin_spec_vmax.setValue(0.0)
        an_form.addRow("Spec vmax:", self.spin_spec_vmax)

        # Run button
        self.btn_run = QPushButton("Run")
        left_layout.addWidget(self.btn_run)

        self.lbl_status = QLabel("Ready.")
        self.lbl_status.setWordWrap(True)
        left_layout.addWidget(self.lbl_status)

        left_layout.addStretch(1)

        # Right: tabs
        right = QWidget()
        layout.addWidget(right, stretch=1)
        right_layout = QVBoxLayout(right)

        self.tabs = QTabWidget()
        right_layout.addWidget(self.tabs)

        # Tab 1: Constellation (TX & RX)
        tab_const = QWidget()
        tab_const_layout = QVBoxLayout(tab_const)
        self.canvas_tx = MplCanvas(width=6.0, height=4.0, dpi=100)
        self.canvas_rx = MplCanvas(width=6.0, height=4.0, dpi=100)
        tab_const_layout.addWidget(self.canvas_tx)
        tab_const_layout.addWidget(self.canvas_rx)
        self.tabs.addTab(tab_const, "Constellation")

        # Tab 2: PSD
        tab_psd = QWidget()
        tab_psd_layout = QVBoxLayout(tab_psd)
        self.canvas_psd = MplCanvas(width=6.0, height=6.0, dpi=100)
        tab_psd_layout.addWidget(self.canvas_psd)
        self.tabs.addTab(tab_psd, "PSD")

        # Tab 3: Spectrogram (RX)
        tab_spec = QWidget()
        tab_spec_layout = QVBoxLayout(tab_spec)
        self.canvas_spec = MplCanvas(width=6.0, height=6.0, dpi=100)
        tab_spec_layout.addWidget(self.canvas_spec)
        self.tabs.addTab(tab_spec, "Spectrogram")

        self._spec_cbar = None

        # signals
        self.btn_run.clicked.connect(self.on_run_clicked)
        self.cmb_signal.currentIndexChanged.connect(self._update_signal_ui)

        # init visibility
        self._update_signal_ui()

    def _update_signal_ui(self) -> None:
        sig = self.cmb_signal.currentText()
        is_mod = (sig == "Modulated")
        is_tone = (sig == "Single Tone")
        is_mt = (sig == "Multi Tone")

        self.mod_box.setVisible(is_mod)
        self.tone_box.setVisible(is_tone)
        self.mt_box.setVisible(is_mt)

        # show BER only for modulated
        self.chk_compute_ber.setEnabled(is_mod)

    def _build_channel_config(self) -> ChannelConfig:
        return ChannelConfig(
            enable_awgn=self.chk_awgn.isChecked(),
            snr_db=float(self.spin_snr.value()),
            enable_cfo=self.chk_cfo.isChecked(),
            f_cfo=float(self.spin_fcfo.value()),
            phase0=float(self.spin_phase0.value()),
            enable_phase_noise=self.chk_pn.isChecked(),
            sigma=float(self.spin_sigma.value()),
            pn_seed=int(self.spin_pn_seed.value()),
            enable_timing_offset=self.chk_to.isChecked(),
            offset=int(self.spin_offset.value()),
            order=("cfo", "phase_noise", "awgn", "timing_offset"),
        )

    def _apply_channel_chain(self, iq: np.ndarray, cfg: ChannelConfig, seed: int) -> np.ndarray:
        """
        Apply channel impairments in the order defined by cfg.order.
        For AWGN: uses rng seeded by 'seed'.
        For Phase Noise: uses cfg.pn_seed (deterministic).
        """
        y = np.asarray(iq)
        rng = np.random.default_rng(seed)

        for name in cfg.order:
            if name == "cfo" and cfg.enable_cfo:
                y = cfo_channel(y, f_cfo=cfg.f_cfo, phase0=cfg.phase0)
            elif name == "phase_noise" and cfg.enable_phase_noise:
                y = phase_noise_channel(y, sigma=cfg.sigma, seed=cfg.pn_seed)
            elif name == "awgn" and cfg.enable_awgn:
                y = awgn_channel(y, snr_db=cfg.snr_db, rng=rng)
            elif name == "timing_offset" and cfg.enable_timing_offset:
                y = timing_offset_channel(y, offset=cfg.offset)
        return y

    def on_run_clicked(self) -> None:
        try:
            sig_type = self.cmb_signal.currentText()
            cfg = self._build_channel_config()

            # analysis params
            nfft = int(self.spin_nfft.value())
            window = self.cmb_window.currentText()
            spec_nfft = int(self.spin_spec_nfft.value())
            spec_hop = int(self.spin_spec_hop.value())
            vmin = float(self.spin_spec_vmin.value())
            vmax = float(self.spin_spec_vmax.value())

            # Build imp text
            imp = []
            if cfg.enable_cfo:
                imp.append(f"CFO(f={cfg.f_cfo:g})")
            if cfg.enable_phase_noise:
                imp.append(f"PN(σ={cfg.sigma:g})")
            if cfg.enable_awgn:
                imp.append(f"AWGN(SNR={cfg.snr_db:g} dB)")
            if cfg.enable_timing_offset:
                imp.append(f"TO(offset={cfg.offset})")
            imp_txt = " + ".join(imp) if imp else "No channel"

            # ---------------- Modulated (existing flow) ----------------
            if sig_type == "Modulated":
                mod = self.cmb_mod.currentText()
                n_bits = int(self.spin_nbits.value())
                seed = int(self.spin_seed.value())
                compute_ber = self.chk_compute_ber.isChecked()

                out = run_chain(
                    modulation=mod,  # type: ignore[arg-type]
                    n_bits=n_bits,
                    seed=seed,
                    cfg=cfg,
                    compute_ber=compute_ber,
                )

                symbols_tx = out["symbols_tx"]
                symbols_rx = out["symbols_rx"]

                # ---- Constellation tab ----
                plot_constellation(
                    self.canvas_tx.ax,
                    symbols_tx,
                    title=f"Constellation (TX) | {mod} | Nbits={n_bits}",
                )
                self.canvas_tx.draw()

                plot_constellation(
                    self.canvas_rx.ax,
                    symbols_rx,
                    title=f"Constellation (RX) | {mod} | {imp_txt}",
                )
                self.canvas_rx.draw()

                # ---- PSD tab (TX vs RX) ----
                f_tx, psd_tx = spectrum(symbols_tx, nfft=nfft, window=window)
                f_rx, psd_rx = spectrum(symbols_rx, nfft=nfft, window=window)

                ax_psd = self.canvas_psd.ax
                ax_psd.clear()
                plot_psd(ax_psd, f_tx, psd_tx, label="TX", title=f"PSD | {mod} | window={window}, nfft={nfft}")
                plot_psd(ax_psd, f_rx, psd_rx, label="RX", title=f"PSD | {mod} | window={window}, nfft={nfft}")
                ax_psd.legend()
                ax_psd.figure.tight_layout()
                self.canvas_psd.draw()

                # ---- Spectrogram tab (RX) ----
                ax = self.canvas_spec.ax
                ax.clear()

                t, f, S_db = spectrogram(symbols_rx, nfft=spec_nfft, hop=spec_hop, window=window)

                im = plot_spectrogram(
                    ax,
                    t=t,
                    f=f,
                    S_db=S_db,
                    title=f"Spectrogram (RX) | {mod} | {imp_txt}",
                    vmin=vmin,
                    vmax=vmax,
                )

                if self._spec_cbar is not None:
                    self._spec_cbar.remove()
                    self._spec_cbar = None

                # יצירת colorbar חדש
                self._spec_cbar = ax.figure.colorbar(
                    im,
                    ax=ax,
                    label="Power [dB] (relative to peak)",
                )

                ax.figure.tight_layout()
                self.canvas_spec.draw()

                # # ---- Spectrogram tab (RX) ----
                # t, f, S_db = spectrogram(symbols_rx, nfft=spec_nfft, hop=spec_hop, window=window)
                # plot_spectrogram(
                #     self.canvas_spec.ax,
                #     t=t,
                #     f=f,
                #     S_db=S_db,
                #     title=f"Spectrogram (RX) | {mod} | {imp_txt}",
                #     vmin=vmin,
                #     vmax=vmax,
                # )
                # self.canvas_spec.draw()

                msg = f"OK | signal=Modulated | mod={mod}"
                if compute_ber and "ber" in out:
                    msg += f" | BER={out['ber']:.3e}"
                self.lbl_status.setText(msg)
                return

            # ---------------- Single Tone ----------------
            if sig_type == "Single Tone":
                seed = int(self.spin_seed.value())  # reuse same seed control
                n = int(self.spin_tone_n.value())
                f0 = float(self.spin_tone_f0.value())
                phase0 = float(self.spin_tone_phase0.value())
                amp = float(self.spin_tone_amp.value())
                normalize = bool(self.chk_tone_norm.isChecked())

                iq_clean = tone_iq(n=n, f0=f0, phase0=phase0, amplitude=amp)
                if normalize:
                    p = float(np.mean(np.abs(iq_clean) ** 2))
                    if p > 0:
                        iq_clean = iq_clean / np.sqrt(p)

                iq_rx = self._apply_channel_chain(iq_clean, cfg, seed=seed)

                # Constellation tab: not applicable
                self.canvas_tx.ax.clear()
                self.canvas_tx.ax.text(0.5, 0.5, "Constellation N/A for Tone", ha="center", va="center")
                self.canvas_rx.ax.clear()
                self.canvas_rx.ax.text(0.5, 0.5, "Constellation N/A for Tone", ha="center", va="center")
                self.canvas_tx.draw()
                self.canvas_rx.draw()

                # PSD tab
                f_tx, psd_tx = spectrum(iq_clean, nfft=nfft, window=window)
                f_rx, psd_rx = spectrum(iq_rx, nfft=nfft, window=window)

                ax_psd = self.canvas_psd.ax
                ax_psd.clear()
                plot_psd(ax_psd, f_tx, psd_tx, label="clean", title=f"PSD | Tone | window={window}, nfft={nfft}")
                plot_psd(ax_psd, f_rx, psd_rx, label=f"rx ({imp_txt})", title=f"PSD | Tone | window={window}, nfft={nfft}")
                ax_psd.legend()
                ax_psd.figure.tight_layout()
                self.canvas_psd.draw()

                # Spectrogram tab (RX)
                t, f, S_db = spectrogram(iq_rx, nfft=spec_nfft, hop=spec_hop, window=window)
                plot_spectrogram(
                    self.canvas_spec.ax,
                    t=t,
                    f=f,
                    S_db=S_db,
                    title=f"Spectrogram (RX) | Tone | {imp_txt}",
                    vmin=vmin,
                    vmax=vmax,
                )
                self.canvas_spec.draw()

                self.lbl_status.setText("OK | signal=Single Tone")
                return

            # ---------------- Multi Tone ----------------
            if sig_type == "Multi Tone":
                seed = int(self.spin_seed.value())  # reuse same seed control
                n = int(self.spin_mt_n.value())
                normalize = bool(self.chk_mt_norm.isChecked())

                f0_list = []
                phase0_list = []
                amp_list = []

                for i in range(3):
                    if self.mt_enable[i].isChecked():
                        f0_list.append(float(self.mt_f0[i].value()))
                        phase0_list.append(float(self.mt_phase0[i].value()))
                        amp_list.append(float(self.mt_amp[i].value()))

                if len(f0_list) == 0:
                    raise ValueError("Multi Tone: enable at least one tone")

                iq_clean = multi_tone_iq(
                    n=n,
                    f0_list=f0_list,
                    phase0_list=phase0_list,
                    amplitude_list=amp_list,
                    normalize=normalize,
                )

                iq_rx = self._apply_channel_chain(iq_clean, cfg, seed=seed)

                # Constellation tab: not applicable
                self.canvas_tx.ax.clear()
                self.canvas_tx.ax.text(0.5, 0.5, "Constellation N/A for Multi Tone", ha="center", va="center")
                self.canvas_rx.ax.clear()
                self.canvas_rx.ax.text(0.5, 0.5, "Constellation N/A for Multi Tone", ha="center", va="center")
                self.canvas_tx.draw()
                self.canvas_rx.draw()

                # PSD tab
                f_tx, psd_tx = spectrum(iq_clean, nfft=nfft, window=window)
                f_rx, psd_rx = spectrum(iq_rx, nfft=nfft, window=window)

                ax_psd = self.canvas_psd.ax
                ax_psd.clear()
                plot_psd(ax_psd, f_tx, psd_tx, label="clean", title=f"PSD | Multi Tone | window={window}, nfft={nfft}")
                plot_psd(ax_psd, f_rx, psd_rx, label=f"rx ({imp_txt})", title=f"PSD | Multi Tone | window={window}, nfft={nfft}")
                ax_psd.legend()
                ax_psd.figure.tight_layout()
                self.canvas_psd.draw()

                # Spectrogram tab (RX)
                t, f, S_db = spectrogram(iq_rx, nfft=spec_nfft, hop=spec_hop, window=window)
                plot_spectrogram(
                    self.canvas_spec.ax,
                    t=t,
                    f=f,
                    S_db=S_db,
                    title=f"Spectrogram (RX) | Multi Tone | {imp_txt}",
                    vmin=vmin,
                    vmax=vmax,
                )
                self.canvas_spec.draw()

                self.lbl_status.setText("OK | signal=Multi Tone")
                return

        except Exception as e:
            self.lbl_status.setText(f"ERROR: {e}")
