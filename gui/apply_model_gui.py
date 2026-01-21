#!/usr/bin/env python3
import sys
import os
import re
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog,
    QComboBox, QSlider
)
from PyQt5.QtCore import Qt

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = (SCRIPT_DIR / "../src").resolve()
sys.path.append(str(SRC_DIR))

from volare import data, features

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
NROWS = 300_000
EPS = 1e-8

MODEL_DIR = SCRIPT_DIR / "../results/models"

# ---------------------------------------------------------------------
# Main GUI
# ---------------------------------------------------------------------
class VolatilityGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("volare – Volatility Forecast Viewer")
        self.resize(1300, 750)

        self.df = None
        self.df_clean = None
        self.timestamps = None
        self.preds = None
        self.baseline = None
        self.realised_vol = None
        self.model = None

        self._load_models()
        self._build_ui()

    # -----------------------------------------------------------------
    def _load_models(self):
        model_files = [
            f for f in MODEL_DIR.iterdir()
            if f.name.startswith("volare_lgb_h") and f.suffix == ".pkl"
        ]

        self.horizons = sorted(
            int(re.search(r"h(\d+)\.pkl", f.name).group(1))
            for f in model_files
        )

        self.horizon_to_file = {
            int(re.search(r"h(\d+)\.pkl", f.name).group(1)): f
            for f in model_files
        }

    # -----------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # ---------------- Controls ----------------
        controls = QHBoxLayout()

        load_btn = QPushButton("Load CSV")
        load_btn.setMinimumHeight(40)
        load_btn.clicked.connect(self.load_csv)

        controls.addWidget(load_btn)

        controls.addWidget(QLabel("Forecast horizon (seconds):"))

        self.horizon_combo = QComboBox()
        for h in self.horizons:
            self.horizon_combo.addItem(str(h))
        self.horizon_combo.currentIndexChanged.connect(self.reapply_model)
        controls.addWidget(self.horizon_combo)

        controls.addStretch()

        main_layout.addLayout(controls)

        # ---------------- Slider ----------------
        slider_layout = QHBoxLayout()
        self.slider_label = QLabel("Display last 500 minutes")

        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setMinimum(10)
        self.time_slider.setMaximum(5000)
        self.time_slider.setValue(500)
        self.time_slider.valueChanged.connect(self.update_plot)

        slider_layout.addWidget(self.slider_label)
        slider_layout.addWidget(self.time_slider)

        main_layout.addLayout(slider_layout)

        # ---------------- Plot area ----------------
        self.plot_container = QVBoxLayout()
        self.placeholder = QLabel("Load a CSV file to view predictions")
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.plot_container.addWidget(self.placeholder)

        main_layout.addLayout(self.plot_container)

    # -----------------------------------------------------------------
    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select candle CSV", "", "CSV files (*.csv)"
        )
        if not path:
            return

        self.df = data.load_candles(path, nrows=NROWS)

        # --- timestamps ---
        self.timestamps = pd.to_datetime(self.df["timestamp"], unit="s")

        self.reapply_model()

    # -----------------------------------------------------------------
    def reapply_model(self):
        if self.df is None:
            return

        horizon = int(self.horizon_combo.currentText())
        model_path = self.horizon_to_file[horizon]

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        self._compute_predictions()
        self.update_plot()

    # -----------------------------------------------------------------
    def _compute_predictions(self):
        df = self.df.copy()

        k, alpha = 8, 1
        horizon = int(self.horizon_combo.currentText())

        df = features.compute_log_return(df)
        df = features.compute_rolling_volatility(df, horizon_seconds=horizon, k=k)
        df = features.compute_lagged_rolling_volatility(df, horizon_seconds=horizon, alpha=alpha, k=k)
        df = features.compute_multi_window_rolling_vol(df, horizon_seconds=horizon)
        df = features.compute_intraday_seasonality(df)
        df = features.compute_volatility_slope(df, horizon_seconds=horizon)
        df = features.compute_volatility_zscore(df, horizon_seconds=horizon)
        df = features.compute_volatility_acceleration(df)

        feature_cols = (
            [c for c in df.columns if c.startswith("rolling_vol")] +
            [c for c in df.columns if c.startswith("tod_")] +
            ["vol_of_vol", "vol_slope", "vol_zscore", "vol_accel"]
        )

        self.df_clean = df[feature_cols].dropna()
        X = self.df_clean[feature_cols]

        self.preds = self.model.predict(X)

        rolling_cols = [c for c in feature_cols if "rolling_vol_" in c and "cand" in c]
        mid = rolling_cols[len(rolling_cols) // 2]

        self.realised_vol = np.log(
            df.loc[self.df_clean.index, mid].values + EPS
        )

        self.ts_feat = pd.to_datetime(df.loc[self.df_clean.index, "timestamp"])

        self.baseline = self.realised_vol.copy()

    # -----------------------------------------------------------------
    def update_plot(self):
        if self.preds is None:
            return

        minutes = self.time_slider.value()
        self.slider_label.setText(f"Display last {minutes} minutes")

        seconds = minutes * 60

        # t_all = self.timestamps.iloc[self.df_clean.index]
        t_all = self.ts_feat
        t_end = t_all.iloc[-1]
        t_start = t_end - pd.Timedelta(seconds=seconds)

        mask = (t_all >= t_start) & (t_all <= t_end)

        # Remove placeholder
        if self.placeholder:
            self.plot_container.removeWidget(self.placeholder)
            self.placeholder.deleteLater()
            self.placeholder = None

        # Clear old canvas
        for i in reversed(range(self.plot_container.count())):
            self.plot_container.itemAt(i).widget().setParent(None)

        assert len(t_all) == len(self.realised_vol) == len(self.preds)

        fig, ax = plt.subplots(figsize=(12, 4))

        dark = self.palette().color(self.backgroundRole()).lightness() < 128
        if dark:
            fig.patch.set_facecolor("#121212")
            ax.set_facecolor("#121212")
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")

        ax.plot(t_all[mask], self.realised_vol[mask],
                label="Realised rolling vol", color="#bbbbbb")

        ax.plot(t_all[mask], self.preds[mask],
                label="Model prediction", color="#4fa3ff")

        ax.plot(t_all[mask], self.baseline[mask],
                label="Rolling baseline", color="#f4a261", alpha=0.8)

        horizon = int(self.horizon_combo.currentText())

        ax.axvspan(
            t_end,
            t_end + pd.Timedelta(seconds=horizon),
            color="gray", alpha=0.25,
            label="Forecast horizon"
        )

        ax.axvline(t_end, linestyle="--", color="white" if dark else "black")

        ax.set_xlabel("Time")
        ax.set_ylabel("Log volatility")
        ax.legend()
        fig.tight_layout()

        canvas = FigureCanvas(fig)
        self.plot_container.addWidget(canvas)

# ---------------------------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = VolatilityGUI()
    win.show()
    sys.exit(app.exec_())
