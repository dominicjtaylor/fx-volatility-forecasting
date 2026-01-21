#!/usr/bin/env python3
import sys
from pathlib import Path
import pickle
import os
import re
import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QFileDialog, QSlider
)
from PyQt5.QtCore import Qt

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from volare import data, features, model

EPS = 1e-8
MAX_CANDLES = 300_000  # Number of last candles to read
HORIZON_SEC_OPTIONS = None  # Filled later

# -------------------------------
# Utility functions
# -------------------------------
def load_models(model_dir):
    model_files = [f for f in os.listdir(model_dir) if f.startswith("volare_lgb_h") and f.endswith(".pkl")]
    horizons = sorted([int(re.search(r"h(\d+)\.pkl", f).group(1)) for f in model_files])
    horizon_to_file = {h: Path(model_dir) / f"volare_lgb_h{h}.pkl" for h in horizons}
    return horizons, horizon_to_file

# -------------------------------
# Main GUI
# -------------------------------
class VolatilityApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Volare Volatility Forecast")
        self.resize(1400, 800)

        self.df = None
        self.df_clean = None
        self.timestamps = None
        self.preds = None
        self.baseline = None
        self.realised_vol = None
        self.forward_pred = None
        self.model = None
        self.horizon_sec = None
        self.horizon_file = None

        self.dark_mode = False  # Could auto-detect system if needed

        # -------------------------------
        # Load available models
        # -------------------------------
        script_dir = Path(__file__).resolve().parent
        model_dir = (script_dir / "../results/models").resolve()
        self.available_horizons, self.horizon_to_file = load_models(model_dir)
        self.horizon_sec = self.available_horizons[0]
        self.horizon_file = self.horizon_to_file[self.horizon_sec]

        # -------------------------------
        # Layout
        # -------------------------------
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Controls
        controls_layout = QHBoxLayout()
        layout.addLayout(controls_layout)

        self.load_button = QPushButton("Load CSV")
        self.load_button.setMinimumHeight(40)
        self.load_button.clicked.connect(self.load_csv)
        controls_layout.addWidget(self.load_button)

        controls_layout.addWidget(QLabel("Horizon (s):"))
        self.horizon_combo = QComboBox()
        self.horizon_combo.setMinimumHeight(30)
        for h in self.available_horizons:
            self.horizon_combo.addItem(str(h))
        self.horizon_combo.currentTextChanged.connect(self.horizon_changed)
        controls_layout.addWidget(self.horizon_combo)

        self.export_button = QPushButton("Export Predictions")
        self.export_button.setMinimumHeight(40)
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_predictions)
        controls_layout.addWidget(self.export_button)

        # Slider for zoom
        self.slider_label = QLabel("Display last 60 min")
        controls_layout.addWidget(self.slider_label)
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setMinimum(1)
        self.time_slider.setMaximum(180)  # max 3 hours
        self.time_slider.setValue(60)
        self.time_slider.valueChanged.connect(self.update_plot)
        controls_layout.addWidget(self.time_slider)

        # Figure
        self.fig = Figure(figsize=(12,6))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

    # -------------------------------
    # Load CSV
    # -------------------------------
    def load_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select CSV file", "", "CSV files (*.csv)")
        if not file_path:
            return
        self.df = data.load_candles(file_path, nrows=MAX_CANDLES)
        self.timestamps = self.df.index

        # Compute features
        self.df = features.compute_log_return(self.df)
        self.df = features.compute_rolling_volatility(self.df, horizon_seconds=self.horizon_sec, k=8)
        self.df = features.compute_lagged_rolling_volatility(self.df, horizon_seconds=self.horizon_sec, alpha=1, k=8)
        self.df = features.compute_multi_window_rolling_vol(self.df, horizon_seconds=self.horizon_sec)
        self.df = features.compute_intraday_seasonality(self.df)
        self.df = features.compute_volatility_slope(self.df, horizon_seconds=self.horizon_sec)
        self.df = features.compute_volatility_zscore(self.df, horizon_seconds=self.horizon_sec)
        self.df = features.compute_volatility_acceleration(self.df)

        feature_cols = [c for c in self.df.columns if c.startswith('rolling_vol')] + \
                       [c for c in self.df.columns if c.startswith('tod_')] + \
                       [c for c in self.df.columns if c in ['vol_of_vol','vol_slope','vol_zscore','vol_accel']]
        self.df_clean = self.df[feature_cols].dropna()

        # Compute realised volatility (full, not baseline)
        rolling_cols = [c for c in feature_cols if 'rolling_vol_' in c and 'cand' in c]
        mid = rolling_cols[len(rolling_cols)//2]
        self.realised_vol = np.log(self.df.loc[self.df_clean.index, mid].values + EPS)
        self.baseline = self.realised_vol.copy()

        # Load model
        with open(self.horizon_file, "rb") as f:
            self.model = pickle.load(f)

        # Compute predictions
        self.preds = self.model.predict(self.df_clean.values)

        # Compute forward prediction for horizon
        self.compute_forward_prediction()

        self.export_button.setEnabled(True)
        self.update_plot()

    # -------------------------------
    # Horizon changed
    # -------------------------------
    def horizon_changed(self, text):
        self.horizon_sec = int(text)
        self.horizon_file = self.horizon_to_file[self.horizon_sec]
        if self.df is not None:
            with open(self.horizon_file, "rb") as f:
                self.model = pickle.load(f)
            self.preds = self.model.predict(self.df_clean.values)
            self.compute_forward_prediction()
            self.update_plot()

    # -------------------------------
    # Compute forward prediction
    # -------------------------------
    def compute_forward_prediction(self):
        X_last = self.df_clean.values[-1].reshape(1, -1)
        self.forward_pred = self.model.predict(X_last)[0]

    # -------------------------------
    # Update plot
    # -------------------------------
    def update_plot(self):
        if self.df is None:
            return

        self.fig.clear()
        ax = self.fig.add_subplot(111)

        # Determine time window from slider
        minutes = self.time_slider.value()
        if minutes <= 60:
            self.slider_label.setText(f"Display last {minutes} min")
        else:
            self.slider_label.setText(f"Display last {minutes/60:.1f} h")
        seconds = minutes * 60

        t_all = self.timestamps[self.df_clean.index]
        t_end = t_all[-1]
        t_start = t_end - pd.Timedelta(seconds=seconds)
        mask = (t_all >= t_start) & (t_all <= t_end)

        t_disp = t_all[mask]
        preds_disp = self.preds[mask]
        baseline_disp = self.baseline[mask]
        realised_disp = self.realised_vol[mask]

        # Plot lines
        color_bg = "#222222" if self.dark_mode else "#ffffff"
        color_grid = "#555555" if self.dark_mode else "#cccccc"
        ax.set_facecolor(color_bg)
        ax.grid(True, color=color_grid, alpha=0.3)

        ax.plot(t_disp, realised_disp, label="Actual Volatility", color="red")
        ax.plot(t_disp, baseline_disp, label="Medium-window Baseline", color="orange")
        ax.plot(t_disp, preds_disp, label="Model Prediction", color="steelblue")

        # Horizon shaded region and forward prediction
        t_horizon_start = t_all.iloc[-1]
        t_horizon_end = t_horizon_start + pd.Timedelta(seconds=self.horizon_sec)
        ax.axvspan(t_horizon_start, t_horizon_end, color="grey", alpha=0.3, label="Prediction Horizon")
        ax.hlines(self.forward_pred, xmin=t_horizon_start, xmax=t_horizon_end, linestyles="dashed",
                  colors="steelblue", label="Forward Prediction")

        ax.set_xlabel("Time")
        ax.set_ylabel("Log Volatility")
        ax.set_title("Volatility Forecast vs Baseline")
        ax.legend()
        self.fig.tight_layout()
        self.canvas.draw()

    # -------------------------------
    # Export predictions
    # -------------------------------
    def export_predictions(self):
        if self.preds is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Predictions", "", "CSV files (*.csv)")
        if not path:
            return
        df_save = pd.DataFrame({
            "timestamp": self.timestamps[self.df_clean.index],
            "realised_vol": self.realised_vol,
            "baseline": self.baseline,
            "prediction": self.preds
        })
        df_save.to_csv(path, index=False)

# -------------------------------
# Run application
# -------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VolatilityApp()
    window.show()
    sys.exit(app.exec_())
