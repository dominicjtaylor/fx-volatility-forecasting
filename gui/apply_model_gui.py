#!/usr/bin/env python3
import sys, os, pickle, re
from pathlib import Path

import numpy as np
import pandas as pd

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# Ensure your src is in path
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = (SCRIPT_DIR / "../src").resolve()
sys.path.append(str(SRC_DIR))

from volare import data, features, model

# --- Model loading ---
MODEL_DIR = SCRIPT_DIR / "../results/models"
model_files = [f for f in os.listdir(MODEL_DIR) if f.startswith("volare_lgb_h") and f.endswith(".pkl")]
available_horizons = sorted([int(re.search(r"h(\d+)\.pkl", f).group(1)) for f in model_files])
horizon_to_file = {int(re.search(r"h(\d+)\.pkl", f).group(1)): MODEL_DIR / f for f in model_files}

EPS = 1e-8
MAX_CANDLES = 300_000

class VolatilityApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Volatility Forecast Viewer")
        self.resize(1400, 800)

        self.df = None
        self.df_clean = None
        self.model = None
        self.preds = None
        self.baseline = None
        self.real_vol = None
        self.horizon_seconds = available_horizons[0]

        self.init_ui()
        self.show()

    def init_ui(self):
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        layout = QtWidgets.QVBoxLayout(central_widget)

        # --- Controls ---
        controls = QtWidgets.QHBoxLayout()
        layout.addLayout(controls)

        self.load_button = QtWidgets.QPushButton("Load CSV")
        self.load_button.clicked.connect(self.load_csv)
        self.load_button.setMinimumHeight(40)
        controls.addWidget(self.load_button)

        controls.addWidget(QtWidgets.QLabel("Horizon (s):"))
        self.horizon_dropdown = QtWidgets.QComboBox()
        self.horizon_dropdown.addItems([str(h) for h in available_horizons])
        self.horizon_dropdown.setCurrentIndex(0)
        self.horizon_dropdown.currentTextChanged.connect(self.horizon_changed)
        self.horizon_dropdown.setMinimumHeight(40)
        controls.addWidget(self.horizon_dropdown)

        self.export_button = QtWidgets.QPushButton("Export Predictions")
        self.export_button.clicked.connect(self.export_predictions)
        self.export_button.setMinimumHeight(40)
        self.export_button.setEnabled(False)
        controls.addWidget(self.export_button)

        # Slider for last N minutes/hours
        self.slider_label = QtWidgets.QLabel("Display last: 60 min")
        self.slider_label.setMinimumHeight(30)
        controls.addWidget(self.slider_label)

        self.time_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.time_slider.setMinimum(1)
        self.time_slider.setMaximum(360)  # 6 hours max
        self.time_slider.setValue(60)
        self.time_slider.valueChanged.connect(self.update_plot)
        controls.addWidget(self.time_slider)

        # --- Plot ---
        self.figure = Figure(figsize=(12,6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

    def load_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select CSV file", "", "CSV Files (*.csv)")
        if not file_path:
            return
        # Load last MAX_CANDLES candles
        df_full = data.load_candles(file_path, nrows=MAX_CANDLES)
        self.df = df_full
        self.apply_model()
        self.export_button.setEnabled(True)

    def apply_model(self):
        # Load model
        horizon = int(self.horizon_dropdown.currentText())
        model_file = horizon_to_file[horizon]
        with open(model_file, "rb") as f:
            self.model = pickle.load(f)

        df = self.df.copy()

        # --- Feature computation ---
        k, alpha = 8, 1
        df = features.compute_log_return(df)
        df = features.compute_rolling_volatility(df, horizon_seconds=horizon, k=k)
        df = features.compute_lagged_rolling_volatility(df, horizon_seconds=horizon, alpha=alpha, k=k)
        df = features.compute_multi_window_rolling_vol(df, horizon_seconds=horizon)
        df = features.compute_intraday_seasonality(df)
        df = features.compute_volatility_slope(df, horizon_seconds=horizon)
        df = features.compute_volatility_zscore(df, horizon_seconds=horizon)
        df = features.compute_volatility_acceleration(df)

        feature_cols = [c for c in df.columns if c.startswith("rolling_vol")] + \
                       [c for c in df.columns if c.startswith("tod_")] + \
                       [c for c in df.columns if c in ["vol_of_vol","vol_slope","vol_zscore","vol_accel"]]
        self.df_clean = df[feature_cols].dropna()
        X = self.df_clean.values

        # Predictions
        self.preds = self.model.predict(X)
        rolling_cols = [c for c in feature_cols if "rolling_vol_" in c and "cand" in c]
        mid = rolling_cols[len(rolling_cols)//2]

        # Medium-window baseline
        self.baseline = np.log(df.loc[self.df_clean.index, mid].values + EPS)
        # Realised volatility (actual)
        self.real_vol = np.log(df.loc[self.df_clean.index, "vol"].values + EPS)

        # Timestamp series
        self.timestamps = df.loc[self.df_clean.index, "timestamp"]

        self.update_plot()

    def horizon_changed(self):
        if self.df is not None:
            self.apply_model()

    def update_plot(self):
        if self.preds is None:
            return

        minutes = self.time_slider.value()
        self.slider_label.setText(f"Display last: {minutes} min")
        seconds = minutes * 60

        t_end = self.timestamps.iloc[-1]
        t_start = t_end - pd.Timedelta(seconds=seconds)
        mask = (self.timestamps >= t_start) & (self.timestamps <= t_end)

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        ax.plot(self.timestamps[mask], self.real_vol[mask], label="Actual Volatility", color="orange")
        ax.plot(self.timestamps[mask], self.preds[mask], label="Model Prediction", color="steelblue")
        ax.plot(self.timestamps[mask], self.baseline[mask], label="Medium-window Baseline", color="green", linestyle="--")

        # Horizon shading
        horizon_sec = int(self.horizon_dropdown.currentText())
        t_horizon_start = t_end
        t_horizon_end = t_end + pd.Timedelta(seconds=horizon_sec)
        ax.axvspan(t_horizon_start, t_horizon_end, color="gray", alpha=0.3, label="Prediction Horizon")

        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Log Volatility")
        ax.legend()
        ax.grid(alpha=0.3)
        self.figure.autofmt_xdate()
        self.canvas.draw()

    def export_predictions(self):
        if self.preds is None:
            QMessageBox.warning(self, "Warning", "No predictions to save yet.")
            return

        save_path, _ = QFileDialog.getSaveFileName(self, "Save Predictions", "", "CSV Files (*.csv)")
        if save_path:
            df_out = pd.DataFrame({
                "timestamp": self.timestamps,
                "actual_vol": self.real_vol,
                "baseline": self.baseline,
                "prediction": self.preds
            })
            df_out.to_csv(save_path, index=False)
            QMessageBox.information(self, "Saved", f"Predictions saved to {save_path}")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    ex = VolatilityApp()
    sys.exit(app.exec_())
