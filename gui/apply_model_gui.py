#!/usr/bin/env python3
import sys
import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import re
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import matplotlib.pyplot as plt

# Add src directory to path
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = (SCRIPT_DIR / "../src").resolve()
sys.path.append(str(SRC_DIR))

from volare import data, features, model

EPS = 1e-8
MAX_CANDLES = 300_000

# Locate models
MODEL_DIR = SCRIPT_DIR / "../results/models"
model_files = [f for f in os.listdir(MODEL_DIR) if f.startswith("volare_lgb_h") and f.endswith(".pkl")]
available_horizons = sorted([int(re.search(r"h(\d+)\.pkl", f).group(1)) for f in model_files])
if not available_horizons:
    raise RuntimeError(f"No LightGBM models found in {MODEL_DIR}")
horizon_to_file = {int(re.search(r"h(\d+)\.pkl", f).group(1)): MODEL_DIR / f for f in model_files}


class VolatilityGUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("volare LightGBM model application")
        self.resize(1400, 800)

        # State
        self.df = None
        self.df_clean = None
        self.timestamps = None
        self.preds = None
        self.realised_vol = None
        self.baseline = None
        self.model = None
        self.horizon_seconds = None

        # Layouts
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.controls_layout = QtWidgets.QHBoxLayout()
        self.plot_layout = QtWidgets.QVBoxLayout()

        # Controls
        self.file_button = QtWidgets.QPushButton("Load CSV")
        self.file_button.setFixedHeight(40)
        self.file_button.clicked.connect(self.load_csv)
        self.controls_layout.addWidget(self.file_button)

        self.controls_layout.addSpacing(20)
        self.controls_layout.addWidget(QtWidgets.QLabel("Horizon (seconds):"))

        self.horizon_combo = QtWidgets.QComboBox()
        self.horizon_combo.addItems([str(h) for h in available_horizons])
        self.horizon_combo.setFixedHeight(35)
        self.horizon_combo.currentIndexChanged.connect(self.horizon_changed)
        self.controls_layout.addWidget(self.horizon_combo)

        self.slider_label = QtWidgets.QLabel("Display last 60 minutes")
        self.slider_label.setFixedHeight(30)
        self.controls_layout.addWidget(self.slider_label)

        self.time_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.time_slider.setMinimum(5)
        self.time_slider.setMaximum(180)
        self.time_slider.setValue(60)
        self.time_slider.setTickInterval(5)
        self.time_slider.valueChanged.connect(self.update_plot)
        self.controls_layout.addWidget(self.time_slider)

        self.export_button = QtWidgets.QPushButton("Export Predictions")
        self.export_button.setFixedHeight(40)
        self.export_button.clicked.connect(self.export_predictions)
        self.export_button.setEnabled(False)
        self.controls_layout.addWidget(self.export_button)

        self.main_layout.addLayout(self.controls_layout)

        # Plot
        self.fig, self.ax = plt.subplots(figsize=(12, 5))
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.plot_layout.addWidget(self.canvas)
        self.main_layout.addLayout(self.plot_layout)

        # Dark mode detection
        palette = self.palette()
        bg_color = palette.color(self.backgroundRole()).name()
        if bg_color.lower() in ['#000000', '#2e2e2e', '#353535', '#1e1e1e']:
            plt.style.use('dark_background')

    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select CSV file", "", "CSV files (*.csv)")
        if not path:
            return

        # Load last MAX_CANDLES
        self.df = data.load_candles(path, nrows=MAX_CANDLES)
        self.timestamps = pd.to_datetime(self.df['timestamp'])
        self.apply_model()
        self.export_button.setEnabled(True)

    def apply_model(self):
        h = int(self.horizon_combo.currentText())
        self.horizon_seconds = h
        model_file = horizon_to_file[h]

        # Load model
        with open(model_file, 'rb') as f:
            self.model = pickle.load(f)

        # Feature engineering
        k, alpha = 8, 1
        df = self.df.copy()
        df = features.compute_log_return(df)
        df = features.compute_rolling_volatility(df, horizon_seconds=h, k=k)
        df = features.compute_lagged_rolling_volatility(df, horizon_seconds=h, alpha=alpha, k=k)
        df = features.compute_multi_window_rolling_vol(df, horizon_seconds=h)
        df = features.compute_intraday_seasonality(df)
        df = features.compute_volatility_slope(df, horizon_seconds=h)
        df = features.compute_volatility_zscore(df, horizon_seconds=h)
        df = features.compute_volatility_acceleration(df)

        feature_cols = [c for c in df.columns if c.startswith('rolling_vol')] + \
                       [c for c in df.columns if c.startswith('tod_')] + \
                       [c for c in df.columns if c in ['vol_of_vol', 'vol_slope', 'vol_zscore', 'vol_accel']]

        self.df_clean = df[feature_cols].dropna()
        X = self.df_clean.values
        self.preds = self.model.predict(X)

        # Realised volatility baseline
        rolling_cols = [c for c in feature_cols if 'rolling_vol_' in c and 'cand' in c]
        mid = rolling_cols[len(rolling_cols) // 2]
        self.realised_vol = np.log(df.loc[self.df_clean.index, mid].values + EPS)
        self.baseline = self.realised_vol.copy()

        self.update_plot()

    def horizon_changed(self):
        if self.df is not None:
            self.apply_model()

    def update_plot(self):
        if self.preds is None:
            return

        # Determine time window
        minutes = self.time_slider.value()
        if minutes > 60:
            label = f"{minutes/60:.1f} hours"
        else:
            label = f"{minutes} minutes"
        self.slider_label.setText(f"Display last {label}")

        seconds = minutes * 60
        t_all = self.timestamps.loc[self.df_clean.index]
        t_end = t_all.iloc[-1]
        t_start = t_end - pd.Timedelta(seconds=seconds)

        mask = (t_all >= t_start) & (t_all <= t_end)
        if mask.sum() == 0:
            return

        self.ax.clear()
        self.ax.plot(t_all[mask], self.realised_vol[mask], label='Realised Volatility')
        self.ax.plot(t_all[mask], self.preds[mask], label='Predicted Volatility')

        # Horizon shading
        horizon_start = t_end
        horizon_end = t_end + pd.Timedelta(seconds=self.horizon_seconds)
        self.ax.axvspan(horizon_start, horizon_end, color='gray', alpha=0.3)
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Log Volatility")
        self.ax.legend()
        self.fig.tight_layout()
        self.canvas.draw()

    def export_predictions(self):
        if self.preds is None:
            QMessageBox.warning(self, "Warning", "No predictions to export.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Predictions", "", "CSV files (*.csv)")
        if not path:
            return
        df_save = pd.DataFrame({
            'timestamp': self.timestamps.loc[self.df_clean.index],
            'predicted_vol': self.preds,
            'realised_vol': self.realised_vol
        })
        df_save.to_csv(path, index=False)
        QMessageBox.information(self, "Saved", f"Predictions saved to {path}")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    gui = VolatilityGUI()
    gui.show()
    sys.exit(app.exec_())
