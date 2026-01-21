#!/usr/bin/env python3
import sys, os, pickle, re
from pathlib import Path

import pandas as pd
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QComboBox, QSlider, QMessageBox
)
from PyQt5.QtCore import Qt

import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

# Use custom style
plt.style.use('../styles/science.mplstyle')

# Ensure dark/light theme for plotting
from matplotlib import rcParams

# ------------------------------
# Paths and model setup
# ------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = (SCRIPT_DIR / "../src").resolve()
sys.path.append(str(SRC_DIR))

from volare import data, features, model  # your modules

MODEL_DIR = SCRIPT_DIR / "../results/models"
model_files = [f for f in os.listdir(MODEL_DIR) if f.startswith("volare_lgb_h") and f.endswith(".pkl")]

available_horizons = sorted([int(re.search(r"h(\d+)\.pkl", f).group(1)) for f in model_files])
horizon_to_file = {int(re.search(r"h(\d+)\.pkl", f).group(1)): MODEL_DIR / f for f in model_files}

EPS = 1e-8
DEFAULT_CANDLES = 300_000
DEFAULT_DISPLAY_MINUTES = 60

# ------------------------------
# GUI Class
# ------------------------------
class VolatilityApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Volatility Forecast GUI")
        self.resize(1200, 700)

        self.model = None
        self.df = None
        self.df_clean = None
        self.feature_cols = None
        self.timestamps = None
        self.preds = None
        self.pred_horizon = None
        self.medium_baseline = None
        self.actual_vol = None
        self.forecast_horizon = None

        self.dark_mode = self.is_dark_mode()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # ---------------- Controls ----------------
        controls = QHBoxLayout()

        self.horizon_label = QLabel("Horizon (s):")
        controls.addWidget(self.horizon_label)

        self.horizon_combo = QComboBox()
        for h in available_horizons:
            self.horizon_combo.addItem(str(h))
        self.horizon_combo.currentIndexChanged.connect(self.on_horizon_change)
        controls.addWidget(self.horizon_combo)

        self.load_btn = QPushButton("Load CSV")
        self.load_btn.clicked.connect(self.load_csv_file)
        controls.addWidget(self.load_btn)

        self.export_btn = QPushButton("Export Predictions")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.export_predictions)
        controls.addWidget(self.export_btn)

        layout.addLayout(controls)

        # ---------------- Slider ----------------
        slider_layout = QHBoxLayout()
        self.slider_label = QLabel(f"Display last {DEFAULT_DISPLAY_MINUTES} min")
        slider_layout.addWidget(self.slider_label)

        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setMinimum(1)
        self.time_slider.setMaximum(180)  # up to 3 hours
        self.time_slider.setValue(DEFAULT_DISPLAY_MINUTES)
        self.time_slider.valueChanged.connect(self.update_plot)
        slider_layout.addWidget(self.time_slider)

        layout.addLayout(slider_layout)

        # ---------------- Plot ----------------
        self.fig, self.ax = plt.subplots(figsize=(12, 4))
        self.ax.set_visible(False)  # hide axes initially
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        self.show()

    # ---------------- File Loading ----------------
    def load_csv_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select CSV", "", "CSV Files (*.csv)")
        if not path:
            return
        self.load_csv(path)

    def load_csv(self, path):
        try:
            self.df = data.load_candles(path, nrows=DEFAULT_CANDLES)
            self.timestamps = self.df['timestamp']
            self.apply_model()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load CSV:\n{e}")

    # ---------------- Model Application ----------------
    def apply_model(self):
        horizon_seconds = int(self.horizon_combo.currentText())
        self.forecast_horizon = horizon_seconds

        # Load model
        model_file = horizon_to_file[horizon_seconds]
        with open(model_file, "rb") as f:
            self.model = pickle.load(f)

        # Feature computation
        df = self.df.copy()
        k, alpha = 8, 1
        df = features.compute_log_return(df)
        df = features.compute_rolling_volatility(df, horizon_seconds=horizon_seconds, k=k)
        df = features.compute_lagged_rolling_volatility(df, horizon_seconds=horizon_seconds, alpha=alpha, k=k)
        df = features.compute_multi_window_rolling_vol(df, horizon_seconds=horizon_seconds)
        df = features.compute_intraday_seasonality(df)
        df = features.compute_volatility_slope(df, horizon_seconds=horizon_seconds)
        df = features.compute_volatility_zscore(df, horizon_seconds=horizon_seconds)
        df = features.compute_volatility_acceleration(df)
        df = features.compute_future_rolling_volatility(df, horizon_seconds=horizon_seconds)  # updated

        self.feature_cols = [c for c in df.columns if c.startswith('rolling_vol')] + \
                            [c for c in df.columns if c.startswith('tod_')] + \
                            [c for c in df.columns if c in ['vol_of_vol', 'vol_slope', 'vol_zscore', 'vol_accel']]

        self.df_clean = df[self.feature_cols].dropna()
        X = self.df_clean.values

        # Model prediction on known data
        self.preds = self.model.predict(X)

        # Medium-window baseline
        rolling_cols = [c for c in self.feature_cols if 'rolling_vol_' in c and 'cand' in c]
        mid_idx = self.feature_cols.index(rolling_cols[len(rolling_cols)//2])
        self.medium_baseline = np.log(X[:, mid_idx] + EPS)

        # Actual volatility (medium rolling window)
        self.actual_vol = np.log(df.loc[self.df_clean.index, 'rolling_vol'].values + EPS)

        # Prediction over horizon (using last X as input)
        # last_features = X[-1].reshape(1, -1)
        # self.pred_horizon = self.model.predict(np.repeat(last_features, horizon_seconds, axis=0))
        print('Computed all features. Now simulating future features..')
        timestamps_clean = self.timestamps.iloc[self.df_clean.index]
        X_future, self.t_horizon = model.simulate_future_features(df=self.df_clean, timestamps=timestamps_clean,
                                                                  horizon_seconds=horizon_seconds,k=k,alpha=alpha)
        print('Done simulating future features')
        # self.pred_horizon = self.model.predict(X_future)
        pred_future = self.model.predict(X_future)
        offset = self.preds[-1] - pred_future[0]
        pred_future[0] += offset
        self.pred_horizon = pred_future

        self.export_btn.setEnabled(True)
        self.update_plot()

    # ---------------- Plotting ----------------
    def update_plot(self):
        if self.preds is None or len(self.df_clean) == 0:
            return

        # Show axes if hidden
        if not self.ax.get_visible():
            self.ax.set_visible(True)

        # Time window
        minutes = self.time_slider.value()
        self.slider_label.setText(f"Display last {minutes} min")
        seconds = minutes * 60

        t_end = self.timestamps.iloc[self.df_clean.index[-1]]
        t_start = t_end - pd.Timedelta(seconds=seconds)
        mask = (self.timestamps.iloc[self.df_clean.index] >= t_start) & (self.timestamps.iloc[self.df_clean.index] <= t_end)

        t_display = self.timestamps.iloc[self.df_clean.index][mask]
        preds_display = self.preds[mask]
        baseline_display = self.medium_baseline[mask]
        actual_display = self.actual_vol[mask]

        # Horizon timestamps
        t_horizon_start = t_display.iloc[-1] + pd.Timedelta(seconds=1)
        t_horizon = pd.date_range(t_horizon_start, periods=self.forecast_horizon, freq='S')

        # Clear previous plot
        self.ax.clear()
        bg_color = "#222222" if self.dark_mode else "white"
        fg_color = "white" if self.dark_mode else "black"
        self.fig.patch.set_facecolor(bg_color)
        self.ax.set_facecolor(bg_color)
        self.ax.tick_params(colors=fg_color)
        self.ax.yaxis.label.set_color(fg_color)
        self.ax.xaxis.label.set_color(fg_color)
        self.ax.title.set_color(fg_color)

        actual_color = 'k'
        model_color = 'firebrick'
        baseline_color = 'steelblue'
        horizon_color = 'orange'
        actual_alpha = 0.6
        model_alpha = 0.8
        baseline_alpha = 0.5

        # Plot
        self.ax.plot(t_display, actual_display, label="Actual Volatility", color=actual_color, alpha=actual_alpha)
        self.ax.plot(t_display, preds_display, label="Model Prediction", color=model_color, alpha=model_alpha)
        self.ax.plot(t_display, baseline_display, label="Medium-window Baseline", color=baseline_color, alpha=baseline_alpha)
        self.ax.plot(self.t_horizon, self.pred_horizon, label="Prediction Horizon", color=model_color, alpha=model_alpha, ls='--')

        # Shaded horizon
        self.ax.axvspan(t_horizon[0], t_horizon[-1], color=horizon_color, alpha=0.2, label='Forecast Horizon')

        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Log Volatility")
        self.ax.legend(loc='lower left')  # fixed legend location
        self.fig.tight_layout()
        self.canvas.draw()

    # ---------------- Export ----------------
    def export_predictions(self):
        if self.preds is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Predictions", "", "CSV Files (*.csv)")
        if not path:
            return

        df_out = pd.DataFrame({
            "timestamp": self.timestamps.iloc[self.df_clean.index],
            "actual_vol": self.actual_vol,
            "baseline": self.medium_baseline,
            "prediction": self.preds
        })
        df_out.to_csv(path, index=False)
        QMessageBox.information(self, "Saved", f"Predictions saved to {path}")

    # ---------------- Dark Mode Detection ----------------
    def is_dark_mode(self):
        # crude detection, adjust if needed
        return QApplication.palette().color(QApplication.palette().Window).value() < 128

    # ---------------- Horizon Change ----------------
    def on_horizon_change(self):
        if self.df is not None:
            self.apply_model()


# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = VolatilityApp()
    sys.exit(app.exec_())
