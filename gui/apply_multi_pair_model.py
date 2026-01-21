#!/usr/bin/env python3
import sys, os, pickle, re
from pathlib import Path

import pandas as pd
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QSlider, QMessageBox
)
from PyQt5.QtCore import Qt

import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

plt.style.use('../styles/science.mplstyle')

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = (SCRIPT_DIR / "../src").resolve()
sys.path.append(str(SRC_DIR))

from volare import data, features, model

MODEL_DIR = SCRIPT_DIR / "../results/models"
EPS = 1e-8
HORIZON_SECONDS = 60 * 60  # Fixed 60-min horizon
DEFAULT_CANDLES = 500_000
DEFAULT_DISPLAY_MINUTES = 60

# ------------------------------
# GUI Class
# ------------------------------
class VolatilityApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Volatility Forecast GUI — Horizon: {HORIZON_SECONDS//60} min")
        self.resize(1200, 700)

        self.model = None
        self.df = None
        self.df_clean = None
        self.feature_cols = None
        self.timestamps = None
        self.preds = None
        self.medium_baseline = None
        self.actual_vol = None
        self.t_horizon = None

        self.dark_mode = self.is_dark_mode()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # ---------------- Controls ----------------
        controls = QHBoxLayout()
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
        self.time_slider.setMaximum(180)
        self.time_slider.setValue(DEFAULT_DISPLAY_MINUTES)
        self.time_slider.valueChanged.connect(self.update_plot)
        slider_layout.addWidget(self.time_slider)
        layout.addLayout(slider_layout)

        # ---------------- Plot ----------------
        self.fig, self.ax = plt.subplots(figsize=(12, 4))
        self.ax.set_visible(False)
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
            # Load last 500_000 rows
            self.df = data.load_candles(path, nrows=None)
            self.df = self.df.iloc[-DEFAULT_CANDLES:]
            self.timestamps = self.df['timestamp']

            # Determine currency pair from filename
            stem = Path(path).stem  # e.g., 'questdb-gbpusd'
            if '-' not in stem:
                raise ValueError("Cannot parse currency pair from filename")
            currencies = stem.split('-')[1]
            model_file = Path(MODEL_DIR) / f"volare_lgb_{currencies}_h{HORIZON_SECONDS}.pkl"

            if not model_file.exists():
                raise FileNotFoundError(f"Model file not found: {model_file}")

            with open(model_file, "rb") as f:
                self.model = pickle.load(f)

            self.apply_model()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load CSV or model:\n{e}")

    # ---------------- Model Application ----------------
    def apply_model(self):
        horizon_seconds = HORIZON_SECONDS
        k, alpha = 8, 1

        df = self.df.copy()
        df = features.compute_log_return(df)
        df = features.compute_rolling_volatility(df, horizon_seconds=horizon_seconds, k=k)
        df = features.compute_lagged_rolling_volatility(df, horizon_seconds=horizon_seconds, alpha=alpha, k=k)
        df = features.compute_multi_window_rolling_vol(df, horizon_seconds=horizon_seconds)
        df = features.compute_intraday_seasonality(df)
        df = features.compute_volatility_slope(df, horizon_seconds=horizon_seconds)
        df = features.compute_volatility_zscore(df, horizon_seconds=horizon_seconds)
        df = features.compute_volatility_acceleration(df)
        df = features.compute_future_rolling_volatility(df, horizon_seconds=horizon_seconds)

        self.feature_cols = [c for c in df.columns if c.startswith('rolling_vol')] + \
                            [c for c in df.columns if c.startswith('tod_')] + \
                            [c for c in ['vol_of_vol', 'vol_slope', 'vol_zscore', 'vol_accel']]

        self.df_clean = df[self.feature_cols].dropna()
        X = self.df_clean.values
        self.preds = self.model.predict(X)

        # Medium-window baseline
        rolling_cols = [c for c in self.feature_cols if 'rolling_vol_' in c and 'cand' in c]
        mid_idx = self.feature_cols.index(rolling_cols[len(rolling_cols)//2])
        self.medium_baseline = np.log(X[:, mid_idx] + EPS)

        # Actual volatility
        self.actual_vol = np.log(df.loc[self.df_clean.index, 'rolling_vol'].values + EPS)

        # Simulate forecast horizon
        print('Simulating future features')
        X_future, self.t_horizon = model.simulate_future_features_conditional(
            df=df, timestamps=df['timestamp'], horizon_seconds=horizon_seconds
        )
        print('Done simulating future features')
        pred_future = self.model.predict(X_future)
        pred_future[0] = self.preds[-1]
        self.pred_horizon = pred_future

        self.export_btn.setEnabled(True)
        self.update_plot()

    # ---------------- Plotting ----------------
    def update_plot(self):
        if self.preds is None or len(self.df_clean) == 0:
            return

        if not self.ax.get_visible():
            self.ax.set_visible(True)

        minutes = self.time_slider.value()
        self.slider_label.setText(f"Display last {minutes} min")
        seconds = minutes * 60

        print('Compute time for axes')
        pos_idx = np.searchsorted(self.df.index.values, self.df_clean.index.values)
        t_end = self.timestamps.iloc[pos_idx[-1]]
        t_start = t_end - pd.Timedelta(seconds=seconds)
        mask = (self.timestamps.iloc[pos_idx] >= t_start) & (self.timestamps.iloc[pos_idx] <= t_end)

        print('Get timestamps')
        t_display = self.timestamps.iloc[pos_idx][mask]
        preds_display = self.preds[mask]
        baseline_display = self.medium_baseline[mask]
        actual_display = self.actual_vol[mask]

        print("Clear axes")
        self.ax.clear()
        bg_color = "#222222" if self.dark_mode else "white"
        fg_color = "white" if self.dark_mode else "black"
        self.fig.patch.set_facecolor(bg_color)
        self.ax.set_facecolor(bg_color)
        self.ax.tick_params(colors=fg_color)
        self.ax.yaxis.label.set_color(fg_color)
        self.ax.xaxis.label.set_color(fg_color)
        self.ax.title.set_color(fg_color)

        self.ax.plot(t_display, actual_display, label="Actual Volatility", color='k', alpha=0.6)
        self.ax.plot(t_display, preds_display, label="Model Prediction", color='firebrick', alpha=0.8)
        self.ax.plot(t_display, baseline_display, label="Medium-window Baseline", color='steelblue', alpha=0.5)
        self.ax.plot(self.t_horizon, self.pred_horizon, label="Prediction Horizon", color='firebrick', alpha=0.8, ls='--')
        self.ax.axvspan(self.t_horizon[0], self.t_horizon[-1], color='orange', alpha=0.2, label='Forecast Horizon')

        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Log Volatility")
        self.ax.legend(loc='lower left')
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

    # ---------------- Dark Mode ----------------
    def is_dark_mode(self):
        return QApplication.palette().color(QApplication.palette().Window).value() < 128


# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = VolatilityApp()
    sys.exit(app.exec_())
