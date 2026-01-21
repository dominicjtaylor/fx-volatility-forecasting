#!/usr/bin/env python3
import sys, os, pickle, re
from pathlib import Path
import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QFileDialog, QSlider, QMessageBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette, QColor

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

# Add src to path
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = (SCRIPT_DIR / "../src").resolve()
sys.path.append(str(SRC_DIR))

from volare import data, features, model

EPS = 1e-8
MAX_CANDLES = 300_000

# --- Locate available models ---
MODEL_DIR = SCRIPT_DIR / "../results/models"
model_files = [f for f in os.listdir(MODEL_DIR) if f.startswith("volare_lgb_h") and f.endswith(".pkl")]
available_horizons = sorted([int(re.search(r"h(\d+)\.pkl", f).group(1)) for f in model_files])
if not available_horizons:
    raise RuntimeError(f"No LightGBM models found in {MODEL_DIR}")

horizon_to_file = {int(re.search(r"h(\d+)\.pkl", f).group(1)): MODEL_DIR / f for f in model_files}

# --- Main GUI Class ---
class VolatilityApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Volare LightGBM Volatility Forecast")
        self.resize(1400, 800)
        self.df = None
        self.df_clean = None
        self.timestamps = None
        self.model = None
        self.preds = None
        self.baseline = None
        self.future_vol = None
        self.forecast_horizon = available_horizons[0]

        # Layouts
        main_layout = QVBoxLayout()
        controls_layout = QHBoxLayout()
        main_layout.addLayout(controls_layout)

        # Horizon selector
        controls_layout.addWidget(QLabel("Forecast Horizon (s):"))
        self.horizon_combo = QComboBox()
        self.horizon_combo.addItems([str(h) for h in available_horizons])
        self.horizon_combo.setCurrentText(str(self.forecast_horizon))
        self.horizon_combo.currentTextChanged.connect(self.change_horizon)
        controls_layout.addWidget(self.horizon_combo)

        # Load CSV button
        self.load_btn = QPushButton("Load CSV")
        self.load_btn.clicked.connect(self.load_csv)
        self.load_btn.setFixedHeight(40)
        controls_layout.addWidget(self.load_btn)

        # Export predictions button
        self.save_btn = QPushButton("Export Predictions")
        self.save_btn.clicked.connect(self.save_predictions)
        self.save_btn.setEnabled(False)
        self.save_btn.setFixedHeight(40)
        controls_layout.addWidget(self.save_btn)

        # Slider for history length in minutes
        self.slider_label = QLabel("Display last 60 minutes")
        main_layout.addWidget(self.slider_label)
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setMinimum(1)
        self.time_slider.setMaximum(360)  # 6 hours max by default
        self.time_slider.setValue(60)
        self.time_slider.valueChanged.connect(self.update_plot)
        main_layout.addWidget(self.time_slider)

        # Matplotlib canvas
        self.fig, self.ax = plt.subplots(figsize=(12, 5))
        self.canvas = FigureCanvas(self.fig)
        main_layout.addWidget(self.canvas)

        self.setLayout(main_layout)

        # Dark mode detection
        palette = QApplication.instance().palette()
        if palette.color(QPalette.Window).value() < 128:
            self.dark_mode = True
            self.fig.patch.set_facecolor("#2e3b4e")
            self.ax.set_facecolor("#2e3b4e")
            self.ax.tick_params(colors="white", labelcolor="white")
            self.ax.yaxis.label.set_color("white")
            self.ax.xaxis.label.set_color("white")
            self.ax.title.set_color("white")
        else:
            self.dark_mode = False

    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select CSV", "", "CSV files (*.csv)")
        if not path:
            return
        self.df = data.load_candles(path, nrows=MAX_CANDLES)
        self.timestamps = self.df["timestamp"]

        self.apply_model()

    def change_horizon(self, text):
        self.forecast_horizon = int(text)
        if self.df is not None:
            self.apply_model()

    def apply_model(self):
        try:
            # Load model
            model_file = horizon_to_file[self.forecast_horizon]
            with open(model_file, "rb") as f:
                self.model = pickle.load(f)

            # Compute features
            k, alpha = 8, 1
            df = self.df.copy()
            df = features.compute_log_return(df)
            df = features.compute_rolling_volatility(df, horizon_seconds=self.forecast_horizon, k=k)
            df = features.compute_lagged_rolling_volatility(df, horizon_seconds=self.forecast_horizon, alpha=alpha, k=k)
            df = features.compute_multi_window_rolling_vol(df, horizon_seconds=self.forecast_horizon)
            df = features.compute_intraday_seasonality(df)
            df = features.compute_volatility_slope(df, horizon_seconds=self.forecast_horizon)
            df = features.compute_volatility_zscore(df, horizon_seconds=self.forecast_horizon)
            df = features.compute_volatility_acceleration(df)
            df = features.compute_future_rolling_volatility(df, horizon_seconds=self.forecast_horizon)

            # Feature selection
            feature_cols = [c for c in df.columns if c.startswith("rolling_vol")] + \
                           [c for c in df.columns if c.startswith("tod_")] + \
                           [c for c in df.columns if c in ["vol_of_vol", "vol_slope", "vol_zscore", "vol_accel"]]
            self.df_clean = df[feature_cols].dropna()
            X = self.df_clean[feature_cols]

            # Predictions
            self.preds = self.model.predict(X)

            # Medium-window baseline
            rolling_cols = [c for c in feature_cols if "rolling_vol_" in c and "cand" in c]
            mid = rolling_cols[len(rolling_cols)//2]
            self.baseline = np.log(X[:, feature_cols.index(mid)] + EPS)

            # Future rolling volatility
            self.future_vol = np.log(df.loc[self.df_clean.index, "rolling_future_vol"].values + EPS)

            self.save_btn.setEnabled(True)
            self.update_plot()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply model:\n{e}")

    def update_plot(self):
        if self.preds is None:
            self.ax.clear()
            self.canvas.draw()
            return

        # Determine history window
        minutes = self.time_slider.value()
        if minutes > 60:
            display_time = f"{minutes/60:.1f} hours"
        else:
            display_time = f"{minutes} minutes"
        self.slider_label.setText(f"Display last {display_time}")
        seconds = minutes * 60

        timestamps = pd.to_datetime(self.timestamps.loc[self.df_clean.index])
        t_end = timestamps.iloc[-1]
        t_start = t_end - pd.Timedelta(seconds=seconds)

        mask = (timestamps >= t_start) & (timestamps <= t_end)

        self.ax.clear()
        self.ax.plot(timestamps[mask], self.baseline[mask], label="Medium-window Baseline", color="orange")
        self.ax.plot(timestamps[mask], self.preds[mask], label="Model Prediction", color="steelblue")

        # Horizon region
        horizon_start = timestamps.iloc[-1]
        horizon_end = horizon_start + pd.Timedelta(seconds=self.forecast_horizon)
        self.ax.axvspan(horizon_start, horizon_end, color="grey", alpha=0.2, label="Forecast Horizon")

        # Future rolling volatility in horizon
        horizon_mask = timestamps >= horizon_start
        self.ax.plot(timestamps[horizon_mask], self.future_vol[horizon_mask], label="Future Rolling Volatility", color="green", linestyle="--")

        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Log Volatility")
        self.ax.legend()
        self.ax.grid(alpha=0.3)
        self.fig.tight_layout()
        self.canvas.draw()

    def save_predictions(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save predictions", "", "CSV files (*.csv)")
        if not path:
            return
        df_save = pd.DataFrame({
            "timestamp": self.timestamps.loc[self.df_clean.index],
            "baseline": self.baseline,
            "prediction": self.preds,
            "future_vol": self.future_vol
        })
        df_save.to_csv(path, index=False)
        QMessageBox.information(self, "Saved", f"Predictions saved to {path}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VolatilityApp()
    window.show()
    sys.exit(app.exec_())
