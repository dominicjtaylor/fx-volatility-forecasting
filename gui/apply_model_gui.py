#!/usr/bin/env python3
import sys
import os
import re
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from PyQt6 import QtWidgets, QtCore, QtGui
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

plt.style.use('../styles/science.mplstyle')

# Add src directory to path
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = (SCRIPT_DIR / "../src").resolve()
sys.path.append(str(SRC_DIR))

from volare import data, features, model

MODEL_DIR = SCRIPT_DIR / "../results/models"
model_files = [f for f in os.listdir(MODEL_DIR) if f.startswith("volare_lgb_h") and f.endswith(".pkl")]
available_horizons = sorted([int(re.search(r"h(\d+)\.pkl", f).group(1)) for f in model_files])
horizon_to_file = {int(re.search(r"h(\d+)\.pkl", f).group(1)): MODEL_DIR / f for f in model_files}


class VolatilityApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Volare LightGBM Model Application")
        self.setGeometry(100, 50, 1400, 800)

        self.df_full = None
        self.preds_storage = None
        self.canvas = None

        self.init_ui()
        self.setAcceptDrops(True)

    def init_ui(self):
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout()
        central_widget.setLayout(layout)

        # --- Controls ---
        controls_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(controls_layout)

        self.load_button = QtWidgets.QPushButton("Load CSV")
        self.load_button.setMinimumHeight(40)
        self.load_button.clicked.connect(self.load_csv)
        controls_layout.addWidget(self.load_button)

        controls_layout.addWidget(QtWidgets.QLabel("Horizon (seconds):"))
        self.horizon_combo = QtWidgets.QComboBox()
        self.horizon_combo.addItems([str(h) for h in available_horizons])
        self.horizon_combo.currentTextChanged.connect(self.on_horizon_change)
        self.horizon_combo.setMinimumWidth(100)
        controls_layout.addWidget(self.horizon_combo)

        self.save_button = QtWidgets.QPushButton("Save Predictions")
        self.save_button.setEnabled(False)
        self.save_button.setMinimumHeight(40)
        self.save_button.clicked.connect(self.save_predictions)
        controls_layout.addWidget(self.save_button)

        # Slider for time window
        slider_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(slider_layout)
        self.time_label = QtWidgets.QLabel("Seconds to display:")
        slider_layout.addWidget(self.time_label)
        self.time_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.time_slider.setMinimum(60)
        self.time_slider.setMaximum(3000*60)  # example max ~3000 mins
        self.time_slider.setValue(5000)  # default
        self.time_slider.setTickInterval(60)
        self.time_slider.valueChanged.connect(self.update_plot)
        slider_layout.addWidget(self.time_slider)

        # --- Plot area ---
        self.plot_frame = QtWidgets.QFrame()
        layout.addWidget(self.plot_frame)
        self.plot_layout = QtWidgets.QVBoxLayout()
        self.plot_frame.setLayout(self.plot_layout)

        # Initial label
        self.info_label = QtWidgets.QLabel("Load a CSV to see predictions")
        self.info_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.plot_layout.addWidget(self.info_label)

    # --- Drag & drop ---
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if path.endswith(".csv"):
                self.load_csv(path)

    # --- Apply model ---
    def apply_model(self):
        if self.df_full is None:
            return

        horizon_seconds = int(self.horizon_combo.currentText())
        model_file = horizon_to_file[horizon_seconds]

        with open(model_file, "rb") as f:
            lgb_model = pickle.load(f)

        df = self.df_full.copy()
        # Feature engineering
        k, alpha = 8, 1
        df = features.compute_log_return(df)
        df = features.compute_rolling_volatility(df, horizon_seconds=horizon_seconds, k=k)
        df = features.compute_lagged_rolling_volatility(df, horizon_seconds=horizon_seconds, alpha=alpha, k=k)
        df = features.compute_multi_window_rolling_vol(df, horizon_seconds=horizon_seconds)
        df = features.compute_intraday_seasonality(df)
        df = features.compute_volatility_slope(df, horizon_seconds=horizon_seconds)
        df = features.compute_volatility_zscore(df, horizon_seconds=horizon_seconds)
        df = features.compute_volatility_acceleration(df)

        # Feature selection
        feature_cols = [c for c in df.columns if c.startswith('rolling_vol')] + \
                       [c for c in df.columns if c.startswith('tod_')] + \
                       [c for c in df.columns if c in ['vol_of_vol', 'vol_slope', 'vol_zscore', 'vol_accel']]
        df_clean = df[feature_cols].dropna()
        X_user = df_clean.values

        # Predict
        preds = lgb_model.predict(X_user, num_threads=os.cpu_count())

        # Baseline
        rolling_cols = [c for c in feature_cols if 'rolling_vol_' in c and 'cand' in c]
        eps = 1e-8
        baseline_medium = None
        if rolling_cols:
            baseline_medium = np.log(df[rolling_cols[len(rolling_cols)//2]].iloc[df_clean.index].values + eps)

        self.preds_storage = {"preds": preds, "baseline": baseline_medium}
        self.save_button.setEnabled(True)

        # Plot
        self.update_plot()

    # --- Load CSV ---
    def load_csv(self, path=None):
        if path is None:  # called by button
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Select CSV file", "", "CSV Files (*.csv)"
            )
        if not path:
            return

        # Load last 300_000 candles
        self.df_full = data.load_candles(path, nrows=300_000)

        # Remove info label if it exists
        if self.info_label:
            self.plot_layout.removeWidget(self.info_label)
            self.info_label.deleteLater()
            self.info_label = None

        # Apply model with selected horizon
        self.apply_model()

    # --- Update plot with dark mode support ---
    def update_plot(self):
        if self.df_full is None or self.preds_storage is None:
            return

        # Remove previous canvas
        if self.canvas:
            self.plot_layout.removeWidget(self.canvas)
            self.canvas.deleteLater()
            self.canvas = None

        preds = self.preds_storage["preds"]
        baseline = self.preds_storage["baseline"]
        df = self.df_full

        timestamps = df['timestamp'].values.astype(float)
        last_time = timestamps[-1]
        display_seconds = self.time_slider.value()
        start_time = max(timestamps[0], last_time - display_seconds)
        mask = (timestamps >= start_time) & (timestamps <= last_time)

        t_disp = timestamps[mask]
        preds_disp = preds[mask]
        baseline_disp = baseline[mask] if baseline is not None else None

        # Detect dark mode (macOS / Windows)
        dark_mode = False
        try:
            import platform
            if platform.system() == "Darwin":
                # macOS: detect dark appearance
                from subprocess import check_output
                out = check_output(
                    ["defaults", "read", "-g", "AppleInterfaceStyle"]
                ).decode().strip()
                dark_mode = out.lower() == "dark"
        except:
            pass

        fig, ax = plt.subplots(figsize=(12,5))

        if dark_mode:
            fig.patch.set_facecolor('#2e3b4e')
            ax.set_facecolor('#2e3b4e')
            ax.tick_params(colors='white')
            ax.yaxis.label.set_color('white')
            ax.xaxis.label.set_color('white')
            ax.title.set_color('white')
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['right'].set_color('white')
            # Light line colors
            pred_color = 'cyan'
            base_color = 'orange'
            horizon_alpha = 0.3
        else:
            pred_color = 'steelblue'
            base_color = 'orange'
            horizon_alpha = 0.3

        ax.plot(t_disp, preds_disp, label="Model Prediction", color=pred_color)
        if baseline_disp is not None:
            ax.plot(t_disp, baseline_disp, label="Medium Rolling Volatility", color=base_color, alpha=0.7)

        horizon_seconds = int(self.horizon_combo.currentText())
        ax.axvspan(last_time, last_time + horizon_seconds, color="grey", alpha=horizon_alpha,
                label=f"Horizon {horizon_seconds}s")

        ax.set_xlabel("Time")
        ax.set_ylabel("Volatility")
        ax.set_title("Predictions vs Baseline")
        ax.legend()

        fig.tight_layout()

        self.canvas = FigureCanvas(fig)
        self.plot_layout.addWidget(self.canvas)
        self.canvas.draw()

    # --- Save predictions ---
    def save_predictions(self):
        if self.preds_storage is None:
            QtWidgets.QMessageBox.warning(self, "Warning", "No predictions to save yet.")
            return

        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save predictions", "", "CSV Files (*.csv)")
        if path:
            df_save = pd.DataFrame({"prediction": self.preds_storage["preds"]})
            if self.preds_storage["baseline"] is not None:
                df_save["baseline_medium"] = self.preds_storage["baseline"]
            df_save.to_csv(path, index=False)
            QtWidgets.QMessageBox.information(self, "Saved", f"Predictions saved to {path}")

    def on_horizon_change(self):
        if self.df_full is not None:
            self.apply_model()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = VolatilityApp()
    window.show()
    sys.exit(app.exec())
