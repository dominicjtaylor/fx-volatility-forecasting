#!/usr/bin/env python3
import sys, os, pickle, re
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

from PyQt6 import QtWidgets, QtCore
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# --- Add src directory ---
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = (SCRIPT_DIR / "../src").resolve()
sys.path.append(str(SRC_DIR))
from volare import data, features

# --- Model files ---
MODEL_DIR = SCRIPT_DIR / "../results/models"
model_files = [f for f in os.listdir(MODEL_DIR) if f.startswith("volare_lgb_h") and f.endswith(".pkl")]
available_horizons = sorted([int(re.search(r"h(\d+)\.pkl", f).group(1)) for f in model_files])
if not available_horizons:
    raise RuntimeError("No LightGBM models found")
horizon_to_file = {int(re.search(r"h(\d+)\.pkl", f).group(1)): MODEL_DIR / f for f in model_files}

# --- Default config ---
MAX_CANDLES = 300_000
DEFAULT_DISPLAY_SECONDS = 3600  # 1 hour display window for slider
EPS = 1e-8

class VolModelApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Volatility Model Viewer")
        self.resize(1400, 800)
        self.csv_path = None
        self.df_full = None
        self.df_features = None
        self.preds = None
        self.baseline = None
        self.timestamps = None
        self.display_seconds = DEFAULT_DISPLAY_SECONDS

        # --- Main widget & layout ---
        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QtWidgets.QVBoxLayout(main_widget)

        # --- Controls ---
        ctrl_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(ctrl_layout)

        self.load_button = QtWidgets.QPushButton("Load CSV")
        self.load_button.setMinimumHeight(40)
        self.load_button.clicked.connect(self.load_csv)
        ctrl_layout.addWidget(self.load_button)

        ctrl_layout.addSpacing(20)
        ctrl_layout.addWidget(QtWidgets.QLabel("Horizon (s):"))
        self.horizon_combo = QtWidgets.QComboBox()
        for h in available_horizons:
            self.horizon_combo.addItem(str(h))
        self.horizon_combo.currentTextChanged.connect(self.apply_model)
        ctrl_layout.addWidget(self.horizon_combo)

        ctrl_layout.addSpacing(20)
        self.save_button = QtWidgets.QPushButton("Save Predictions")
        self.save_button.setEnabled(False)
        self.save_button.setMinimumHeight(40)
        self.save_button.clicked.connect(self.save_predictions)
        ctrl_layout.addWidget(self.save_button)

        # Slider to zoom into recent data in seconds
        ctrl_layout.addSpacing(20)
        self.slider_label = QtWidgets.QLabel(f"Display last {self.display_seconds}s")
        ctrl_layout.addWidget(self.slider_label)
        self.time_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.time_slider.setMinimum(60)  # 1 minute
        self.time_slider.setMaximum(3600*24)  # 24 hours
        self.time_slider.setValue(self.display_seconds)
        self.time_slider.valueChanged.connect(self.update_plot)
        ctrl_layout.addWidget(self.time_slider)

        # --- Plot area ---
        self.canvas = None
        self.plot_widget = QtWidgets.QWidget()
        main_layout.addWidget(self.plot_widget)
        self.plot_layout = QtWidgets.QVBoxLayout(self.plot_widget)
        self.loading_label = QtWidgets.QLabel("Load a CSV to see predictions")
        self.loading_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.plot_layout.addWidget(self.loading_label)

    # -------------------
    # Load CSV
    # -------------------
    def load_csv(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select CSV file", filter="CSV Files (*.csv)")
        if not path:
            return
        self.csv_path = path
        self.df_full = data.load_candles(path, nrows=MAX_CANDLES)
        self.apply_model()

    # -------------------
    # Compute features & model predictions
    # -------------------
    def apply_model(self):
        if self.csv_path is None:
            return
        horizon = int(self.horizon_combo.currentText())
        model_file = horizon_to_file[horizon]

        # Load model
        with open(model_file, "rb") as f:
            lgb_model = pickle.load(f)

        # Feature computation
        df = self.df_full.copy()
        df = features.compute_log_return(df)
        df = features.compute_rolling_volatility(df, horizon_seconds=horizon, k=8)
        df = features.compute_lagged_rolling_volatility(df, horizon_seconds=horizon, alpha=1, k=8)
        df = features.compute_multi_window_rolling_vol(df, horizon_seconds=horizon)
        df = features.compute_intraday_seasonality(df)
        df = features.compute_volatility_slope(df, horizon_seconds=horizon)
        df = features.compute_volatility_zscore(df, horizon_seconds=horizon)
        df = features.compute_volatility_acceleration(df)

        feature_cols = [c for c in df.columns if c.startswith("rolling_vol")] + \
                       [c for c in df.columns if c.startswith("tod_")] + \
                       [c for c in df.columns if c in ["vol_of_vol","vol_slope","vol_zscore","vol_accel"]]

        df_clean = df[feature_cols].dropna()
        self.df_features = df_clean
        X_user = df_clean  # DataFrame keeps feature names
        self.preds = lgb_model.predict(X_user, num_threads=os.cpu_count())

        rolling_cols = [c for c in feature_cols if "rolling_vol_" in c and "cand" in c]
        self.baseline = None
        if rolling_cols:
            self.baseline = np.log(df[rolling_cols[len(rolling_cols)//2]].iloc[df_clean.index].values + EPS)

        self.timestamps = df.index[df_clean.index].astype(np.int64) / 1e9  # timestamps in seconds

        self.save_button.setEnabled(True)
        self.update_plot()

    # -------------------
    # Plotting
    # -------------------
    def update_plot(self):
        if self.preds is None or self.timestamps is None:
            return

        # Slider controls displayed seconds
        self.display_seconds = self.time_slider.value()
        self.slider_label.setText(f"Display last {self.display_seconds}s")

        t_end = self.timestamps[-1]
        t_start = max(t_end - self.display_seconds, self.timestamps[0])
        mask = (self.timestamps >= t_start) & (self.timestamps <= t_end)

        t_disp = self.timestamps[mask]
        preds_disp = self.preds[mask]
        baseline_disp = self.baseline[mask] if self.baseline is not None else None

        # Clear previous plot
        for i in reversed(range(self.plot_layout.count())):
            widget = self.plot_layout.itemAt(i).widget()
            widget.setParent(None)

        # Figure
        fig, ax = plt.subplots(figsize=(12,5))
        # Dark mode detection
        dark_bg = self.palette().color(self.backgroundRole()).value() < 128
        if dark_bg:
            fig.patch.set_facecolor("#2e3b4e")
            ax.set_facecolor("#2e3b4e")
            ax.tick_params(colors="white")
            ax.yaxis.label.set_color("white")
            ax.xaxis.label.set_color("white")
            ax.title.set_color("white")
            legend_color = "white"
        else:
            legend_color = "black"

        ax.plot(t_disp, preds_disp, label="Model Prediction", color="cyan" if dark_bg else "steelblue")
        if baseline_disp is not None:
            ax.plot(t_disp, baseline_disp, label="Baseline", color="orange", alpha=0.7)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Volatility")
        ax.set_title("Volatility Predictions vs Baseline")
        ax.legend(facecolor=fig.get_facecolor(), labelcolor=legend_color)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # Shaded horizon region (prediction horizon)
        horizon = int(self.horizon_combo.currentText())
        ax.axvspan(t_disp[-1], t_disp[-1]+horizon, color="gray", alpha=0.2)

        self.canvas = FigureCanvas(fig)
        self.plot_layout.addWidget(self.canvas)

    # -------------------
    # Save predictions
    # -------------------
    def save_predictions(self):
        if self.preds is None:
            QtWidgets.QMessageBox.warning(self, "Warning", "No predictions yet.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Predictions", filter="CSV Files (*.csv)")
        if not path:
            return
        df_save = pd.DataFrame({"prediction": self.preds})
        if self.baseline is not None:
            df_save["baseline"] = self.baseline
        df_save.to_csv(path, index=False)
        QtWidgets.QMessageBox.information(self, "Saved", f"Predictions saved to {path}")

# -------------------
# Run application
# -------------------
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = VolModelApp()
    window.show()
    sys.exit(app.exec())
