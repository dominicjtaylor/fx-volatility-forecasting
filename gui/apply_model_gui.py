#!/usr/bin/env python3
import sys
import os
import re
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

from PyQt6 import QtWidgets, QtCore, QtGui
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

plt.style.use('../styles/science.mplstyle')

# Add src directory to path
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = (SCRIPT_DIR / "../src").resolve()
sys.path.append(str(SRC_DIR))

from volare import data, features, model

# --- Locate available models ---
MODEL_DIR = SCRIPT_DIR / "../results/models"
model_files = [f for f in os.listdir(MODEL_DIR) if f.startswith("volare_lgb_h") and f.endswith(".pkl")]
if not model_files:
    raise RuntimeError(f"No LightGBM models found in {MODEL_DIR}")

available_horizons = sorted([int(re.search(r"h(\d+)\.pkl", f).group(1)) for f in model_files])
horizon_to_file = {int(re.search(r"h(\d+)\.pkl", f).group(1)): MODEL_DIR / f for f in model_files}


class VolareApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Volare LightGBM Model Application")
        self.resize(1400, 800)

        # Storage
        self.preds_storage = None
        self.df_full = None
        self.baseline_full = None
        self.canvas = None

        self._init_ui()
        self.show()

    def _init_ui(self):
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)

        # --- Controls ---
        controls_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(controls_layout)

        horizon_label = QtWidgets.QLabel("Horizon (s):")
        controls_layout.addWidget(horizon_label)

        self.horizon_combo = QtWidgets.QComboBox()
        self.horizon_combo.addItems([str(h) for h in available_horizons])
        self.horizon_combo.setFixedWidth(120)
        controls_layout.addWidget(self.horizon_combo)

        self.load_button = QtWidgets.QPushButton("Select CSV / Apply Model")
        self.load_button.setFixedHeight(40)
        self.load_button.clicked.connect(self.load_csv)
        controls_layout.addWidget(self.load_button)

        self.save_button = QtWidgets.QPushButton("Save Predictions")
        self.save_button.setFixedHeight(40)
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_predictions)
        controls_layout.addWidget(self.save_button)

        controls_layout.addStretch()

        # --- Status label ---
        self.status_label = QtWidgets.QLabel("Drag a CSV here or click 'Select CSV'.")
        self.status_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("font-size: 14px; color: gray;")
        layout.addWidget(self.status_label)

        # --- Plot area ---
        self.plot_frame = QtWidgets.QFrame()
        self.plot_layout = QtWidgets.QVBoxLayout(self.plot_frame)
        layout.addWidget(self.plot_frame)

        # --- Slider for zooming last N candles ---
        slider_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(slider_layout)
        self.slider_label = QtWidgets.QLabel("Number of candles to display:")
        slider_layout.addWidget(self.slider_label)
        self.candle_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.candle_slider.setMinimum(100)
        self.candle_slider.setMaximum(300_000)
        self.candle_slider.setValue(5000)
        self.candle_slider.setSingleStep(100)
        self.candle_slider.valueChanged.connect(self.update_plot_zoom)
        slider_layout.addWidget(self.candle_slider)

        # Enable drag-and-drop
        self.setAcceptDrops(True)

    # --- Drag-and-drop events ---
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if file_path.endswith(".csv"):
                self.load_csv(file_path)

    # --- Load CSV and apply model ---
    def load_csv(self, file_path=None):
        if not file_path:
            file_dialog = QtWidgets.QFileDialog(self)
            file_path, _ = file_dialog.getOpenFileName(self, "Select CSV file with your candle data", "", "CSV Files (*.csv)")
            if not file_path:
                return

        self.status_label.setText("Computing predictions, please wait...")
        QtWidgets.QApplication.processEvents()  # force GUI update

        horizon_seconds = int(self.horizon_combo.currentText())
        model_file = horizon_to_file[horizon_seconds]

        try:
            # Load model
            with open(model_file, "rb") as f:
                lgb_model = pickle.load(f)

            # Load last 300_000 candles
            self.df_full = data.load_candles(file_path, nrows=300_000)

            # Feature computation
            k, alpha = 8, 1
            df = self.df_full.copy()
            df = features.compute_log_return(df)
            df = features.compute_rolling_volatility(df, horizon_seconds=horizon_seconds, k=k)
            df = features.compute_lagged_rolling_volatility(df, horizon_seconds=horizon_seconds, alpha=alpha, k=k)
            df = features.compute_multi_window_rolling_vol(df, horizon_seconds=horizon_seconds)
            df = features.compute_intraday_seasonality(df)
            df = features.compute_volatility_slope(df, horizon_seconds=horizon_seconds)
            df = features.compute_volatility_zscore(df, horizon_seconds=horizon_seconds)
            df = features.compute_volatility_acceleration(df)

            # Features selection
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

            # Store predictions
            self.preds_storage = {"preds": preds, "baseline": baseline_medium}
            self.df_full = df_clean
            self.baseline_full = baseline_medium
            self.save_button.setEnabled(True)

            # Update plot
            self.plot_predictions()

            self.status_label.setText("Predictions computed successfully.")

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to apply model:\n{e}")
            self.status_label.setText("Error computing predictions.")

    # --- Plotting function ---
    def plot_predictions(self):
        if self.preds_storage is None:
            return

        # Clear previous plot
        for i in reversed(range(self.plot_layout.count())):
            widget = self.plot_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        preds = self.preds_storage["preds"]
        baseline = self.baseline_full
        n_display = self.candle_slider.value()

        t_hist = np.arange(max(0, len(preds)-n_display), len(preds))
        preds_disp = preds[-n_display:]
        baseline_disp = baseline[-n_display:] if baseline is not None else None

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(t_hist, preds_disp, label="Model Prediction", color="steelblue")
        if baseline_disp is not None:
            ax.plot(t_hist, baseline_disp, label="Medium Rolling Volatility", color="orange", alpha=0.7)

        # Forward horizon line
        ax.axvline(t_hist[-1], linestyle='dashed', color='black', label=f"Horizon +{self.horizon_combo.currentText()}s")

        ax.set_xlabel("Time step")
        ax.set_ylabel("Volatility")
        ax.set_title(f"Predictions vs Baseline")
        ax.legend()
        fig.tight_layout()

        self.canvas = FigureCanvas(fig)
        self.plot_layout.addWidget(self.canvas)
        self.canvas.draw()

    # --- Update plot zoom ---
    def update_plot_zoom(self):
        self.plot_predictions()

    # --- Save predictions ---
    def save_predictions(self):
        if self.preds_storage is None:
            QtWidgets.QMessageBox.warning(self, "Warning", "No predictions to save yet.")
            return

        file_dialog = QtWidgets.QFileDialog(self)
        save_path, _ = file_dialog.getSaveFileName(self, "Save predictions", "", "CSV Files (*.csv)")
        if save_path:
            df_save = pd.DataFrame({"prediction": self.preds_storage["preds"]})
            if self.baseline_full is not None:
                df_save["baseline_medium"] = self.baseline_full
            df_save.to_csv(save_path, index=False)
            QtWidgets.QMessageBox.information(self, "Saved", f"Predictions saved to {save_path}")


# --- Run Application ---
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = VolareApp()
    sys.exit(app.exec())
