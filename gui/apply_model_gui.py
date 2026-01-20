#!/usr/bin/env python3
import sys
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import re

from PyQt6 import QtWidgets, QtCore, QtGui
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

# Add your src directory
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = (SCRIPT_DIR / "../src").resolve()
sys.path.append(str(SRC_DIR))

from volare import data, features, model

plt.style.use('../styles/science.mplstyle')

# --- Locate available models ---
MODEL_DIR = SCRIPT_DIR / "../results/models"
model_files = [f for f in MODEL_DIR.iterdir() if f.name.startswith("volare_lgb_h") and f.name.endswith(".pkl")]
available_horizons = sorted([int(re.search(r"h(\d+)\.pkl", f.name).group(1)) for f in model_files])
horizon_to_file = {int(re.search(r"h(\d+)\.pkl", f.name).group(1)): f for f in model_files}

# --- GUI Application ---
class VolareApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Volare LightGBM Model Application")
        self.resize(1300, 800)
        self.setAcceptDrops(True)

        # Storage
        self.preds_storage = None

        # --- Central widget ---
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QtWidgets.QVBoxLayout(central_widget)

        # --- Controls ---
        controls_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(controls_layout)

        # Horizon label and dropdown
        horizon_label = QtWidgets.QLabel("Select Horizon (seconds):")
        horizon_label.setMinimumWidth(180)
        horizon_label.setStyleSheet("font-weight:bold; font-size:14px;")
        controls_layout.addWidget(horizon_label)

        self.horizon_combo = QtWidgets.QComboBox()
        self.horizon_combo.addItems([str(h) for h in available_horizons])
        self.horizon_combo.setMinimumWidth(120)
        self.horizon_combo.setStyleSheet("font-size:14px;")
        controls_layout.addWidget(self.horizon_combo)

        # Apply model button
        self.apply_btn = QtWidgets.QPushButton("Select CSV / Apply Model")
        self.apply_btn.setStyleSheet("font-size:16px; padding:8px 12px;")
        self.apply_btn.clicked.connect(self.select_csv)
        controls_layout.addWidget(self.apply_btn)

        # Save predictions button
        self.save_btn = QtWidgets.QPushButton("Save Predictions")
        self.save_btn.setStyleSheet("font-size:16px; padding:8px 12px;")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_predictions)
        controls_layout.addWidget(self.save_btn)

        controls_layout.addStretch()

        # --- Plot area ---
        self.figure, self.ax = plt.subplots(figsize=(12, 5))
        self.canvas = FigureCanvas(self.figure)
        main_layout.addWidget(self.canvas)

    # --- Drag and drop ---
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.endswith(".csv"):
                self.process_file(file_path)
                break

    # --- Select CSV dialog ---
    def select_csv(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select CSV file", "", "CSV Files (*.csv)")
        if file_path:
            self.process_file(file_path)

    # --- Main computation and plotting ---
    def process_file(self, file_path):
        self.ax.clear()
        self.ax.text(0.5, 0.5, "Computing predictions...", ha='center', va='center', transform=self.ax.transAxes)
        self.canvas.draw()
        QtWidgets.QApplication.processEvents()  # force GUI update

        try:
            horizon_seconds = int(self.horizon_combo.currentText())
            model_file = horizon_to_file[horizon_seconds]

            # Load model
            with open(model_file, "rb") as f:
                lgb_model = pickle.load(f)

            # Load data (full last N candles)
            df = data.load_candles(file_path, nrows=300_000)

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

            # Select features
            feature_cols = [c for c in df.columns if c.startswith('rolling_vol')] + \
                           [c for c in df.columns if c.startswith('tod_')] + \
                           [c for c in df.columns if c in ['vol_of_vol','vol_slope','vol_zscore','vol_accel']]
            df_clean = df[feature_cols].dropna()
            X_user = df_clean.values

            # Predict
            preds = lgb_model.predict(X_user, num_threads=os.cpu_count())

            # Baseline
            rolling_cols = [c for c in feature_cols if 'rolling_vol_' in c and 'cand' in c]
            eps = 1e-8
            baseline = None
            if rolling_cols:
                baseline = np.log(df[rolling_cols[len(rolling_cols)//2]].iloc[df_clean.index].values + eps)

            self.preds_storage = {"preds": preds, "baseline": baseline}
            self.save_btn.setEnabled(True)

            # --- Plot ---
            self.ax.clear()
            self.ax.plot(preds, label='Model Prediction', color='steelblue')
            if baseline is not None:
                self.ax.plot(baseline, label='Medium Rolling Volatility', color='orange', alpha=0.7)

            # Forward-horizon prediction at the end
            forward_pred = lgb_model.predict(X_user[-1].reshape(1,-1))[0]
            self.ax.axvline(len(preds)-1, color='black', linestyle='--', label=f'Horizon {horizon_seconds}s Forward')

            self.ax.set_xlabel("Time step")
            self.ax.set_ylabel("Volatility")
            self.ax.set_title(f"Volatility Prediction (Last {len(preds)} candles + horizon)")
            self.ax.legend()
            self.figure.tight_layout()
            self.canvas.draw()

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to apply model:\n{e}")

    # --- Save predictions ---
    def save_predictions(self):
        if self.preds_storage is None:
            QtWidgets.QMessageBox.warning(self, "Warning", "No predictions to save yet.")
            return

        save_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Predictions", "", "CSV Files (*.csv)")
        if save_path:
            df_save = pd.DataFrame({"prediction": self.preds_storage["preds"]})
            if self.preds_storage["baseline"] is not None:
                df_save["baseline_medium"] = self.preds_storage["baseline"]
            df_save.to_csv(save_path, index=False)
            QtWidgets.QMessageBox.information(self, "Saved", f"Predictions saved to {save_path}")


# --- Run Application ---
app = QtWidgets.QApplication(sys.argv)
window = VolareApp()
window.show()
window.raise_()  # bring to front initially
sys.exit(app.exec())
