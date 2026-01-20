# #!/usr/bin/env python3
# import tkinter as tk
# from tkinter import filedialog, messagebox
# import pandas as pd
# import numpy as np
# import pickle
# import os
# import re
# from pathlib import Path
# import sys
# import matplotlib

# # Ensure Tkinter backend
# matplotlib.use("TkAgg")
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# plt.style.use('../styles/science.mplstyle')

# # Add src directory to path
# SCRIPT_DIR = Path(__file__).resolve().parent
# SRC_DIR = (SCRIPT_DIR / "../src").resolve()
# sys.path.append(str(SRC_DIR))

# from volare import data, features, model

# # --- Locate available models ---
# MODEL_DIR = SCRIPT_DIR / "../results/models"
# model_files = [f for f in os.listdir(MODEL_DIR) if f.startswith("volare_lgb_h") and f.endswith(".pkl")]

# available_horizons = sorted([int(re.search(r'h(\d+)\.pkl', f).group(1)) for f in model_files])
# if not available_horizons:
#     raise RuntimeError(f"No LightGBM models found in {MODEL_DIR}")

# horizon_to_file = {int(re.search(r'h(\d+)\.pkl', f).group(1)): MODEL_DIR / f for f in model_files}

# # --- Storage for predictions ---
# preds_storage = None

# def compute_predictions(file_path,model_file,horizon_seconds,loading_label):
#     global preds_storage, canvas_storage

#     # Load model
#     with open(model_file, "rb") as f:
#         lgb_model = pickle.load(f)

#     # Load data
#     df = data.load_candles(file_path, nrows=1000)

#     # Compute features
#     k, alpha = 8, 1
#     df = features.compute_log_return(df)
#     df = features.compute_rolling_volatility(df, horizon_seconds=horizon_seconds, k=k)
#     df = features.compute_lagged_rolling_volatility(df, horizon_seconds=horizon_seconds, alpha=alpha, k=k)
#     df = features.compute_multi_window_rolling_vol(df, horizon_seconds=horizon_seconds)
#     df = features.compute_intraday_seasonality(df)
#     df = features.compute_volatility_slope(df, horizon_seconds=horizon_seconds)
#     df = features.compute_volatility_zscore(df, horizon_seconds=horizon_seconds)
#     df = features.compute_volatility_acceleration(df)

#     # Extract feature columns and drop rows with NaNs
#     feature_cols = [c for c in df.columns if c.startswith('rolling_vol')] + \
#                     [c for c in df.columns if c.startswith('tod_')] + \
#                     [c for c in df.columns if c in ['vol_of_vol', 'vol_slope', 'vol_zscore', 'vol_accel']]
#     df_clean = df[feature_cols].dropna()
#     X_user = df_clean.values

#     # Predict
#     preds = lgb_model.predict(X_user, num_threads=os.cpu_count())

#     # Baseline
#     rolling_cols = [c for c in feature_cols if 'rolling_vol_' in c and 'cand' in c]
#     eps = 1e-8
#     baseline_medium = None
#     if rolling_cols:
#         baseline_medium = np.log(df[rolling_cols[len(rolling_cols)//2]].iloc[df_clean.index].values + eps)

#     # Store predictions for saving
#     preds_storage = {"preds": preds, "baseline": baseline_medium}

#     # Enable save button
#     save_button.config(state='normal')

#     if loading_label.winfo_exists():
#         loading_label.destroy()

#     # --- Plot inside GUI ---
#     max_points = 1000
#     step = max(1, len(preds)//max_points)
#     fig, ax = plt.subplots(figsize=(10,4))
#     ax.plot(preds[::step], label='Model Prediction', color='steelblue')
#     if baseline_medium is not None:
#         ax.plot(baseline_medium[::step], label='Medium Rolling Volatility', color='orange', alpha=0.7)
#     ax.set_xlabel('Time step')
#     ax.set_ylabel('Volatility')
#     ax.set_title(f'Predictions vs Baseline (Horizon {horizon_seconds}s)')
#     ax.legend()
#     fig.tight_layout()

#     canvas = FigureCanvasTkAgg(fig, master=plot_frame)
#     canvas.draw()
#     canvas.get_tk_widget().pack(fill='both', expand=True)
#     canvas.get_tk_widget().update_idletasks()
#     canvas_storage = canvas

# # --- Function to save predictions ---
# def save_predictions():
#     global preds_storage
#     if preds_storage is None:
#         messagebox.showwarning("Warning", "No predictions to save yet.")
#         return

#     save_path = filedialog.asksaveasfilename(
#         title="Save predictions",
#         defaultextension=".csv",
#         filetypes=[("CSV files", "*.csv")]
#     )
#     if save_path:
#         df_save = pd.DataFrame({"prediction": preds_storage["preds"]})
#         if preds_storage["baseline"] is not None:
#             df_save["baseline_medium"] = preds_storage["baseline"]
#         df_save.to_csv(save_path, index=False)
#         messagebox.showinfo("Saved", f"Predictions saved to {save_path}")

# # --- Function to apply model and plot ---
# def apply_model():
#     global preds_storage
#     try:
#         horizon_seconds = int(horizon_var.get())
#         model_file = horizon_to_file[horizon_seconds]

#         # Ask user to select CSV
#         file_path = filedialog.askopenfilename(
#             title="Select CSV file with your candle data",
#             filetypes=[("CSV files", "*.csv")]
#         )
#         if not file_path:
#             return

#         # Clear previous plot
#         for widget in plot_frame.winfo_children():
#             widget.destroy()
        
#         print('Destroyed widgets')
#         print('Computing label should appear here')
#         loading_label = tk.Label(plot_frame, text="Computing predictions, please wait...")
#         loading_label.pack()

#         # Defer actual computation slightly so GUI renders first
#         root.after(50, lambda: compute_predictions(file_path, model_file, horizon_seconds, loading_label))
        
#     except Exception as e:
#         messagebox.showerror("Error", f"Failed to apply model:\n{e}")

# # --- GUI setup ---
# root = tk.Tk()
# root.title("volare LightGBM model application")
# root.geometry("1200x700")

# Show initially on top
# root.lift()
# root.attributes("-topmost", True)
# root.after(500, lambda: root.attributes("-topmost", False))

# # Plot frame
# plot_frame = tk.Frame(root)
# plot_frame.pack(fill='both', expand=True, padx=10, pady=10)

# # Controls frame
# controls_frame = tk.Frame(root)
# controls_frame.pack(fill='x', padx=10, pady=5)

# tk.Label(controls_frame, text="Select Horizon (seconds):").pack(side='left')
# horizon_var = tk.StringVar(root)
# horizon_var.set(str(available_horizons[0]))
# dropdown = tk.OptionMenu(controls_frame, horizon_var, *available_horizons)
# dropdown.pack(side='left', padx=5)

# # Apply model button
# # tk.Button(controls_frame, text="Select CSV and Apply Model", command=apply_model).pack(side='left', padx=10)
# tk.Button(controls_frame, text="Select CSV and Apply Model", command=lambda: root.after(50, apply_model)).pack(side='left', padx=10)

# # Save predictions button (initially disabled)
# save_button = tk.Button(controls_frame, text="Save Predictions", command=save_predictions, state='disabled')
# save_button.pack(side='left', padx=10)

# root.mainloop()

#!/usr/bin/env python3
import sys
from pathlib import Path
import os
import re
import pickle
import numpy as np
import pandas as pd

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QLabel,
    QComboBox, QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

plt.style.use('../styles/science.mplstyle')

# Add src directory to path
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = (SCRIPT_DIR / "../src").resolve()
sys.path.append(str(SRC_DIR))

from volare import data, features, model

# --- Locate available models ---
MODEL_DIR = SCRIPT_DIR / "../results/models"
model_files = [f for f in os.listdir(MODEL_DIR) if f.startswith("volare_lgb_h") and f.endswith(".pkl")]

available_horizons = sorted([int(re.search(r'h(\d+)\.pkl', f).group(1)) for f in model_files])
if not available_horizons:
    raise RuntimeError(f"No LightGBM models found in {MODEL_DIR}")

horizon_to_file = {int(re.search(r'h(\d+)\.pkl', f).group(1)): MODEL_DIR / f for f in model_files}


# ---------------- Worker Thread ----------------
class ComputeThread(QThread):
    finished = pyqtSignal(np.ndarray, np.ndarray, int)  # preds, baseline, horizon
    error = pyqtSignal(str)

    def __init__(self, file_path, model_file, horizon_seconds):
        super().__init__()
        self.file_path = file_path
        self.model_file = model_file
        self.horizon_seconds = horizon_seconds

    def run(self):
        try:
            # Load model
            with open(self.model_file, "rb") as f:
                lgb_model = pickle.load(f)

            # Load data
            df = data.load_candles(self.file_path, nrows=1000)

            # Compute features
            k, alpha = 8, 1
            df = features.compute_log_return(df)
            df = features.compute_rolling_volatility(df, horizon_seconds=self.horizon_seconds, k=k)
            df = features.compute_lagged_rolling_volatility(df, horizon_seconds=self.horizon_seconds, alpha=alpha, k=k)
            df = features.compute_multi_window_rolling_vol(df, horizon_seconds=self.horizon_seconds)
            df = features.compute_intraday_seasonality(df)
            df = features.compute_volatility_slope(df, horizon_seconds=self.horizon_seconds)
            df = features.compute_volatility_zscore(df, horizon_seconds=self.horizon_seconds)
            df = features.compute_volatility_acceleration(df)

            # Extract feature columns
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

            # Emit results
            self.finished.emit(preds, baseline_medium, self.horizon_seconds)

        except Exception as e:
            self.error.emit(str(e))


# ---------------- Main Window ----------------
class VolareGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("volare LightGBM model application")
        self.setGeometry(100, 100, 1200, 700)

        # Storage for predictions
        self.preds_storage = None

        # Central widget and layout
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout()
        central.setLayout(main_layout)

        # Controls
        controls_layout = QHBoxLayout()
        main_layout.addLayout(controls_layout)

        controls_layout.addWidget(QLabel("Select Horizon (seconds):"))
        self.horizon_combo = QComboBox()
        for h in available_horizons:
            self.horizon_combo.addItem(str(h))
        controls_layout.addWidget(self.horizon_combo)

        self.load_button = QPushButton("Select CSV and Apply Model")
        self.load_button.clicked.connect(self.select_file)
        controls_layout.addWidget(self.load_button)

        self.save_button = QPushButton("Save Predictions")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_predictions)
        controls_layout.addWidget(self.save_button)

        # Status label
        self.status_label = QLabel("")
        main_layout.addWidget(self.status_label)

        # Plot canvas
        self.plot_canvas = None
        self.plot_container = QWidget()
        main_layout.addWidget(self.plot_container)
        self.plot_layout = QVBoxLayout()
        self.plot_container.setLayout(self.plot_layout)

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select CSV file", "", "CSV files (*.csv)")
        if not file_path:
            return

        horizon = int(self.horizon_combo.currentText())
        self.status_label.setText("Computing predictions, please wait...")
        QApplication.processEvents()  # Force GUI update

        model_file = horizon_to_file[horizon]
        self.thread = ComputeThread(file_path, model_file, horizon)
        self.thread.finished.connect(self.display_results)
        self.thread.error.connect(self.show_error)
        self.thread.start()

    def display_results(self, preds, baseline, horizon):
        self.preds_storage = {"preds": preds, "baseline": baseline}

        # Enable save button
        self.save_button.setEnabled(True)

        # Clear previous plot
        for i in reversed(range(self.plot_layout.count())):
            widget = self.plot_layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)

        # Plot
        max_points = 1000
        step = max(1, len(preds)//max_points)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(preds[::step], label='Model Prediction', color='steelblue')
        if baseline is not None:
            ax.plot(baseline[::step], label='Medium Rolling Volatility', color='orange', alpha=0.7)
        ax.set_xlabel('Time step')
        ax.set_ylabel('Volatility')
        ax.set_title(f'Predictions vs Baseline (Horizon {horizon}s)')
        ax.legend()
        fig.tight_layout()

        self.plot_canvas = FigureCanvas(fig)
        self.plot_layout.addWidget(self.plot_canvas)
        self.plot_canvas.draw()

        self.status_label.setText("Done.")

    def save_predictions(self):
        if self.preds_storage is None:
            QMessageBox.warning(self, "Warning", "No predictions to save yet.")
            return

        save_path, _ = QFileDialog.getSaveFileName(self, "Save predictions", "", "CSV files (*.csv)")
        if save_path:
            df_save = pd.DataFrame({"prediction": self.preds_storage["preds"]})
            if self.preds_storage["baseline"] is not None:
                df_save["baseline_medium"] = self.preds_storage["baseline"]
            df_save.to_csv(save_path, index=False)
            QMessageBox.information(self, "Saved", f"Predictions saved to {save_path}")

    def show_error(self, msg):
        QMessageBox.critical(self, "Error", f"Failed to apply model:\n{msg}")


# ---------------- Run Application ----------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VolareGUI()
    window.show()
    sys.exit(app.exec())
