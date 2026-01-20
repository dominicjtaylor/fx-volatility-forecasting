#!/usr/bin/env python3
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import pickle
import os
import re
from pathlib import Path
import sys
import matplotlib
matplotlib.use("TkAgg")

# Matplotlib for embedding in Tkinter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

plt.style.use('../styles/science.mplstyle')

# Add src directory to path
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = (SCRIPT_DIR / "../src").resolve()
sys.path.append(str(SRC_DIR))

from volare import data, features, model

# --- Locate available models ---
MODEL_DIR = SCRIPT_DIR / "../results/models"
model_files = [f for f in os.listdir(MODEL_DIR) if f.startswith("volare_lgb_h") and f.endswith(".pkl")]

# Extract horizon in seconds from filename
available_horizons = sorted([int(re.search(r'h(\d+)\.pkl', f).group(1)) for f in model_files])
if not available_horizons:
    raise RuntimeError(f"No LightGBM models found in {MODEL_DIR}")

horizon_to_file = {int(re.search(r'h(\d+)\.pkl', f).group(1)): MODEL_DIR / f for f in model_files}

# --- Function to apply model ---
def apply_model():
    try:
        horizon_seconds = int(horizon_var.get())
        model_file = horizon_to_file[horizon_seconds]

        # Load model
        with open(model_file, "rb") as f:
            lgb_model = pickle.load(f)

        # Ask user to select CSV
        file_path = filedialog.askopenfilename(
            title="Select CSV file with your candle data",
            filetypes=[("CSV files", "*.csv")]
        )
        if not file_path:
            return

        # Load data
        df = data.load_candles(file_path, nrows=300000)

        # Compute features
        k, alpha = 8, 1
        df = features.compute_log_return(df)
        df = features.compute_rolling_volatility(df, horizon_seconds=horizon_seconds, k=k)
        df = features.compute_lagged_rolling_volatility(df, horizon_seconds=horizon_seconds, alpha=alpha, k=k)
        df = features.compute_multi_window_rolling_vol(df, horizon_seconds=horizon_seconds)
        df = features.compute_intraday_seasonality(df)
        df = features.compute_volatility_slope(df, horizon_seconds=horizon_seconds)
        df = features.compute_volatility_zscore(df, horizon_seconds=horizon_seconds)
        df = features.compute_volatility_acceleration(df)

        # Extract feature columns and drop rows with NaNs
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
        if rolling_cols:
            baseline_medium = np.log(df[rolling_cols[len(rolling_cols)//2]].iloc[df_clean.index].values + eps)
        else:
            baseline_medium = None

        # --- Plot inside GUI ---
        # Downsample for speed if too many points
        max_points = 1000
        step = max(1, len(preds)//max_points)
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(preds[::step], label='Model Prediction', color='steelblue')
        if baseline_medium is not None:
            ax.plot(baseline_medium[::step], label='Medium Rolling Volatility', color='orange', alpha=0.7)
        ax.set_xlabel('Time step')
        ax.set_ylabel('Volatility')
        ax.set_title(f'Predictions vs Baseline (Horizon {horizon_seconds}s)')
        ax.legend()
        fig.tight_layout()

        # Clear previous canvas if exists
        for widget in plot_frame.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        canvas.get_tk_widget().update_idletasks()
        # root.update()

        # --- Optionally save predictions ---
        save_path = filedialog.asksaveasfilename(
            title="Save predictions",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if save_path:
            df_save = pd.DataFrame({"prediction": preds})
            if baseline_medium is not None:
                df_save["baseline_medium"] = baseline_medium
            df_save.to_csv(save_path, index=False)
            messagebox.showinfo("Saved", f"Predictions saved to {save_path}")

    except Exception as e:
        messagebox.showerror("Error", f"Failed to apply model:\n{e}")

# --- GUI ---
root = tk.Tk()
root.title("volare LightGBM model application")
root.geometry("1200x700")

root.lift()                # bring to front
root.attributes("-topmost", True)  # temporarily on top
root.after(500, lambda: root.attributes("-topmost", False)) # remove always-on-top after 0.5s 

# Plot frame
plot_frame = tk.Frame(root)
plot_frame.pack(fill='both', expand=True, padx=10, pady=10)

# Controls frame
controls_frame = tk.Frame(root)
controls_frame.pack(fill='x', padx=10, pady=5)

tk.Label(controls_frame, text="Select Horizon (seconds):").pack(side='left')
horizon_var = tk.StringVar(root)
horizon_var.set(str(available_horizons[0]))
dropdown = tk.OptionMenu(controls_frame, horizon_var, *available_horizons)
dropdown.pack(side='left', padx=5)

tk.Button(controls_frame, text="Select CSV and Apply Model", command=apply_model).pack(side='left', padx=10)

root.mainloop()