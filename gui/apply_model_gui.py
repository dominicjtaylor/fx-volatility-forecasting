#!/usr/bin/env python3
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import pickle
import os
import re
from pathlib import Path

# Add src directory to path
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = (SCRIPT_DIR / "../src").resolve()
os.sys.path.append(str(SRC_DIR))

from volare import data, features, model
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --- Locate available models ---
MODEL_DIR = SCRIPT_DIR / "../results/models"
model_files = [f for f in os.listdir(MODEL_DIR) if f.startswith("volare_lgb_h") and f.endswith(".pkl")]

# Extract horizon in seconds from filename
available_horizons = sorted([int(re.search(r'h(\d+)\.pkl', f).group(1)) for f in model_files])

if not available_horizons:
    raise RuntimeError(f"No LightGBM models found in {MODEL_DIR}")

# Map horizon to model file
horizon_to_file = {int(re.search(r'h(\d+)\.pkl', f).group(1)): MODEL_DIR / f for f in model_files}

# --- Function to apply model to user data ---
def apply_model():
    try:
        horizon_seconds = int(horizon_var.get())
        model_file = horizon_to_file[horizon_seconds]

        # Load model
        with open(model_file, "rb") as f:
            lgb_model = pickle.load(f)
        print(f"Loaded model: {model_file.name}")

        # Ask user to select CSV
        file_path = filedialog.askopenfilename(
            title="Select CSV file with your candle data",
            filetypes=[("CSV files", "*.csv")]
        )
        if not file_path:
            return

        # Load data
        df = data.load_candles(file_path)

        # Compute features for this horizon
        k, alpha = 8, 1
        df = features.compute_log_return(df)
        df = features.compute_rolling_volatility(df, horizon_seconds=horizon_seconds, k=k)
        df = features.compute_lagged_rolling_volatility(df, horizon_seconds=horizon_seconds, alpha=alpha, k=k)
        df = features.compute_multi_window_rolling_vol(df, horizon_seconds=horizon_seconds)
        df = features.compute_intraday_seasonality(df)
        df = features.compute_volatility_slope(df, horizon_seconds=horizon_seconds)
        df = features.compute_volatility_zscore(df, horizon_seconds=horizon_seconds)
        df = features.compute_volatility_acceleration(df)

        # Extract feature columns
        feature_cols = [col for col in df.columns if col.startswith('rolling_vol')] + \
                       [col for col in df.columns if col.startswith('tod_')] + \
                       [col for col in df.columns if col in ['vol_of_vol', 'vol_slope', 'vol_zscore', 'vol_accel']]

        X_user = df[feature_cols].dropna().values

        # --- Predict ---
        preds = lgb_model.predict(X_user)

        # --- Compute baselines ---
        rolling_cols = [c for c in feature_cols if 'rolling_vol_' in c and 'cand' in c]
        eps = 1e-8
        if rolling_cols:
            baseline_medium = np.log(df[rolling_cols[len(rolling_cols)//2]].values + eps)
            rmse_med = np.sqrt(mean_squared_error(preds, baseline_medium))
            mae_med = mean_absolute_error(preds, baseline_medium)
            rmse = np.sqrt(mean_squared_error(preds, preds))  # trivial
            mae = mean_absolute_error(preds, preds)
            rmse_improve = 100 * (rmse_med - rmse) / rmse_med if rmse_med != 0 else np.nan
            mae_improve = 100 * (mae_med - mae) / mae_med if mae_med != 0 else np.nan
        else:
            baseline_medium = None
            rmse_improve = mae_improve = np.nan

        # --- Show results ---
        msg = f"First 10 predictions:\n{preds[:10]}"
        if baseline_medium is not None:
            msg += f"\n\nImprovement vs medium rolling vol baseline:\nRMSE: {rmse_improve:.2f}%\nMAE: {mae_improve:.2f}%"
        messagebox.showinfo("Predictions & Stats", msg)

        # Save option
        save_path = filedialog.asksaveasfilename(
            title="Save predictions",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if save_path:
            pd.DataFrame({
                "prediction": preds,
                "baseline_medium": baseline_medium if baseline_medium is not None else np.nan
            }).to_csv(save_path, index=False)
            messagebox.showinfo("Saved", f"Predictions saved to {save_path}")

    except Exception as e:
        messagebox.showerror("Error", f"Failed to apply model:\n{e}")

# --- GUI ---
root = tk.Tk()
root.title("Apply LightGBM Model")

tk.Label(root, text="Select Horizon (seconds):").pack(pady=5)

horizon_var = tk.StringVar(root)
horizon_var.set(str(available_horizons[0]))  # default value

dropdown = tk.OptionMenu(root, horizon_var, *available_horizons)
dropdown.pack(pady=5)

tk.Button(root, text="Select CSV and Apply Model", command=apply_model).pack(pady=20)

root.mainloop()
