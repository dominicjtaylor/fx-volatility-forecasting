#!/usr/bin/env python3
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import pickle
import os
import sys
import re
from pathlib import Path

# Add src directory to path
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = (SCRIPT_DIR / "../src").resolve()
sys.path.append(str(SRC_DIR))

from volare import features, model
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --- Parse horizon from command line ---
if len(sys.argv) != 2:
    print("Usage: python apply_model_gui.py <horizon_in_seconds>")
    sys.exit(1)

try:
    horizon_sec = int(sys.argv[1])
except ValueError:
    print("Horizon must be an integer number of seconds.")
    sys.exit(1)

# --- Locate model ---
MODEL_DIR = SCRIPT_DIR / "../results/models"
pattern = re.compile(rf"volare_lgb_h{horizon_sec}\.pkl$")
matching_models = [f for f in os.listdir(MODEL_DIR) if pattern.match(f)]

if not matching_models:
    print(f"No model found for horizon {horizon_sec} seconds in {MODEL_DIR}")
    sys.exit(1)

model_file = MODEL_DIR / matching_models[0]

try:
    with open(model_file, "rb") as f:
        lgb_model = pickle.load(f)
    print(f"Loaded model: {matching_models[0]}")
except Exception as e:
    print(f"Failed to load model: {e}")
    sys.exit(1)

# --- Function to apply model to user data ---
def apply_model():
    file_path = filedialog.askopenfilename(
        title="Select CSV file with your candle data",
        filetypes=[("CSV files", "*.csv")]
    )
    if not file_path:
        return

    try:
        # Load data
        df_user = pd.read_csv(file_path)

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
        feature_cols = [col for col in df_user.columns if col.startswith('rolling_vol')] + \
                       [col for col in df_user.columns if col.startswith('tod_')] + \
                       [col for col in df_user.columns if col in ['vol_of_vol', 'vol_slope', 'vol_zscore', 'vol_accel']]

        X_user = df_user[feature_cols].values

        # --- Predict ---
        preds = lgb_model.predict(X_user)

        # --- Compute baselines ---
        rolling_cols = [c for c in feature_cols if 'rolling_vol_' in c and 'cand' in c]
        eps = 1e-8
        if rolling_cols:
            # Take medium window as baseline
            baseline_medium = np.log(df_user[rolling_cols[len(rolling_cols)//2]].values + eps)
            rmse_med = np.sqrt(mean_squared_error(preds, baseline_medium))
            mae_med = mean_absolute_error(preds, baseline_medium)
            rmse = np.sqrt(mean_squared_error(preds, preds))  # RMSE of model vs itself (trivial)
            mae = mean_absolute_error(preds, preds)           # MAE of model vs itself (trivial)
            # Improvement over medium baseline
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
root.title(f"Apply LightGBM Model (Horizon: {horizon_sec}s)")

tk.Label(root, text=f"Model horizon: {horizon_sec} seconds").pack(pady=10)
tk.Button(root, text="Select CSV and Apply Model", command=apply_model).pack(pady=20)

root.mainloop()