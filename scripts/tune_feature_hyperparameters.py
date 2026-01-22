#!/usr/bin/env python3
import argparse
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path
import matplotlib.pyplot as plt
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = (SCRIPT_DIR / "../src").resolve()
sys.path.append(str(SRC_DIR))
from volare import data, features, model

plt.style.use('../styles/science.mplstyle')

# ---------------- Feature tuning ----------------
def evaluate_features(df, horizon_seconds, window_factors, window_scales, lag_scales):
    """
    Grid search over feature-level hyperparameters.
    Returns a dict: {pair_name: best_params, ...}
    """
    results = {}

    # --- Define grid ---
    grid = [
        (wf, ws, ls) for wf in window_factors
                     for ws in window_scales
                     for ls in lag_scales
    ]

    for file in sorted(df.glob("*.csv")):
        df0 = data.load_candles(file, nrows=1_000_000)
        stem = file.stem
        currencies = stem.split('-')[1]
        base_currency = currencies[:3].upper()
        quote_currency = currencies[3:].upper()
        pair_name = f"{base_currency}-{quote_currency}"

        best_rmse = -np.inf
        best_params = None
        best_model = None
        print(f"\nTuning features for {pair_name}..")

        for wf, ws, ls in grid:
            print(f"Testing window_factor={wf}, window_scale={ws}, lag_scale={ls}")
            try:
                model_final, rmse_improve, mae_improve = train_single_pair(
                    df0, horizon_seconds,
                    window_factor=wf,
                    window_scale=ws,
                    lag_scale=ls
                )
                if rmse_improve > best_rmse:
                    best_rmse = rmse_improve
                    best_params = {"window_factor": wf, "window_scale": ws, "lag_scale": ls,
                                   "RMSE_vs_medium(%)": rmse_improve,
                                   "MAE_vs_medium(%)": mae_improve}
                    best_model = model_final
            except Exception as e:
                print("Error for this combination:", e)
                continue

        results[pair_name] = best_params

    return results

# ---------------- Single pair training (unchanged) ----------------
def train_single_pair(df, horizon_seconds, window_scale=0.75, window_factor=8, lag_scale=1):
    df = features.compute_log_return(df)
    df = features.compute_rolling_volatility(df, horizon_seconds=horizon_seconds,
                                             window_scale=window_scale, window_factor=window_factor)
    df = features.compute_lagged_rolling_volatility(df, horizon_seconds=horizon_seconds,
                                                    lag_scale=lag_scale, window_factor=window_factor)
    df = features.compute_multi_window_rolling_vol(df, horizon_seconds=horizon_seconds)
    df = features.compute_intraday_seasonality(df)
    df = features.compute_volatility_slope(df, horizon_seconds=horizon_seconds)
    df = features.compute_volatility_zscore(df, horizon_seconds=horizon_seconds)
    df = features.compute_volatility_acceleration(df)

    df = features.compute_future_rolling_volatility(df,horizon_seconds=horizon_seconds)

    feature_cols = [c for c in df.columns if c.startswith('rolling_vol')] + \
                   [c for c in df.columns if c.startswith('tod_')] + \
                   [c for c in ['vol_of_vol', 'vol_slope', 'vol_zscore', 'vol_accel']]

    X_train_full, X_test, y_train_full, y_test = model.split_data(df, feature_cols, target_col='rolling_log_future_vol', train_frac=0.8)
    X_train, X_val, y_train, y_val = model.split_training(X_train_full, y_train_full)

    gb_params = {
        'n_estimators': 500,
        'learning_rate': 0.1,
        'max_depth': 3,
        'random_state': 42,
        'num_leaves': 31,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbosity': -1
    }
    lgb_model, _ = model.train_model(X_train, y_train, X_val, y_val, **gb_params)
    model_final = model.retrain_model(np.vstack([X_train, X_val]), np.concatenate([y_train, y_val]), lgb_model)

    y_pred = model_final.predict(X_test)
    eps = 1e-8
    rolling_cols = [c for c in feature_cols if 'rolling_vol_' in c and 'cand' in c]
    col_idx = feature_cols.index(rolling_cols[len(rolling_cols)//2])
    baseline_medium = np.log(np.clip(X_test[:, col_idx], eps, None))
    # baseline_medium = np.log(X_test[:, feature_cols.index(rolling_cols[len(rolling_cols)//2])] + eps)

    rmse_model = np.sqrt(mean_squared_error(y_test, y_pred))
    mae_model  = mean_absolute_error(y_test, y_pred)
    rmse_base  = np.sqrt(mean_squared_error(y_test, baseline_medium))
    mae_base   = mean_absolute_error(y_test, baseline_medium)

    rmse_improve = 100 * (rmse_base - rmse_model) / rmse_base
    mae_improve  = 100 * (mae_base - mae_model) / mae_base

    return model_final, rmse_improve, mae_improve

# ---------------- Main ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True,
                        help="Directory containing candle CSV files (e.g., questdb-gbpusd.csv)")
    args = parser.parse_args()

    candle_dir = Path(args.dir)

    # --- Feature-level grid ---
    window_factors = [4, 6, 8]     # e.g., multiples of horizon
    window_scales  = [0.5, 0.75, 1.0]
    lag_scales     = [0.5, 1, 2]   # multiplies horizon for lags

    results = evaluate_features(candle_dir, horizon_seconds=60*60,
                                window_factors=window_factors,
                                window_scales=window_scales,
                                lag_scales=lag_scales)

    # --- Save summary ---
    os.makedirs('../results/feature_tuning', exist_ok=True)
    summary_file = '../results/feature_tuning/best_params.csv'
    df_summary = pd.DataFrame(results).T
    df_summary.to_csv(summary_file)
    print(f"\nSaved best feature hyperparameters summary to {summary_file}")
