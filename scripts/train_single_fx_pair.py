import argparse
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = (SCRIPT_DIR / "../src").resolve()
sys.path.append(str(SRC_DIR))
from volare import data, features, model

import matplotlib.pyplot as plt
plt.style.use('../styles/science.mplstyle')

HORIZON_SECONDS = 60 * 60  # 60 minutes

def train_single_horizon(df, horizon_seconds, k=8, alpha=1):
    """
    Train a LightGBM model for a single forecast horizon.
    """
    # --- Features ---
    df = features.compute_log_return(df)
    df = features.compute_rolling_volatility(df, horizon_seconds=horizon_seconds, k=k)
    df = features.compute_lagged_rolling_volatility(df, horizon_seconds=horizon_seconds, alpha=alpha, k=k)
    df = features.compute_multi_window_rolling_vol(df, horizon_seconds=horizon_seconds)
    df = features.compute_intraday_seasonality(df)
    df = features.compute_volatility_slope(df, horizon_seconds=horizon_seconds)
    df = features.compute_volatility_zscore(df, horizon_seconds=horizon_seconds)
    df = features.compute_volatility_acceleration(df)

    # --- Target ---
    df = features.compute_future_rolling_volatility(df, horizon_seconds=horizon_seconds)

    # --- Feature selection ---
    df_columns = df.columns
    feature_cols = [col for col in df_columns if col.startswith('rolling_vol')] + \
                   [col for col in df_columns if col.startswith('tod_')] + \
                   [col for col in ['vol_of_vol', 'vol_slope', 'vol_zscore', 'vol_accel']]

    # --- Split data ---
    X_train_full, X_test, y_train_full, y_test = model.split_data(
        df, feature_cols, target_col='rolling_log_future_vol', train_frac=0.8)
    X_train, X_val, y_train, y_val = model.split_training(X_train_full, y_train_full)

    # --- Stage 1: initial training ---
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

    # --- Stage 2: retrain on full train + val ---
    X_train_combined = np.vstack([X_train, X_val])
    y_train_combined = np.concatenate([y_train, y_val])
    model_final = model.retrain_model(X_train_combined, y_train_combined, lgb_model)

    return model_final, df, X_test, y_test, feature_cols


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True,
                        help="Directory containing candle CSV files (e.g., questdb-gbpusd.csv)")
    args = parser.parse_args()

    candle_dir = Path(args.dir)
    csv_files = sorted(candle_dir.glob("*.csv"))

    models = {}
    results = {}

    for file in csv_files:
        stem = file.stem  # e.g., 'questdb-gbpusd'
        if '-' not in stem:
            print(f"Skipping {file}, cannot parse currency pair")
            continue
        currencies = stem.split('-')[1]  # 'gbpusd'
        base_currency = currencies[:3].upper()
        quote_currency = currencies[3:].upper()
        pair_name = f"{base_currency}-{quote_currency}"

        print(f"\nProcessing {pair_name}...")

        # Load first 500,000 rows
        df0 = data.load_candles(file, nrows=500_000)

        # Train model
        lgb_model, df, X_test, y_test, feature_cols = train_single_horizon(df0, horizon_seconds=HORIZON_SECONDS)
        models[pair_name] = lgb_model

        # --- Compute model performance vs medium-window baseline ---
        y_pred = lgb_model.predict(X_test)
        eps = 1e-8
        rolling_cols = [c for c in feature_cols if 'rolling_vol_' in c and 'cand' in c]
        baseline_medium = np.log(X_test[:, feature_cols.index(rolling_cols[len(rolling_cols)//2])] + eps)

        rmse_model = np.sqrt(mean_squared_error(y_test, y_pred))
        mae_model  = mean_absolute_error(y_test, y_pred)
        rmse_base  = np.sqrt(mean_squared_error(y_test, baseline_medium))
        mae_base   = mean_absolute_error(y_test, baseline_medium)

        rmse_improve = 100 * (rmse_base - rmse_model) / rmse_base
        mae_improve  = 100 * (mae_base - mae_model) / mae_base

        results[pair_name] = {"RMSE_vs_medium(%)": rmse_improve, "MAE_vs_medium(%)": mae_improve}

        # Save model
        os.makedirs('../results/models', exist_ok=True)
        model_file = f'../results/models/volare_lgb_{currencies}_h{int(HORIZON_SECONDS)}.pkl'
        with open(model_file, 'wb') as f:
            pickle.dump(lgb_model, f)
        print(f"Saved model to {model_file}")

    # --- Summary ---
    df_results = pd.DataFrame(results).T
    print("\nPerformance vs medium-window baseline (% improvement):\n", df_results)

    # --- Bar plot: % improvement ---
    fig, ax = plt.subplots(figsize=(12,5))
    width = 0.35
    pairs = df_results.index.tolist()
    x = np.arange(len(pairs))

    ax.bar(x - width/2, df_results['RMSE_vs_medium(%)'], width, label='RMSE improvement', color='steelblue')
    ax.bar(x + width/2, df_results['MAE_vs_medium(%)'], width, label='MAE improvement', color='skyblue')

    ax.set_xticks(x)
    ax.set_xticklabels(pairs, rotation=45, ha='right')
    ax.set_ylabel('% improvement vs medium baseline')
    ax.set_title('Model Performance vs FX Currency Pair (60-min horizon)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs('../results/plots', exist_ok=True)
    plt.savefig('../results/plots/performance_vs_pair.png')
    print("\nSaved bar plot to ../results/plots/performance_vs_pair.png")