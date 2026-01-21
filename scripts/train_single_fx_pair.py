import argparse
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = (SCRIPT_DIR / "../src").resolve()
sys.path.append(str(SRC_DIR))
from volare import data, features, model
plt.style.use('../styles/science.mplstyle')

def train_single_pair(df, horizon_seconds, k=8, alpha=1):
    """
    Train a LightGBM model for a single currency pair.
    Stage 1: training + validation to find best_iteration
    Stage 2: retraining on training+validation
    """

    # --- Compute features for this horizon ---
    print('\nComputing features..')
    df = features.compute_log_return(df)
    df = features.compute_rolling_volatility(df, horizon_seconds=horizon_seconds, k=k)
    df = features.compute_lagged_rolling_volatility(df, horizon_seconds=horizon_seconds, alpha=alpha, k=k)
    df = features.compute_multi_window_rolling_vol(df, horizon_seconds=horizon_seconds)
    df = features.compute_intraday_seasonality(df)
    df = features.compute_volatility_slope(df, horizon_seconds=horizon_seconds)
    df = features.compute_volatility_zscore(df, horizon_seconds=horizon_seconds)
    df = features.compute_volatility_acceleration(df)
    print('Finished computing features!\n')

    # --- Compute target for this horizon ---
    print('Computing target..')
    df = features.compute_future_rolling_volatility(df,horizon_seconds=horizon_seconds)
    print('Finished computing target!\n')

    print('Splitting data')
    df_columns = df.columns
    feature_cols = [col for col in df_columns if col.startswith('rolling_vol')] + \
                [col for col in df_columns if col.startswith('tod_')] + \
                [col for col in df_columns if col in ['vol_of_vol', 'vol_slope', 'vol_zscore', 'vol_accel']]
    print('Feature cols:',feature_cols,'\n')

    X_train_full, X_test, y_train_full, y_test = model.split_data(df, feature_cols, target_col='rolling_log_future_vol', train_frac=0.8)

    # Further split training into train/val for early stopping
    X_train, X_val, y_train, y_val = model.split_training(X_train_full, y_train_full)

    # --- Stage 1: initial training to find best_iteration ---
    print(f"Training horizon {horizon_seconds}s: Stage 1 (train + validation)")
    gb_params = {
        'n_estimators': 500,      
        'learning_rate': 0.1,     
        'max_depth': 3,            
        'random_state': 42,        
        'num_leaves': 31,          # default leaf-wise growth
        'feature_fraction': 0.9,   # subsample features per tree
        'bagging_fraction': 0.8,   # subsample rows per iteration
        'bagging_freq': 5,         # perform bagging every 5 rounds
        'verbosity': -1
    }
    lgb_model, evals_result = model.train_model(X_train, y_train, X_val, y_val, **gb_params)
    best_iter = lgb_model.best_iteration_
    print(f"Best iteration found: {best_iter}\n")

    # --- Stage 2: retrain on train + validation ---
    print(f"Stage 2: retrain on full training + validation set")
    X_train_combined = np.vstack([X_train, X_val])
    y_train_combined = np.concatenate([y_train, y_val])
    model_final = model.retrain_model(X_train_combined, y_train_combined, lgb_model)

    y_pred = model_final.predict(X_test)
    eps = 1e-8
    rolling_cols = [c for c in feature_cols if 'rolling_vol_' in c and 'cand' in c]
    baseline_medium = np.log(X_test[:, feature_cols.index(rolling_cols[len(rolling_cols)//2])] + eps)

    rmse_model = np.sqrt(mean_squared_error(y_test, y_pred))
    mae_model  = mean_absolute_error(y_test, y_pred)
    rmse_base  = np.sqrt(mean_squared_error(y_test, baseline_medium))
    mae_base   = mean_absolute_error(y_test, baseline_medium)

    rmse_improve = 100 * (rmse_base - rmse_model) / rmse_base
    mae_improve  = 100 * (mae_base - mae_model) / mae_base

    return model_final, rmse_improve, mae_improve

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True,
                        help="Directory containing candle CSV files (e.g., questdb-gbpusd.csv)")
    args = parser.parse_args()

    candle_dir = Path(args.dir)
    csv_files = sorted(candle_dir.glob("*.csv"))

    HORIZON_SECONDS = 60 * 60
    NROWS = 1_000_000

    # --- Load and prepare data ---

    results = {}
    for file in csv_files:

        print('Loading data..')
        df0 = data.load_candles(file,nrows=NROWS)
        print(df0.head())

        stem = file.stem  # e.g., 'questdb-gbpusd'
        currencies = stem.split('-')[1]  # 'gbpusd'
        base_currency = currencies[:3].upper()
        quote_currency = currencies[3:].upper()
        pair_name = f"{base_currency}-{quote_currency}"

        lgb_model, rmse_improve, mae_improve = train_single_pair(df0,horizon_seconds=HORIZON_SECONDS)
        results[pair_name] = {"RMSE_vs_medium(%)": rmse_improve, "MAE_vs_medium(%)": mae_improve}

        # --- Save model ---
        print('\nSaving model..')
        os.makedirs('../results/models', exist_ok=True)
        model_file = f'../results/models/volare_lgb_{currencies}_h{int(HORIZON_SECONDS)}.pkl'
        with open(model_file, 'wb') as f:
            pickle.dump(lgb_model, f)
        print(f"Saved final model to {model_file}\n")

    #Plot
    # Convert to DataFrame
    df_results = pd.DataFrame(results).T
    print("\nPerformance vs medium-window baseline (% improvement):\n", df_results)

    # --- Plot performance ---
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