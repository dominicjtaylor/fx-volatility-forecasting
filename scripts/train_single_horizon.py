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

def train_single_horizon(df, horizon_seconds, k=8, alpha=1):
    """
    Train a LightGBM model for a single forecast horizon.
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

    # --- Save model ---
    print('\nSaving model..')
    os.makedirs('results/models', exist_ok=True)
    model_file = f'results/models/volare_lgb_h{int(horizon_seconds)}.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump(model_final, f)
    print(f"Saved final model to {model_file}\n")

    return model_final, df, X_test, y_test, feature_cols

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True,
                        help="CSV file of candle data")
    parser.add_argument("--horizons", nargs='+', type=float, required=True,
                        help="Forecast horizons in seconds")
    args = parser.parse_args()

    # print('File to load:')
    currencies = args.file.split('-')[1].split('.')[0]  # 'eurgbp'
    base_currency = currencies[:3].upper()   # 'EUR'
    quote_currency = currencies[3:].upper()
    print(f"Using candle data for {base_currency}-{quote_currency}")

    # --- Load and prepare data ---
    print('Loading data..')
    nrows = 300000
    df0 = data.load_candles(args.file,nrows=nrows)
    if nrows is None:
        nrows = 'all'
    print(df0.head())

    models = {}
    dfs = {}
    X_tests = {}
    y_tests = {}
    feature_cols = {}
    for H in args.horizons:
        lgb_model, df, X_test_h, y_test_h, feature_cols_h = train_single_horizon(df0,horizon_seconds=H)
        models[H] = lgb_model
        dfs[H] = df
        X_tests[H] = X_test_h
        y_tests[H] = y_test_h
        feature_cols[H] = feature_cols_h

    results = []
    for H in args.horizons:
        # Predict
        y_pred = models[H].predict(X_tests[H])

        # Compute metrics
        rmse = np.sqrt(mean_squared_error(y_tests[H], y_pred))
        mae  = mean_absolute_error(y_tests[H], y_pred)

        # Get baselines
        # baselines = {}

        # lagged_col = [c for c in feature_cols[H] if 'lag' in c][0] # Lagged baseline
        # baselines['Lagged'] = np.log(X_tests[H][:, feature_cols[H].index(lagged_col)] + eps)

        rolling_cols = [c for c in feature_cols[H] if 'rolling_vol_' in c and 'cand' in c] # Short / Medium / Long rolling vol baselines
        # baselines['Short-window']  = np.log(X_tests[H][:, feature_cols.index(rolling_cols[0])] + eps)
        eps = 1e-8
        baseline_medium = np.log(X_tests[H][:, feature_cols[H].index(rolling_cols[len(rolling_cols)//2])] + eps)
        # baselines['Long-window']   = np.log(X_tests[H][:, feature_cols.index(rolling_cols[-1])] + eps)

        # Example: improvement over medium rolling vol baseline
        # baseline_col = [c for c in df0.columns if 'rolling_vol_' in c and 'cand' in c]
        # baseline_medium = np.log(df0[baseline_col[len(baseline_col)//2]].iloc[-len(y_test):] + 1e-8)
        rmse_med = np.sqrt(mean_squared_error(y_tests[H], baseline_medium))
        mae_med  = mean_absolute_error(y_tests[H], baseline_medium)
        rmse_improve = 100*(rmse_med - rmse)/rmse_med
        mae_improve  = 100*(mae_med - mae)/mae_med

        results.append({
            "horizon_sec": H,
            "RMSE": rmse,
            "MAE": mae,
            "RMSE_vs_medium(%)": rmse_improve,
            "MAE_vs_medium(%)": mae_improve
        })

    # Convert to DataFrame
    df_results = pd.DataFrame(results).sort_values("horizon_sec")
    print("\nMulti-horizon performance:\n", df_results)

    # --- Plot performance ---
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1,2,figsize=(12,4))

    ax[0].plot(df_results['horizon_sec']/60, df_results['RMSE'], marker='o', label='RMSE')
    ax[0].plot(df_results['horizon_sec']/60, df_results['MAE'], marker='s', label='MAE')
    ax[0].set_xlabel('Horizon (minutes)')
    ax[0].set_ylabel('Error')
    ax[0].set_title('Absolute performance vs horizon')
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()

    ax[1].plot(df_results['horizon_sec']/60, df_results['RMSE_vs_medium(%)'], marker='o', label='RMSE improvement')
    ax[1].plot(df_results['horizon_sec']/60, df_results['MAE_vs_medium(%)'], marker='s', label='MAE improvement')
    ax[1].set_xlabel('Horizon (minutes)')
    ax[1].set_ylabel('% improvement vs medium baseline')
    ax[1].set_title('Relative improvement vs horizon')
    ax[1].grid(True, alpha=0.3)
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(f'../results/plots/multi_horizon_performance_{nrows}_cand.png')
    # plt.show()
