from lightgbm import LGBMRegressor, early_stopping, log_evaluation, record_evaluation
import numpy as np
import pandas as pd
from . import features

def split_data(df, feature_cols, target_col='rolling_future_vol', train_frac=0.8):
    """
    Split the data into training and testing sets.
    Splits chronologically because past values influence future values (dependent data).
    train_frac: Fraction of data to be used for training (high).
    Returns X_train, X_test, y_train, y_test.
    """
    df_model = df[feature_cols + [target_col]].dropna()
    
    X = df_model[feature_cols].values
    y = df_model[target_col].values
    
    # Chronological split
    n = len(X)
    split_idx = int(n * train_frac)
    
    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_test = X[split_idx:]
    y_test = y[split_idx:]
    
    return X_train, X_test, y_train, y_test

def split_training(X_train,y_train,val_frac=0.1):
    val_index = int(len(X_train) * (1 - val_frac))
    X_tr, X_val = X_train[:val_index], X_train[val_index:]
    y_tr, y_val = y_train[:val_index], y_train[val_index:]
    return X_tr, X_val, y_tr, y_val

def train_model(X_train,y_train,X_val,y_val,**kwargs):
    """
    Train a Light Gradient Boosting Machine to predict volatility
    Returns trained model.
    kwargs: Additional parameters for the model.
    """
    model = LGBMRegressor(**kwargs)
    # model.fit(X_train, y_train)#eval_set=[(X_val, y_val)],eval_metric='rmse',early_stopping_rounds=50,verbose=50)

    evals_result = {}
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
        callbacks=[
            early_stopping(stopping_rounds=50),
            log_evaluation(period=50),
            record_evaluation(evals_result)
        ]
    )
    # print(evals_result)
    return model, evals_result

def retrain_model(X_train,y_train,model):
    model_refit = LGBMRegressor(**model.get_params())
    model_refit.n_estimators = model.best_iteration_
    model_refit.fit(X_train,y_train)
    return model

def evaluate_model(model,X_test,y_test_log,eps=1e-8):
    """
    Evaluate the trained model on test data.
    Returns Mean Squared Error (MSE) of the predictions.
    May return plots.
    """
    y_pred_log = model.predict(X_test)

    # #transform back from log to linear volatility
    y_pred_vol = np.exp(y_pred_log) - eps

    rmse_log = np.sqrt(np.mean((y_test_log - y_pred_log)**2))
    mae_log  = np.mean(np.abs(y_test_log - y_pred_log))

    return y_pred_log, y_pred_vol, rmse_log, mae_log

# def simulate_future_features(df, timestamps, horizon_seconds, k, alpha):
#     """
#     Generate future feature vectors over a given horizon in seconds,
#     at the same time resolution as the historical data, updating only
#     time-dependent features. Assumes rolling window > horizon.

#     Parameters
#     ----------
#     df : pd.DataFrame
#         Full historical DataFrame with all computed features.
#     timestamps : pd.Series
#         Corresponding timestamps for the df rows.
#     horizon_seconds : int
#         Total number of seconds to forecast into the future.
#     k : int
#         Rolling window multiplier (kept for signature consistency, unused here).
#     alpha : float
#         Lagged volatility parameter (kept for signature consistency, unused here).

#     Returns
#     -------
#     X_future : np.ndarray
#         Array of shape (num_steps, n_features) containing simulated future features.
#     t_future : pd.DatetimeIndex
#         Future timestamps corresponding to each row in X_future.
#     """
#     if df.empty or len(df) < 2:
#         raise ValueError("Input DataFrame must have at least 2 rows")

#     # Compute time resolution from historical data
#     time_res = (timestamps.iloc[1] - timestamps.iloc[0]).total_seconds()

#     # Determine number of prediction steps
#     num_steps = int(np.ceil(horizon_seconds / time_res))

#     # Generate future timestamps
#     t_future_start = timestamps.iloc[-1] 
#     t_future = pd.date_range(start=t_future_start, periods=num_steps, freq=pd.Timedelta(seconds=time_res))

#     # Start from the last row
#     last_row = df.iloc[-1].copy()

#     # Identify time-dependent features (e.g., intraday seasonality)
#     time_cols = [c for c in df.columns if c.startswith("tod_")]

#     # Build future feature matrix by repeating the last row
#     X_future = np.tile(last_row.values, (num_steps, 1))

#     # Update only time-dependent features
#     for i, ts in enumerate(t_future):
#         for c in time_cols:
#             col_idx = df.columns.get_loc(c)
#             if c.endswith("_sin"):
#                 X_future[i, col_idx] = np.sin(2 * np.pi * ts.hour / 24)
#             elif c.endswith("_cos"):
#                 X_future[i, col_idx] = np.cos(2 * np.pi * ts.hour / 24)

#     return X_future, t_future

def simulate_future_features_autoregressive(df, timestamps, horizon_seconds, k, alpha):
    """
    Generate future feature vectors over a given horizon in seconds using a full autoregressive simulation.

    Parameters
    ----------
    df : pd.DataFrame
        Full historical DataFrame with all computed features and a 'timestamp' column.
        Must have at least 2 rows to compute time resolution.
    timestamps : pd.Series
        Series of timestamps corresponding to df rows.
    horizon_seconds : int
        Total number of seconds to forecast into the future.
    k : int
        Window size multiplier for rolling features.
    alpha : float
        Parameter for lagged rolling volatility.

    Returns
    -------
    X_future : np.ndarray
        Array of shape (num_steps, n_features) containing simulated future features.
    t_future : pd.DatetimeIndex
        Future timestamps corresponding to each row in X_future.
    """
    if df.empty or len(df) < 2:
        raise ValueError("Input DataFrame must have at least 2 rows")

    # Compute time resolution from historical data
    time_res = (timestamps.iloc[1] - timestamps.iloc[0]).total_seconds()
    num_steps = int(np.ceil(horizon_seconds / time_res))

    # Generate future timestamps
    t_future_start = timestamps.iloc[-1]  # start exactly at last observed timestamp
    t_future = pd.date_range(start=t_future_start, periods=num_steps, freq=pd.Timedelta(seconds=time_res))

    # Start from last historical row
    df_current = df.iloc[[-1]].copy()
    df_current = df_current.reset_index(drop=True)

    future_features = []

    for ts in t_future:
        # Update timestamp-dependent features
        df_current['timestamp'] = ts
        df_current['tod_sin'] = np.sin(2 * np.pi * ts.hour / 24 + 2 * np.pi * ts.minute / 1440)
        df_current['tod_cos'] = np.cos(2 * np.pi * ts.hour / 24 + 2 * np.pi * ts.minute / 1440)

        # Compute rolling features using previous values including last predicted volatility
        df_current = features.compute_rolling_volatility(df_current, horizon_seconds=time_res, k=k)
        df_current = features.compute_lagged_rolling_volatility(df_current, horizon_seconds=time_res, alpha=alpha, k=k)
        df_current = features.compute_multi_window_rolling_vol(df_current, horizon_seconds=time_res)
        df_current = features.compute_volatility_slope(df_current, horizon_seconds=time_res)
        df_current = features.compute_volatility_zscore(df_current, horizon_seconds=time_res)
        df_current = features.compute_volatility_acceleration(df_current)
        df_current = features.compute_future_rolling_volatility(df_current, horizon_seconds=time_res)

        # Extract features in the same order as the model expects
        feature_cols = [c for c in df_current.columns if c.startswith('rolling_vol')] + \
                       [c for c in df_current.columns if c.startswith('tod_')] + \
                       [c for c in df_current.columns if c in ['vol_of_vol', 'vol_slope', 'vol_zscore', 'vol_accel']]

        features_vector = df_current[feature_cols].values[0]
        future_features.append(features_vector)

        # Predict next step using your trained model outside this function
        # The user will call: pred_next = model.predict(features_vector.reshape(1, -1))
        # Then update df_current with this predicted volatility for next step if desired
        # For example:
        # df_current['rolling_vol'] = pred_next  # anchor rolling_vol to predicted value

    X_future = np.vstack(future_features)
    return X_future, t_future
