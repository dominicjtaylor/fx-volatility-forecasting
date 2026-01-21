from lightgbm import LGBMRegressor, early_stopping, log_evaluation, record_evaluation
import numpy as np
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

def simulate_future_features(df, horizon_seconds, k=8, alpha=1):
    """
    Generate future feature vectors over a given horizon in seconds,
    at the same time resolution as the historical data.

    Parameters
    ----------
    df : pd.DataFrame
        Full historical DataFrame with all computed features and a 'timestamp' column.
        Must have at least 2 rows to compute time resolution.
    horizon_seconds : int
        Total number of seconds to forecast into the future.
    k : int
        Window size for rolling features.
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
        raise ValueError("Input DataFrame must have at least 2 rows with a 'timestamp' column")

    # Compute time resolution from historical data
    time_res = (df['timestamp'].iloc[1] - df['timestamp'].iloc[0]).total_seconds()

    # Determine number of prediction steps
    num_steps = int(np.ceil(horizon_seconds / time_res))

    # Generate future timestamps
    t_future_start = df['timestamp'].iloc[-1] + pd.Timedelta(seconds=time_res)
    t_future = pd.date_range(start=t_future_start, periods=num_steps, freq=pd.Timedelta(seconds=time_res))

    # Start simulation from the last row
    df_last = df.iloc[[-1]].copy()
    df_current = df_last.copy()

    future_features = []

    for ts in t_future:
        # Assign future timestamp
        df_current['timestamp'] = ts

        # Iteratively compute features
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

        future_features.append(df_current[feature_cols].values[0])

    X_future = np.vstack(future_features)
    return X_future, t_future
