import numpy as np
import pandas as pd

def compute_log_return(df):
    """
    Compute log returns from open and close prices.
    Log returns are additive over time.
    Adds column to dataframe.
    """
    rt = np.log(df['close'] / df['open'])
    df = df.copy()
    df['log_return'] = rt
    return df

def compute_high_low_range(df):
    """
    Compute high-low range from high and low prices.
    Adds column to dataframe.
    """
    df = df.copy()
    df['high_low_range'] = (df['high'] - df['low']) / df['open']
    return df

def compute_open_close_diff(df):
    """
    Compute open-close difference.
    Adds columns to dataframe.
    """
    df = df.copy()
    df['open_close'] = df['close'] - df['open']
    return df

def logged_lag_return(df, lag_seconds):
    """
    Compute lagged log returns for a given number of previous candles.
    Receives a dataframe containing a 'log_return' column.
    n_lags: Number of previous candles to include as lagged features (default is 4).
    Adds a column to dataframe.
    """
    df = df.copy()

    time_res = (df['timestamp'].iloc[1] - df['timestamp'].iloc[0]).total_seconds()
    lag_steps = [int(ls / time_res) for ls in lag_seconds]

    for lag in lag_steps:
        df[f'log_return_lag_{lag}'] = df['log_return'].shift(lag)

    return df

def compute_rolling_volatility(df, horizon_seconds, k):
    """
    Compute rolling volatility of log returns.
    Windows overlap to capture short-term changes.
    window: time to consider for rolling calculation (default is 60s).
    Adds column to dataframe.
    """
    df = df.copy()

    window_seconds = k * horizon_seconds

    print('Computing past rolling volatility..')
    time_res = (df['timestamp'].iloc[1] - df['timestamp'].iloc[0]).total_seconds()
    window_candles = int(window_seconds / time_res)

    df['rolling_vol'] = np.sqrt(
        df['log_return'].pow(2).rolling(window=window_candles).mean()
    )

    return df

def compute_lagged_rolling_volatility(df,horizon_seconds,alpha,k,lag_multipliers=[0.25, 0.5, 1, 2, 4]):
    """
    Compute lagged rolling volatility.
    Adds column to dataframe.
    """
    df = df.copy()

    print('Computing lagged rolling volatility..')
    time_res = (df['timestamp'].iloc[1] - df['timestamp'].iloc[0]).total_seconds()
    window_seconds = k * horizon_seconds
    lag_seconds = [alpha * f * horizon_seconds for f in lag_multipliers]
    lag_steps = [int(ls / time_res) for ls in lag_seconds]
    window_candles = int(window_seconds / time_res)

    for lag in lag_steps:
        df[f'rolling_vol_lag_{lag}'] = df['rolling_vol'].shift(lag)

    # volatility-of-volatility
    df['vol_of_vol'] = df['rolling_vol'].rolling(window=window_candles).std()

    return df

def compute_rolling_stats(df, window_seconds=3*60):
    """
    Compute rolling statistics of log returns over a specified window.
    Captures recent average returns and short-term volatility.
    NaNs appear at the beginning of the series for positions where the full window is not available.
    Receives dataframe containing a 'log_return' column.
    window: Number of periods to use for rolling calculations (default is 18).
    Adds columns to dataframe.
    """
    df = df.copy()
    time_res = (df['timestamp'].iloc[1] - df['timestamp'].iloc[0]).total_seconds()
    window_candles = int(window_seconds / time_res)
    df['roll_mean'] = df['log_return'].rolling(window=window_candles).mean()
    df['roll_std'] = df['log_return'].rolling(window=window_candles).std()
    return df

def compute_multi_window_rolling_vol(df, horizon_seconds, window_multipliers=[0.05, 0.1, 0.25, 0.5, 1, 3, 10], base_col='log_return'):
    """
    Compute rolling volatilities over multiple windows (HAR-RV style) as features.
    window_multipliers: Multiples of forecast horizon for the rolling windows
    base_col: Column name to compute rolling vol on (default 'log_return')
    Adds new column to the dataframe.
    """
    df = df.copy()
    
    print('Computing multi-window rolling volatility..')
    time_res = (df['timestamp'].iloc[1] - df['timestamp'].iloc[0]).total_seconds()
    
    for k in window_multipliers:
        window_candles = max(int(k * horizon_seconds / time_res), 2)
        col_name = f'rolling_vol_{window_candles}_cand'
        df[col_name] = df[base_col].rolling(window_candles).std()
    
    return df

def compute_intraday_seasonality(df):
    """
    Compute intraday (time-of-day) seasonality.
    Add column to dataframe.
    """
    df = df.copy()

    print('Computing intra-day seasonality..')
    minutes = df['timestamp'].dt.hour * 60 + df['timestamp'].dt.minute
    df['tod_sin'] = np.sin(2 * np.pi * minutes / 1440)
    df['tod_cos'] = np.cos(2 * np.pi * minutes / 1440)

    return df

def compute_volatility_slope(df, horizon_seconds):
    """
    Compute slope of rolling volatility over one forecast horizon.
    """
    df = df.copy()

    print('Computing volatility slope..')
    time_res = (df['timestamp'].iloc[1] - df['timestamp'].iloc[0]).total_seconds()
    H = max(int(horizon_seconds / time_res), 1)

    df['vol_slope'] = df['rolling_vol'] - df['rolling_vol'].shift(H)

    return df

def compute_volatility_zscore(df, horizon_seconds, z_window_multiplier=2.0, eps=1e-8):
    """
    Compute volatility z-score relative to recent regime.
    Window length is aligned to the forecast horizon:
        window = z_window_multiplier * forecast_horizon
    Adds column to dataframe.
    """
    df = df.copy()

    print('Computing volatility z-score..')
    time_res = (df['timestamp'].iloc[1] - df['timestamp'].iloc[0]).total_seconds()
    window_candles = max(int(z_window_multiplier * horizon_seconds / time_res),2)

    vol_mean = df['rolling_vol'].rolling(window=window_candles).mean()
    vol_std  = df['rolling_vol'].rolling(window=window_candles).std()

    df['vol_zscore'] = (df['rolling_vol'] - vol_mean) / (vol_std + eps)

    return df

def compute_volatility_acceleration(df):
    df = df.copy()

    print('Computing volatility acceleration..')
    df['vol_accel'] = df['rolling_vol'] * df['vol_slope']

    return df

def compute_future_rolling_volatility(df, horizon_seconds=2*60, eps=1e-8):
    """
    Compute future volatility of log returns over a given horizon.
    horizon: time ahead to compute the return [s].
    Adds column to dataframe.
    """
    df = df.copy()
    time_res = (df['timestamp'].iloc[1] - df['timestamp'].iloc[0]).total_seconds()
    H = int(horizon_seconds/time_res)

    r = np.log(df['close'].values[1:] / df['close'].values[:-1])
    # lr = np.log(df['close'] / df['close'].shift(1))

    r2 = r**2

    csum = np.cumsum(np.insert(r2,0,0))
    rms = np.sqrt((csum[H:] - csum[:-H]) / H)

    df['rolling_future_vol'] = np.nan
    df.iloc[:len(rms), df.columns.get_loc('rolling_future_vol')] = rms

    df['rolling_log_future_vol'] = np.log(df['rolling_future_vol'] + eps)

    return df