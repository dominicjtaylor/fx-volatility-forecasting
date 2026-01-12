import pandas as pd

def load_candles(filepath,dtype=None,nrows=None):
    """
    Load candlestick data from a CSV file.
    Returns pandas DataFrame with parsed dates.
    """
    if dtype is None:
        dtype = {
            'open': 'float32',
            'high': 'float32',
            'low': 'float32',
            'close': 'float32',
            'symbol': 'category'
            }
    return pd.read_csv(filepath, parse_dates=['timestamp'], dtype=dtype, nrows=nrows)