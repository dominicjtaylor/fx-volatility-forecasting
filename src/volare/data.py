import pandas as pd

def load_candles(filepath):
    """
    Load candlestick data from a CSV file.
    Returns pandas DataFrame with parsed dates.
    """
    return pd.load_csv(filepath, parse_dates=['timestamp'])

def clean_data(df):
    """
    Clean data:
    - remove duplicates
    - handle missing values
    - ensure correct data types
    """
    pass

