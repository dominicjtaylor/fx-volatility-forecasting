import pandas as pd
from io import BytesIO
from pathlib import Path

def load_candles(file,dtype=None,nrows=None):
    """
    Load candlestick data from a CSV file.
    Returns pandas DataFrame with parsed dates.
    """
    if dtype is None:
        dtype = {
            'symbol': 'category',
            'open': 'float32',
            'high': 'float32',
            'low': 'float32',
            'close': 'float32'
            }

    df_head = pd.read_csv(file, parse_dates=['timestamp'], dtype=dtype, nrows=0) 

    expected_columns = ["timestamp", "symbol", "open", "high", "low", "close"]
    assert list(df_head.columns) == expected_columns, (
        f"CSV columns mismatch! Expected {expected_columns}, "
        f"but found {list(df_head.columns)}"
    )

    df = pd.read_csv(file, parse_dates=['timestamp'], dtype=dtype, nrows=nrows)

    return df 

def load_candles_st(file, dtype=None, nrows=None):
    if dtype is None:
        dtype = {
            'symbol': 'category',
            'open': 'float32',
            'high': 'float32',
            'low': 'float32',
            'close': 'float32'
        }

    # If file is a Streamlit UploadedFile, wrap in BytesIO
    if not isinstance(file, (str, Path)):
        file_bytes = BytesIO(file.getvalue())
    else:
        file_bytes = file

    # Read header
    df_head = pd.read_csv(file_bytes, parse_dates=['timestamp'], dtype=dtype, nrows=0)

    expected_columns = ["timestamp", "symbol", "open", "high", "low", "close"]
    assert list(df_head.columns) == expected_columns, (
        f"CSV columns mismatch! Expected {expected_columns}, "
        f"but found {list(df_head.columns)}"
    )

    # Reset pointer if file-like
    if hasattr(file_bytes, "seek"):
        file_bytes.seek(0)

    # Read full data
    df = pd.read_csv(file_bytes, parse_dates=['timestamp'], dtype=dtype, nrows=nrows)

    return df
