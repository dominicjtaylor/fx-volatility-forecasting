"""
Test script — runs a single agent cycle with real data.
Usage: python scripts/run_cycle.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.agent.loop import run_cycle

DATA_DIR = "data"

pair_files = {
    "EURGBP": "questdb-eurgbp.csv",
    "GBPUSD": "questdb-gbpusd.csv",
}

data = {}
for pair, filename in pair_files.items():
    df = pd.read_csv(
        f"{DATA_DIR}/{filename}",
        parse_dates=["timestamp"],
        index_col="timestamp"
    )
    data[pair] = df
    print(f"Loaded {pair}: {len(df)} rows")

entry = run_cycle(data, user_hint="explore regime transition signals")