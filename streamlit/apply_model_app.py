import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os, sys
import re
from io import BytesIO
from pathlib import Path
import matplotlib.pyplot as plt

plt.style.use('../styles/science.mplstyle')

# --- Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = (SCRIPT_DIR / "../src").resolve()
sys.path.append(str(SRC_DIR))
MODEL_DIR = SCRIPT_DIR / "../results/models"

from volare import data, features

# --- Locate available models ---
model_files = [f for f in os.listdir(MODEL_DIR) if f.startswith("volare_lgb_h") and f.endswith(".pkl")]
if not model_files:
    st.error(f"No LightGBM models found in {MODEL_DIR}")
    st.stop()

available_horizons = sorted([int(re.search(r"h(\d+)\.pkl", f).group(1)) for f in model_files])
horizon_to_file = {int(re.search(r"h(\d+)\.pkl", f).group(1)): MODEL_DIR / f for f in model_files}

# --- Streamlit UI ---
st.title("Volare LightGBM Model Application")

# Upload CSV
uploaded_file = st.file_uploader("Upload your candle CSV file", type=["csv"])

# Select horizon
horizon_seconds = st.select_slider("Prediction Horizon (s)", options=available_horizons)

# Slider for time range to display
time_range_display = st.slider("Time range to display (seconds)", min_value=10, max_value=3600, value=600, step=10)

# Button to compute predictions
compute = st.button("Compute Predictions")

# --- Functions ---
def load_model(horizon):
    model_file = horizon_to_file[horizon]
    with open(model_file, "rb") as f:
        return pickle.load(f)

def compute_features(df, horizon_seconds):

    df = features.compute_log_return(df)
    df = features.compute_rolling_volatility(df, horizon_seconds=horizon_seconds, k=8)
    df = features.compute_lagged_rolling_volatility(df, horizon_seconds=horizon_seconds, alpha=1, k=8)
    df = features.compute_multi_window_rolling_vol(df, horizon_seconds=horizon_seconds)
    df = features.compute_intraday_seasonality(df)
    df = features.compute_volatility_slope(df, horizon_seconds=horizon_seconds)
    df = features.compute_volatility_zscore(df, horizon_seconds=horizon_seconds)
    df = features.compute_volatility_acceleration(df)

    feature_cols = [c for c in df.columns if c.startswith('rolling_vol')] + \
                   [c for c in df.columns if c.startswith('tod_')] + \
                   [c for c in df.columns if c in ['vol_of_vol', 'vol_slope', 'vol_zscore', 'vol_accel']]
    df_clean = df[feature_cols].dropna()
    return df_clean

def plot_predictions(df_time, preds, baseline, horizon, display_range):
    fig, ax = plt.subplots(figsize=(12, 5))

    # Only display the last 'display_range' seconds
    mask = df_time >= (df_time.max() - display_range)
    t_plot = df_time[mask]
    preds_plot = preds[mask]
    baseline_plot = baseline[mask] if baseline is not None else None

    ax.plot(t_plot, preds_plot, label="Model Prediction", color="steelblue")
    if baseline_plot is not None:
        ax.plot(t_plot, baseline_plot, label="Medium Rolling Volatility", color="orange", alpha=0.7)

    # Shaded horizon
    ax.axvspan(t_plot.max(), t_plot.max() + horizon, color='gray', alpha=0.3, label=f"Horizon +{horizon}s")

    ax.set_xlabel("Time (s)")  # replace with actual datetime if available
    ax.set_ylabel("Volatility")
    ax.set_title("Predictions vs Baseline")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)

# --- Main logic ---
if uploaded_file and compute:
    st.info("Computing predictions...")
    df_raw = data.load_candles_st(uploaded_file,nrows=300_000)

    # Load model
    lgb_model = load_model(horizon_seconds)

    # Limit rows for performance
    # df_raw = df_raw.tail(300_000)

    # Compute features
    df_features = compute_features(df_raw, horizon_seconds)
    X_user = df_features.values

    # Predict
    preds = lgb_model.predict(X_user, num_threads=os.cpu_count())

    # Compute baseline (medium rolling vol)
    rolling_cols = [c for c in df_features.columns if 'rolling_vol_' in c and 'cand' in c]
    baseline = np.log(df_features[rolling_cols[len(rolling_cols)//2]].values + 1e-8) if rolling_cols else None

    # Time axis
    if "time" in df_raw.columns:
        df_time = df_raw["time"].iloc[df_features.index].values
    else:
        df_time = np.arange(len(preds))

    # Plot
    plot_predictions(df_time, preds, baseline, horizon_seconds, display_range=time_range_display)

    # Option to download predictions
    output_df = pd.DataFrame({"time": df_time, "prediction": preds})
    if baseline is not None:
        output_df["baseline_medium"] = baseline

    st.download_button(
        label="Download Predictions CSV",
        data=output_df.to_csv(index=False),
        file_name="predictions.csv",
        mime="text/csv"
    )
