# volare: FX Volatility Forecasting with Machine Learning

This project forecasts FX spot volatility using **LightGBM**, a gradient boosting machine learning model, trained on high-frequency candle data in Python.  
It demonstrates a complete workflow of feature engineering, model training, validation, and forecasting, with practical applications for liquidity providers and FX traders.

---

## Workflow

The pipeline consists of:

- **Data ingestion** (`data.py`): load candle data in CSV format  
- **Feature generation** (`features.py`): log returns, high-low ranges, rolling volatility, multi-window rolling volatilities, volatility slope, z-score, and acceleration  
- **Train/test split**: chronological split to respect time dependencies  
- **Model training** (`model.py`): **LightGBM** trained on engineered features with optimal fitting  
- **Model validation**: RMSE, MAE, and visual inspection of predictions  
- **Forecasting** (`predict.py`): generate short- to medium-term volatility forecasts on new data  
- **Visualisation** (`visualisation.py` and GUI): plots and performance metrics, with interactive forecast exploration  

---

## Features

- Customisable forecast horizons (e.g., 10–60 minutes)  
- Multi-window rolling volatility features (short, medium, long windows)  
- Volatility-of-volatility, slope, and acceleration features  
- Chronological train/test split and optional walk-forward validation  
- Model persistence: save and reload **LightGBM** models for downstream use  

> **Caution:** Features are currently tuned for short horizons (~1 hour). Longer horizons may require re-tuning windows or adding features capturing slower volatility dynamics.

---

## Motivation

Short- to medium-term FX volatility forecasts are critical for:  

- Managing risk and setting spreads  
- Optimising liquidity provision  
- Supporting quantitative trading strategies  

By forecasting volatility directly rather than prices, this project highlights actionable insights for trading and risk teams.

---

## Data

This repository does not include raw FX data.

To run the pipeline, users must provide historical FX candle data formatted consistently with the training setup:

- Fixed time-resolution candles (e.g., 10-second intervals)  
- Columns: `timestamp`, `open`, `high`, `low`, `close`  
- Continuous time series without gaps  

Applying the model to differently formatted data requires retraining or feature redefinition.

---

## Methodology

1. Compute features and target from candle data  
2. Chronological train-test split  
3. Train **LightGBM** on engineered features  
4. Compare **LightGBM** model to simple baselines:  
   - **Short-window rolling volatility:** recent backward-looking volatility  
   - **Medium-window rolling volatility:** intra-hour trends  
   - **Long-window rolling volatility:** longer-term trends  
5. Evaluate with RMSE, MAE, and visual inspection  
6. Optionally, retrain the model on the full training data using the **optimal number of iterations** for improved performance  

> **Interactive GUI:** Allows users to load CSVs, visualize predicted vs actual volatility, compare forecasts with baselines, and export results. Forecast improvements are displayed for the selected horizon.

---

## Results

### Within-Sample Performance

The figures below show **within-sample performance** on the training data (first 800,000 candles of a single FX pair).  
> These results indicate how well the model learns historical patterns and **do not represent out-of-sample performance**.

**Features used:** past rolling volatility, lagged rolling volatility, multi-window rolling volatility, intra-day seasonality, volatility slope, z-score, acceleration.

### Predicted vs Actual Volatility with Baseline Comparisons

![Predicted vs Actual Volatility Baseline Compare](results/plots/predicted_vs_actual_volatility_baseline_compare.png)

> X-axis shows candle index rather than raw timestamps to anonymize data.  
> The model learns feature mappings beyond simple historical smoothing.

### Model Residuals

![Model Residuals for Regime Handling](results/plots/model_residual_volatility.png)

> Residuals help identify volatility regime changes and periods where the model may under- or over-predict.

### Performance vs Baselines

![Performance vs Baselines](results/plots/performance_baseline_compare.png)

- RMSE and MAE improvements over the medium-window rolling volatility baseline  
- Model forecast errors are typically 20–35% smaller than the simple baseline

### Performance vs Horizon

![Performance vs Horizon](results/plots/multi_horizon_performance_300000_cand.png)

- Forecasting error varies with horizon: short-term forecasts (~60 min) may see marginal or even negative improvements, while medium-term horizons can capture stronger predictive signals  
- Percentage improvement shows the model’s predictive signal relative to the medium-window baseline  

### Out-of-Sample / Unseen Data

Once feature hyperparameters are tuned, the model can be evaluated on an **unseen test set** of 1,000,000 candles (not included in training) using a 60-minute horizon.

![Performance vs FX Pair](results/plots/tuned_performance_vs_pair.png)

- Shows **percentage improvement in RMSE and MAE** over the baseline across multiple FX pairs  
- Highlights which pairs benefit most from machine learning forecasts  

---

## Usage

### Installation

Clone the repo and install the package in editable mode:

```bash
git clone https://github.com/dominicjtaylor/fx-volatility-forecasting.git
cd fx-volatility-forecasting
pip install -r requirements.txt
pip install -e .
```

### Python

Once the package is installed, you can import it in your Python scripts or interactive sessions:

```python
import volare
from volare import data, features, model

# Load candle data
df = data.load_last_candles("path/to/your/data.csv")

# Compute multi-window rolling volatility
df_features = features.compute_multi_window_rolling_vol(df, horizon_seconds=3600)

# Train/test split
X_train, X_test, y_train, y_test = model.split_data(df_features, feature_cols=[c for c in df_features.columns if 'rolling_vol' in c])

# Train LightGBM
lgb_model, _ = model.train_model(X_train, y_train, X_test, y_test)

# Forecast
preds = lgb_model.predict(X_test)
```

### Interactive GUI

![Applied Model](results/plots/applied_model.png)

To explore forecasts interactively:

1. Navigate to the `gui` folder:

```bash
cd gui
```

2. Run the GUI script:

```bash
python apply_multi_pair_model.py
```