# volare: FX Volatility Forecasting with Machine Learning

This project investigates whether short-horizon **FX spot volatility** contains forecastable structure beyond simple historical smoothing.  
Using high-frequency candle data, it implements a **LightGBM-based volatility forecasting pipeline** with strict chronological validation, baseline comparisons, and explicit attention to **regime robustness**.

> The goal is **not** to claim tradable “alpha”, but to assess when and why ML forecasts add value over rolling volatility estimators, and where they fail.

---

## Problem & Motivation

Short- to medium-term FX volatility forecasts are central to:

- Risk management and setting spreads  
- Optimising liquidity provision  
- Supporting quantitative trading strategies  

By forecasting volatility directly rather than prices, this project demonstrates how machine learning can highlight actionable signals while respecting the limits of high-frequency FX data.

---

## Validation & Assumptions

- All splits are **chronological**; random cross-validation is avoided due to temporal dependencies  
- Model performance is benchmarked against simple rolling-volatility baselines (short, medium, long windows)  
- **Out-of-sample performance** is reported where possible  
- Performance varies across horizons, FX pairs, and volatility regimes; short horizons show limited or sometimes negative gains  
- Results should be interpreted **relative to baselines**, not in absolute terms  

> Explicitly identifying model failure points strengthens trust in the approach.

---

## Methodology

1. Compute features and target from candle data  
2. Chronological train-test split  
3. Train **LightGBM** on engineered features  
4. Compare **LightGBM** forecasts to rolling-volatility baselines:  
   - Short-window: recent backward-looking volatility  
   - Medium-window: intra-hour trends  
   - Long-window: slower trends  
5. Evaluate with RMSE, MAE, and visual inspection  
6. Optionally, retrain the model on the full training data using the **optimal number of iterations**  

> **Interactive GUI:** Allows loading CSVs, visualising predicted vs actual volatility, comparing forecasts with baselines, and exporting results. Improvements are displayed for the selected horizon.

---

## Workflow

The code pipeline consists of:

- **Data ingestion** (`data.py`)  
- **Feature generation** (`features.py`): log returns, high-low ranges, rolling volatility, multi-window volatilities, slope, z-score, acceleration  
- **Model training** (`model.py`)  
- **Forecasting** (`predict.py`)  
- **Visualization** (`visualisation.py` and GUI)  

---

## Features

- Customisable forecast horizons (e.g., 10–60 minutes)  
- Multi-window rolling volatility features  
- Volatility-of-volatility, slope, and acceleration features  
- Chronological train/test split and optional walk-forward validation  
- Model persistence: save and reload **LightGBM** models for downstream use  

> **Caution:** Features are currently tuned for short horizons (~1 hour). Longer horizons may require re-tuning windows or adding features capturing slower volatility dynamics.

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
