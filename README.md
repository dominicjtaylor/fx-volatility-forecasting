# volare: Tech Finance FX Volatility Forecasting

This project forecasts FX spot volatility using **LightGBM**, a gradient boosting machine learning model, trained on high-frequency candle data in Python.  
It demonstrates a complete workflow of feature engineering, model training, validation, and forecasting, with practical applications for liquidity providers and FX traders.

---

## Workflow

The pipeline consists of:

- **Data ingestion:** import candle data (`data.py`)
- **Feature generation** (`features.py`): log returns, high-low ranges, rolling volatility, rolling future volatility, volatility slope, z-score  
- **Train/test split:** chronological split to respect time dependencies  
- **Model training** (`model.py`): **LightGBM** trained on engineered features  
- **Model validation** (`model.py`): RMSE, MAE, and visual inspection  
- **Forecasting** (`predict.py`): generate volatility forecasts on new data  
- **Visualisation** (`visualisation.py`): plots and performance metrics  

---

## Features

- Customisable forecast horizons (e.g., 10–30 minutes)  
- Multi-window rolling volatility features  
- Volatility-of-volatility and slope features  
- Chronological train/test split and optional walk-forward validation  
- Model persistence: save and reload **LightGBM** models for downstream use  

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

To run the pipeline, users must provide their own historical FX candle data formatted consistently with the training setup used in this project:

- Fixed time-resolution candles (e.g. 10-second)
- Columns in order: `timestamp`, `open`, `high`, `low`, `close`
- Continuous time series without gaps

The trained model and feature pipeline assume this structure. Applying the model to differently formatted data would require retraining or feature redefinition.

---

## Methodology

1. Compute features and target from candle data  
2. Train-test split in chronological order  
3. Compare **LightGBM** model to simple baselines:  
   - **Persistence:** future volatility = current volatility  
   - **Rolling historical volatility:** backward-looking volatility  
4. Train **LightGBM** on features  
5. Evaluate with RMSE, MAE, and visual inspection of predictions  

---

## Results

| Model | RMSE | MAE |
|-------|------|-----|
| Persistence | 0.0012 | 0.0009 |
| Rolling Vol | 0.0010 | 0.0008 |
| **LightGBM** | **0.0008** | **0.0006** |

Visualisations are below and are available in `results/plots/`.

### Predicted vs Actual Volatility

**Figure:** Predicted vs actual log-volatility for a single FX currency pair.  
The model is trained **only on the first 300,000 candles** of the time series, with predictions evaluated on subsequent, unseen data.

*Note:* Results should be interpreted as within-pair temporal generalisation rather than cross-asset performance.

![Predicted vs Actual Volatility](results/plots/predicted_vs_actual_volatility.pdf)

---

## Usage

Clone the repo and install dependencies:

```bash
git clone https://github.com/dominicjtaylor/fx-volatility-forecasting.git
cd fx-volatility-forecasting
pip install -r requirements.txt

import volare