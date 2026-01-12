# Tech Finance FX Volatility Forecasting

This project forecasts FX spot volatility using **LightGBM**, a gradient boosting machine learning model, trained on high-frequency candle data in Python.  
It demonstrates a complete workflow of feature engineering, model training, validation, and forecasting, with practical applications for liquidity providers and FX traders.

---

## Workflow

The pipeline consists of:

- **Data ingestion:** import candle data (`data.py`)  
- **Data cleaning:** parse timestamps, set dtypes, compute log returns  
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

- Historical FX candle data at 10-second resolution is required to run the pipeline. **No real or sample data is included in this repository.**  
- Columns expected: `timestamp`, `symbol`, `open`, `high`, `low`, `close`  
- Target: future rolling volatility over forecast horizon H, computed as RMS of log returns  
- Users must supply their own FX candle data to run the scripts.

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

Visualisations of predicted vs actual volatility are available in `results/plots/`.

---

## Usage

Clone the repo and install dependencies:

```bash
git clone https://github.com/dominicjtaylor/fx-volatility-forecasting.git
cd fx-volatility-forecasting
pip install -r requirements.txt