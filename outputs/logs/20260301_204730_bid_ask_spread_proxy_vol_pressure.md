# bid_ask_spread_proxy_vol_pressure

**Timestamp:** 2026-03-01 20:47
**Verdict:** REJECTED

## Hypothesis
The intrabar high-low range relative to the open-close body, measured as a rolling asymmetry ratio, captures latent liquidity stress and order flow imbalance. When the wick-to-body ratio is persistently elevated, market makers are widening effective spreads and price discovery is increasingly uncertain, which precedes elevated realised volatility. Conversely, compressed wick ratios indicate orderly two-way flow and predict calm periods. This proxy for effective spread pressure should be regime-robust because liquidity stress manifests in both crisis and elevated-vol regimes.

## Construction
For each 10-second bar, compute the total wick length as (high - low) minus abs(close - open), which isolates the shadow portions of the candle. Divide by (high - low + epsilon) to normalise, yielding a wick dominance ratio in [0,1]. Then compute the exponentially weighted moving average of this ratio over a 360-bar lookback (60 minutes) with a 180-bar half-life (30 minutes). Subtract the longer-term EWMA over a 2160-bar lookback (6 hours) to produce a mean-deviation signal that captures recent stress relative to the local baseline. This difference is the feature: positive values indicate rising wick dominance (liquidity stress building), negative values indicate compression.

## Rejection Criteria
Reject if: (1) out-of-sample RMSE improvement over rolling vol baseline is absent or negative across any two consecutive regimes; (2) feature importance drift exceeds 40% across walk-forward windows; (3) signal correlation with future realised vol (3600s horizon) degrades monotonically across chronological folds, indicating decay; (4) the signal shows near-zero variance during the 2022 inflation regime, suggesting it fails under persistent directional vol pressure; (5) performance in the COVID March 2020 regime degrades by more than 30% relative to the post-2022 normalisation regime.

## Test Results
```json
{
  "per_pair": {
    "EURGBP": {
      "folds": [
        {
          "fold": 1,
          "rmse_model": 0.0492,
          "rmse_baseline": 0.060797,
          "rmse_improvement_pct": 19.076,
          "feature_target_correlation": 0.006
        },
        {
          "fold": 2,
          "rmse_model": 0.079422,
          "rmse_baseline": 0.090833,
          "rmse_improvement_pct": 12.563,
          "feature_target_correlation": -0.098
        },
        {
          "fold": 3,
          "rmse_model": 0.042412,
          "rmse_baseline": 0.055868,
          "rmse_improvement_pct": 24.085,
          "feature_target_correlation": -0.0449
        },
        {
          "fold": 4,
          "rmse_model": 0.024434,
          "rmse_baseline": 0.027012,
          "rmse_improvement_pct": 9.542,
          "feature_target_correlation": 0.1165
        },
        {
          "fold": 5,
          "rmse_model": 0.022089,
          "rmse_baseline": 0.023379,
          "rmse_improvement_pct": 5.517,
          "feature_target_correlation": 0.1961
        }
      ],
      "overall_rmse_improvement_pct": 14.157,
      "importance_drift_pct": 206.0,
      "monotonic_decay": false,
      "n_folds_completed": 5
    },
    "GBPUSD": {
      "folds": [
        {
          "fold": 1,
          "rmse_model": 0.052026,
          "rmse_baseline": 0.047596,
          "rmse_improvement_pct": -9.307,
          "feature_target_correlation": 0.2492
        },
        {
          "fold": 2,
          "rmse_model": 0.054235,
          "rmse_baseline": 0.045772,
          "rmse_improvement_pct": -18.491,
          "feature_target_correlation": 0.0288
        },
        {
          "fold": 3,
          "rmse_model": 0.035637,
          "rmse_baseline": 0.041923,
          "rmse_improvement_pct": 14.993,
          "feature_target_correlation": 0.1804
        },
        {
          "fold": 4,
          "rmse_model": 0.034586,
          "rmse_baseline": 0.037264,
          "rmse_improvement_pct": 7.185,
          "feature_target_correlation": 0.209
        },
        {
          "fold": 5,
          "rmse_model": 0.030244,
          "rmse_baseline": 0.031943,
          "rmse_improvement_pct": 5.317,
          "feature_target_correlation": 0.2305
        }
      ],
      "overall_rmse_improvement_pct": -0.061,
      "importance_drift_pct": 122.7,
      "monotonic_decay": false,
      "n_folds_completed": 5
    }
  },
  "aggregate": {
    "mean_rmse_improvement_pct": 7.048,
    "mean_importance_drift_pct": 164.3,
    "any_monotonic_decay": false,
    "pairs_tested": [
      "EURGBP",
      "GBPUSD"
    ],
    "errors": []
  }
}
```

## Triggered Principles
P01, P03, P04

## Summary
This feature fails on multiple hard criteria and cannot be promoted in its current form. The most damning result is extreme importance drift of 164% on average across pairs, peaking at 206% for EURGBP, which is more than four times the 40% rejection threshold — this tells us the model is fitting noise, not a persistent signal. GBPUSD shows essentially zero net RMSE improvement (-0.061%) with two early folds delivering losses of -9.3% and -18.5%, triggering the pre-stated criterion of absent or negative improvement across consecutive regimes. The correlations between the signal and future realised vol are near-zero and inconsistent across folds for both pairs, meaning the feature lacks a stable directional relationship with what it is supposed to predict. The economic rationale is coherent and the signal does not exhibit monotonic decay, but structural instability and cross-pair inconsistency make this unsuitable for production.

## Next Action
Investigate why GBPUSD and EURGBP diverge so sharply — specifically test whether the wick dominance signal is being dominated by microstructure noise in more liquid pairs (GBPUSD), and consider whether the 10-second bar granularity is too fine for this construction; retest at 1-minute bars with a pair-specific normalisation to assess whether importance drift reduces to an acceptable level before any further evaluation.
