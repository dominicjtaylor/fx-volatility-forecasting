# high_low_close_skewness_asymmetry

**Timestamp:** 2026-03-12 18:13
**Verdict:** REJECTED

## Hypothesis
Intrabar price action skewness — measured as the asymmetry between where price closes within the high-low range over a rolling window — captures informed order flow directionality and microstructure pressure. When price consistently closes near the extremes of bars (high or low), it signals persistent directional momentum or exhaustion, both of which are precursors to volatility expansion. A shift in this skewness measure from neutral toward extreme values predicts elevated realised volatility over the next hour, as the market is absorbing directional pressure that has not yet fully repriced.

## Construction
For each bar, compute the normalised close position within the bar range: close_position = (close - low) / (high - low), handling zero-range bars by assigning 0.5. Over a rolling window of 360 bars (60 minutes of 10s data), compute the rolling mean of this position series as a baseline. Then compute the rolling standard deviation of the same series over the same window. The feature is defined as the absolute deviation of the 30-bar (5-minute) rolling mean close position from the 360-bar rolling mean, normalised by the 360-bar rolling standard deviation of close positions. This z-score captures how far the recent short-term close-position skewness has drifted from the medium-term baseline — a large absolute z-score indicates emerging directional pressure in microstructure, signalling imminent volatility expansion.

## Rejection Criteria
Reject if: (1) Out-of-sample RMSE improvement over rolling volatility baseline is zero or negative across any two consecutive walk-forward folds; (2) Feature importance degrades monotonically across chronological test folds, indicating signal decay; (3) Predictive power in the March 2020 crisis regime or September 2022 GBP mini-budget regime degrades more than 30% relative to the calmer 2023 regime, indicating regime dependence; (4) The feature produces NaN for more than 20% of the live prediction window due to insufficient history at session opens; (5) Correlation with already-rejected range-expansion or vol-ratio features exceeds 0.85, indicating it is capturing the same signal under a different construction.

## Test Results
```json
{
  "per_pair": {
    "EURGBP": {
      "error": "Insufficient data after NaN removal"
    },
    "GBPUSD": {
      "error": "Insufficient data after NaN removal"
    }
  },
  "aggregate": {
    "mean_improvement_vs_lgbm_pct": null,
    "mean_improvement_vs_naive_pct": null,
    "mean_importance_drift_pct": null,
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
P02, P04

## Summary
The feature failed to produce any usable test results — both currency pairs returned errors due to insufficient data after NaN removal, meaning the feature could not even be evaluated. This directly triggers the pre-stated rejection criterion around NaN production (criterion 4), and means no principles around out-of-sample improvement, regime robustness, or importance stability can be assessed. A feature that cannot survive basic data pipeline validation is not ready for consideration, regardless of how plausible the underlying hypothesis may be. The NaN issue is likely rooted in the long 360-bar initialisation window combined with session boundary gaps, which the current construction does not handle adequately. Until the construction is made robust to data gaps and session opens, no meaningful signal evaluation is possible.

## Next Action
Diagnose the NaN propagation: audit how zero-range bars, session-open gaps, and the 360-bar initialisation period interact — then add explicit forward-fill or minimum-history guards so the feature produces valid values for at least 80% of the live prediction window before re-running the full walk-forward evaluation.
