# range_expansion_acceleration

**Timestamp:** 2026-03-12 18:21
**Verdict:** REJECTED

## Hypothesis
When the true range of recent bars is expanding at an accelerating rate (second derivative of range is positive and growing), it signals that volatility is building momentum and is likely to remain elevated or increase over the next hour. Conversely, decelerating range expansion signals compression and lower forward volatility. This is grounded in the microstructure concept of volatility clustering: large moves beget large moves, but the rate of change of that clustering provides an earlier signal than the level of volatility alone.

## Construction
Compute the true range (TR) for each bar as max(high-low, |high-prev_close|, |low-prev_close|). Smooth TR over a short window (18 bars, ~3 minutes) to reduce noise. Compute the first difference of smoothed TR (velocity of range expansion). Then compute the rolling mean of that first difference over a medium window (60 bars, ~10 minutes) to get a stable acceleration estimate. Normalise by dividing by the rolling mean of the smoothed TR itself (clipped to avoid zero division) so the signal is scale-free and comparable across different volatility regimes. The result is a dimensionless ratio representing the proportional rate of acceleration of range expansion.

## Rejection Criteria
Reject if: (1) Out-of-sample RMSE improvement over rolling-vol baseline is absent or negative in any two or more volatility regimes (P01, P04). (2) Feature importance drift across rolling walk-forward windows exceeds 40% (P03). (3) Predictive power shows monotonic decay across chronological test folds, suggesting the signal has structurally deteriorated (P06). (4) Correlation with forward realised vol is not statistically distinguishable from zero in the post-2023 normalisation regime, suggesting the acceleration signal is regime-specific.

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
P01, P04

## Summary
The test produced no usable results whatsoever — both currency pairs failed with insufficient data after NaN removal, meaning the feature construction itself is broken. There is no evidence of predictive power, no baseline comparison, and no importance metrics to evaluate. All pre-stated rejection criteria relating to out-of-sample improvement (P04) and regime robustness (P01) are effectively triggered by default because the feature cannot even be evaluated. A feature that cannot survive basic data pipeline integrity checks has no path to promotion regardless of the underlying hypothesis. The construction must be fixed before any research verdict on the signal's merit is possible.

## Next Action
Diagnose the NaN propagation issue in the feature construction pipeline — likely caused by the compound smoothing and differencing steps producing excessive leading NaNs — then reduce window sizes or implement NaN-tolerant rolling functions, and re-run the full test suite before reconsidering the hypothesis.
