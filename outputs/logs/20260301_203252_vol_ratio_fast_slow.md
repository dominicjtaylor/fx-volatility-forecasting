# vol_ratio_fast_slow

**Timestamp:** 2026-03-01 20:32
**Verdict:** REJECTED

## Hypothesis
When short-term realised volatility begins to diverge from medium-term realised volatility — specifically when the ratio of fast-window vol to slow-window vol crosses above a threshold — this signals a regime transition from calm to elevated volatility (or vice versa when the ratio compresses). Such transitions precede sustained changes in 1-hour forward volatility, as markets reprice risk during structural shifts. The ratio captures the acceleration of vol rather than its level, providing an early-warning signal for regime change that is more timely than level-based vol measures.

## Construction
Using 10s mid-price returns (computed as log(close_t / close_{t-1}) for each 10s bar): (1) Compute fast realised vol as the rolling standard deviation of log returns over the past 60 bars (600 seconds / 10 minutes), annualised by sqrt(6*24*365). (2) Compute slow realised vol as the rolling standard deviation of log returns over the past 360 bars (3600 seconds / 60 minutes), annualised by the same factor. (3) Feature = fast_vol / slow_vol. Apply independently to both EURGBP and GBPUSD. All windows are strictly backward-looking. Clip extreme values at the 99th percentile to prevent distortion during flash events. Compute at every 10s timestamp using only data available at that timestamp.

## Rejection Criteria
Reject if: (1) Out-of-sample RMSE improvement over the 60-minute rolling vol baseline is less than 1% for both currency pairs. (2) Feature importance degrades by more than 40% across consecutive 6-month walk-forward windows (P03). (3) Performance in any single named regime (e.g., COVID crisis March 2020, 2022 inflation regime) is more than 30% worse than the aggregate OOS performance (P01). (4) Predictive power shows monotonic decay across chronological test folds, suggesting the signal has structurally degraded (P06). (5) Correlation with next 3600s realised vol is not statistically significant (p > 0.05) in at least two distinct regimes.

## Test Results
```json
{
  "per_pair": {},
  "aggregate": {
    "mean_rmse_improvement_pct": null,
    "mean_importance_drift_pct": null,
    "any_monotonic_decay": false,
    "pairs_tested": [],
    "errors": [
      "EURGBP: No implementation found for feature: vol_ratio_fast_slow",
      "GBPUSD: No implementation found for feature: vol_ratio_fast_slow"
    ]
  }
}
```

## Triggered Principles
P01, P03, P04, P06

## Summary
The feature could not be evaluated at all because no implementation was found for vol_ratio_fast_slow on either EURGBP or GBPUSD. This is a hard implementation failure, not a marginal result — there is literally no evidence to assess, no RMSE improvement, no importance scores, and no regime performance data. Every quantitative rejection criterion is technically triggered by default since no results were produced. The underlying hypothesis is economically coherent and worth revisiting, but a feature that fails to execute cannot be promoted under any standard.

## Next Action
Implement the vol_ratio_fast_slow feature in the feature registry so it can be computed and tested — verify the function name, module path, and parameter signatures are correctly registered before re-running the full evaluation pipeline.
