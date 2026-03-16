# volume_clock_volatility_clustering

**Timestamp:** 2026-03-12 18:13
**Verdict:** REJECTED

## Hypothesis
Price moves that occur within compressed time intervals (rapid succession of bar-to-bar changes) signal heightened market activity and precede elevated realised volatility over the next hour. When many consecutive 10-second bars each show above-average absolute returns, the market is in an active participation regime where volatility clusters. Conversely, long stretches of near-zero bar returns indicate absorbed liquidity and compressed forthcoming volatility. This is grounded in the market microstructure literature on volatility clustering and the Mandelbrot-Clark subordinated random walk, where clock time is less informative than activity time.

## Construction
1. Compute the absolute bar return for each 10-second bar: abs(close - open) / open. 2. Compute a rolling median of this absolute bar return over a short backward window (360 bars = 60 minutes) to establish a local activity baseline. 3. For each bar, flag it as 'active' if its absolute return exceeds the rolling median. 4. Compute a rolling sum of these active flags over a medium backward window (720 bars = 120 minutes) to get an activity density score. 5. Normalise this score by dividing by the window length to get a [0,1] bounded activity rate. 6. Subtract the long-run rolling mean of this activity rate (2160 bars = 6 hours) to centre the signal, producing a z-scored excess activity rate that is positive when recent activity density is elevated relative to its own recent history.

## Rejection Criteria
Reject if: (1) Out-of-sample RMSE does not improve over rolling volatility baseline on either EURGBP or GBPUSD. (2) Feature importance degrades by more than 30% between the low-vol 2023 regime and the elevated-vol 2022 regime, indicating regime dependence. (3) Walk-forward importance drift exceeds 40% across quarterly rolling windows. (4) Performance shows monotonic degradation across chronological test folds, suggesting the signal has decayed. (5) The signal correlation with future realised volatility is not statistically significant (p > 0.05) in at least two of the four named volatility regimes.

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
P01, P03, P04, P06

## Summary
The test produced no usable results for either EURGBP or GBPUSD due to insufficient data after NaN removal, meaning no performance metrics, importance scores, or correlation statistics could be evaluated. Every pre-stated rejection criterion is effectively triggered by default: there is no out-of-sample improvement demonstrated, no regime robustness evidence, no importance stability data, and no signal correlation figures. The feature's economic rationale grounded in volatility clustering and the subordinated random walk is coherent, so the hypothesis itself is not the problem. The failure is entirely in the construction pipeline, most likely caused by the cascading NaN generation from the long chained rolling windows (360, 720, and 2160 bars) consuming the majority of the available data before any valid observations remain. This is a data engineering failure, not a signal failure, but it still constitutes a clear rejection under current evidence.

## Next Action
Diagnose the NaN propagation by logging the row counts at each rolling window step, then either shorten the longest window (2160 bars) substantially, switch to an expanding window for the long-run mean centering, or ensure the input dataset covers a sufficiently long history to absorb all burn-in periods before the test period begins.
