# intrabar_momentum_reversal_asymmetry

**Timestamp:** 2026-03-12 18:59
**Verdict:** REJECTED

## Hypothesis
When price consistently opens near one extreme of its bar range but closes near the opposite extreme (intrabar reversal), it signals active two-way liquidity absorption and microstructure noise that tends to suppress subsequent directional volatility. Conversely, when price opens and closes near the same extreme (momentum persistence within bars), it signals one-sided order flow that precedes elevated realised volatility. The asymmetry between these two regimes — measured as the rolling balance of momentum bars versus reversal bars — is a forward-looking signal for volatility clustering.

## Construction
For each bar, compute: (1) open_position = (open - low) / (high - low).clip(1e-8), measuring where open sits in the bar range [0=low, 1=high]; (2) close_position = (close - low) / (high - low).clip(1e-8), measuring where close sits in the bar range; (3) intrabar_direction = close_position - open_position, which is positive for bullish bars (price moved from low-end open to high-end close) and negative for bearish bars; (4) reversal_flag = 1 if abs(intrabar_direction) < 0.3 (price opened and closed near midpoint, indicating absorption/indecision) else 0; (5) momentum_flag = 1 if abs(intrabar_direction) > 0.6 (price traversed most of the bar range directionally) else 0; (6) asymmetry_ratio = rolling_sum(momentum_flag, 180 bars) / (rolling_sum(reversal_flag, 180 bars) + 1e-8), capped at a maximum of 10 to prevent outliers. High asymmetry_ratio (many momentum bars, few reversal bars) predicts elevated upcoming volatility; low ratio predicts suppressed volatility.

## Rejection Criteria
Reject if: (1) Out-of-sample RMSE improvement over rolling vol baseline is absent or negative in any two consecutive walk-forward folds; (2) Feature importance drifts by more than 40% across rolling windows, indicating noise-fitting; (3) Performance degrades by more than 30% in any single named regime (COVID shock, 2022 inflation cycle, or post-normalisation 2023); (4) The signal shows monotonic degradation across chronological test folds, consistent with decay rather than persistence; (5) More than 10% of rows produce NaN after warmup.

## Test Results
```json
{
  "per_pair": {},
  "aggregate": {
    "mean_improvement_vs_lgbm_pct": null,
    "mean_improvement_vs_naive_pct": null,
    "mean_importance_drift_pct": null,
    "any_monotonic_decay": false,
    "pairs_tested": [],
    "errors": [
      "EURGBP \u2014 validation error: operands could not be broadcast together with shapes (386956,2) (386956,) ",
      "GBPUSD \u2014 validation error: operands could not be broadcast together with shapes (386996,2) (386996,) "
    ]
  }
}
```

## Triggered Principles
P01, P03, P04, P06

## Summary
The feature failed to produce any usable test results due to shape mismatch errors on both tested currency pairs (EURGBP and GBPUSD), meaning no out-of-sample performance data exists at all. With zero per-pair results and no improvement metrics available, every quantitative rejection criterion is effectively triggered by default — we cannot confirm OOS improvement, importance stability, regime robustness, or absence of signal decay. The construction errors suggest the feature is returning a 2-column output instead of a scalar series, pointing to a bug in the asymmetry_ratio computation or its integration into the validation pipeline. Even setting aside the implementation failure, the hypothesis — while economically plausible — relies on a coarse threshold-based categorisation that may not generalise across FX pairs with different tick structures and liquidity profiles. There is nothing in these results that justifies promotion or even a provisional pass.

## Next Action
Fix the implementation bug causing the (N,2) output shape — likely the asymmetry_ratio or one of its intermediate series is returning a DataFrame instead of a Series — then re-run the full validation pipeline before any further evaluation of the signal's merit.
