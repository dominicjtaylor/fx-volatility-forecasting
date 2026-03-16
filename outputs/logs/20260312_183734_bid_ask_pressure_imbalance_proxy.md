# bid_ask_pressure_imbalance_proxy

**Timestamp:** 2026-03-12 18:37
**Verdict:** REJECTED

## Hypothesis
When price consistently closes near the high of each bar (buying pressure) or near the low (selling pressure), directional order flow imbalance is elevated. Sustained one-sided pressure compresses realised spreads and reduces two-way liquidity, which precedes volatility expansion as the market searches for the opposing side. Conversely, balanced closes within bars indicate orderly two-way flow and predict lower near-term volatility. This is the microstructure 'order flow toxicity' concept applied to OHLC-only data.

## Construction
For each bar, compute the close position within the bar range: close_position = (close - low) / (high - low).clip(lower=1e-8), yielding a value in [0,1] where 1=closed at high, 0=closed at low. Then compute a 60-bar (~10 minute) rolling mean of this position, call it 'flow_bias' (0.5 = balanced). The imbalance signal is abs(flow_bias - 0.5) * 2, normalised to [0,1], representing directional pressure magnitude. To capture persistence of this imbalance (which matters more than single-bar extremes), compute the 30-bar rolling standard deviation of close_position and divide the imbalance signal by it, clipped at 1e-8. Higher values mean strong persistent directional pressure relative to bar-level noise — a regime shift precursor. Finally, z-score normalise over a 180-bar window to make it stationary across regimes.

## Rejection Criteria
Reject if: (1) out-of-sample RMSE versus rolling vol baseline does not improve by any margin on either EURGBP or GBPUSD; (2) feature importance drops more than 40% between the 2020-2021 crisis regime and the 2022-2023 elevated vol regime; (3) feature produces NaN for more than 10% of rows in live evaluation; (4) signal shows monotonic decay across chronological walk-forward folds suggesting the microstructure relationship has broken down post-2022.

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
The feature could not be evaluated at all — both currency pairs threw shape broadcast errors during validation, meaning no performance metrics, importance scores, or baseline comparisons were produced. With zero usable results, every quantitative rejection criterion is effectively triggered: there is no out-of-sample improvement, no importance data, and no regime comparison possible. The underlying economic hypothesis around order flow toxicity is coherent and worth preserving, but the current implementation has a clear engineering defect — the feature is likely returning a 2D array where a 1D series is expected, probably due to a DataFrame/Series mismatch in the z-score normalisation or division step. Signal decay was the one criterion that returned a result (no monotonic decay detected), but this is meaningless given the pipeline failed entirely. This feature must be fixed before any research judgement can be made.

## Next Action
Debug the shape mismatch: inspect the output of the z-score normalisation step to confirm it returns a 1D Series rather than a 2D DataFrame, likely caused by passing a DataFrame column through a rolling operation that returns a multi-column object — fix this, then re-run the full evaluation pipeline before any further research assessment.
