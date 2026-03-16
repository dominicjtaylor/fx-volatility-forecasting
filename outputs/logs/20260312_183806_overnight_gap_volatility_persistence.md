# overnight_gap_volatility_persistence

**Timestamp:** 2026-03-12 18:38
**Verdict:** REJECTED

## Hypothesis
When price gaps significantly at session open (relative to the prior close), it signals a regime shift in realized volatility that persists for the subsequent hour. Large gaps indicate that information accumulated during illiquid/closed periods was material, and the subsequent price discovery process elevates intraday volatility as the market re-anchors. This is economically grounded in the open-to-close information asymmetry documented in market microstructure literature.

## Construction
For each bar, compute the absolute gap between current open and prior close, normalized by the rolling median true range (last 180 bars) to make it scale-invariant across regimes. Then compute a short-term exponentially weighted moving average (EWM span=36 bars, ~6 minutes) of these normalized gaps to capture persistent gap clustering. The resulting feature reflects whether recent opens have been consistently gapping away from prior closes, indicating heightened uncertainty and elevated vol persistence ahead.

## Rejection Criteria
Reject if: (1) Out-of-sample RMSE improvement over rolling vol baseline is absent in more than one regime (P04); (2) Feature importance drifts more than 40% across rolling walk-forward windows (P03); (3) Predictive power is concentrated only in the COVID March 2020 spike and absent in the 2022 inflation regime, indicating crisis-only dependency (P01); (4) Monotonic decay in predictive power across chronological folds from 2020 to 2024 (P06); (5) The normalized gap is near-zero for extended periods during low-volatility regimes, making the signal degenerate.

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
P04, P03

## Summary
The test produced no usable results: both currency pairs threw shape mismatch errors during validation, meaning the feature never actually ran to completion. There are no RMSE figures, no importance scores, and no per-pair breakdowns to evaluate against any of the research principles. With zero valid output, every quantitative rejection criterion is effectively untestable, but the implementation failure itself is disqualifying — a feature that cannot be computed reliably across pairs has no place in production. The economic hypothesis is coherent, but that cannot rescue broken code.

## Next Action
Fix the broadcast shape error in the feature construction pipeline — likely caused by a DataFrame vs Series mismatch when computing the normalized gap or EWM output — then re-run the full test suite before any further evaluation of the hypothesis.
