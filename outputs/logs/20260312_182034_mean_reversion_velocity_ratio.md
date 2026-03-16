# mean_reversion_velocity_ratio

**Timestamp:** 2026-03-12 18:20
**Verdict:** REJECTED

## Hypothesis
When price repeatedly reverts toward its recent mean at high speed, it signals a low-volatility, liquidity-rich regime. Conversely, when price moves persistently away from its mean without reversion, it indicates directional momentum and elevated volatility. The ratio of mean-reversion speed (how quickly close returns toward the rolling mean) relative to recent range provides a regime-aware volatility predictor: low ratio → trending/volatile, high ratio → mean-reverting/calm.

## Construction
1. Compute a short-term rolling mean of close over 18 bars (~3 minutes). 2. Compute the signed deviation of close from that mean. 3. Compute the one-bar change in that deviation (delta_deviation): if price is moving toward the mean, delta_deviation has opposite sign to deviation (negative product). 4. Compute a 'reversion force' as the rolling mean of (-deviation * delta_deviation) over 36 bars, capturing how consistently price is pulled back toward mean. 5. Normalise by the rolling standard deviation of close over 36 bars (clipped to avoid zero division) to make the signal scale-invariant. The result is the mean-reversion velocity ratio: positive values indicate strong mean reversion (calm), negative values indicate trending divergence (volatile).

## Rejection Criteria
Reject if: (1) Out-of-sample RMSE improvement over rolling vol baseline is absent or negative; (2) Feature importance degrades by more than 30% in any single named regime (especially COVID March 2020 vs 2022 inflation period); (3) Importance drift exceeds 40% across rolling walk-forward windows; (4) Signal shows monotonic performance decay across chronological folds, suggesting the structural reversion dynamic has dissipated post-2022; (5) More than 10% of rows produce NaN values in live computation.

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
P04, P01, P03

## Summary
The feature could not be evaluated at all: both currency pairs failed with insufficient data after NaN removal, meaning no performance metrics, importance scores, or regime comparisons could be produced. This is a hard technical failure, not a marginal result. Pre-stated rejection criteria P04 is triggered because there is zero out-of-sample improvement evidence, and the NaN issue alone likely breaches the stated 10% NaN row threshold given it rendered entire datasets unusable. The underlying hypothesis about mean-reversion velocity is economically coherent, but a signal that cannot survive basic data preprocessing has no path to production in its current form.

## Next Action
Diagnose the NaN propagation: audit the rolling window sizes (18-bar mean, 36-bar reversion force, 36-bar std) against actual available history per pair, then add warm-up period handling so the feature only activates after sufficient bars have accumulated, and re-run the full test suite before any further evaluation.
