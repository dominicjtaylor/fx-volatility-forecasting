# range_expansion_acceleration

**Timestamp:** 2026-03-12 19:45
**Verdict:** REJECTED

## Hypothesis
The second derivative of the high-low range (acceleration of range expansion or contraction) captures whether volatility is not just increasing but doing so at an accelerating pace. When range acceleration is strongly positive, the market is entering a burst of volatility, providing a leading signal for elevated realised vol over the next hour. Conversely, decelerating range expansion (negative second derivative) suggests a volatility peak may be forming. This mirrors the physics intuition: velocity (first derivative = range change rate) tells you direction of vol movement; acceleration (second derivative) tells you the momentum behind that movement, which is more predictive of near-term vol persistence.

## Construction
1. Compute the raw high-low range for each bar: range = high - low, clipped at 1e-8 to avoid zeros. 2. Smooth the raw range with a short exponential moving average (span=6 bars, ~60 seconds) to reduce microstructure noise while preserving responsiveness. 3. Compute the first derivative (velocity) as the difference of the smoothed range over a medium window (12 bars, ~2 minutes): velocity = smoothed_range.diff(12). 4. Compute the second derivative (acceleration) as the difference of the velocity over the same window: acceleration = velocity.diff(12). 5. Normalise the acceleration by the rolling median of the absolute smoothed range over 120 bars (~20 minutes) to make the signal scale-invariant and comparable across different volatility regimes. Clip the denominator at 1e-8. Return the normalised acceleration series.

## Rejection Criteria
Reject if: (1) out-of-sample RMSE improvement over rolling vol baseline is absent or negative [P04]; (2) feature importance drifts more than 40% across chronological walk-forward windows [P03]; (3) predictive power is isolated to a single regime (e.g. only crisis, not elevated or normal) with >30% degradation elsewhere [P01]; (4) monotonic decay in performance across test folds indicating signal decay rather than persistent structure [P06]; (5) the distribution of the feature collapses to near-zero across most regimes, indicating the second derivative is too noisy at 10s resolution to carry information.

## Test Results
```json
{
  "per_pair": {
    "EURGBP": {
      "folds": [
        {
          "fold": 1,
          "rmse_naive_baseline": 0.559614,
          "rmse_lgbm_without_candidate": 0.449828,
          "rmse_lgbm_with_candidate": 0.362697,
          "improvement_vs_lgbm_baseline_pct": 19.37,
          "improvement_vs_naive_baseline_pct": 35.188,
          "candidate_importance_pct": 0.56
        },
        {
          "fold": 2,
          "rmse_naive_baseline": 0.497649,
          "rmse_lgbm_without_candidate": 0.433257,
          "rmse_lgbm_with_candidate": 0.394112,
          "improvement_vs_lgbm_baseline_pct": 9.035,
          "improvement_vs_naive_baseline_pct": 20.805,
          "candidate_importance_pct": 0.81
        },
        {
          "fold": 3,
          "rmse_naive_baseline": 0.491519,
          "rmse_lgbm_without_candidate": 0.379528,
          "rmse_lgbm_with_candidate": 0.308165,
          "improvement_vs_lgbm_baseline_pct": 18.803,
          "improvement_vs_naive_baseline_pct": 37.304,
          "candidate_importance_pct": 0.6
        },
        {
          "fold": 4,
          "rmse_naive_baseline": 0.445499,
          "rmse_lgbm_without_candidate": 0.343847,
          "rmse_lgbm_with_candidate": 0.234392,
          "improvement_vs_lgbm_baseline_pct": 31.832,
          "improvement_vs_naive_baseline_pct": 47.387,
          "candidate_importance_pct": 0.74
        },
        {
          "fold": 5,
          "rmse_naive_baseline": 0.393702,
          "rmse_lgbm_without_candidate": 0.313895,
          "rmse_lgbm_with_candidate": 0.250025,
          "improvement_vs_lgbm_baseline_pct": 20.348,
          "improvement_vs_naive_baseline_pct": 36.494,
          "candidate_importance_pct": 0.77
        }
      ],
      "overall_improvement_vs_lgbm_pct": 19.878,
      "overall_improvement_vs_naive_pct": 35.436,
      "mean_candidate_importance_pct": 0.7,
      "importance_drift_pct": 35.9,
      "monotonic_decay": false,
      "n_folds_completed": 5
    },
    "GBPUSD": {
      "folds": [
        {
          "fold": 1,
          "rmse_naive_baseline": 0.377095,
          "rmse_lgbm_without_candidate": 0.337691,
          "rmse_lgbm_with_candidate": 0.315446,
          "improvement_vs_lgbm_baseline_pct": 6.588,
          "improvement_vs_naive_baseline_pct": 16.349,
          "candidate_importance_pct": 0.3
        },
        {
          "fold": 2,
          "rmse_naive_baseline": 0.350893,
          "rmse_lgbm_without_candidate": 0.304227,
          "rmse_lgbm_with_candidate": 0.278716,
          "improvement_vs_lgbm_baseline_pct": 8.385,
          "improvement_vs_naive_baseline_pct": 20.569,
          "candidate_importance_pct": 0.33
        },
        {
          "fold": 3,
          "rmse_naive_baseline": 0.419607,
          "rmse_lgbm_without_candidate": 0.332167,
          "rmse_lgbm_with_candidate": 0.279966,
          "improvement_vs_lgbm_baseline_pct": 15.715,
          "improvement_vs_naive_baseline_pct": 33.279,
          "candidate_importance_pct": 0.19
        },
        {
          "fold": 4,
          "rmse_naive_baseline": 0.432321,
          "rmse_lgbm_without_candidate": 0.331152,
          "rmse_lgbm_with_candidate": 0.240652,
          "improvement_vs_lgbm_baseline_pct": 27.329,
          "improvement_vs_naive_baseline_pct": 44.335,
          "candidate_importance_pct": 0.37
        },
        {
          "fold": 5,
          "rmse_naive_baseline": 0.397478,
          "rmse_lgbm_without_candidate": 0.312415,
          "rmse_lgbm_with_candidate": 0.254682,
          "improvement_vs_lgbm_baseline_pct": 18.48,
          "improvement_vs_naive_baseline_pct": 35.925,
          "candidate_importance_pct": 0.49
        }
      ],
      "overall_improvement_vs_lgbm_pct": 15.299,
      "overall_improvement_vs_naive_pct": 30.091,
      "mean_candidate_importance_pct": 0.34,
      "importance_drift_pct": 89.3,
      "monotonic_decay": false,
      "n_folds_completed": 5
    }
  },
  "aggregate": {
    "mean_improvement_vs_lgbm_pct": 17.588,
    "mean_improvement_vs_naive_pct": 32.764,
    "mean_candidate_importance_pct": 0.52,
    "mean_importance_drift_pct": 62.6,
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
P03

## Summary
The feature shows genuinely strong predictive improvement — over 32% gain versus the naive rolling vol baseline and 17% versus LightGBM — and no monotonic decay was detected across folds, which is encouraging. However, the importance drift criterion is clearly breached: the aggregate drift is 62.6%, well above the 40% threshold, and GBPUSD alone shows 89.3% drift, which is severe. This means the model is not consistently relying on this feature in the same way across time windows, suggesting it is capturing noise or regime-specific artefacts rather than a persistent structural signal. The feature's construction at 10-second resolution with a second derivative amplifies noise by design, and the high drift on GBPUSD versus acceptable drift on EURGBP hints that the signal degrades on more liquid, noisier pairs. Until importance stability can be demonstrated, the strong aggregate performance numbers cannot be trusted as evidence of a robust signal.

## Next Action
Increase the smoothing span and derivative window lengths — for example, doubling the EMA span to 12 bars and the diff window to 24 bars — to reduce noise amplification in the second derivative, then re-run the walk-forward test specifically monitoring per-pair importance drift to confirm it falls below 40% before reconsidering promotion.
