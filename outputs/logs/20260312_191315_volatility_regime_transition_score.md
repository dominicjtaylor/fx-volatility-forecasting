# volatility_regime_transition_score

**Timestamp:** 2026-03-12 19:13
**Verdict:** REJECTED

## Hypothesis
When the short-term realised volatility is diverging rapidly from its longer-term baseline, the market is transitioning between volatility regimes. The ratio of a fast volatility estimate to a slow volatility estimate, when it crosses meaningfully above or below unity, signals a regime shift in progress. Such transitions tend to precede sustained elevated or depressed volatility over the subsequent hour, as institutional positioning adjustments propagate through the market. The rate-of-change of this ratio (its momentum) captures the acceleration of the transition and should provide additional predictive power beyond the level alone.

## Construction
1. Compute bar-level true range: TR = max(high-low, abs(high-prev_close), abs(low-prev_close)). 2. Compute fast realised vol as EWM of TR with span=18 bars (~3 minutes). 3. Compute slow realised vol as EWM of TR with span=180 bars (~30 minutes). 4. Form the vol ratio = fast_vol / slow_vol (clipped denominator at 1e-8 to avoid division by zero). 5. Compute the 36-bar rolling z-score of this ratio (subtract rolling mean, divide by rolling std) to normalise across regimes. 6. Compute the 18-bar momentum of this z-score (current minus 18-bars-ago value) to capture the acceleration of regime transition. 7. The feature is this z-score momentum, capturing how quickly the fast/slow vol ratio is moving away from its recent norm.

## Rejection Criteria
Reject if: (1) Out-of-sample RMSE improvement over rolling vol baseline is absent in any two or more of the four named regimes (P01, P04). (2) Feature importance drift across walk-forward windows exceeds 40% (P03). (3) Predictive power shows monotonic decay across chronological test folds, suggesting the signal is historical artefact rather than persistent structure (P06). (4) The feature produces identical or near-zero variance during the 2020 COVID crisis period, indicating insensitivity to the most extreme regime transition in the sample (P01). (5) Any evidence of look-ahead contamination detected via future-bar correlation checks (P02).

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
          "rmse_lgbm_with_candidate": 0.364376,
          "improvement_vs_lgbm_baseline_pct": 18.997,
          "improvement_vs_naive_baseline_pct": 34.888,
          "candidate_importance_pct": 0.14
        },
        {
          "fold": 2,
          "rmse_naive_baseline": 0.497649,
          "rmse_lgbm_without_candidate": 0.433257,
          "rmse_lgbm_with_candidate": 0.395784,
          "improvement_vs_lgbm_baseline_pct": 8.649,
          "improvement_vs_naive_baseline_pct": 20.469,
          "candidate_importance_pct": 0.16
        },
        {
          "fold": 3,
          "rmse_naive_baseline": 0.491519,
          "rmse_lgbm_without_candidate": 0.379528,
          "rmse_lgbm_with_candidate": 0.309388,
          "improvement_vs_lgbm_baseline_pct": 18.481,
          "improvement_vs_naive_baseline_pct": 37.055,
          "candidate_importance_pct": 0.29
        },
        {
          "fold": 4,
          "rmse_naive_baseline": 0.445499,
          "rmse_lgbm_without_candidate": 0.343847,
          "rmse_lgbm_with_candidate": 0.234697,
          "improvement_vs_lgbm_baseline_pct": 31.744,
          "improvement_vs_naive_baseline_pct": 47.318,
          "candidate_importance_pct": 0.28
        },
        {
          "fold": 5,
          "rmse_naive_baseline": 0.393702,
          "rmse_lgbm_without_candidate": 0.313895,
          "rmse_lgbm_with_candidate": 0.250592,
          "improvement_vs_lgbm_baseline_pct": 20.167,
          "improvement_vs_naive_baseline_pct": 36.35,
          "candidate_importance_pct": 0.35
        }
      ],
      "overall_improvement_vs_lgbm_pct": 19.608,
      "overall_improvement_vs_naive_pct": 35.216,
      "mean_candidate_importance_pct": 0.24,
      "importance_drift_pct": 86.1,
      "monotonic_decay": false,
      "n_folds_completed": 5
    },
    "GBPUSD": {
      "folds": [
        {
          "fold": 1,
          "rmse_naive_baseline": 0.377095,
          "rmse_lgbm_without_candidate": 0.337691,
          "rmse_lgbm_with_candidate": 0.31496,
          "improvement_vs_lgbm_baseline_pct": 6.731,
          "improvement_vs_naive_baseline_pct": 16.477,
          "candidate_importance_pct": 0.09
        },
        {
          "fold": 2,
          "rmse_naive_baseline": 0.350893,
          "rmse_lgbm_without_candidate": 0.304227,
          "rmse_lgbm_with_candidate": 0.279795,
          "improvement_vs_lgbm_baseline_pct": 8.031,
          "improvement_vs_naive_baseline_pct": 20.262,
          "candidate_importance_pct": 0.12
        },
        {
          "fold": 3,
          "rmse_naive_baseline": 0.419607,
          "rmse_lgbm_without_candidate": 0.332167,
          "rmse_lgbm_with_candidate": 0.280502,
          "improvement_vs_lgbm_baseline_pct": 15.554,
          "improvement_vs_naive_baseline_pct": 33.151,
          "candidate_importance_pct": 0.02
        },
        {
          "fold": 4,
          "rmse_naive_baseline": 0.432321,
          "rmse_lgbm_without_candidate": 0.331152,
          "rmse_lgbm_with_candidate": 0.241545,
          "improvement_vs_lgbm_baseline_pct": 27.059,
          "improvement_vs_naive_baseline_pct": 44.128,
          "candidate_importance_pct": 0.14
        },
        {
          "fold": 5,
          "rmse_naive_baseline": 0.397478,
          "rmse_lgbm_without_candidate": 0.312415,
          "rmse_lgbm_with_candidate": 0.254774,
          "improvement_vs_lgbm_baseline_pct": 18.45,
          "improvement_vs_naive_baseline_pct": 35.902,
          "candidate_importance_pct": 0.15
        }
      ],
      "overall_improvement_vs_lgbm_pct": 15.165,
      "overall_improvement_vs_naive_pct": 29.984,
      "mean_candidate_importance_pct": 0.1,
      "importance_drift_pct": 125.0,
      "monotonic_decay": false,
      "n_folds_completed": 5
    }
  },
  "aggregate": {
    "mean_improvement_vs_lgbm_pct": 17.386,
    "mean_improvement_vs_naive_pct": 32.6,
    "mean_importance_drift_pct": 105.5,
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
The feature shows strong aggregate out-of-sample improvement over both baselines and no monotonic decay, which are genuinely encouraging signs. However, importance drift is catastrophically high in both pairs tested — 86.1% for EURGBP and 125.0% for GBPUSD — far exceeding the 40% rejection threshold, which means the model is not consistently relying on this feature across walk-forward windows and the aggregate improvement numbers are unreliable. This level of drift strongly suggests the feature is fitting noise in specific sub-periods rather than capturing a persistent structural signal. Additionally, the mean candidate importance is extremely low (0.24% and 0.10%), meaning the improvement claims may be driven almost entirely by other features in the model, with this feature contributing negligibly and erratically. The COVID regime sensitivity and look-ahead checks are not explicitly reported in the results, leaving two rejection criteria unverifiable, which alone warrants caution.

## Next Action
Investigate why importance drift is so extreme by decomposing which specific walk-forward windows show high versus low importance, then test whether simplifying the construction — removing the z-score normalisation layer or the momentum step — stabilises importance before re-evaluating.
