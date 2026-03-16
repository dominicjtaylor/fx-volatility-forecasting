# intrabar_momentum_reversal_asymmetry

**Timestamp:** 2026-03-12 19:50
**Verdict:** REJECTED

## Hypothesis
When the open-to-close direction (net momentum) consistently opposes the high-low midpoint direction (intrabar range momentum), it signals that price is being rejected at intrabar extremes — a structure associated with imminent volatility expansion. Conversely, when open-to-close direction aligns with high-low midpoint direction, price is accepting new levels, suggesting lower forthcoming realised volatility. The asymmetry between these two directional signals captures latent tension in the microstructure that precedes volatility regime shifts.

## Construction
For each bar compute: (1) OC_dir = sign(close - open), capturing net directional momentum within the bar. (2) HL_mid_dir = sign((high + low)/2 - rolling_mean_of_HL_mid over past N bars), capturing whether the intrabar range midpoint is above or below its recent average. (3) Asymmetry_raw = OC_dir * HL_mid_dir — this is +1 when both signals agree (trending acceptance) and -1 when they conflict (rejection/reversal tension). (4) Smooth this over a short rolling window (30 bars, ~5 minutes) to reduce noise: the rolling mean of Asymmetry_raw. (5) Normalise by subtracting a longer-term rolling mean (180 bars) and dividing by a rolling std (180 bars) to produce a z-score that is regime-adaptive and mean-stationary. Window sizes: HL_mid baseline = 60 bars, smoothing = 30 bars, normalisation = 180 bars.

## Rejection Criteria
Reject if: (1) Out-of-sample RMSE improvement over rolling volatility baseline is absent or negative across either EURGBP or GBPUSD. (2) Feature importance degrades by more than 30% in any single named regime (especially COVID March 2020 or UK mini-budget Sep 2022). (3) Importance drift exceeds 40% across rolling walk-forward windows. (4) The signal produces NaN for more than 10% of rows. (5) The sign relationship inverts across regimes — i.e. negative z-score predicts vol expansion in one regime but vol contraction in another without a structural explanation, indicating the feature is regime-dependent noise.

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
          "rmse_lgbm_with_candidate": 0.363142,
          "improvement_vs_lgbm_baseline_pct": 19.271,
          "improvement_vs_naive_baseline_pct": 35.108,
          "candidate_importance_pct": 0.31
        },
        {
          "fold": 2,
          "rmse_naive_baseline": 0.497649,
          "rmse_lgbm_without_candidate": 0.433257,
          "rmse_lgbm_with_candidate": 0.394168,
          "improvement_vs_lgbm_baseline_pct": 9.022,
          "improvement_vs_naive_baseline_pct": 20.794,
          "candidate_importance_pct": 0.23
        },
        {
          "fold": 3,
          "rmse_naive_baseline": 0.491519,
          "rmse_lgbm_without_candidate": 0.379528,
          "rmse_lgbm_with_candidate": 0.308827,
          "improvement_vs_lgbm_baseline_pct": 18.629,
          "improvement_vs_naive_baseline_pct": 37.169,
          "candidate_importance_pct": 0.49
        },
        {
          "fold": 4,
          "rmse_naive_baseline": 0.445499,
          "rmse_lgbm_without_candidate": 0.343847,
          "rmse_lgbm_with_candidate": 0.234592,
          "improvement_vs_lgbm_baseline_pct": 31.774,
          "improvement_vs_naive_baseline_pct": 47.342,
          "candidate_importance_pct": 0.44
        },
        {
          "fold": 5,
          "rmse_naive_baseline": 0.393702,
          "rmse_lgbm_without_candidate": 0.313895,
          "rmse_lgbm_with_candidate": 0.251595,
          "improvement_vs_lgbm_baseline_pct": 19.847,
          "improvement_vs_naive_baseline_pct": 36.095,
          "candidate_importance_pct": 0.25
        }
      ],
      "overall_improvement_vs_lgbm_pct": 19.709,
      "overall_improvement_vs_naive_pct": 35.302,
      "mean_candidate_importance_pct": 0.34,
      "importance_drift_pct": 75.6,
      "monotonic_decay": false,
      "n_folds_completed": 5
    },
    "GBPUSD": {
      "folds": [
        {
          "fold": 1,
          "rmse_naive_baseline": 0.377095,
          "rmse_lgbm_without_candidate": 0.337691,
          "rmse_lgbm_with_candidate": 0.315387,
          "improvement_vs_lgbm_baseline_pct": 6.605,
          "improvement_vs_naive_baseline_pct": 16.364,
          "candidate_importance_pct": 0.07
        },
        {
          "fold": 2,
          "rmse_naive_baseline": 0.350893,
          "rmse_lgbm_without_candidate": 0.304227,
          "rmse_lgbm_with_candidate": 0.278568,
          "improvement_vs_lgbm_baseline_pct": 8.434,
          "improvement_vs_naive_baseline_pct": 20.612,
          "candidate_importance_pct": 0.17
        },
        {
          "fold": 3,
          "rmse_naive_baseline": 0.419607,
          "rmse_lgbm_without_candidate": 0.332167,
          "rmse_lgbm_with_candidate": 0.279875,
          "improvement_vs_lgbm_baseline_pct": 15.743,
          "improvement_vs_naive_baseline_pct": 33.301,
          "candidate_importance_pct": 0.05
        },
        {
          "fold": 4,
          "rmse_naive_baseline": 0.432321,
          "rmse_lgbm_without_candidate": 0.331152,
          "rmse_lgbm_with_candidate": 0.24086,
          "improvement_vs_lgbm_baseline_pct": 27.266,
          "improvement_vs_naive_baseline_pct": 44.287,
          "candidate_importance_pct": 0.16
        },
        {
          "fold": 5,
          "rmse_naive_baseline": 0.397478,
          "rmse_lgbm_without_candidate": 0.312415,
          "rmse_lgbm_with_candidate": 0.255029,
          "improvement_vs_lgbm_baseline_pct": 18.369,
          "improvement_vs_naive_baseline_pct": 35.838,
          "candidate_importance_pct": 0.27
        }
      ],
      "overall_improvement_vs_lgbm_pct": 15.283,
      "overall_improvement_vs_naive_pct": 30.08,
      "mean_candidate_importance_pct": 0.14,
      "importance_drift_pct": 152.8,
      "monotonic_decay": false,
      "n_folds_completed": 5
    }
  },
  "aggregate": {
    "mean_improvement_vs_lgbm_pct": 17.496,
    "mean_improvement_vs_naive_pct": 32.691,
    "mean_candidate_importance_pct": 0.24,
    "mean_importance_drift_pct": 114.2,
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
The feature shows genuine out-of-sample improvement across both currency pairs — clearing the baseline hurdle comfortably — and has a coherent microstructure rationale. However, importance drift is catastrophically high: 114.2% in aggregate, 152.8% on GBPUSD alone, both far exceeding the 40% rejection threshold. This means the model is not reliably leaning on this feature in a consistent way across time windows; what looks like signal in some folds is likely noise-fitting in others. The per-fold importance on GBPUSD swings from 0.05% to 0.27%, a fivefold range, which is structurally incompatible with the stability required for deployment. The RMSE improvements are encouraging enough to warrant further investigation, but the feature cannot be promoted in its current form.

## Next Action
Investigate whether the importance instability is driven by the normalisation window (180 bars) being too short to stabilise the z-score across different volatility regimes — experiment with longer normalisation windows (360–720 bars) and test whether collapsing the HL_mid baseline from 60 to 30 bars reduces noise-sensitivity before re-running the walk-forward evaluation.
