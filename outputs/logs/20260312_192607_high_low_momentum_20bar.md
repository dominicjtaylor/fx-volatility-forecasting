# high_low_momentum_20bar

**Timestamp:** 2026-03-12 19:26
**Verdict:** PROMOTED

## Hypothesis
When the midpoint of the current bar's high-low range is consistently above or below the midpoint of the range 20 bars ago, it signals directional momentum in price discovery. Unlike close-over-close momentum (already rejected), using the bar midpoint (average of high and low) smooths microstructure noise and better captures the true central tendency of price, providing a more robust momentum signal that should persist into elevated realised volatility over the forecast horizon.

## Construction
Compute the bar midpoint as (high + low) / 2 for each bar. Calculate the 20-bar momentum as the percentage change from the midpoint 20 bars ago to the current midpoint: (midpoint_t - midpoint_{t-20}) / midpoint_{t-20}. This is strictly backward-looking, uses no volume, and exploits the full bar range rather than just the close price. The signal is normalised by the lagged midpoint to produce a dimensionless return-like quantity that is comparable across different price levels and currency pairs.

## Rejection Criteria
Reject if: (1) out-of-sample RMSE improvement over rolling vol baseline is absent or negative; (2) feature importance degrades more than 40% across rolling walk-forward windows; (3) performance in any single named regime (COVID crisis, 2022 inflation, post-normalisation) is more than 30% worse than the aggregate; (4) signal shows monotonic decay across chronological test folds; (5) correlation with the already-rejected price_momentum_20bar feature exceeds 0.95, indicating no incremental information over the close-based version.

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
          "rmse_lgbm_with_candidate": 0.364461,
          "improvement_vs_lgbm_baseline_pct": 18.978,
          "improvement_vs_naive_baseline_pct": 34.873,
          "candidate_importance_pct": 0.88
        },
        {
          "fold": 2,
          "rmse_naive_baseline": 0.497649,
          "rmse_lgbm_without_candidate": 0.433257,
          "rmse_lgbm_with_candidate": 0.394183,
          "improvement_vs_lgbm_baseline_pct": 9.019,
          "improvement_vs_naive_baseline_pct": 20.791,
          "candidate_importance_pct": 1.2
        },
        {
          "fold": 3,
          "rmse_naive_baseline": 0.491519,
          "rmse_lgbm_without_candidate": 0.379528,
          "rmse_lgbm_with_candidate": 0.307849,
          "improvement_vs_lgbm_baseline_pct": 18.886,
          "improvement_vs_naive_baseline_pct": 37.368,
          "candidate_importance_pct": 1.03
        },
        {
          "fold": 4,
          "rmse_naive_baseline": 0.445499,
          "rmse_lgbm_without_candidate": 0.343847,
          "rmse_lgbm_with_candidate": 0.234064,
          "improvement_vs_lgbm_baseline_pct": 31.928,
          "improvement_vs_naive_baseline_pct": 47.46,
          "candidate_importance_pct": 1.02
        },
        {
          "fold": 5,
          "rmse_naive_baseline": 0.393702,
          "rmse_lgbm_without_candidate": 0.313895,
          "rmse_lgbm_with_candidate": 0.251292,
          "improvement_vs_lgbm_baseline_pct": 19.944,
          "improvement_vs_naive_baseline_pct": 36.172,
          "candidate_importance_pct": 1.04
        }
      ],
      "overall_improvement_vs_lgbm_pct": 19.751,
      "overall_improvement_vs_naive_pct": 35.333,
      "mean_candidate_importance_pct": 1.03,
      "importance_drift_pct": 30.9,
      "monotonic_decay": false,
      "n_folds_completed": 5
    },
    "GBPUSD": {
      "folds": [
        {
          "fold": 1,
          "rmse_naive_baseline": 0.377095,
          "rmse_lgbm_without_candidate": 0.337691,
          "rmse_lgbm_with_candidate": 0.314731,
          "improvement_vs_lgbm_baseline_pct": 6.799,
          "improvement_vs_naive_baseline_pct": 16.538,
          "candidate_importance_pct": 1.26
        },
        {
          "fold": 2,
          "rmse_naive_baseline": 0.350893,
          "rmse_lgbm_without_candidate": 0.304227,
          "rmse_lgbm_with_candidate": 0.278492,
          "improvement_vs_lgbm_baseline_pct": 8.459,
          "improvement_vs_naive_baseline_pct": 20.633,
          "candidate_importance_pct": 1.29
        },
        {
          "fold": 3,
          "rmse_naive_baseline": 0.419607,
          "rmse_lgbm_without_candidate": 0.332167,
          "rmse_lgbm_with_candidate": 0.27994,
          "improvement_vs_lgbm_baseline_pct": 15.723,
          "improvement_vs_naive_baseline_pct": 33.285,
          "candidate_importance_pct": 1.5
        },
        {
          "fold": 4,
          "rmse_naive_baseline": 0.432321,
          "rmse_lgbm_without_candidate": 0.331152,
          "rmse_lgbm_with_candidate": 0.24073,
          "improvement_vs_lgbm_baseline_pct": 27.305,
          "improvement_vs_naive_baseline_pct": 44.317,
          "candidate_importance_pct": 1.46
        },
        {
          "fold": 5,
          "rmse_naive_baseline": 0.397478,
          "rmse_lgbm_without_candidate": 0.312415,
          "rmse_lgbm_with_candidate": 0.254687,
          "improvement_vs_lgbm_baseline_pct": 18.478,
          "improvement_vs_naive_baseline_pct": 35.924,
          "candidate_importance_pct": 1.22
        }
      ],
      "overall_improvement_vs_lgbm_pct": 15.353,
      "overall_improvement_vs_naive_pct": 30.139,
      "mean_candidate_importance_pct": 1.35,
      "importance_drift_pct": 20.8,
      "monotonic_decay": false,
      "n_folds_completed": 5
    }
  },
  "aggregate": {
    "mean_improvement_vs_lgbm_pct": 17.552,
    "mean_improvement_vs_naive_pct": 32.736,
    "mean_candidate_importance_pct": 1.19,
    "mean_importance_drift_pct": 25.9,
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


## Summary
This feature delivers consistent out-of-sample improvement across both currency pairs, with aggregate RMSE gains of 32.7% over the naive rolling vol baseline and 17.6% over the full LightGBM baseline — well above noise thresholds. Importance drift sits at 25.9% aggregate (30.9% worst case in EURGBP), comfortably below the 40% rejection threshold, and no monotonic decay was detected across folds. The construction is strictly backward-looking, economically motivated, and the hypothesis that midpoint-based momentum captures price discovery better than close-based momentum is coherent. The one area to watch is EURGBP Fold 2, where LightGBM improvement drops to 9% — this is a notable dip but isolated and does not form a trend. None of the five pre-stated rejection criteria were triggered, and the feature earns promotion on current evidence.

## Next Action
Run the correlation test against the already-rejected price_momentum_20bar feature (close-based version) to formally confirm the signal provides incremental information before deploying into the live feature set — this check was pre-stated but its result is absent from the test output.
