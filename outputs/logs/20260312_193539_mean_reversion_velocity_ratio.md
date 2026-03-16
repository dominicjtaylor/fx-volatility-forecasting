# mean_reversion_velocity_ratio

**Timestamp:** 2026-03-12 19:35
**Verdict:** REJECTED

## Hypothesis
When price deviates from its rolling mean and then snaps back quickly, the velocity (speed) of that reversion relative to the initial deviation magnitude signals how strongly mean-reverting the current regime is. High reversion velocity implies a liquid, efficient market actively correcting mispricings — a structural feature of low-vol, range-bound FX conditions. This velocity ratio should predict near-term realised volatility: fast reversion → suppressed vol (mean-reverting regime), slow or absent reversion → elevated vol (trending or crisis regime).

## Construction
1. Compute rolling mean of close over a medium window (e.g. 120 bars = 20 minutes). 2. For each bar, measure the signed deviation of close from the rolling mean. 3. Compute the change in deviation over a short lag (e.g. 6 bars = 1 minute) — this is the reversion delta: if price deviated +X one minute ago and is now +Y, the delta is Y-X. Negative delta when deviation was positive (or positive delta when deviation was negative) means reversion is occurring. 4. Reversion velocity = -(deviation_lag * sign_adjustment) / max(abs(deviation_lag), 1e-8), bounded to [-1, 1]. More precisely: velocity = -delta / abs(deviation_lag).clip(1e-8), where delta = deviation_now - deviation_lag6. 5. Smooth over a short window (12 bars) to reduce noise. 6. The output is this smoothed ratio — positive values indicate active mean reversion, negative values indicate trend continuation or momentum.

## Rejection Criteria
Reject if: (1) Out-of-sample RMSE vs rolling vol baseline does not improve (P04). (2) Feature importance degrades more than 30% in the March 2020 COVID crisis regime or the 2022 inflation regime versus the baseline calm period (P01). (3) Importance drift across rolling walk-forward windows exceeds 40% (P03). (4) Monotonic degradation in predictive power across chronological folds — e.g. strong signal pre-2021, progressively weaker through 2022-2023 — suggesting the mean-reversion structure has permanently broken down (P06). (5) Feature produces NaN in more than 10% of rows after warmup.

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
          "rmse_lgbm_with_candidate": 0.362923,
          "improvement_vs_lgbm_baseline_pct": 19.32,
          "improvement_vs_naive_baseline_pct": 35.148,
          "candidate_importance_pct": 0.12
        },
        {
          "fold": 2,
          "rmse_naive_baseline": 0.497649,
          "rmse_lgbm_without_candidate": 0.433257,
          "rmse_lgbm_with_candidate": 0.394327,
          "improvement_vs_lgbm_baseline_pct": 8.985,
          "improvement_vs_naive_baseline_pct": 20.762,
          "candidate_importance_pct": 0.04
        },
        {
          "fold": 3,
          "rmse_naive_baseline": 0.491519,
          "rmse_lgbm_without_candidate": 0.379528,
          "rmse_lgbm_with_candidate": 0.308597,
          "improvement_vs_lgbm_baseline_pct": 18.689,
          "improvement_vs_naive_baseline_pct": 37.216,
          "candidate_importance_pct": 0.32
        },
        {
          "fold": 4,
          "rmse_naive_baseline": 0.445499,
          "rmse_lgbm_without_candidate": 0.343847,
          "rmse_lgbm_with_candidate": 0.234411,
          "improvement_vs_lgbm_baseline_pct": 31.827,
          "improvement_vs_naive_baseline_pct": 47.382,
          "candidate_importance_pct": 0.41
        },
        {
          "fold": 5,
          "rmse_naive_baseline": 0.393702,
          "rmse_lgbm_without_candidate": 0.313895,
          "rmse_lgbm_with_candidate": 0.251193,
          "improvement_vs_lgbm_baseline_pct": 19.976,
          "improvement_vs_naive_baseline_pct": 36.197,
          "candidate_importance_pct": 0.34
        }
      ],
      "overall_improvement_vs_lgbm_pct": 19.759,
      "overall_improvement_vs_naive_pct": 35.341,
      "mean_candidate_importance_pct": 0.25,
      "importance_drift_pct": 150.4,
      "monotonic_decay": false,
      "n_folds_completed": 5
    },
    "GBPUSD": {
      "folds": [
        {
          "fold": 1,
          "rmse_naive_baseline": 0.377095,
          "rmse_lgbm_without_candidate": 0.337691,
          "rmse_lgbm_with_candidate": 0.315392,
          "improvement_vs_lgbm_baseline_pct": 6.603,
          "improvement_vs_naive_baseline_pct": 16.363,
          "candidate_importance_pct": 0.31
        },
        {
          "fold": 2,
          "rmse_naive_baseline": 0.350893,
          "rmse_lgbm_without_candidate": 0.304227,
          "rmse_lgbm_with_candidate": 0.27802,
          "improvement_vs_lgbm_baseline_pct": 8.614,
          "improvement_vs_naive_baseline_pct": 20.768,
          "candidate_importance_pct": 0.23
        },
        {
          "fold": 3,
          "rmse_naive_baseline": 0.419607,
          "rmse_lgbm_without_candidate": 0.332167,
          "rmse_lgbm_with_candidate": 0.279567,
          "improvement_vs_lgbm_baseline_pct": 15.835,
          "improvement_vs_naive_baseline_pct": 33.374,
          "candidate_importance_pct": 0.15
        },
        {
          "fold": 4,
          "rmse_naive_baseline": 0.432321,
          "rmse_lgbm_without_candidate": 0.331152,
          "rmse_lgbm_with_candidate": 0.241052,
          "improvement_vs_lgbm_baseline_pct": 27.208,
          "improvement_vs_naive_baseline_pct": 44.242,
          "candidate_importance_pct": 0.48
        },
        {
          "fold": 5,
          "rmse_naive_baseline": 0.397478,
          "rmse_lgbm_without_candidate": 0.312415,
          "rmse_lgbm_with_candidate": 0.254666,
          "improvement_vs_lgbm_baseline_pct": 18.485,
          "improvement_vs_naive_baseline_pct": 35.93,
          "candidate_importance_pct": 0.69
        }
      ],
      "overall_improvement_vs_lgbm_pct": 15.349,
      "overall_improvement_vs_naive_pct": 30.135,
      "mean_candidate_importance_pct": 0.37,
      "importance_drift_pct": 145.2,
      "monotonic_decay": false,
      "n_folds_completed": 5
    }
  },
  "aggregate": {
    "mean_improvement_vs_lgbm_pct": 17.554,
    "mean_improvement_vs_naive_pct": 32.738,
    "mean_candidate_importance_pct": 0.31,
    "mean_importance_drift_pct": 147.8,
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
The feature shows genuinely strong out-of-sample predictive improvement — 17-20% over the LightGBM baseline and 30-35% over the naive rolling vol baseline — and there is no monotonic decay across folds, which is encouraging. However, the importance drift across rolling walk-forward windows is catastrophically high at 147-150% against a 40% rejection threshold, meaning the model is leaning on this feature inconsistently and unpredictably across time. This level of drift indicates the signal is not being used as a stable structural input but is instead being picked up opportunistically in certain windows and ignored in others, which is a hallmark of noise fitting rather than genuine regime information. The low mean importance (0.25-0.37%) combined with extreme drift suggests the aggregate improvement numbers may be driven by a small number of high-contribution folds masking near-zero contribution elsewhere. On the basis of P03 alone, this feature cannot be promoted in its current form.

## Next Action
Investigate whether the importance drift is driven by the normalization in step 4 becoming unstable during low-deviation regimes — replace the raw velocity ratio with a rank-transformed or percentile-normalized version over a rolling window to stabilize the feature's distribution across time, then re-run the walk-forward test to check if drift falls below the 40% threshold.
