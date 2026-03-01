# range_expansion_persistence

**Timestamp:** 2026-03-01 20:39
**Verdict:** REJECTED

## Hypothesis
Regime transitions are often preceded by a sustained expansion in intrabar price ranges that persists across multiple consecutive bars — a signature distinct from isolated volatility spikes. When the ratio of recent short-window true range to a longer baseline true range begins to expand AND this expansion is persistent (i.e., the short/long ratio has been above 1.0 for several consecutive bars), this indicates the market is entering a new volatility regime rather than experiencing a transient spike. This persistence-weighted expansion should forecast elevated realised volatility over the next 3600 seconds because structural regime shifts involve sustained demand/supply imbalances rather than instantaneous price adjustments.

## Construction
1. Compute True Range (TR) for each 10s bar: max(high-low, abs(high-prev_close), abs(low-prev_close)). 2. Compute a short-window smoothed TR using a 30-bar (5-minute) rolling mean. 3. Compute a long-window smoothed TR using a 360-bar (1-hour) rolling mean. 4. Compute the TR ratio: short_tr / long_tr. 5. Compute a persistence score: rolling count of consecutive bars where TR ratio > 1.0, capped at the long window length (360 bars). 6. The feature is: TR_ratio * log1p(persistence_count), which amplifies the ratio signal only when it has been sustained. All windows are strictly backward-looking.

## Rejection Criteria
Reject if: (1) Out-of-sample RMSE improvement over rolling vol baseline is not statistically meaningful (per P04). (2) Feature importance degrades by more than 30% in any single named volatility regime — particularly if it only activates during COVID crisis and fails in 2022 elevated-vol or post-normalisation periods (per P01). (3) Importance drift across rolling walk-forward windows exceeds 40% (per P03). (4) The feature shows monotonically declining predictive power across chronological test folds (per P06). (5) Any evidence of look-ahead contamination found in implementation review (per P02).

## Test Results
```json
{
  "per_pair": {
    "EURGBP": {
      "folds": [
        {
          "fold": 1,
          "rmse_model": 0.048889,
          "rmse_baseline": 0.060797,
          "rmse_improvement_pct": 19.588,
          "feature_target_correlation": 0.1073
        },
        {
          "fold": 2,
          "rmse_model": 0.079036,
          "rmse_baseline": 0.090833,
          "rmse_improvement_pct": 12.988,
          "feature_target_correlation": 0.0613
        },
        {
          "fold": 3,
          "rmse_model": 0.04213,
          "rmse_baseline": 0.055868,
          "rmse_improvement_pct": 24.59,
          "feature_target_correlation": 0.1005
        },
        {
          "fold": 4,
          "rmse_model": 0.02402,
          "rmse_baseline": 0.027012,
          "rmse_improvement_pct": 11.076,
          "feature_target_correlation": 0.2058
        },
        {
          "fold": 5,
          "rmse_model": 0.022005,
          "rmse_baseline": 0.023379,
          "rmse_improvement_pct": 5.879,
          "feature_target_correlation": 0.1665
        }
      ],
      "overall_rmse_improvement_pct": 14.824,
      "importance_drift_pct": 112.6,
      "monotonic_decay": false,
      "n_folds_completed": 5
    },
    "GBPUSD": {
      "folds": [
        {
          "fold": 1,
          "rmse_model": 0.052697,
          "rmse_baseline": 0.047596,
          "rmse_improvement_pct": -10.718,
          "feature_target_correlation": 0.147
        },
        {
          "fold": 2,
          "rmse_model": 0.053567,
          "rmse_baseline": 0.045772,
          "rmse_improvement_pct": -17.031,
          "feature_target_correlation": 0.1308
        },
        {
          "fold": 3,
          "rmse_model": 0.03583,
          "rmse_baseline": 0.041923,
          "rmse_improvement_pct": 14.532,
          "feature_target_correlation": 0.148
        },
        {
          "fold": 4,
          "rmse_model": 0.034721,
          "rmse_baseline": 0.037264,
          "rmse_improvement_pct": 6.823,
          "feature_target_correlation": 0.1823
        },
        {
          "fold": 5,
          "rmse_model": 0.03068,
          "rmse_baseline": 0.031943,
          "rmse_improvement_pct": 3.952,
          "feature_target_correlation": 0.1578
        }
      ],
      "overall_rmse_improvement_pct": -0.488,
      "importance_drift_pct": 33.6,
      "monotonic_decay": false,
      "n_folds_completed": 5
    }
  },
  "aggregate": {
    "mean_rmse_improvement_pct": 7.168,
    "mean_importance_drift_pct": 73.1,
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
P01, P03, P04

## Summary
This feature fails on multiple hard rejection criteria and cannot be promoted in its current form. The mean importance drift of 73.1% across all folds far exceeds the 40% threshold, with EURGBP alone showing 112.6% drift — a clear sign the model is fitting noise rather than a stable signal. More critically, GBPUSD shows an aggregate RMSE deterioration of -0.488%, with the first two folds losing over 10% and 17% respectively, meaning the feature actively harms predictions on one of the two tested pairs. While EURGBP shows encouraging RMSE improvement of 14.8%, a feature that works on one pair but hurts another is not robust — it is pair-specific, which is a form of regime dependence that violates P01 and P03. The economic hypothesis is coherent and worth preserving, but the implementation produces an unstable, pair-inconsistent signal that fails the stability and generalization bar required for production use.

## Next Action
Investigate why GBPUSD diverges so sharply from EURGBP — specifically whether the persistence count mechanism is being dominated by microstructure noise on higher-liquidity pairs, and consider replacing the raw consecutive-bar count with a decay-weighted persistence score or testing alternative short/long window ratios to find a more stable parametrization before re-running walk-forward validation across a broader set of pairs.
