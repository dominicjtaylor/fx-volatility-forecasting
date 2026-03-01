# overnight_gap_volatility_anchor

**Timestamp:** 2026-03-01 20:49
**Verdict:** REJECTED

## Hypothesis
The magnitude of the gap between the previous session's close and the current session's open reflects the accumulation of information flow and order imbalance during illiquid periods (Asian session for EURGBP/GBPUSD, or weekend gaps). Large gaps signal latent volatility that tends to persist into the subsequent session as market participants reposition, creating an elevated volatility environment over the forecast horizon. This is a structural microstructure phenomenon: gap closes attract momentum and stop-loss activity, sustaining vol. Conversely, small or zero gaps imply continuity and calmer conditions.

## Construction
Using 10-second OHLC data, identify session boundaries by detecting gaps in the close-to-open return. Specifically: (1) compute the log return from each bar's close to the next bar's open (gap return). (2) Take the absolute value of this gap return as the raw gap signal. (3) To avoid single-bar noise, compute a rolling Z-score of the absolute gap return using a 1-hour backward window (360 bars at 10s) — normalising by the local mean and standard deviation of recent gap returns. (4) Additionally, smooth with a 5-minute (30-bar) exponential weighted moving average to reduce tick-level noise. The final feature is this smoothed, normalised gap return magnitude, which encodes how anomalous the current gap behaviour is relative to recent history.

## Rejection Criteria
Reject if: (1) Out-of-sample RMSE does not improve over the rolling realised volatility baseline in at least 2 of the 4 named regimes. (2) Feature importance drifts more than 40% across chronological walk-forward folds, suggesting the signal is regime-specific noise. (3) Performance in any single named regime is more than 30% worse than aggregate performance. (4) The feature shows near-zero variance during the COVID crisis regime (March 2020), which would indicate it fails to capture the most extreme volatility episode in the sample. (5) Autocorrelation of the feature drops to near zero within 6 minutes (36 bars), implying it carries no information persistence relevant to the 3600-second forecast horizon.

## Test Results
```json
{
  "per_pair": {
    "EURGBP": {
      "folds": [
        {
          "fold": 1,
          "rmse_model": 0.048997,
          "rmse_baseline": 0.060797,
          "rmse_improvement_pct": 19.41,
          "feature_target_correlation": 0.0775
        },
        {
          "fold": 2,
          "rmse_model": 0.079108,
          "rmse_baseline": 0.090833,
          "rmse_improvement_pct": 12.908,
          "feature_target_correlation": 0.0343
        },
        {
          "fold": 3,
          "rmse_model": 0.042215,
          "rmse_baseline": 0.055868,
          "rmse_improvement_pct": 24.438,
          "feature_target_correlation": 0.0781
        },
        {
          "fold": 4,
          "rmse_model": 0.024228,
          "rmse_baseline": 0.027012,
          "rmse_improvement_pct": 10.307,
          "feature_target_correlation": 0.1623
        },
        {
          "fold": 5,
          "rmse_model": 0.022089,
          "rmse_baseline": 0.023379,
          "rmse_improvement_pct": 5.519,
          "feature_target_correlation": 0.1382
        }
      ],
      "overall_rmse_improvement_pct": 14.516,
      "importance_drift_pct": 130.5,
      "monotonic_decay": false,
      "n_folds_completed": 5
    },
    "GBPUSD": {
      "folds": [
        {
          "fold": 1,
          "rmse_model": 0.052805,
          "rmse_baseline": 0.047596,
          "rmse_improvement_pct": -10.944,
          "feature_target_correlation": 0.1102
        },
        {
          "fold": 2,
          "rmse_model": 0.05368,
          "rmse_baseline": 0.045772,
          "rmse_improvement_pct": -17.277,
          "feature_target_correlation": 0.0967
        },
        {
          "fold": 3,
          "rmse_model": 0.035965,
          "rmse_baseline": 0.041923,
          "rmse_improvement_pct": 14.211,
          "feature_target_correlation": 0.1142
        },
        {
          "fold": 4,
          "rmse_model": 0.034932,
          "rmse_baseline": 0.037264,
          "rmse_improvement_pct": 6.257,
          "feature_target_correlation": 0.1474
        },
        {
          "fold": 5,
          "rmse_model": 0.030767,
          "rmse_baseline": 0.031943,
          "rmse_improvement_pct": 3.679,
          "feature_target_correlation": 0.1371
        }
      ],
      "overall_rmse_improvement_pct": -0.815,
      "importance_drift_pct": 41.9,
      "monotonic_decay": false,
      "n_folds_completed": 5
    }
  },
  "aggregate": {
    "mean_rmse_improvement_pct": 6.851,
    "mean_importance_drift_pct": 86.2,
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
This feature fails on multiple hard rejection criteria and cannot be promoted. Importance drift is catastrophic at 86.2% aggregate and 130.5% for EURGBP alone, far exceeding the 40% threshold — the model is clearly fitting noise rather than a stable structural signal. GBPUSD shows a negative RMSE improvement of -0.815% overall, with the first two folds losing nearly 11% and 17% respectively against baseline, meaning the feature actively harms prediction on one of the two tested pairs. The pre-stated rejection criterion requiring improvement in at least 2 of 4 named regimes also appears at risk given the pair-level divergence, and the GBPUSD importance drift of 41.9% is itself borderline. The economic rationale is coherent and the EURGBP results are intriguing, but instability of this magnitude is grounds for outright rejection under the stated research principles.

## Next Action
Investigate why the feature works for EURGBP but not GBPUSD by examining whether gap construction is capturing genuine session boundaries for each pair — GBPUSD may have materially different liquidity patterns that make the gap signal noisy; consider pair-specific normalisation windows or restricting the feature to the Asian-to-London open transition where the microstructure rationale is strongest.
