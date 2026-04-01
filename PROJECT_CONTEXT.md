# Project Overview
`volare` — autonomous FX volatility forecasting research agent. Combines a stable LightGBM forecasting pipeline with an AI-driven feature discovery loop that compounds improvements across cycles.

# Original Idea
Short-horizon FX spot volatility has forecastable structure beyond simple rolling-vol estimates. The agent continuously proposes, tests, and promotes new features to improve forecast quality, while explicitly identifying failure modes.

# Current State
- Core pipeline (`src/volare/`): data loading, feature engineering, LightGBM training, walk-forward eval, predict utilities — stable, untouched unless asked.
- Autonomous agent (`src/agent/`): end-to-end cycle operational — propose → test (sandboxed) → reason (principles P01–P06) → registry → active features accumulation.
- CLI (`volare`): `init`, `run`, `status`, `inspect`, `chat` all functional.
- Streamlit apps: agent research dashboard (`streamlit/agent_app.py`) and apply-model app (`streamlit/apply_model_app.py`).
- Persistent state: `outputs/registry.json` (full audit trail), `outputs/active_features.json` (promoted features re-included in all future cycles).
- Economic value simulation: `scripts/vol_simulation.py` — compares constant-weight vs vol-scaled strategies on OOS data; outputs metrics table + 4-panel plot to `outputs/vol_simulation/`.
- No test suite — verification via `scripts/run_cycle.py` end-to-end.

# Goal
- Quantify when/how/by how much ML forecasts beat rolling-vol baselines
- Continuously compound discovered features across agent cycles
- Explicitly identify and document failure modes (regime transitions, short horizons)

# Constraints
- Python 3.9 syntax only in `src/agent/` — no `X | Y` union types
- Feature code runs in sandboxed namespace (numpy + pandas only)
- Rolling windows capped at 360 bars max; all ops need `min_periods`; no zero denominators; NaN < 10%
- All train/test splits strictly chronological — no random CV
- Baseline comparison: RMSE delta (full model vs LightGBM baseline), evaluated OOS
- Data columns: `timestamp, symbol, open, high, low, close` — no volume column

# Gaps
- No automated test suite
- No regime-aware adaptive mechanism (P01 regime robustness is a common rejection trigger)
- Feature search is purely LLM-driven with no structured exploration strategy
- No stronger baselines (EWMA, GARCH) for contextualising ML gains
- `PROJECT_CONTEXT.md` was missing (now created)

# Next Steps
1. Review `outputs/registry.json` to assess recent cycle quality (promotion rate, rejection reasons)
2. Review `outputs/active_features.json` to see what has been accumulated
3. Identify dominant rejection causes and tune principles or proposal strategy accordingly
4. Consider regime-robustness improvements to reduce P01 rejections
5. Evaluate adding EWMA/GARCH to the baseline comparison framework
