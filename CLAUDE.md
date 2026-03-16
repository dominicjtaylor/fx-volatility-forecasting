# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (exposes `volare` CLI)
pip install -r requirements.txt && pip install -e .

# CLI — configure data paths and API key (writes .env and volare_config.yaml)
volare init

# CLI — run N agent cycles
volare run --cycles 5 --hint "explore session overlap signals"

# CLI — show active features and registry state
volare status

# CLI — inspect a specific feature (partial name match or exact ID)
volare inspect <name>
volare inspect <name> --code   # also prints the feature code

# CLI — research chat (interactive or single-shot)
volare chat
volare chat "why are session overlap features being rejected?"

# Run one agent cycle (legacy script)
python scripts/run_cycle.py

# Agent research dashboard
streamlit run streamlit/agent_app.py

# Apply a trained model to new data
streamlit run streamlit/apply_model_app.py
```

No test suite exists. Verify changes by running `scripts/run_cycle.py` end-to-end.

## Configuration

After `volare init`:
- API key is written to `.env` (gitignored), loaded automatically at startup
- Data paths are written to `volare_config.yaml` (gitignored)
- Data files must follow the naming convention: `questdb-<pair>.csv` (e.g. `questdb-eurgbp.csv`) unless overridden per-pair in `volare_config.yaml`

## Architecture

The repo has two independent layers:

**`src/volare/`** — The existing forecasting pipeline. Treat as stable/untouched unless explicitly asked. Contains: data loading (`data.py`), feature engineering (`features.py`), LightGBM training & walk-forward evaluation (`model.py`), and forecasting utilities (`predict.py`).

**`src/agent/`** — The autonomous research agent built on top. One cycle: `loop.py` orchestrates → `propose.py` calls Claude to generate a candidate feature (returns executable `compute_feature(df) -> pd.Series`) → `test.py` executes it in a sandboxed namespace, trains two LightGBM models (baseline vs baseline+candidate) using 5-fold walk-forward validation, returns per-fold RMSE delta and importance stats → `reason.py` calls Claude to issue a verdict (promoted / rejected / modified) against the principles register → `registry.py` logs the entry to `outputs/registry.json` → `active_features.py` adds promoted features to `outputs/active_features.json` for inclusion in all future cycles.

- `context.py` — central config loader; all agent modules read from here. Loads `config/context.yaml` (data context, pairs, regimes) and `config/principles.yaml` (rejection criteria P01–P06).
- `propose.py` and `reason.py` both call `claude-sonnet-4-6` via the Anthropic SDK using `ANTHROPIC_API_KEY`.

**`config/`** — Two YAML files drive the agent. `context.yaml`: pairs (EURGBP, GBPUSD), 10s candles, 3600s forecast horizon, known regimes and structural breaks. `principles.yaml`: six rejection criteria (P01 regime robustness, P02 no look-ahead, P03 importance stability, P04 OOS improvement, P05 economic motivation, P06 signal decay).

**`outputs/`** — All persistent state lives here. `registry.json` is the full audit trail. `active_features.json` is the promoted feature set that compounds across cycles. `logs/` holds per-cycle markdown reasoning.

**`streamlit/agent_app.py`** — Three-tab UI: Explore (launch cycles), Research Chat (Claude grounded in registry), Registry (browsable audit trail with verdict filtering).

**`context/`** — Untracked markdown documents (`volare_architecture.md`, `volare_research_proposal.md`) providing background on the project. Useful reading for understanding research intent.

## Key Constraints

- **Python compatibility**: Use Python 3.9 syntax throughout `src/agent/`. Do NOT use `X | Y` union type syntax — use `Optional[X]` from `typing`.
- **Sandboxed code execution**: LLM-generated feature code runs via `execute_feature_code()` in `test.py` — a restricted namespace with only numpy and pandas. `validate_code()` checks for forbidden patterns before execution.
- **Chronological integrity**: All train/test splits are chronological. Never use random CV.
- **Baseline comparison**: Feature improvement is always measured as RMSE delta of the full model vs the LightGBM baseline (not just vs rolling vol), evaluated out-of-sample.
- **Active feature accumulation**: Promoted features are re-included in every subsequent test cycle. Changes to `active_features.py` or `outputs/active_features.json` affect all future cycles.
- **Data files** in `data/` are gitignored. Columns: `timestamp, symbol, open, high, low, close`. No volume column exists — do not reference it in feature code.
- **Feature code constraints**: rolling windows capped at 360 bars max; all rolling ops must use `min_periods`; zero denominators must be clipped before division; NaN must not exceed 10% of rows.
