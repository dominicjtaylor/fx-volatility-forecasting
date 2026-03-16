# Volare — Project Context & Architecture Notes

This document is a persistent reference for Claude Code sessions working on the Volare FX Volatility Research Agent. It covers research motivation, design decisions, agent architecture, and implementation details.

---

## Project Identity

**Name:** Volare  
**Repo:** `dominicjtaylor/fx-volatility-forecasting`  
**Type:** Local research tool + quant portfolio piece  
**Owner:** Dom Taylor — late-stage PhD researcher (Astronomy & Astrophysics, Durham University) transitioning into systematic quantitative research  

---

## Research Motivation

Volatility forecasting is central to risk management, position sizing, options pricing, and portfolio construction. Traditional approaches (rolling standard deviations, GARCH-style models) rely on parametric assumptions and linear structure.

The core research question is:

> Can machine learning models — specifically LightGBM — capture non-linear relationships in FX data to improve short-horizon volatility forecasts? And can an autonomous agent systematically discover which features drive that improvement?

The goal is **not** to claim tradable alpha. It is to:
- Build a rigorous research framework that enforces institutional-grade validation discipline
- Systematically evaluate candidate features against a principled rejection criteria
- Accumulate a structured, falsifiable theory of what predicts FX volatility

This reframes the project from "I trained an ML model" to "I built a systematic research framework that enforces robustness."

---

## Data

| Property | Value |
|---|---|
| Pairs | EURGBP, GBPUSD |
| Resolution | 10-second candles |
| Rows per pair | ~3.8M |
| File format | CSV (`questdb-eurgbp.csv`, `questdb-gbpusd.csv`) |
| Location | `data/` (gitignored) |
| Columns | `timestamp, symbol, open, high, low, close` |
| Forecast horizon | 3,600 seconds (1 hour) |
| Target variable | `rolling_log_future_vol` |
| Baseline | Medium-window rolling vol |

**Known structural breaks and regimes (defined in `config/context.yaml`):**
- Low volatility: 2018–2019 pre-COVID
- Crisis volatility: March 2020 COVID shock
- Elevated volatility: 2022 inflation/rate cycle
- Post-normalisation: 2023 onwards
- Structural breaks: COVID dislocation (2020-03-15), Russia-Ukraine EUR shock (2022-02-24), UK mini-budget GBP flash move (2022-09-23)

---

## Architecture Overview

```
fx-volatility-forecasting/
├── src/
│   ├── volare/               # Existing pipeline — untouched
│   │   ├── data.py           # load_candles, load_last_candles
│   │   ├── features.py       # Feature engineering functions
│   │   ├── model.py          # LightGBM training, evaluation
│   │   └── predict.py        # Forecasting utilities
│   └── agent/                # New agent layer
│       ├── __init__.py
│       ├── context.py        # Loads context.yaml and principles.yaml
│       ├── registry.py       # Hypothesis registry — audit trail
│       ├── propose.py        # Feature proposal via Claude API
│       ├── test.py           # LightGBM-based feature testing
│       ├── reason.py         # Reasoning and verdict via Claude API
│       ├── loop.py           # Orchestrates full propose→test→reason→log cycle
│       └── active_features.py # Manages promoted feature set
├── config/
│   ├── context.yaml          # Data context, pairs, regimes, quality flags
│   └── principles.yaml       # Research principles (P01–P06)
├── outputs/
│   ├── registry.json         # Persistent hypothesis registry
│   ├── active_features.json  # Currently promoted feature set
│   └── logs/                 # Per-run markdown reasoning logs
├── streamlit/
│   ├── agent_app.py          # Volare agent control panel
│   └── apply_model_app.py    # Existing model application GUI
├── scripts/
│   └── run_cycle.py          # CLI entry point for running a cycle
└── data/                     # Gitignored — CSV files live here
```

---

## Agent Core Loop

Each cycle executes the following sequence:

```
1. Load context + principles + registry + active features
2. propose_feature()     — Claude proposes a candidate feature with Python implementation
3. run_feature_test()    — LightGBM trains on (active features + candidate) vs baseline
4. reason_and_verdict()  — Claude interprets results against principles register
5. Log to registry.json + outputs/logs/
6. If promoted → add to active_features.json
```

### Key design decisions

**Code generation in proposal:**  
The agent proposes features by generating executable Python code, not just descriptions. `propose.py` asks Claude to return a `compute_feature(df) -> pd.Series` function as part of the JSON proposal. This makes the agent genuinely open-ended — it can test any feature expressible in numpy/pandas.

**Safe code execution:**  
`test.py` runs LLM-generated code in a restricted namespace containing only numpy and pandas. A `validate_code()` function checks for forbidden patterns (os, sys, subprocess, eval, exec etc.) before execution.

**LightGBM as the test engine:**  
Features are not tested in isolation. Each cycle trains two LightGBM models:
- **Baseline model:** trained on multi-window rolling vol features only
- **Full model:** trained on baseline features + all active promoted features + candidate

Improvement is measured as RMSE reduction of full model vs baseline model, both evaluated out-of-sample. This mirrors the actual research question: does adding this feature improve the combined model?

**Active feature set accumulation:**  
Promoted features are stored in `outputs/active_features.json` and included in every subsequent training run. The agent systematically builds the best feature combination over time, not just tests features individually.

**Principles-based reasoning:**  
Verdicts are not purely statistical. Claude reasons against a principles register that encodes research philosophy explicitly. A feature that improves RMSE but shows unstable importance or regime dependence can still be rejected.

---

## Research Principles Register

Defined in `config/principles.yaml`:

| ID | Name | Rejection Trigger |
|---|---|---|
| P01 | Regime robustness required | Performance degrades >30% in any single named regime |
| P02 | No look-ahead tolerance | Any feature referencing data after prediction timestamp |
| P03 | Importance stability required | Importance drift >40% across rolling windows |
| P04 | Out-of-sample improvement over baseline | No OOS RMSE improvement over medium-window rolling vol |
| P05 | Economically motivated construction | No coherent economic rationale can be stated |
| P06 | Signal decay awareness | Monotonic performance degradation across chronological folds |

Verdict options: `promoted`, `rejected`, `modified`

---

## Module Interfaces

### `context.py`
```python
load_context() -> dict
load_principles() -> list
save_context(context: dict)
format_context_for_prompt(context: dict) -> str
format_principles_for_prompt(principles: list) -> str
```

### `registry.py`
```python
load_registry() -> list
save_entry(entry: dict)
get_entry(feature_id: str) -> Optional[dict]
filter_by_verdict(verdict: str) -> list
make_entry(feature, test_results, reasoning) -> dict
format_registry_for_prompt(registry: list, max_entries: int) -> str
```

### `propose.py`
```python
propose_feature(
    context: dict,
    principles: list,
    registry: list,
    user_hint: str = None,
    active_features: list = None
) -> dict
# Returns: name, hypothesis, construction, code, rejection_criteria, rationale
```

### `test.py`
```python
run_feature_test(
    feature: dict,
    context: dict,
    data: Dict[str, pd.DataFrame],
    active_features: list = None
) -> dict
# Returns: per_pair results + aggregate metrics
```

### `reason.py`
```python
reason_and_verdict(
    feature: dict,
    test_results: dict,
    principles: list,
    context: dict
) -> dict
# Returns: verdict, summary, triggered_principles, next_action
```

### `loop.py`
```python
run_cycle(
    data: Dict[str, pd.DataFrame],
    user_hint: str = None
) -> dict
# Orchestrates full cycle, writes to registry, returns entry
```

### `active_features.py`
```python
load_active_features() -> list
save_active_features(features: list)
add_active_feature(feature: dict)
get_active_feature_names() -> list
format_active_features_for_prompt(active_features: list) -> str
```

---

## Streamlit App

**File:** `streamlit/agent_app.py`  
**Run:** `streamlit run streamlit/agent_app.py` from repo root  
**Streamlit binary:** `/Users/dominictaylor/Library/Python/3.9/bin/streamlit`

Three tabs:

**Explore** — Launchpad. Large text input for direction hints, "Explore" button triggers a full agent cycle, stats row (total/promoted/rejected/modified), recent activity feed.

**Research Chat** — Claude-powered chat interface grounded in registry and context. Allows Dom to guide research direction, discuss findings, and ask questions about what has been tested. Uses `claude-sonnet-4-6` with a system prompt that includes registry summary and data context.

**Registry** — Browsable audit trail. Filter by verdict. Per-entry expandable cards showing: Trading 212-style plain-language summary, performance metrics, hypothesis, construction, triggered principles, next action recommendation, per-pair fold-level results table.

---

## Technical Environment

| Property | Value |
|---|---|
| Python | 3.9 (for agent/Streamlit); 3.8.18 for legacy volare work |
| LLM | `claude-sonnet-4-6` via Anthropic API |
| ML | LightGBM with walk-forward validation |
| Key dependencies | `anthropic`, `lightgbm`, `streamlit`, `pandas`, `numpy`, `pyyaml` |
| API key | Set as `ANTHROPIC_API_KEY` environment variable in `~/.zshrc` |
| OS | Mac |
| Git | SSH-based |

**Important Python compatibility note:**  
Do not use `X | Y` union type syntax — use `Optional[X]` from `typing` instead. Applies to all agent modules for 3.8 compatibility.

---

## Running the Agent

**Single cycle via CLI:**
```bash
cd /Users/dominictaylor/fx-volatility-forecasting
python scripts/run_cycle.py
```

**With direction hint:**
```python
entry = run_cycle(data, user_hint="explore regime transition signals")
```

**Via Streamlit:**
```bash
streamlit run streamlit/agent_app.py
```

---

## Strategic Positioning

This project demonstrates:
- Research thinking (hypothesis → test → falsifiable verdict)
- Risk awareness (overfitting detection, regime dependence checks)
- Model governance logic (principled rejection criteria, audit trail)
- Automation of quant workflow (autonomous feature search)
- Engineering discipline (safe code execution, modular architecture)

The framing is not "I built a model that forecasts FX volatility well" but "I built a framework that systematically evaluates whether a forecasting signal is real, stable, and robust — and I understand why most signals aren't." This is the framing that resonates with systematic quant research environments.

---

## Open Questions & Next Steps

- [ ] Streamlit app testing and refinement
- [ ] README update to reflect agent architecture
- [ ] Consider adding regime-conditional evaluation as a separate diagnostic layer
- [ ] Consider CLI subcommands (`volare run`, `volare status`, `volare inspect`) for a cleaner interface
- [ ] Evaluate whether walk-forward fold count (currently 5) is appropriate given data size
- [ ] Add progress indicators for long-running LightGBM training cycles
