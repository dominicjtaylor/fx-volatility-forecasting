"""
vol_simulation.py — Economic value simulation for the volare volatility model.

Compares two strategies on out-of-sample (OOS) test data:
  A) Baseline:    constant position size (w_t = 1)
  B) Vol-scaled:  w_t = target_vol / predicted_vol_t, clipped to [0, 3]

target_vol defaults to median predicted vol so average leverage is identical
across both strategies — isolates risk distribution impact, not leverage.

Usage:
    python scripts/vol_simulation.py --pair eurgbp
    python scripts/vol_simulation.py --pair gbpusd --cost 0.0001
    python scripts/vol_simulation.py --pair eurgbp --nrows 300000
"""

import argparse
import pickle
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
SRC_DIR = ROOT_DIR / "src"
sys.path.append(str(SRC_DIR))

from volare import data, features

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HORIZON_SECONDS = 3600
TRAIN_FRAC = 0.8
NROWS = 500_000
WEIGHT_CLIP = (0.0, 3.0)
EPS = 1e-8

# FX trades 24/5: 24h × 5 days/week × 52 weeks × 360 bars/h
BARS_PER_YEAR = int(24 * 5 * 52 * 3600 / 10)

STYLE_PATH = str(ROOT_DIR / "styles" / "science.mplstyle")
MODEL_DIR = ROOT_DIR / "results" / "models"
PARAMS_FILE = ROOT_DIR / "results" / "feature_tuning" / "best_params.csv"
OUTPUT_DIR = ROOT_DIR / "outputs" / "vol_simulation"

COLORS = {
    "baseline": "#0C5DA5",
    "scaled":   "#FF2C00",
    "gray":     "#474747",
    "violet":   "#845B97",
}


# ---------------------------------------------------------------------------
# Data & features
# ---------------------------------------------------------------------------

def load_data(pair: str, nrows: int = NROWS):
    """
    Load raw candles and compute the full feature + target pipeline,
    matching train_single_fx_pair.py exactly.

    Returns
    -------
    df_model : pd.DataFrame
        NaN-dropped, contains feature_cols + rolling_log_future_vol +
        log_return + timestamp.
    feature_cols : list[str]
    """
    data_file = ROOT_DIR / "data" / f"questdb-{pair}.csv"
    print(f"Loading {data_file} ({nrows:,} rows)...")
    df = data.load_candles(str(data_file), nrows=nrows)

    # Look up tuned params for this pair; fall back to safe defaults
    window_factor, window_scale, lag_scale = 8, 0.75, 1
    if PARAMS_FILE.exists():
        best = pd.read_csv(PARAMS_FILE, index_col=0)
        # best_params.csv index: 'EUR-GBP'; pair arg: 'eurgbp'
        pair_key = f"{pair[:3].upper()}-{pair[3:].upper()}"
        if pair_key in best.index:
            row = best.loc[pair_key]
            window_factor = float(row["window_factor"])
            window_scale  = float(row["window_scale"])
            lag_scale     = float(row["lag_scale"])
            print(f"Using tuned params for {pair_key}: wf={window_factor}, ws={window_scale}, ls={lag_scale}")
        else:
            print(f"No tuned params for {pair_key} — using defaults")
    else:
        print("best_params.csv not found — using defaults")

    print("Computing features...")
    df = features.compute_log_return(df)
    df = features.compute_rolling_volatility(
        df, horizon_seconds=HORIZON_SECONDS,
        window_scale=window_scale, window_factor=window_factor
    )
    df = features.compute_lagged_rolling_volatility(
        df, horizon_seconds=HORIZON_SECONDS,
        lag_scale=lag_scale, window_factor=window_factor
    )
    df = features.compute_multi_window_rolling_vol(df, horizon_seconds=HORIZON_SECONDS)
    df = features.compute_intraday_seasonality(df)
    df = features.compute_volatility_slope(df, horizon_seconds=HORIZON_SECONDS)
    df = features.compute_volatility_zscore(df, horizon_seconds=HORIZON_SECONDS)
    df = features.compute_volatility_acceleration(df)
    df = features.compute_future_rolling_volatility(df, horizon_seconds=HORIZON_SECONDS)
    print("Features done.\n")

    # Feature columns — identical definition to train_single_fx_pair.py
    feature_cols = (
        [c for c in df.columns if c.startswith("rolling_vol")] +
        [c for c in df.columns if c.startswith("tod_")] +
        [c for c in df.columns if c in ["vol_of_vol", "vol_slope", "vol_zscore", "vol_accel"]]
    )

    # Keep only what we need, drop NaNs to match split_data() behaviour
    keep_cols = feature_cols + ["rolling_log_future_vol", "log_return", "timestamp"]
    df_model = df[keep_cols].dropna(subset=feature_cols + ["rolling_log_future_vol"]).copy()
    df_model = df_model.reset_index(drop=True)

    print(f"Post-NaN-drop rows: {len(df_model):,}")
    return df_model, feature_cols


# ---------------------------------------------------------------------------
# OOS split
# ---------------------------------------------------------------------------

def get_oos_split(df_model: pd.DataFrame, train_frac: float = TRAIN_FRAC) -> pd.DataFrame:
    """Return the out-of-sample (test) portion, matching the training split."""
    split_idx = int(len(df_model) * train_frac)
    return df_model.iloc[split_idx:]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(pair: str):
    """Load the pre-trained LightGBM model for this pair."""
    model_path = MODEL_DIR / f"volare_lgb_{pair}_h{HORIZON_SECONDS}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Run train_single_fx_pair.py first to generate it."
        )
    with open(model_path, "rb") as f:
        lgb_model = pickle.load(f)
    print(f"Loaded model: {model_path.name}")
    return lgb_model


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

def compute_weights(
    predicted_vol: np.ndarray,
    target_vol: float,
    clip: tuple = WEIGHT_CLIP,
) -> np.ndarray:
    """
    Vol-targeting weights: w_t = target_vol / predicted_vol_t.
    Clipped to avoid extreme leverage.
    """
    safe_vol = np.clip(predicted_vol, 1e-10, None)
    weights = target_vol / safe_vol
    return np.clip(weights, clip[0], clip[1])


def compute_pnl(
    returns: np.ndarray,
    weights: np.ndarray,
    cost: float = 0.0,
) -> np.ndarray:
    """
    PnL per bar: w_t * r_t minus optional transaction cost.
    cost: charged per unit change in weight (|Δw|).
    """
    gross = weights * returns
    tc = cost * np.abs(np.diff(weights, prepend=weights[0]))
    return gross - tc


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _drawdown(pnl: np.ndarray) -> np.ndarray:
    """Drawdown series: peak cumulative PnL minus current."""
    cum = np.cumsum(pnl)
    return np.maximum.accumulate(cum) - cum


def compute_metrics(pnl: np.ndarray, label: str) -> dict:
    """Annualised performance metrics."""
    ann_ret = float(np.mean(pnl) * BARS_PER_YEAR)
    ann_vol = float(np.std(pnl) * np.sqrt(BARS_PER_YEAR))
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan
    max_dd  = float(_drawdown(pnl).max())

    return {
        "label":   label,
        "sharpe":  sharpe,
        "ann_vol": ann_vol,
        "max_dd":  max_dd,
        "ann_ret": ann_ret,
    }


def print_metrics_table(metrics_base: dict, metrics_scaled: dict) -> None:
    """Print a clean side-by-side comparison table."""
    b, s = metrics_base, metrics_scaled
    w = 20
    sep = "-" * (w * 3 + 4)
    print(f"\n{'Metric':<{w}}  {'Baseline':>{w}}  {'Vol-Scaled':>{w}}")
    print(sep)
    print(f"{'Annualised Return':<{w}}  {b['ann_ret']:>{w}.4f}  {s['ann_ret']:>{w}.4f}")
    print(f"{'Annualised Vol':<{w}}  {b['ann_vol']:>{w}.4f}  {s['ann_vol']:>{w}.4f}")
    print(f"{'Sharpe Ratio':<{w}}  {b['sharpe']:>{w}.3f}  {s['sharpe']:>{w}.3f}")
    print(f"{'Max Drawdown':<{w}}  {b['max_dd']:>{w}.4f}  {s['max_dd']:>{w}.4f}")
    print(sep)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(
    timestamps: pd.Series,
    pnl_base: np.ndarray,
    pnl_scaled: np.ndarray,
    weights_scaled: np.ndarray,
    predicted_vol: np.ndarray,
    baseline_vol: np.ndarray,
    pair: str,
    output_dir: Path,
) -> None:
    """
    4-panel figure matching repo aesthetics:
      1. Cumulative PnL
      2. Vol series (model vs rolling baseline)
      3. Drawdown
      4. Position weights
    """
    plt.style.use(STYLE_PATH)

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    fig.suptitle(f"Vol-Targeting Simulation — {pair.upper()} (OOS)", fontsize=14, y=1.01)

    ts = timestamps.values  # numpy array for x-axis

    cum_base   = np.cumsum(pnl_base)
    cum_scaled = np.cumsum(pnl_scaled)
    dd_base    = np.maximum.accumulate(cum_base)   - cum_base
    dd_scaled  = np.maximum.accumulate(cum_scaled) - cum_scaled

    # -- Panel 1: Cumulative PnL --
    ax = axes[0]
    ax.plot(ts, cum_base,   color=COLORS["baseline"], lw=1.3, label="Baseline (constant w=1)", alpha=0.85)
    ax.plot(ts, cum_scaled, color=COLORS["scaled"],   lw=1.3, label="Vol-scaled",              alpha=0.85)
    ax.axhline(0, color=COLORS["gray"], lw=0.6, ls="--")
    ax.set_ylabel("Cumulative log-return")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # -- Panel 2: Vol series --
    ax = axes[1]
    ax.plot(ts, baseline_vol,  color=COLORS["gray"],   lw=0.9, label="Rolling vol (baseline)", alpha=0.75)
    ax.plot(ts, predicted_vol, color=COLORS["scaled"],  lw=1.0, label="Model predicted vol",    alpha=0.80)
    ax.set_ylabel("Vol (per-bar)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # -- Panel 3: Drawdown --
    ax = axes[2]
    ax.fill_between(ts, -dd_base,   color=COLORS["baseline"], alpha=0.25, label="Baseline")
    ax.fill_between(ts, -dd_scaled, color=COLORS["scaled"],   alpha=0.25, label="Vol-scaled")
    ax.plot(ts, -dd_base,   color=COLORS["baseline"], lw=0.8)
    ax.plot(ts, -dd_scaled, color=COLORS["scaled"],   lw=0.8)
    ax.set_ylabel("Drawdown")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    # -- Panel 4: Weights --
    ax = axes[3]
    ax.plot(ts, weights_scaled, color=COLORS["violet"], lw=0.8, alpha=0.75, label="Vol-scaled weight")
    ax.axhline(1.0, color=COLORS["gray"], lw=0.8, ls="--", label="w = 1 (baseline)")
    ax.set_ylabel("Position weight")
    ax.set_xlabel("Time")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Rotate x-tick labels on bottom panel only (sharex=True)
    plt.setp(axes[-1].get_xticklabels(), rotation=30, ha="right")

    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"vol_simulation_{pair}.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {out_path}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_simulation(pair: str, cost: float = 0.0, nrows: int = NROWS) -> None:
    """End-to-end simulation for a single FX pair."""
    print(f"\n{'='*60}")
    print(f"  Vol-Targeting Simulation — {pair.upper()}")
    print(f"{'='*60}\n")

    # 1. Data + features
    df_model, feature_cols = load_data(pair, nrows)

    # 2. OOS split
    df_test = get_oos_split(df_model)
    print(f"OOS test rows: {len(df_test):,}  "
          f"({df_test['timestamp'].iloc[0]} → {df_test['timestamp'].iloc[-1]})\n")

    # 3. Load model
    lgb_model = load_model(pair)

    # 4. Predict (log space → vol)
    X_test = df_test[feature_cols].values
    y_pred_log    = lgb_model.predict(X_test)
    predicted_vol = np.clip(np.exp(y_pred_log), EPS, None)

    # Baseline vol: medium-window rolling_vol_*_cand feature
    rolling_cols = [
        c for c in feature_cols
        if "rolling_vol_" in c and "_cand" in c
        and "_slope" not in c and "_over_" not in c
    ]
    baseline_vol = df_test[rolling_cols[len(rolling_cols) // 2]].values

    # 5. Returns
    returns = df_test["log_return"].values

    # 6. Weights
    target_vol = float(np.median(predicted_vol))  # keeps avg weight ≈ 1
    print(f"target_vol (median predicted): {target_vol:.6f}")

    w_scaled = compute_weights(predicted_vol, target_vol)

    print(f"Weight stats — mean: {w_scaled.mean():.3f}, "
          f"std: {w_scaled.std():.3f}, "
          f"max: {w_scaled.max():.3f}, "
          f"clip%: {(w_scaled >= WEIGHT_CLIP[1]).mean()*100:.1f}%\n")

    # 7. PnL — baseline is unscaled returns (w=1 constant, no cost)
    pnl_base   = returns
    pnl_scaled = compute_pnl(returns, w_scaled, cost=cost)

    # 8. Metrics
    metrics_base   = compute_metrics(pnl_base,   "Baseline")
    metrics_scaled = compute_metrics(pnl_scaled, "Vol-Scaled")
    print_metrics_table(metrics_base, metrics_scaled)

    # 9. Plots
    plot_results(
        timestamps    = df_test["timestamp"].reset_index(drop=True),
        pnl_base      = pnl_base,
        pnl_scaled    = pnl_scaled,
        weights_scaled= w_scaled,
        predicted_vol = predicted_vol,
        baseline_vol  = baseline_vol,
        pair          = pair,
        output_dir    = OUTPUT_DIR,
    )


# ---------------------------------------------------------------------------
# Audit extensions: discrete rebalancing & multi-variant comparison
# ---------------------------------------------------------------------------

BAR_SECONDS      = 10
BARS_PER_HORIZON = HORIZON_SECONDS // BAR_SECONDS   # 360 bars per 1h horizon

COST_NONE = 0.0
COST_LOW  = 0.00005   # 0.5 bps
COST_HIGH = 0.0001    # 1.0 bp

OUTPUT_DIR_AUDIT = ROOT_DIR / "outputs" / "vol_simulation_final"

AUDIT_COLORS = {
    "A": "#0C5DA5",   # blue   — continuous, no cost
    "B": "#00B945",   # green  — discrete, no cost
    "C": "#FF9500",   # orange — discrete + 0.5 bps
    "D": "#FF2C00",   # red    — discrete + 1 bp
}


def discrete_weights(
    predicted_vol: np.ndarray,
    target_vol: float,
    bars_per_horizon: int,
    clip: tuple = WEIGHT_CLIP,
) -> np.ndarray:
    """
    Vol-targeting weights updated only at rebalance intervals.
    Computes a new weight every bars_per_horizon steps; holds it constant
    (forward-fill) until the next rebalance.
    """
    n = len(predicted_vol)
    raw = compute_weights(predicted_vol, target_vol, clip)
    w = np.full(n, np.nan)
    w[np.arange(0, n, bars_per_horizon)] = raw[np.arange(0, n, bars_per_horizon)]
    return pd.Series(w).ffill().to_numpy()


def _shift_weights(weights: np.ndarray) -> np.ndarray:
    """
    Shift weights forward by 1 bar: weight computed at bar t is applied to
    return at bar t+1. Guarantees predicted_vol[t] does not affect pnl[t].

    Features at bar t include log_return[t] = log(close[t]/open[t]), so without
    this shift the weight at bar t has a same-bar dependency on return[t].
    The 1-bar shift eliminates that dependency at the cost of one 10s edge.
    """
    return np.concatenate([[weights[0]], weights[:-1]])


def compute_turnover(weights: np.ndarray) -> dict:
    """
    Compute portfolio turnover entering from flat (weight = 0).
    First bar cost = weights[0] (cost of entering the initial position).
    """
    diffs = np.abs(np.diff(weights, prepend=0.0))
    total = float(diffs.sum())
    n_changes = int((diffs > 1e-12).sum())
    avg_per_change = total / n_changes if n_changes > 0 else 0.0
    return {"total": total, "n_changes": n_changes, "avg_per_change": avg_per_change}


def print_audit_table(variants: list) -> None:
    """Print multi-variant results table: Variant | Sharpe | Ann Vol | Max DD | Turnover."""
    cw = 22
    sep = "-" * (cw * 5 + 8)
    print(f"\n{'Variant':<{cw}}  {'Sharpe':>{cw}}  {'Ann Vol':>{cw}}  {'Max DD':>{cw}}  {'Turnover':>{cw}}")
    print(sep)
    for v in variants:
        print(
            f"{v['label']:<{cw}}  "
            f"{v['sharpe']:>{cw}.3f}  "
            f"{v['ann_vol']:>{cw}.4f}  "
            f"{v['max_dd']:>{cw}.4f}  "
            f"{v['turnover']:>{cw}.1f}"
        )
    print(sep)


def plot_audit_results(
    timestamps: pd.Series,
    pnl_dict: dict,
    weights_dict: dict,
    pair: str,
    output_dir: Path,
) -> None:
    """
    3-panel audit figure:
      1. Cumulative PnL — all variants
      2. Drawdown — all variants
      3. Weight series — continuous vs discrete
    """
    plt.style.use(STYLE_PATH)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(
        f"Vol-Targeting Simulation Audit — {pair.upper()} (OOS)",
        fontsize=14, y=1.01,
    )

    ts = timestamps.values

    label_map = {
        "A": "A: Continuous, no cost",
        "B": "B: Discrete, no cost",
        "C": "C: Discrete + 0.5 bps",
        "D": "D: Discrete + 1 bp",
    }

    # Panel 1: Cumulative PnL
    ax = axes[0]
    for key, pnl in pnl_dict.items():
        ax.plot(ts, np.cumsum(pnl), color=AUDIT_COLORS[key], lw=1.3,
                label=label_map[key], alpha=0.85)
    ax.axhline(0, color=COLORS["gray"], lw=0.6, ls="--")
    ax.set_ylabel("Cumulative log-return")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: Drawdown
    ax = axes[1]
    for key, pnl in pnl_dict.items():
        dd = _drawdown(pnl)
        ax.fill_between(ts, -dd, color=AUDIT_COLORS[key], alpha=0.15)
        ax.plot(ts, -dd, color=AUDIT_COLORS[key], lw=0.8, label=label_map[key])
    ax.set_ylabel("Drawdown")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: Weight series — continuous vs discrete
    ax = axes[2]
    for key, label in (("A", "Continuous"), ("B", "Discrete")):
        if key in weights_dict:
            ax.plot(ts, weights_dict[key], color=AUDIT_COLORS[key],
                    lw=0.8, alpha=0.8, label=label)
    ax.axhline(1.0, color=COLORS["gray"], lw=0.6, ls="--")
    ax.set_ylabel("Position weight")
    ax.set_xlabel("Time")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.setp(axes[-1].get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"vol_simulation_audit_{pair}.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {out_path}")


def run_audit(pair: str, nrows: int = NROWS) -> None:
    """
    4-variant audit:
      A) Continuous weights, no cost    — current behaviour (with shift)
      B) Discrete rebalancing, no cost  — weight updated every HORIZON bars
      C) Discrete + 0.5 bps             — low transaction cost
      D) Discrete + 1 bp                — high transaction cost

    All variants apply a 1-bar forward shift to weights so that
    predicted_vol[t] (computed from features at bar t, which include
    log_return[t]) is never applied to return[t] — only return[t+1] onward.
    """
    label_map = {
        "A": "A: Continuous, no cost",
        "B": "B: Discrete, no cost",
        "C": "C: Discrete + 0.5 bps",
        "D": "D: Discrete + 1 bp",
    }

    print(f"\n{'='*60}")
    print(f"  Vol-Targeting Simulation Audit — {pair.upper()}")
    print(f"{'='*60}")
    print(f"\n  Bar size:          {BAR_SECONDS}s")
    print(f"  Forecast horizon:  {HORIZON_SECONDS}s = {BARS_PER_HORIZON} bars")
    print(f"  Periods / year:    {BARS_PER_YEAR:,}")
    print(f"  Rebalance freq:    every {BARS_PER_HORIZON} bars "
          f"({HORIZON_SECONDS // 60:.0f} min)\n")

    # --- Data & model ---
    df_model, feature_cols = load_data(pair, nrows)
    df_test = get_oos_split(df_model)
    print(f"OOS rows: {len(df_test):,}  "
          f"({df_test['timestamp'].iloc[0]} → {df_test['timestamp'].iloc[-1]})\n")

    lgb_model = load_model(pair)

    # --- Predictions ---
    X_test        = df_test[feature_cols].values
    predicted_vol = np.clip(np.exp(lgb_model.predict(X_test)), EPS, None)
    returns       = df_test["log_return"].values
    target_vol    = float(np.median(predicted_vol))
    print(f"target_vol (median predicted): {target_vol:.6f}\n")

    # --- Weights (compute raw, then shift by 1 bar) ---
    #
    # Features at bar t include log_return[t] = log(close[t]/open[t]).
    # Without shifting, w[t] × return[t] creates a same-bar dependency.
    # After shifting: w_applied[t] = w_raw[t-1], so predicted_vol[t-1]
    # drives pnl[t] — no same-bar overlap.
    #
    w_cont = _shift_weights(compute_weights(predicted_vol, target_vol))
    w_disc = _shift_weights(discrete_weights(predicted_vol, target_vol, BARS_PER_HORIZON))

    assert len(w_cont) == len(returns) == len(w_disc), "Weight/return length mismatch"

    # --- Turnover ---
    to_cont = compute_turnover(w_cont)
    to_disc = compute_turnover(w_disc)
    print(f"Turnover — continuous: {to_cont['total']:.1f} total, "
          f"{to_cont['n_changes']:,} changes, "
          f"avg {to_cont['avg_per_change']:.4f}/change")
    print(f"Turnover — discrete:   {to_disc['total']:.1f} total, "
          f"{to_disc['n_changes']:,} changes, "
          f"avg {to_disc['avg_per_change']:.4f}/change\n")

    # --- PnL ---
    pnl_A = compute_pnl(returns, w_cont, cost=COST_NONE)
    pnl_B = compute_pnl(returns, w_disc, cost=COST_NONE)
    pnl_C = compute_pnl(returns, w_disc, cost=COST_LOW)
    pnl_D = compute_pnl(returns, w_disc, cost=COST_HIGH)

    # --- Metrics + table ---
    variants = []
    for key, pnl, turnover in (
        ("A", pnl_A, to_cont["total"]),
        ("B", pnl_B, to_disc["total"]),
        ("C", pnl_C, to_disc["total"]),
        ("D", pnl_D, to_disc["total"]),
    ):
        m = compute_metrics(pnl, label_map[key])
        m["turnover"] = turnover
        variants.append(m)

    print_audit_table(variants)

    # --- Plot ---
    plot_audit_results(
        timestamps   = df_test["timestamp"].reset_index(drop=True),
        pnl_dict     = {"A": pnl_A, "B": pnl_B, "C": pnl_C, "D": pnl_D},
        weights_dict = {"A": w_cont, "B": w_disc},
        pair         = pair,
        output_dir   = OUTPUT_DIR_AUDIT,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Economic value simulation for the volare vol model."
    )
    parser.add_argument(
        "--pair", type=str, default="eurgbp",
        help="FX pair (lowercase, no dash): eurgbp, gbpusd, eurusd, etc."
    )
    parser.add_argument(
        "--cost", type=float, default=0.0,
        help="Transaction cost per unit weight change |Δw| (default: 0)"
    )
    parser.add_argument(
        "--nrows", type=int, default=NROWS,
        help=f"Max rows to load (default: {NROWS:,})"
    )
    parser.add_argument(
        "--audit", action="store_true",
        help="Run 4-variant audit: continuous vs discrete rebalancing, with costs",
    )
    args = parser.parse_args()
    if args.audit:
        run_audit(args.pair, args.nrows)
    else:
        run_simulation(args.pair, args.cost, args.nrows)
