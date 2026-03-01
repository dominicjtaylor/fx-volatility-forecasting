"""
test.py
Feature testing engine.
Executes LLM-generated feature code safely and runs walk-forward validation
against the existing volare pipeline. Returns structured test results.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any


# ---------------------------------------------------------------------------
# Safe code execution
# ---------------------------------------------------------------------------

ALLOWED_MODULES = {"numpy", "pandas", "np", "pd", "math"}


def validate_code(code: str) -> None:
    """
    Basic safety validation of LLM-generated code.
    Raises ValueError if suspicious patterns are detected.
    """
    forbidden = [
        "import os", "import sys", "import subprocess",
        "import shutil", "import socket", "import requests",
        "open(", "exec(", "eval(", "__import__",
        "globals()", "locals()", "compile(",
        "os.path", "os.system", "pathlib"
    ]
    for pattern in forbidden:
        if pattern in code:
            raise ValueError(f"Forbidden pattern detected in generated code: '{pattern}'")


def execute_feature_code(code: str, df: pd.DataFrame) -> pd.Series:
    """
    Safely execute LLM-generated feature code.
    Runs the code in a restricted namespace and returns the result.
    """
    validate_code(code)

    # Restricted execution namespace
    namespace = {
        "numpy": np,
        "pandas": pd,
        "np": np,
        "pd": pd,
    }

    # Execute the function definition
    exec(code, namespace)

    if "compute_feature" not in namespace:
        raise ValueError("Generated code does not define a 'compute_feature' function")

    result = namespace["compute_feature"](df)

    if not isinstance(result, pd.Series):
        raise ValueError(f"compute_feature must return a pandas Series, got {type(result)}")

    if len(result) != len(df):
        raise ValueError(
            f"compute_feature returned Series of length {len(result)}, expected {len(df)}"
        )

    return result


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------

def walk_forward_validate(
    df: pd.DataFrame,
    feature_series: pd.Series,
    horizon_seconds: int = 3600,
    n_folds: int = 5,
    min_train_size: float = 0.5
) -> Dict[str, Any]:
    """
    Runs walk-forward validation of a feature against a rolling vol baseline.
    """
    freq_seconds = 10
    horizon_bars = horizon_seconds // freq_seconds

    log_returns = np.log(df["close"] / df["close"].shift(1))

    # Target: forward realised volatility
    target = (
        log_returns
        .shift(-horizon_bars)
        .rolling(horizon_bars)
        .std()
        * np.sqrt(252 * 8640)
    )

    # Baseline: 1-hour rolling vol (360 bars at 10s)
    baseline = log_returns.rolling(1440).std() * np.sqrt(252 * 8640)

    combined = pd.DataFrame({
        "feature": feature_series,
        "target": target,
        "baseline": baseline
    }).dropna()

    if len(combined) < 500:
        return {"error": "Insufficient data after alignment and NaN removal"}

    n = len(combined)
    min_train = int(n * min_train_size)
    fold_size = (n - min_train) // n_folds

    fold_results = []

    for i in range(n_folds):
        train_end = min_train + i * fold_size
        test_start = train_end
        test_end = test_start + fold_size

        if test_end > n:
            break

        train = combined.iloc[:train_end]
        test = combined.iloc[test_start:test_end]

        X_train = train["feature"].values.reshape(-1, 1)
        y_train = train["target"].values
        X_test = test["feature"].values.reshape(-1, 1)
        y_test = test["target"].values
        baseline_test = test["baseline"].values

        coeffs = np.linalg.lstsq(
            np.column_stack([np.ones(len(X_train)), X_train]),
            y_train,
            rcond=None
        )[0]

        preds = coeffs[0] + coeffs[1] * X_test.flatten()

        rmse_model = np.sqrt(np.mean((preds - y_test) ** 2))
        rmse_baseline = np.sqrt(np.mean((baseline_test - y_test) ** 2))
        rmse_improvement = (rmse_baseline - rmse_model) / rmse_baseline * 100

        correlation = np.corrcoef(X_test.flatten(), y_test)[0, 1]

        fold_results.append({
            "fold": i + 1,
            "rmse_model": round(float(rmse_model), 6),
            "rmse_baseline": round(float(rmse_baseline), 6),
            "rmse_improvement_pct": round(float(rmse_improvement), 3),
            "feature_target_correlation": round(float(correlation), 4)
        })

    if not fold_results:
        return {"error": "No folds completed"}

    improvements = [f["rmse_improvement_pct"] for f in fold_results]
    correlations = [abs(f["feature_target_correlation"]) for f in fold_results]

    monotonic_decay = all(
        improvements[i] >= improvements[i + 1]
        for i in range(len(improvements) - 1)
    )

    mean_corr = np.mean(correlations)
    importance_drift_pct = (
        (max(correlations) - min(correlations)) / mean_corr * 100
        if mean_corr > 0 else 0.0
    )

    return {
        "folds": fold_results,
        "overall_rmse_improvement_pct": round(float(np.mean(improvements)), 3),
        "importance_drift_pct": round(float(importance_drift_pct), 1),
        "monotonic_decay": monotonic_decay,
        "n_folds_completed": len(fold_results)
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_feature_test(
    feature: dict,
    context: dict,
    data: Dict[str, pd.DataFrame]
) -> Dict[str, Any]:
    """
    Main entry point for feature testing.
    Executes the LLM-generated code and tests across all pairs.
    """
    pairs = context["data"]["pairs"]
    horizon = context["data"]["horizon_seconds"]
    code = feature.get("code")

    if not code:
        return {
            "per_pair": {},
            "aggregate": {
                "error": "No code provided in feature proposal",
                "mean_rmse_improvement_pct": None,
                "mean_importance_drift_pct": None,
                "any_monotonic_decay": None,
                "pairs_tested": [],
                "errors": ["No code provided in feature proposal"]
            }
        }

    per_pair_results = {}
    errors = []

    for pair in pairs:
        if pair not in data:
            errors.append(f"No data found for {pair}")
            continue

        df = data[pair].copy()

        try:
            feature_series = execute_feature_code(code, df)
            results = walk_forward_validate(
                df, feature_series, horizon_seconds=horizon
            )
            per_pair_results[pair] = results
        except ValueError as e:
            errors.append(f"{pair} — code validation error: {str(e)}")
        except Exception as e:
            errors.append(f"{pair} — execution error: {str(e)}")

    all_improvements = [
        per_pair_results[p]["overall_rmse_improvement_pct"]
        for p in per_pair_results
        if "overall_rmse_improvement_pct" in per_pair_results[p]
    ]
    all_drifts = [
        per_pair_results[p]["importance_drift_pct"]
        for p in per_pair_results
        if "importance_drift_pct" in per_pair_results[p]
    ]

    return {
        "per_pair": per_pair_results,
        "aggregate": {
            "mean_rmse_improvement_pct": round(float(np.mean(all_improvements)), 3) if all_improvements else None,
            "mean_importance_drift_pct": round(float(np.mean(all_drifts)), 1) if all_drifts else None,
            "any_monotonic_decay": any(
                per_pair_results[p].get("monotonic_decay", False)
                for p in per_pair_results
            ),
            "pairs_tested": list(per_pair_results.keys()),
            "errors": errors
        }
    }