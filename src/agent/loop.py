"""
loop.py
Orchestrates a single full agent cycle:
propose -> test -> reason -> log
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import pandas as pd

from src.agent.context import load_context, load_principles
from src.agent.registry import load_registry, save_entry, make_entry
from src.agent.propose import propose_feature
from src.agent.test import run_feature_test
from src.agent.reason import reason_and_verdict

LOG_DIR = Path(__file__).resolve().parents[2] / "outputs" / "logs"


def save_log(entry: dict) -> None:
    """Save a markdown reasoning log for this cycle."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = LOG_DIR / f"{timestamp}_{entry['name']}.md"

    with open(filename, "w") as f:
        f.write(f"# {entry['name']}\n\n")
        f.write(f"**Timestamp:** {entry['timestamp']}\n")
        f.write(f"**Verdict:** {entry['verdict'].upper()}\n\n")
        f.write(f"## Hypothesis\n{entry['hypothesis']}\n\n")
        f.write(f"## Construction\n{entry['construction']}\n\n")
        f.write(f"## Rejection Criteria\n{entry['rejection_criteria']}\n\n")
        f.write(f"## Test Results\n```json\n{json.dumps(entry['test_results'], indent=2)}\n```\n\n")
        f.write(f"## Triggered Principles\n{', '.join(entry['triggered_principles'])}\n\n")
        f.write(f"## Summary\n{entry['summary']}\n\n")
        f.write(f"## Next Action\n{entry['next_action']}\n")


def run_cycle(
    data: Dict[str, pd.DataFrame],
    user_hint: Optional[str] = None
) -> dict:
    """
    Run a single full agent cycle.

    Args:
        data: dict of {pair: DataFrame} with OHLC columns
        user_hint: optional natural language direction from the user

    Returns:
        The completed registry entry for this cycle
    """
    print("Loading context and principles...")
    context = load_context()
    principles = load_principles()
    registry = load_registry()

    print("Proposing feature...")
    feature = propose_feature(context, principles, registry, user_hint=user_hint)
    print(f"  → {feature['name']}")

    print("Running feature tests...")
    test_results = run_feature_test(feature, context, data)
    print(f"  → Mean RMSE improvement: {test_results['aggregate'].get('mean_rmse_improvement_pct')}%")

    print("Reasoning and issuing verdict...")
    reasoning = reason_and_verdict(feature, test_results, principles, context)
    print(f"  → Verdict: {reasoning['verdict'].upper()}")

    print("Logging to registry...")
    entry = make_entry(feature, test_results, reasoning)
    save_entry(entry)
    save_log(entry)

    print(f"\nCycle complete. Entry saved: {entry['id']}")
    print(f"Summary: {reasoning['summary']}")

    return entry