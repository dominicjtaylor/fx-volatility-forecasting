"""
reason.py
Reasoning and verdict module.
Calls Claude to interpret test results against the principles register
and produce a structured plain-language verdict.
"""

import os
import json
import anthropic
from src.agent.context import format_principles_for_prompt

CLIENT = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
MODEL = "claude-sonnet-4-6"


def format_test_results_for_prompt(test_results: dict) -> str:
    """Converts test results dict into a readable prompt block."""
    agg = test_results.get("aggregate", {})
    per_pair = test_results.get("per_pair", {})

    lines = ["AGGREGATE RESULTS:"]
    lines.append(f"  Mean RMSE improvement over baseline: {agg.get('mean_rmse_improvement_pct')}%")
    lines.append(f"  Mean importance drift across folds: {agg.get('mean_importance_drift_pct')}%")
    lines.append(f"  Any monotonic decay detected: {agg.get('any_monotonic_decay')}")
    lines.append(f"  Pairs tested: {', '.join(agg.get('pairs_tested', []))}")

    if agg.get("errors"):
        lines.append(f"  Errors: {'; '.join(agg['errors'])}")

    lines.append("\nPER-PAIR RESULTS:")
    for pair, results in per_pair.items():
        lines.append(f"\n  {pair}:")
        if "error" in results:
            lines.append(f"    Error: {results['error']}")
            continue
        lines.append(f"    Overall RMSE improvement: {results.get('overall_rmse_improvement_pct')}%")
        lines.append(f"    Importance drift: {results.get('importance_drift_pct')}%")
        lines.append(f"    Monotonic decay: {results.get('monotonic_decay')}")
        lines.append(f"    Folds completed: {results.get('n_folds_completed')}")
        for fold in results.get("folds", []):
            lines.append(
                f"      Fold {fold['fold']}: RMSE improvement {fold['rmse_improvement_pct']}%, "
                f"correlation {fold['feature_target_correlation']}"
            )

    return "\n".join(lines)


def reason_and_verdict(
    feature: dict,
    test_results: dict,
    principles: list,
    context: dict
) -> dict:
    """
    Ask Claude to reason about test results and issue a structured verdict.

    Returns a dict with:
        - verdict: "promoted", "rejected", or "modified"
        - summary: plain-language Trading212-style summary (2-4 sentences)
        - triggered_principles: list of principle IDs that influenced the verdict
        - next_action: concrete recommendation for what to investigate next
    """

    principles_str = format_principles_for_prompt(principles)
    results_str = format_test_results_for_prompt(test_results)

    prompt = f"""
You are a systematic quantitative research agent reviewing the test results for a candidate FX volatility feature.

Your role is to reason carefully against the research principles and issue a clear, defensible verdict.
You must be sceptical. Marginal improvements do not justify promotion. Instability is grounds for rejection.

FEATURE PROPOSAL:
Name: {feature.get('name')}
Hypothesis: {feature.get('hypothesis')}
Construction: {feature.get('construction')}
Pre-stated rejection criteria: {feature.get('rejection_criteria')}

TEST RESULTS:
{results_str}

RESEARCH PRINCIPLES:
{principles_str}

Instructions:
1. Reason through each relevant principle against the test results.
2. Check whether any pre-stated rejection criteria have been triggered.
3. Issue one of three verdicts: promoted, rejected, or modified.
   - promoted: feature passes all principles and shows robust improvement
   - rejected: one or more rejection criteria triggered, or results are unconvincing
   - modified: signal hypothesis is plausible but construction needs adjustment
4. Write a plain-language summary (3-5 sentences) as if briefing a senior researcher.
   Be direct. State what the results show, which principles were relevant, and why you reached your verdict.
   Do not use jargon. Do not hedge excessively. Write like a confident analyst, not an academic.
5. State the single most useful next action.

Respond ONLY with a valid JSON object in exactly this structure, no preamble, no markdown:
{{
  "verdict": "promoted" | "rejected" | "modified",
  "summary": "plain language 3-5 sentence verdict summary",
  "triggered_principles": ["P01", "P03"],
  "next_action": "single concrete recommendation for what to do next"
}}
""".strip()

    response = CLIENT.messages.create(
        model=MODEL,
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    return json.loads(raw)