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
from src.agent.active_features import get_level_counts

CLIENT = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
MODEL = "claude-sonnet-4-6"


def format_test_results_for_prompt(test_results: dict) -> str:
    """Converts test results dict into a readable prompt block."""
    agg = test_results.get("aggregate", {})
    per_pair = test_results.get("per_pair", {})

    lines = ["AGGREGATE RESULTS:"]
    lines.append(f"  Improvement vs LightGBM baseline: {agg.get('mean_improvement_vs_lgbm_pct')}%")
    lines.append(f"  Improvement vs naive rolling vol baseline: {agg.get('mean_improvement_vs_naive_pct')}%")
    lines.append(f"  Mean candidate importance: {agg.get('mean_candidate_importance_pct')}%")
    lines.append(f"  Mean importance drift across folds: {agg.get('mean_importance_drift_pct')}%")
    lines.append(f"  Any monotonic decay detected: {agg.get('any_monotonic_decay')}")

    if agg.get("errors"):
        lines.append(f"  Errors: {'; '.join(agg['errors'])}")

    lines.append("\nPER-PAIR RESULTS:")
    for pair, results in per_pair.items():
        lines.append(f"\n  {pair}:")
        if "error" in results:
            lines.append(f"    Error: {results['error']}")
            continue
        lines.append(f"    Overall improvement vs LightGBM baseline: {results.get('overall_improvement_vs_lgbm_pct')}%")
        lines.append(f"    Overall improvement vs naive baseline: {results.get('overall_improvement_vs_naive_pct')}%")
        lines.append(f"    Mean candidate importance: {results.get('mean_candidate_importance_pct')}%")
        lines.append(f"    Importance drift: {results.get('importance_drift_pct')}%")
        lines.append(f"    Monotonic decay: {results.get('monotonic_decay')}")
        lines.append(f"    Folds completed: {results.get('n_folds_completed')}")
        for fold in results.get("folds", []):
            lines.append(
                f"      Fold {fold['fold']}: improvement vs LightGBM {fold['improvement_vs_lgbm_baseline_pct']}%, "
                f"vs naive {fold['improvement_vs_naive_baseline_pct']}%, "
                f"importance {fold['candidate_importance_pct']}%"
            )

    return "\n".join(lines)


def reason_and_verdict(
    feature: dict,
    test_results: dict,
    principles: list,
    context: dict,
    active_features: list = None
) -> dict:
    """
    Ask Claude to reason about test results and issue a structured verdict.

    Returns a dict with:
        - verdict: "promoted", "rejected", or "modified"
        - summary: plain-language Trading212-style summary (2-4 sentences)
        - triggered_principles: list of principle IDs that influenced the verdict
        - next_action: concrete recommendation for what to investigate next
    """

    if active_features is None:
        active_features = []

    principles_str = format_principles_for_prompt(principles)
    results_str = format_test_results_for_prompt(test_results)
    counts = get_level_counts(active_features)

    candidate_level = feature.get("level", "primitive")
    hierarchy_violation = None
    if candidate_level == "transform" and counts["primitive"] == 0:
        hierarchy_violation = "Candidate is Level 2 (Transform) but no Level 1 (Primitive) features are active. Hierarchy violation — reject."
    elif candidate_level == "composite" and counts["transform"] == 0:
        hierarchy_violation = "Candidate is Level 3 (Composite) but no Level 2 (Transform) features are active. Hierarchy violation — reject."

    hierarchy_block = (
        f"\nHIERARCHY GUARDRAIL: {hierarchy_violation}\n"
        if hierarchy_violation else ""
    )

    prompt = f"""
You are a systematic quantitative research agent reviewing the test results for a candidate FX volatility feature.

Your role is to reason carefully against the research principles and issue a clear, defensible verdict.
You must be sceptical. Marginal improvements do not justify promotion. Instability is grounds for rejection.
{hierarchy_block}
FEATURE PROPOSAL:
Name: {feature.get('name')}
Level: {candidate_level}
Motivation: {feature.get('motivation')}
Hypothesis: {feature.get('hypothesis')}
Construction: {feature.get('construction')}
Pre-stated rejection criteria: {feature.get('rejection_criteria')}

TEST RESULTS:
{results_str}

RESEARCH PRINCIPLES:
{principles_str}

Instructions:
1. Check results against each relevant principle and any pre-stated rejection criteria.
2. Issue one of three verdicts: promoted, rejected, or modified.
   - promoted: feature passes all principles and shows robust improvement
   - rejected: one or more rejection criteria triggered, or results are unconvincing
   - modified: signal hypothesis is plausible but construction needs adjustment
3. Write a summary of exactly 2-3 sentences: state the verdict, the single most important reason for it, and (if promoted or modified) what to watch.
4. State the single most useful next action.

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