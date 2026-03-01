"""
propose.py
Feature proposal module.
Calls Claude to propose a candidate feature including a Python implementation.
"""

import os
import json
import anthropic
from src.agent.context import format_context_for_prompt, format_principles_for_prompt
from src.agent.registry import format_registry_for_prompt

CLIENT = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
MODEL = "claude-sonnet-4-6"


def propose_feature(
    context: dict,
    principles: list,
    registry: list,
    user_hint: str = None
) -> dict:
    """
    Ask Claude to propose a candidate feature including a Python implementation.

    Returns a dict with:
        - name: short descriptive name
        - hypothesis: the signal hypothesis in plain language
        - construction: how to compute it from raw data
        - code: a self-contained Python function implementing the feature
        - rejection_criteria: pre-stated conditions under which it should be rejected
        - rationale: why this feature given current context and registry history
    """

    context_str = format_context_for_prompt(context)
    principles_str = format_principles_for_prompt(principles)
    registry_str = format_registry_for_prompt(registry)

    hint_block = (
        f"\nThe researcher has provided the following direction hint:\n{user_hint}\n"
        if user_hint else ""
    )

    prompt = f"""
You are a systematic quantitative research agent specialising in FX volatility forecasting.

Your task is to propose ONE candidate feature for testing, including a working Python implementation.

The feature must be:
- Economically motivated with a clear signal hypothesis
- Constructable from the available data described below
- Not already tested (check the registry history)
- Consistent with the research principles
- Implementable as a pure function of a pandas DataFrame with OHLC columns

{hint_block}

DATA CONTEXT:
{context_str}

RESEARCH PRINCIPLES:
{principles_str}

RECENT REGISTRY HISTORY (avoid repeating these):
{registry_str}

The code field must contain a single self-contained Python function with this exact signature:
    def compute_feature(df: pandas.DataFrame) -> pandas.Series

Requirements for the code:
- df has columns: open, high, low, close (float64), indexed by timestamp
- Data is at 10-second resolution
- The function must return a pandas Series of the same length as df
- All computations must be strictly backward-looking (no look-ahead)
- Use only: numpy, pandas — no other imports
- Handle NaNs gracefully — return NaN where insufficient history exists
- Include numpy and pandas imports inside the function body

Respond ONLY with a valid JSON object in exactly this structure, no preamble, no markdown:
{{
  "name": "short_descriptive_feature_name",
  "hypothesis": "plain language explanation of the signal hypothesis",
  "construction": "precise description of how to compute this feature",
  "code": "def compute_feature(df):\\n    import numpy as np\\n    import pandas as pd\\n    # implementation here\\n    return result",
  "rejection_criteria": "specific pre-stated conditions under which this feature should be rejected",
  "rationale": "why this feature is worth testing given current context and registry history"
}}
""".strip()

    response = CLIENT.messages.create(
        model=MODEL,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    return json.loads(raw)