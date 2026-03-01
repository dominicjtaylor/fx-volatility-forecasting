"""
agent_app.py
Volare Agent — Streamlit control panel.
Run with: streamlit run streamlit/agent_app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import streamlit as st
import pandas as pd
import anthropic
import os

from src.agent.context import load_context, load_principles
from src.agent.registry import load_registry, filter_by_verdict
from src.agent.loop import run_cycle

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Volare Agent",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

PAIR_FILES = {
    "EURGBP": "questdb-eurgbp.csv",
    "GBPUSD": "questdb-gbpusd.csv",
}

@st.cache_data(show_spinner=False)
def load_data() -> dict:
    data = {}
    for pair, filename in PAIR_FILES.items():
        path = DATA_DIR / filename
        if path.exists():
            df = pd.read_csv(
                path,
                parse_dates=["timestamp"],
                index_col="timestamp"
            )
            data[pair] = df
    return data

# ---------------------------------------------------------------------------
# Helper: registry stats
# ---------------------------------------------------------------------------

def registry_stats(registry: list) -> dict:
    return {
        "total": len(registry),
        "promoted": len([e for e in registry if e.get("verdict") == "promoted"]),
        "rejected": len([e for e in registry if e.get("verdict") == "rejected"]),
        "modified": len([e for e in registry if e.get("verdict") == "modified"]),
    }

def verdict_colour(verdict: str) -> str:
    return {
        "promoted": "green",
        "rejected": "red",
        "modified": "orange"
    }.get(verdict, "grey")

# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    # Load data once
    with st.spinner("Loading market data..."):
        data = load_data()

    # Header
    st.markdown("## 📡 Volare Agent")
    st.markdown("*FX Volatility Research Agent*")
    st.divider()

    # Tabs
    tab_explore, tab_chat, tab_log = st.tabs(["Explore", "Research Chat", "Registry"])

    # -----------------------------------------------------------------------
    # TAB 1: EXPLORE (launchpad)
    # -----------------------------------------------------------------------
    with tab_explore:

        # Stats row
        registry = load_registry()
        stats = registry_stats(registry)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Cycles Run", stats["total"])
        col2.metric("Promoted", stats["promoted"])
        col3.metric("Rejected", stats["rejected"])
        col4.metric("Modified", stats["modified"])

        st.divider()

        # Launchpad
        st.markdown("### What should the agent explore?")
        st.markdown(
            "Describe a signal direction, or leave blank to let the agent decide "
            "based on registry history and context."
        )

        hint = st.text_area(
            label="Direction hint",
            placeholder="e.g. explore regime transition signals, investigate cross-pair volatility spillover, focus on intraday seasonality patterns...",
            height=100,
            label_visibility="collapsed"
        )

        explore_btn = st.button("Explore", type="primary", use_container_width=False)

        if explore_btn:
            if not data:
                st.error("No market data found. Check that data/ directory contains the expected CSV files.")
            else:
                with st.status("Running agent cycle...", expanded=True) as status:
                    st.write("Loading context and principles...")
                    st.write("Proposing feature...")

                    try:
                        # Run the full cycle
                        entry = run_cycle(data, user_hint=hint if hint.strip() else None)

                        st.write(f"Feature proposed: **{entry['name']}**")
                        st.write("Running walk-forward validation...")
                        st.write("Reasoning against principles register...")
                        st.write(f"Verdict: **{entry['verdict'].upper()}**")
                        status.update(label="Cycle complete", state="complete")

                        # Show result
                        st.divider()
                        verdict_col, _ = st.columns([1, 3])
                        with verdict_col:
                            st.markdown(
                                f"**:{verdict_colour(entry['verdict'])}[{entry['verdict'].upper()}]** — {entry['name']}"
                            )
                        st.markdown(entry["summary"])

                        agg = entry.get("test_results", {}).get("aggregate", {})
                        if agg.get("mean_rmse_improvement_pct") is not None:
                            m1, m2, m3 = st.columns(3)
                            m1.metric(
                                "Mean RMSE Improvement",
                                f"{agg['mean_rmse_improvement_pct']}%"
                            )
                            m2.metric(
                                "Mean Importance Drift",
                                f"{agg['mean_importance_drift_pct']}%"
                            )
                            m3.metric(
                                "Monotonic Decay",
                                "Yes" if agg.get("any_monotonic_decay") else "No"
                            )

                    except Exception as e:
                        status.update(label="Cycle failed", state="error")
                        st.error(f"Cycle failed: {str(e)}")

        # Recent activity
        if registry:
            st.divider()
            st.markdown("### Recent Activity")
            for entry in reversed(registry[-5:]):
                with st.container():
                    c1, c2, c3 = st.columns([3, 1, 2])
                    c1.markdown(f"**{entry['name']}**")
                    c2.markdown(
                        f":{verdict_colour(entry['verdict'])}[{entry['verdict'].upper()}]"
                    )
                    c3.markdown(f"*{entry.get('timestamp', '')}*")

    # -----------------------------------------------------------------------
    # TAB 2: RESEARCH CHAT
    # -----------------------------------------------------------------------
    with tab_chat:
        st.markdown("### Research Direction")
        st.markdown(
            "Discuss signal ideas, ask about registry findings, or guide the "
            "agent's next exploration. The agent has full context of what has "
            "been tested and the current data environment."
        )

        # Initialise chat history
        if "chat_history" not in st.session_state:
            registry_now = load_registry()
            stats_now = registry_stats(registry_now)
            st.session_state.chat_history = [
                {
                    "role": "assistant",
                    "content": (
                        f"Hello. I have {stats_now['total']} features in the registry — "
                        f"{stats_now['promoted']} promoted, {stats_now['rejected']} rejected, "
                        f"{stats_now['modified']} pending modification. "
                        "What direction should I explore next?"
                    )
                }
            ]

        # Display chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        if user_input := st.chat_input("Guide the agent..."):
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input
            })
            with st.chat_message("user"):
                st.markdown(user_input)

            # Build context for the LLM
            context = load_context()
            principles = load_principles()
            registry_now = load_registry()

            system_prompt = f"""
You are a systematic FX volatility research agent assistant.
You help a quantitative researcher (Dom) guide the direction of feature research.
You have full knowledge of what has been tested, the data context, and the research principles.

You are concise, direct, and research-focused. You do not pad responses.
You speak like a senior quant researcher, not a customer service bot.

REGISTRY SUMMARY:
{json.dumps([{
    'name': e['name'],
    'verdict': e['verdict'],
    'summary': e.get('summary', '')[:150]
} for e in registry_now], indent=2)}

DATA CONTEXT:
Pairs: {', '.join(context['data']['pairs'])}
Frequency: {context['data']['frequency']}
Known regimes: {', '.join(r['name'] for r in context.get('known_regimes', []))}
""".strip()

            # Call Claude
            client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

            messages = [
                {"role": m["role"] if m["role"] != "assistant" else "assistant",
                 "content": m["content"]}
                for m in st.session_state.chat_history
                if m["role"] in ("user", "assistant")
            ]

            with st.chat_message("assistant"):
                with st.spinner(""):
                    response = client.messages.create(
                        model="claude-sonnet-4-6",
                        max_tokens=500,
                        system=system_prompt,
                        messages=messages
                    )
                    reply = response.content[0].text
                    st.markdown(reply)

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": reply
            })

    # -----------------------------------------------------------------------
    # TAB 3: REGISTRY LOG
    # -----------------------------------------------------------------------
    with tab_log:
        st.markdown("### Research Registry")

        registry = load_registry()

        if not registry:
            st.info("No features tested yet. Run a cycle from the Explore tab.")
        else:
            # Filter controls
            filter_col, _ = st.columns([2, 4])
            with filter_col:
                verdict_filter = st.selectbox(
                    "Filter by verdict",
                    ["All", "Promoted", "Rejected", "Modified"],
                    label_visibility="collapsed"
                )

            filtered = (
                registry if verdict_filter == "All"
                else [e for e in registry if e.get("verdict") == verdict_filter.lower()]
            )

            st.markdown(f"*Showing {len(filtered)} of {len(registry)} entries*")
            st.divider()

            # Entry list
            for entry in reversed(filtered):
                with st.expander(
                    f"**{entry['name']}** — "
                    f":{verdict_colour(entry['verdict'])}[{entry['verdict'].upper()}] "
                    f"— {entry.get('timestamp', '')}"
                ):
                    # Summary
                    st.markdown("**Summary**")
                    st.markdown(entry.get("summary", "No summary available."))

                    # Metrics
                    agg = entry.get("test_results", {}).get("aggregate", {})
                    if agg.get("mean_rmse_improvement_pct") is not None:
                        m1, m2, m3 = st.columns(3)
                        m1.metric(
                            "RMSE Improvement",
                            f"{agg['mean_rmse_improvement_pct']}%"
                        )
                        m2.metric(
                            "Importance Drift",
                            f"{agg['mean_importance_drift_pct']}%"
                        )
                        m3.metric(
                            "Monotonic Decay",
                            "Yes" if agg.get("any_monotonic_decay") else "No"
                        )

                    # Hypothesis and construction
                    st.markdown("**Hypothesis**")
                    st.markdown(entry.get("hypothesis", ""))

                    st.markdown("**Construction**")
                    st.markdown(entry.get("construction", ""))

                    # Triggered principles
                    if entry.get("triggered_principles"):
                        st.markdown(
                            f"**Triggered Principles:** "
                            f"{', '.join(entry['triggered_principles'])}"
                        )

                    # Next action
                    if entry.get("next_action"):
                        st.markdown("**Next Action**")
                        st.info(entry.get("next_action"))

                    # Per-pair fold detail
                    per_pair = entry.get("test_results", {}).get("per_pair", {})
                    if per_pair:
                        st.markdown("**Per-Pair Fold Results**")
                        for pair, results in per_pair.items():
                            if "folds" in results:
                                fold_df = pd.DataFrame(results["folds"])
                                st.markdown(f"*{pair}*")
                                st.dataframe(fold_df, use_container_width=True)


if __name__ == "__main__":
    main()