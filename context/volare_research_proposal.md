**VOLARE**

FX Volatility Research Intelligence

*Research Proposal & System Design*

**Author:** Dominic Taylor

**Repository:** dominicjtaylor/fx-volatility-forecasting

**Date:** March 2026

**Abstract**

+-----------------------------------------------------------------------+
| **Core Proposition**                                                  |
|                                                                       |
| *Volare is not a feature engineering tool. It is a self-updating,     |
| evidence-backed theory of how a specific currency pair behaves ---    |
| and an agent that actively works to prove itself wrong.*              |
+-----------------------------------------------------------------------+

Existing approaches to FX volatility research treat signal discovery as
an optimisation problem: find features, test them, accumulate the ones
that work. Volare challenges this framing. Rather than accumulating
signals, it accumulates understanding --- building a structured,
falsifiable theory of market microstructure that evolves as conditions
change, argues with its own prior conclusions, and tells the researcher
when it has reached the limits of what is learnable from the data.

The system runs as a self-hosted, CLI-first research agent. A quant
researcher provides their own data and API key, directs the agent from
the terminal, and receives a compounding research log that no single
session with a general-purpose AI could produce.

**1. Motivation & Problem Statement**

**1.1 The gap in existing tooling**

A quant researcher studying FX volatility today has access to powerful
components in isolation: high-quality data vendors, backtesting
frameworks, AutoML libraries, and general-purpose AI assistants. What
they lack is a system that connects these components into a coherent,
persistent research process.

General-purpose AI tools such as ChatGPT or Claude Code can propose and
test features on demand, but every session begins from zero. There is no
memory of what was tried, no accumulated quality bar, and no compounding
library of validated signals. The researcher must manage continuity
themselves.

AutoML frameworks optimise aggressively for in-sample performance,
frequently producing brittle models that fail to generalise across
market regimes. They offer no mechanism for encoding economic intuition,
and no transparency about why a signal was selected.

The result is a research process that is fragmented, non-reproducible,
and unable to build on itself over time.

**1.2 The specific research target**

Volare focuses on a well-defined and practically important forecasting
problem: predicting rolling log future volatility at a 3,600-second
horizon, using high-frequency OHLCV data at 10-second resolution.
Initial development targets EURGBP and GBPUSD, chosen for their
liquidity, distinct microstructure characteristics, and availability of
deep tick history.

The baseline model is a medium-window rolling volatility estimate. All
candidate features are evaluated against this baseline using
walk-forward validation across five folds, ensuring that performance
claims are grounded in genuine out-of-sample evidence.

**2. The Central Idea: A Living Theory of a Market**

The foundational claim of Volare is that feature engineering and
theoretical understanding are not separate activities --- they are the
same activity, done well or done poorly.

When a feature is promoted, it is not merely added to a model. It is
treated as evidence for a hypothesis about how the market works: that
session overlap creates predictable volatility clustering, that spread
widening anticipates directional moves, that order flow imbalance decays
at a measurable rate. The registry accumulates not just signals but the
reasoning behind them.

This framing has three consequences that distinguish Volare from all
existing tools:

-   **Theory falsification.** The agent can detect when a previously
    validated hypothesis begins to break down --- when a promoted
    feature\'s importance decays, when its regime robustness degrades,
    or when the economic logic it encoded no longer holds. This is not
    treated as a model failure but as a research finding.

-   **Saturation detection.** Rather than optimising indefinitely, the
    agent monitors its own performance ceiling. When incremental feature
    additions no longer produce meaningful out-of-sample improvement, it
    signals that the current data is approaching saturation --- and that
    the researcher should seek richer inputs rather than more complex
    models.

-   **Reasoning provenance.** Each promoted feature is connected to the
    features that preceded it, forming a graph of causal dependencies
    and regime conditions. The researcher can trace why the current
    model believes what it believes.

**3. System Architecture**

**3.1 The research loop**

The core of Volare is an autonomous agent loop that runs continuously
until interrupted. Each cycle consists of four stages:

-   **Propose.** The agent proposes a new candidate feature with Python
    implementation code and an explicit economic rationale. The proposal
    is grounded in the current active feature set, the principles
    register, and the reasoning history of prior cycles.

-   **Test.** The candidate feature is executed in a sandboxed
    environment. A LightGBM model is trained using walk-forward
    validation, comparing a baseline model (rolling volatility only)
    against an enriched model that includes the candidate alongside all
    currently active features.

-   **Reason & Verdict.** The agent interprets the results against the
    six principles (see Section 3.3). It issues a verdict --- promoted,
    rejected, or modified --- with explicit reasoning that references
    which principles were satisfied or violated.

-   **Log & Accumulate.** All results, metrics, code, and reasoning are
    written to the registry. Promoted features are added to the active
    feature set for future cycles, making each cycle more informed than
    the last.

**3.2 Technology stack**

The system is built on a deliberately lean stack: Python for
orchestration and feature computation, LightGBM for model training, the
Anthropic API (claude-sonnet-4-6) for proposal and reasoning, and
Streamlit for the optional monitoring interface. All state is managed
through flat files --- registry.json for the feature log,
active_features.json for the current feature set --- ensuring full
transparency and portability.

**3.3 The principles register**

Every feature verdict is evaluated against six quality principles. These
principles encode the hard-won intuitions of quantitative research
practice and cannot be overridden by raw performance metrics:

  ---------- ------------------ -------------------------------------------
  **Code**   **Principle**      **Description**

  **P01**    **Regime           Features must demonstrate consistent
             Robustness**       performance across different market
                                regimes, not just aggregate fold averages.

  **P02**    **No Look-Ahead**  All feature construction must be strictly
                                causal. No future information may be used
                                in the computation of any signal.

  **P03**    **Importance       Feature importance scores must be stable
             Stability**        across folds. Highly variable importance
                                suggests overfitting to a specific market
                                period.

  **P04**    **OOS              The enriched model must demonstrate
             Improvement**      meaningful out-of-sample improvement over
                                the rolling volatility baseline.

  **P05**    **Economic         Every feature must have a plausible
             Motivation**       economic rationale. Statistical patterns
                                without theoretical grounding are rejected.

  **P06**    **Signal Decay     Features must be evaluated for decay
             Awareness**        characteristics. Signals with short
                                half-lives require explicit documentation.
  ---------- ------------------ -------------------------------------------

**4. Volare as a Self-Hosted Product**

**4.1 Design philosophy**

Volare is designed for a specific user: a quant researcher who is
comfortable in a terminal, skeptical of black boxes, and building toward
a systematic trading edge. The product does not attempt to abstract away
complexity --- it surfaces complexity clearly and makes it navigable.

The primary interface is a CLI. The researcher interacts with the agent
through subcommands, reads structured output in the terminal, and
integrates results into their own workflow. A Streamlit monitoring
interface is available for overview and review, but is not required for
the core research loop.

**4.2 Interaction model**

The CLI is designed to feel like a conversation with a knowledgeable
collaborator. Key subcommands include:

-   **volare run \--cycles N** Runs N agent cycles autonomously. Output
    is colour-coded: promoted features in green, rejected in red,
    modified in amber.

-   **volare status** Shows current state: active features, cycles
    completed, baseline performance, and saturation signal.

-   **volare inspect \--feature \<name\>** Displays full detail for a
    specific feature: code, reasoning, fold metrics, and verdict
    history.

-   **volare chat \"\<question\>\"** Opens a grounded research chat in
    the terminal, with the agent responding based on the full registry
    context.

-   **volare init** Configures the data source, pair, forecast horizon,
    and API key for a new research session.

**4.3 Deployment**

The system is packaged as a Docker image with a single
docker-compose.yml entry point. The researcher mounts their data
directory, sets their Anthropic API key as an environment variable, and
runs one command. There is no cloud dependency, no subscription, and no
data leaves the local environment.

A config-driven data ingestion layer accepts any OHLCV CSV and maps it
to the internal schema, making the system pair- and broker-agnostic.

**5. Novel Contributions**

Volare makes three contributions that, to the author\'s knowledge, have
not been combined in any existing research tool:

+-----------------------------------------------------------------------+
| **Contribution 1: Falsifiable theory accumulation**                   |
|                                                                       |
| *Rather than accumulating features, the system accumulates a          |
| structured, evidence-backed theory of market behaviour. Each promoted |
| signal is a testable hypothesis. The agent actively monitors for      |
| hypothesis breakdown and treats regime change as a research finding   |
| rather than a model failure.*                                         |
+-----------------------------------------------------------------------+

+-----------------------------------------------------------------------+
| **Contribution 2: Autonomous saturation detection**                   |
|                                                                       |
| *The system monitors its own performance ceiling and signals when the |
| current data is approaching the limits of what is learnable. This     |
| directs the researcher toward better data acquisition rather than     |
| more complex models --- a form of epistemic honesty that no existing  |
| AutoML tool provides.*                                                |
+-----------------------------------------------------------------------+

+-----------------------------------------------------------------------+
| **Contribution 3: Persistent institutional memory via CLI**           |
|                                                                       |
| *The research loop maintains full continuity across sessions, pairs,  |
| and time. The CLI interaction model makes this accessible to          |
| researchers in their natural working environment, without requiring a |
| GUI or cloud infrastructure.*                                         |
+-----------------------------------------------------------------------+

**6. Development Roadmap**

**Phase 1 --- Core hardening (current)**

-   Config-driven data ingestion and schema mapping

-   Docker packaging and single-command deployment

-   CLI subcommand implementation (run, status, inspect, chat, init)

-   Rich terminal output with colour-coded verdicts and fold summaries

**Phase 2 --- Theory layer**

-   Hypothesis graph: linking features to the economic claims they
    encode

-   Regime change detection and hypothesis falsification alerts

-   Saturation signal: monitoring incremental OOS improvement across
    cycles

-   Reasoning provenance: tracing why the current model holds its
    beliefs

**Phase 3 --- Extensibility**

-   Python API for notebook integration (import volare)

-   Multi-pair support with cross-pair signal comparison

-   Optional anonymised concept-sharing layer for cross-researcher
    signal evolution

**7. Conclusion**

The ambition of Volare is not to make feature engineering faster. It is
to make FX volatility research cumulative --- to build a tool where each
session is more informed than the last, where the agent\'s understanding
of a market deepens over time, and where the researcher is told the
truth about what the data can and cannot support.

The combination of a persistent principles register, a falsifiable
theory architecture, autonomous saturation detection, and a CLI-first
interface for quant researchers represents a meaningfully new approach
to systematic market research. The system is not a wrapper around
existing AI capabilities --- it is a research methodology implemented in
software.

***\"A research partner that never forgets what it tried and why it
failed.\"***
