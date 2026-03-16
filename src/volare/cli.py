"""
cli.py
Volare CLI — terminal-first interface for the FX volatility research agent.

Commands:
  volare run --cycles N [--hint TEXT] [--data-dir PATH]
  volare status
  volare inspect <name>
  volare chat ["<question>"]
  volare init
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

# ── ANSI colour helpers ────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[32m"
RED    = "\033[31m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
DIM    = "\033[2m"

def green(s: str) -> str:  return f"{GREEN}{s}{RESET}"
def red(s: str) -> str:    return f"{RED}{s}{RESET}"
def yellow(s: str) -> str: return f"{YELLOW}{s}{RESET}"
def cyan(s: str) -> str:   return f"{CYAN}{s}{RESET}"
def bold(s: str) -> str:   return f"{BOLD}{s}{RESET}"
def dim(s: str) -> str:    return f"{DIM}{s}{RESET}"

VERDICT_COLOUR = {
    "promoted": green,
    "rejected": red,
    "modified": yellow,
}

def colour_verdict(verdict: str) -> str:
    fn = VERDICT_COLOUR.get(verdict, lambda x: x)
    return fn(verdict.upper())


# ── Repo root resolution ───────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "volare_config.yaml"


def _ensure_repo_on_path() -> None:
    """Ensure the repo root is on sys.path so agent modules are importable."""
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


# ── Config ────────────────────────────────────────────────────────────────────

def load_cli_config() -> dict:
    """Load volare_config.yaml if it exists, otherwise return defaults."""
    if CONFIG_PATH.exists():
        import yaml
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    return {}


def save_cli_config(cfg: dict) -> None:
    import yaml
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)


def default_data_files(data_dir: Path, pairs: list) -> dict:
    """Build pair -> file mapping using the questdb naming convention."""
    return {
        pair: data_dir / f"questdb-{pair.lower()}.csv"
        for pair in pairs
    }


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(data_dir: Optional[str] = None) -> dict:
    """Load OHLCV CSVs for all configured pairs. Returns {pair: DataFrame}."""
    import pandas as pd

    _ensure_repo_on_path()
    from src.agent.context import load_context

    context = load_context()
    pairs = context["data"]["pairs"]

    cfg = load_cli_config()
    resolved_dir = Path(data_dir) if data_dir else Path(cfg.get("data_dir", str(REPO_ROOT / "data")))
    file_map = cfg.get("data_files", {})

    data = {}
    for pair in pairs:
        path_str = file_map.get(pair)
        if path_str:
            csv_path = Path(path_str)
        else:
            csv_path = resolved_dir / f"questdb-{pair.lower()}.csv"

        if not csv_path.exists():
            print(red(f"Data file not found for {pair}: {csv_path}"))
            print(dim(f"  Run 'volare init' to configure data paths."))
            sys.exit(1)

        df = pd.read_csv(csv_path, parse_dates=["timestamp"], index_col="timestamp")
        data[pair] = df
        print(dim(f"  Loaded {pair}: {len(df):,} rows ({csv_path.name})"))

    return data


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_run(args: argparse.Namespace) -> None:
    """Run N agent cycles autonomously."""
    _ensure_repo_on_path()
    from src.agent.loop import run_cycle

    cycles = args.cycles
    hint = args.hint

    print(bold(f"\nVolare — running {cycles} cycle{'s' if cycles > 1 else ''}"))
    if hint:
        print(dim(f"  Hint: {hint}"))
    print()

    print("Loading data...")
    data = load_data(args.data_dir)
    print()

    for i in range(cycles):
        if cycles > 1:
            print(bold(f"── Cycle {i + 1}/{cycles} ──────────────────────────────"))

        entry = run_cycle(data, user_hint=hint)

        verdict = entry.get("verdict", "unknown")
        name = entry.get("name", "unknown")
        summary = entry.get("summary", "")

        print()
        print(f"  {bold(name)}  [{colour_verdict(verdict)}]")
        print(f"  {summary}")
        print()


def cmd_status(args: argparse.Namespace) -> None:
    """Show current research state."""
    _ensure_repo_on_path()
    from src.agent.registry import load_registry
    from src.agent.active_features import load_active_features

    registry = load_registry()
    active = load_active_features()

    total = len(registry)
    promoted = sum(1 for e in registry if e.get("verdict") == "promoted")
    rejected = sum(1 for e in registry if e.get("verdict") == "rejected")
    modified = sum(1 for e in registry if e.get("verdict") == "modified")

    print(bold("\nVolare Status"))
    print("─" * 40)
    print(f"  Cycles completed : {total}")
    print(f"  Promoted         : {green(str(promoted))}")
    print(f"  Rejected         : {red(str(rejected))}")
    print(f"  Modified         : {yellow(str(modified))}")
    print()

    print(bold("Active Features") + f"  ({len(active)})")
    print("─" * 40)
    if active:
        for f in active:
            print(f"  {green('✓')}  {f['name']}")
    else:
        print(dim("  No features promoted yet."))
    print()

    if registry:
        print(bold("Recent Cycles"))
        print("─" * 40)
        recent = registry[-5:]
        for e in reversed(recent):
            verdict = e.get("verdict", "unknown")
            ts = e.get("timestamp", "")
            name = e.get("name", "?")
            print(f"  {colour_verdict(verdict):20s}  {dim(ts)}  {name}")
        print()

        # Baseline performance trend from last few promoted
        promoted_entries = [e for e in registry if e.get("verdict") == "promoted"]
        if len(promoted_entries) >= 2:
            improvements = []
            for e in promoted_entries[-5:]:
                agg = e.get("test_results", {}).get("aggregate", {})
                val = agg.get("mean_improvement_vs_lgbm_pct")
                if val is not None:
                    improvements.append(val)
            if improvements:
                print(bold("OOS Improvement (promoted, recent)"))
                print("─" * 40)
                for i, val in enumerate(improvements):
                    bar = "█" * max(0, int(abs(float(val)) * 2))
                    colour = green if float(val) > 0 else red
                    print(f"  {colour(f'{float(val):+.2f}%'):20s}  {colour(bar)}")
                print()


def cmd_inspect(args: argparse.Namespace) -> None:
    """Show full detail for a specific feature."""
    _ensure_repo_on_path()
    from src.agent.registry import load_registry

    name = args.name
    registry = load_registry()

    # Match by name (partial, case-insensitive) or exact ID
    matches = [
        e for e in registry
        if name.lower() in e.get("name", "").lower()
        or e.get("id", "") == name
    ]

    if not matches:
        print(red(f"No feature found matching '{name}'."))
        print(dim("  Try 'volare status' to see feature names."))
        sys.exit(1)

    if len(matches) > 1:
        print(yellow(f"Multiple matches for '{name}':"))
        for m in matches:
            print(f"  {m['id']}  {m['name']}")
        print(dim("  Use the exact ID or a more specific name."))
        sys.exit(1)

    e = matches[0]
    verdict = e.get("verdict", "unknown")

    print()
    print(bold(e["name"]) + f"  [{colour_verdict(verdict)}]  " + dim(e.get("id", "")))
    print(dim(e.get("timestamp", "")))
    print("─" * 60)

    print()
    print(bold("Hypothesis"))
    print(f"  {e.get('hypothesis', '')}")

    print()
    print(bold("Construction"))
    print(f"  {e.get('construction', '')}")

    print()
    print(bold("Rejection Criteria"))
    print(f"  {e.get('rejection_criteria', '')}")

    # Test results summary
    agg = e.get("test_results", {}).get("aggregate", {})
    if agg:
        print()
        print(bold("Test Results"))
        print("─" * 40)
        lgbm_imp = agg.get("mean_improvement_vs_lgbm_pct")
        naive_imp = agg.get("mean_improvement_vs_naive_pct")
        importance = agg.get("mean_importance_drift_pct")
        decay = agg.get("any_monotonic_decay")

        def _fmt(val: Optional[float]) -> str:
            if val is None:
                return dim("n/a")
            fval = float(val)
            return green(f"{fval:+.3f}%") if fval > 0 else red(f"{fval:+.3f}%")

        print(f"  vs LightGBM baseline : {_fmt(lgbm_imp)}")
        print(f"  vs naive baseline    : {_fmt(naive_imp)}")
        print(f"  Mean importance      : {_fmt(importance)}")
        print(f"  Monotonic decay      : {yellow('yes') if decay else green('no')}")

        per_pair = e.get("test_results", {}).get("per_pair", {})
        if per_pair:
            print()
            for pair, results in per_pair.items():
                if "error" in results:
                    print(f"  {pair}: {red(results['error'])}")
                    continue
                print(f"  {cyan(pair)}: {_fmt(results.get('overall_rmse_improvement_pct'))}"
                      f"  ({results.get('n_folds_completed', '?')} folds)")
                for fold in results.get("folds", []):
                    fi = fold.get("rmse_improvement_pct")
                    print(f"    fold {fold['fold']}: {_fmt(fi)}"
                          f"  corr={fold.get('feature_target_correlation', 'n/a')}")

    triggered = e.get("triggered_principles", [])
    if triggered:
        print()
        print(bold("Triggered Principles"))
        print(f"  {', '.join(triggered)}")

    print()
    print(bold("Summary"))
    print(f"  {e.get('summary', '')}")

    print()
    print(bold("Next Action"))
    print(f"  {e.get('next_action', '')}")

    if args.code:
        code = e.get("code") or (
            # code may be nested under the feature sub-dict in older entries
            {}
        )
        print()
        print(bold("Code"))
        print("─" * 60)
        print(code if code else dim("  (code not stored in this entry)"))

    print()


def cmd_chat(args: argparse.Namespace) -> None:
    """Grounded research chat — single-shot or interactive."""
    _ensure_repo_on_path()
    import anthropic
    from src.agent.registry import load_registry
    from src.agent.context import load_context, load_principles
    from src.agent.active_features import load_active_features

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        _try_load_dotenv()
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(red("ANTHROPIC_API_KEY not set. Run 'volare init' or export it."))
        sys.exit(1)

    registry = load_registry()
    context = load_context()
    active = load_active_features()

    system_prompt = _build_chat_system_prompt(registry, context, active)
    client = anthropic.Anthropic(api_key=api_key)

    if args.question:
        # Single-shot mode
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=600,
            system=system_prompt,
            messages=[{"role": "user", "content": args.question}]
        )
        print()
        print(response.content[0].text)
        print()
    else:
        # Interactive mode
        history = []
        total = len(registry)
        promoted = sum(1 for e in registry if e.get("verdict") == "promoted")
        print()
        print(bold("Volare Research Chat"))
        print(dim(f"  {total} features in registry — {promoted} promoted."))
        print(dim("  Type your question, or 'exit' / Ctrl-C to quit."))
        print()

        while True:
            try:
                user_input = input(cyan("you> ")).strip()
            except (KeyboardInterrupt, EOFError):
                print()
                break

            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                break

            history.append({"role": "user", "content": user_input})

            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=600,
                system=system_prompt,
                messages=history
            )
            reply = response.content[0].text
            history.append({"role": "assistant", "content": reply})

            print()
            print(bold("agent>"))
            print(reply)
            print()


def _build_chat_system_prompt(registry: list, context: dict, active: list) -> str:
    registry_summary = json.dumps([
        {
            "name": e["name"],
            "verdict": e["verdict"],
            "summary": e.get("summary", "")[:150]
        }
        for e in registry
    ], indent=2)

    active_names = ", ".join(f["name"] for f in active) if active else "none"

    return f"""You are a systematic FX volatility research agent assistant.
You help a quantitative researcher guide the direction of feature research.
You have full knowledge of what has been tested, the data context, and the research principles.

You are concise, direct, and research-focused. You do not pad responses.
You speak like a senior quant researcher, not a customer service bot.

ACTIVE FEATURES: {active_names}

REGISTRY SUMMARY:
{registry_summary}

DATA CONTEXT:
Pairs: {', '.join(context['data']['pairs'])}
Frequency: {context['data']['frequency']}
Known regimes: {', '.join(r['name'] for r in context.get('known_regimes', []))}
""".strip()


def cmd_init(args: argparse.Namespace) -> None:
    """Interactive setup — configure data paths and API key."""
    print(bold("\nVolare Init"))
    print("─" * 40)
    print("Configure your data source and API key.\n")

    cfg = load_cli_config()

    # API key
    current_key = os.environ.get("ANTHROPIC_API_KEY", "")
    masked = f"...{current_key[-6:]}" if len(current_key) > 6 else ("set" if current_key else "not set")
    print(f"  ANTHROPIC_API_KEY [{dim(masked)}]")
    new_key = input("  New value (leave blank to keep): ").strip()
    if new_key:
        _write_dotenv("ANTHROPIC_API_KEY", new_key)
        print(green("  ✓ Written to .env"))

    print()

    # Data directory
    current_dir = cfg.get("data_dir", str(REPO_ROOT / "data"))
    print(f"  Data directory [{dim(current_dir)}]")
    new_dir = input("  New value (leave blank to keep): ").strip()
    if new_dir:
        cfg["data_dir"] = new_dir
        print(green(f"  ✓ Set to {new_dir}"))
    else:
        cfg.setdefault("data_dir", current_dir)

    print()

    # Per-pair file overrides
    _ensure_repo_on_path()
    from src.agent.context import load_context
    context = load_context()
    pairs = context["data"]["pairs"]
    data_dir = Path(cfg["data_dir"])

    print("  Per-pair CSV file paths (leave blank for default questdb-<pair>.csv):")
    file_map = cfg.get("data_files", {})
    for pair in pairs:
        default = str(data_dir / f"questdb-{pair.lower()}.csv")
        current = file_map.get(pair, default)
        val = input(f"    {pair} [{dim(current)}]: ").strip()
        if val:
            file_map[pair] = val

    if file_map:
        cfg["data_files"] = file_map

    save_cli_config(cfg)
    print()
    print(green("✓ Config saved to volare_config.yaml"))
    print(dim("  Run 'volare status' to verify the setup."))
    print()


def _write_dotenv(key: str, value: str) -> None:
    """Write or update a key in .env at the repo root."""
    env_path = REPO_ROOT / ".env"
    lines = []
    found = False
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.startswith(f"{key}="):
                    lines.append(f"{key}={value}\n")
                    found = True
                else:
                    lines.append(line)
    if not found:
        lines.append(f"{key}={value}\n")
    with open(env_path, "w") as f:
        f.writelines(lines)


def _try_load_dotenv() -> None:
    """Best-effort load of .env without requiring python-dotenv."""
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    _try_load_dotenv()

    parser = argparse.ArgumentParser(
        prog="volare",
        description="Volare — FX Volatility Research Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
commands:
  run      Run N agent cycles autonomously
  status   Show active features and registry state
  inspect  Show full detail for a specific feature
  chat     Grounded research chat (interactive or single-shot)
  init     Configure data paths and API key
""",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # run
    p_run = sub.add_parser("run", help="Run N agent cycles")
    p_run.add_argument("--cycles", "-n", type=int, default=1, help="Number of cycles (default: 1)")
    p_run.add_argument("--hint", "-H", type=str, default=None, help="Research direction hint for the agent")
    p_run.add_argument("--data-dir", type=str, default=None, help="Override data directory")

    # status
    sub.add_parser("status", help="Show registry state and active features")

    # inspect
    p_inspect = sub.add_parser("inspect", help="Show detail for a specific feature")
    p_inspect.add_argument("name", help="Feature name (partial match) or ID")
    p_inspect.add_argument("--code", action="store_true", help="Also print the feature code")

    # chat
    p_chat = sub.add_parser("chat", help="Grounded research chat")
    p_chat.add_argument("question", nargs="?", default=None, help="Question (omit for interactive mode)")

    # init
    sub.add_parser("init", help="Configure data paths and API key")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "inspect":
        cmd_inspect(args)
    elif args.command == "chat":
        cmd_chat(args)
    elif args.command == "init":
        cmd_init(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
