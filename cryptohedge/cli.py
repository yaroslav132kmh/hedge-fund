"""Command-line entry point and programmatic runner for the hedging pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from cryptohedge.agents import build_pipeline
from cryptohedge.core.config import load_config
from cryptohedge.core.context import AgentContext
from cryptohedge.core.orchestrator import RunReport


def run_pipeline(
    config_dir: str | Path = "config",
    root: str | Path = ".",
    overrides: Optional[Dict[str, Any]] = None,
    fail_fast: bool = True,
) -> Tuple[AgentContext, RunReport]:
    """Build the context, wire the agents and run the full pipeline.

    Returns the populated :class:`AgentContext` (with all blackboard artifacts)
    and the :class:`RunReport`. Designed to be called from the notebook too.
    """
    config = load_config(config_dir, overrides=overrides)
    context = AgentContext(config, root=root)
    orchestrator = build_pipeline(context, fail_fast=fail_fast)
    report = orchestrator.run()
    return context, report


def _print_summary(report: RunReport) -> None:
    print("\n================ PIPELINE SUMMARY ================")
    for r in report.results:
        flag = "OK " if r.success else "FAIL"
        skip = " (restored)" if r.skipped else ""
        print(f"  [{flag}] {r.agent:<24} {r.duration_s:6.2f}s{skip}"
              + (f"  error={r.error}" if not r.success else ""))
    print(f"  total: {report.total_seconds():.2f}s | success={report.success}")
    print("=================================================\n")


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="CryptoHedge multi-agent hedging pipeline")
    parser.add_argument("--config", default="config", help="configuration directory")
    parser.add_argument("--root", default=".", help="project root for artifacts")
    parser.add_argument("--provider", default=None, help="override data provider")
    parser.add_argument("--reset", action="store_true", help="ignore checkpoints and rerun all stages")
    parser.add_argument("--no-fail-fast", action="store_true", help="continue after a stage fails")
    args = parser.parse_args(argv)

    overrides: Dict[str, Any] = {}
    if args.provider:
        overrides["data"] = {"provider": args.provider}
    if args.reset:
        overrides.setdefault("runtime", {})["resume"] = False

    context, report = run_pipeline(args.config, args.root, overrides, fail_fast=not args.no_fail_fast)
    _print_summary(report)
    if context.has("dashboard_path"):
        print(f"Dashboard: {context.get('dashboard_path')}")
    return 0 if report.success else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
