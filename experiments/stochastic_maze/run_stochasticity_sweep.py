#!/usr/bin/env python3
"""Run the first stochastic-maze research experiment from a versioned config."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from rllab.cli import build_preflight_report
from rllab.experiments import Experiment

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "stochasticity_sweep.yaml"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Sweep movement reliability and measure tabular convergence to exact Q*."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume pending or failed trials in an existing Protocol-v2 run directory.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run two seeds and 30 episodes as an installation smoke test.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the expanded resource estimate without running trials.",
    )
    parser.add_argument(
        "--allow-large-run",
        action="store_true",
        help="Acknowledge a run that exceeds the preflight safety limits.",
    )
    args = parser.parse_args(argv)

    experiment = Experiment.from_yaml(args.config)
    config = experiment.config
    if args.workers is not None:
        config = config.with_parallel_workers(args.workers)
    if args.quick:
        config = replace(config, seeds=config.seeds[:2], episodes=30).with_parallel_workers(1)
    report = build_preflight_report(config)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.dry_run:
        return
    if report["requires_large_run_override"] and not args.allow_large_run:
        parser.error(
            "run exceeds preflight safety limits; review the estimate and pass "
            "--allow-large-run to acknowledge it"
        )
    result = Experiment(config).run(resume_from=args.resume)
    print(f"Completed {result.metadata['trial_count']} trials")
    print(f"Results: {result.run_directory}")


if __name__ == "__main__":
    main()
