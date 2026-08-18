#!/usr/bin/env python3
"""Run the Notebook 05 shortcut-or-shelter studies from versioned configs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from rllab.cli import build_preflight_report
from rllab.experiments import Experiment

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = {
    "recoverable": ROOT / "configs" / "shortcut_or_shelter_recoverable.yaml",
    "lethal": ROOT / "configs" / "shortcut_or_shelter_lethal.yaml",
    "annealed": ROOT / "configs" / "shortcut_or_shelter_annealed.yaml",
}


def _quick_experiment(experiment: Experiment) -> Experiment:
    """Return a bounded smoke version without changing the checked-in config."""

    config = experiment.config
    policy_evaluation = replace(
        config.policy_evaluation,
        interval_episodes=1_000_000,
        episodes_per_checkpoint=3,
        include_initial=False,
        include_final=True,
    )
    config = replace(
        config,
        seeds=config.seeds[:2],
        total_interaction_steps=1_000,
        snapshot_interval=1_000_000,
        snapshot_step_interval=250,
        policy_evaluation=policy_evaluation,
    ).with_parallel_workers(1)
    return Experiment(config)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the recoverable, lethal, or annealed shortcut-or-shelter "
            "study used by Notebook 05."
        )
    )
    parser.add_argument(
        "--study",
        choices=("all", *CONFIGS),
        default="all",
        help="Study to run; the default preflights all three in sequence.",
    )
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use two seeds and 1,000 interactions per trial as a smoke run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print all selected resource estimates without running trials.",
    )
    parser.add_argument(
        "--allow-large-run",
        action="store_true",
        help="Acknowledge selected studies that exceed the preflight safety limits.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress display.",
    )
    args = parser.parse_args(argv)

    selected = tuple(CONFIGS) if args.study == "all" else (args.study,)
    experiments: list[tuple[str, Path, Experiment, dict[str, object]]] = []
    for study in selected:
        path = CONFIGS[study]
        experiment = Experiment.from_yaml(path)
        if args.quick:
            experiment = _quick_experiment(experiment)
        elif args.workers is not None:
            experiment = Experiment(experiment.config.with_parallel_workers(args.workers))
        report = build_preflight_report(experiment.config)
        experiments.append((study, path, experiment, report))

    for study, path, _experiment, report in experiments:
        print(f"[{study}] {path}")
        print(json.dumps(report, indent=2, sort_keys=True))

    oversized = [
        study
        for study, _path, _experiment, report in experiments
        if bool(report["requires_large_run_override"])
    ]
    if oversized and not args.dry_run and not args.allow_large_run:
        parser.error(
            "selected studies exceed the preflight limits: "
            + ", ".join(oversized)
            + "; review the estimates and pass --allow-large-run"
        )
    if args.dry_run:
        return

    for study, _path, experiment, _report in experiments:
        result = experiment.run(progress=not args.no_progress)
        print(f"[{study}] completed {result.metadata['trial_count']} trials")
        print(f"[{study}] results: {result.run_directory}")


if __name__ == "__main__":
    main()
