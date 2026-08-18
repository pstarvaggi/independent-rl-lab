"""Command-line entry points for experiments and result summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from rllab.evaluation import final_performance
from rllab.experiments import Experiment, ExperimentConfig
from rllab.experiments.persistence import read_table
from rllab.experiments.preflight import estimate_run
from rllab.metrics import UnsafeAggregationError, distribution_summary

DEFAULT_MAXIMUM_TRANSITIONS = 10_000_000
DEFAULT_MAXIMUM_RETAINED_STEP_ROWS = 5_000_000

app = typer.Typer(
    name="rl-lab",
    help="Run reproducible reinforcement-learning experiments and inspect their results.",
    no_args_is_help=True,
)


def build_preflight_report(
    config: ExperimentConfig,
    *,
    maximum_transitions: int = DEFAULT_MAXIMUM_TRANSITIONS,
    maximum_retained_step_rows: int = DEFAULT_MAXIMUM_RETAINED_STEP_ROWS,
) -> dict[str, Any]:
    """Return a JSON-ready resource estimate and explicit large-run risks."""

    if maximum_transitions < 1 or maximum_retained_step_rows < 1:
        raise ValueError("preflight limits must be positive")
    estimate = estimate_run(config)
    risks: list[str] = []
    if estimate.maximum_transition_rows is None:
        risks.append("the maximum number of environment transitions is unbounded")
    elif estimate.maximum_transition_rows > maximum_transitions:
        risks.append(
            f"maximum transitions {estimate.maximum_transition_rows:,} exceed "
            f"the safety limit {maximum_transitions:,}"
        )
    if estimate.estimated_retained_step_rows is None:
        risks.append("retained step-table size cannot be bounded")
    elif estimate.estimated_retained_step_rows > maximum_retained_step_rows:
        risks.append(
            f"retained step rows {estimate.estimated_retained_step_rows:,} exceed "
            f"the safety limit {maximum_retained_step_rows:,}"
        )
    return {
        **estimate.as_dict(),
        "safety_limits": {
            "maximum_transitions": maximum_transitions,
            "maximum_retained_step_rows": maximum_retained_step_rows,
        },
        "risk_reasons": risks,
        "requires_large_run_override": bool(risks),
    }


def _group_key(columns: tuple[str, ...], keys: Any) -> dict[str, Any]:
    values = keys if isinstance(keys, tuple) else (keys,)
    return dict(zip(columns, values, strict=True))


@app.command("run")
def run_experiment(
    config: Annotated[
        Path,
        typer.Argument(
            help="YAML experiment configuration.", exists=True, dir_okay=False, readable=True
        ),
    ],
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Override the results root.")
    ] = None,
    workers: Annotated[
        int | None, typer.Option("--workers", "-j", min=1, help="Override process count.")
    ] = None,
    resume: Annotated[
        Path | None,
        typer.Option(
            "--resume",
            help="Resume pending/failed trials in an existing Protocol-v2 run directory.",
            exists=True,
            file_okay=False,
            readable=True,
        ),
    ] = None,
    no_progress: Annotated[
        bool, typer.Option("--no-progress", help="Disable progress display.")
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Expand the configuration and report resource bounds without executing it.",
        ),
    ] = False,
    allow_large_run: Annotated[
        bool,
        typer.Option(
            "--allow-large-run",
            help="Acknowledge and execute a run that exceeds the preflight safety limits.",
        ),
    ] = False,
) -> None:
    """Execute every agent/environment/seed/sweep combination in CONFIG."""

    experiment = Experiment.from_yaml(config)
    updated = experiment.config
    if resume is not None and output is not None:
        raise typer.BadParameter("--output cannot be combined with --resume")
    if output is not None:
        updated = updated.with_output_dir(output.resolve())
    if workers is not None:
        updated = updated.with_parallel_workers(workers)
    report = build_preflight_report(updated)
    typer.echo(json.dumps(report, indent=2, sort_keys=True))
    if dry_run:
        return
    if report["requires_large_run_override"] and not allow_large_run:
        typer.echo(
            "Run blocked by preflight safety limits. Review the estimate and pass "
            "--allow-large-run to acknowledge it.",
            err=True,
        )
        raise typer.Exit(code=2)
    run_options: dict[str, Any] = {"progress": not no_progress}
    if resume is not None:
        run_options["resume_from"] = resume
    result = Experiment(updated).run(**run_options)
    typer.echo(f"Completed {result.metadata['trial_count']} trials.")
    typer.echo(str(result.run_directory))


@app.command("summarize")
def summarize_results(
    results: Annotated[
        Path,
        typer.Argument(
            help="Run directory or episodes CSV/Parquet table.", exists=True, readable=True
        ),
    ],
    metric: Annotated[
        str, typer.Option("--metric", "-m", help="Episode metric to summarize.")
    ] = "episode_return",
    last_episodes: Annotated[
        int, typer.Option("--last", min=1, help="Final episodes per run.")
    ] = 100,
    group: Annotated[
        list[str] | None,
        typer.Option(
            "--group",
            "-g",
            help="Comparison column; repeat for every varying condition factor.",
        ),
    ] = None,
) -> None:
    """Print distribution-aware final-window summaries across independent seeds."""

    frame = read_table(results)
    if metric not in frame:
        raise typer.BadParameter(
            f"Metric {metric!r} is absent; choices include {sorted(frame.columns)}"
        )
    groups = tuple(dict.fromkeys(group or ["agent"]))
    missing_groups = set(groups) - set(frame.columns)
    if missing_groups:
        raise typer.BadParameter(f"Group columns are absent: {sorted(missing_groups)}")
    try:
        per_trial = final_performance(
            frame,
            metrics=(metric,),
            last_episodes=last_episodes,
            groups=groups,
        )
    except UnsafeAggregationError as error:
        raise typer.BadParameter(str(error), param_hint="--group") from error

    grouper: str | list[str] = groups[0] if len(groups) == 1 else list(groups)
    rows: list[dict[str, Any]] = []
    for keys, sample in per_trial.groupby(grouper, dropna=False, sort=True):
        row = _group_key(groups, keys)
        row.update(distribution_summary(sample[metric]))
        row["n_units"] = int(sample["trial_id"].nunique()) if "trial_id" in sample else len(sample)
        row["n_seeds"] = int(sample["seed"].nunique()) if "seed" in sample else None
        rows.append(row)
    payload = {
        "metric": metric,
        "last_episodes": last_episodes,
        "groups": list(groups),
        "summaries": rows,
    }
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    app()
