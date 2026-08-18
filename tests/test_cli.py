from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import yaml
from typer.testing import CliRunner

from rllab.cli import app
from rllab.experiments import Experiment

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "stochastic_maze" / "run_stochasticity_sweep.py"


def _write_config(
    path: Path,
    *,
    episodes: int = 2,
    seeds: list[int] | None = None,
    horizon: int = 5,
) -> Path:
    value: dict[str, Any] = {
        "config_schema_version": 2,
        "experiment": {
            "name": "cli-test",
            "episodes": episodes,
            "seeds": seeds or [0, 1, 2],
            "environment": {
                "name": "tiny",
                "kind": "stochastic_maze",
                "parameters": {
                    "shape": [1, 2],
                    "start": [0, 0],
                    "goals": [[0, 1]],
                    "max_episode_steps": horizon,
                },
            },
            "agent": {"name": "q", "kind": "q_learning"},
        },
        "execution": {"parallel_workers": 1},
        "artifacts": {
            "output_dir": "results",
            "step_retention": {
                "mode": "none",
                "keep_terminal": False,
                "keep_events": False,
            },
        },
    }
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def _episode_table() -> pd.DataFrame:
    rows = []
    for reliability in (0.8, 0.9):
        for seed in (0, 1):
            for episode in (0, 1):
                rows.append(
                    {
                        "trial_id": f"p{reliability}-{seed}",
                        "condition_id": f"q-p{reliability}",
                        "agent": "q",
                        "seed": seed,
                        "episode": episode,
                        "episode_return": 10 * reliability + seed + episode,
                        "sweep_environment_action_reliability": reliability,
                    }
                )
    return pd.DataFrame(rows)


def test_run_dry_run_applies_v2_mutators_without_execution(
    tmp_path: Path, monkeypatch: Any
) -> None:
    config = _write_config(tmp_path / "experiment.yaml")
    output = tmp_path / "overridden-results"
    executed = False

    def fail_if_run(*_args: Any, **_kwargs: Any) -> None:
        nonlocal executed
        executed = True
        raise AssertionError("dry-run executed the experiment")

    monkeypatch.setattr(Experiment, "run", fail_if_run)
    result = CliRunner().invoke(
        app,
        ["run", str(config), "--dry-run", "--workers", "3", "--output", str(output)],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["trial_count"] == 3
    assert report["workers"] == 3
    assert report["maximum_transition_rows"] == 30
    assert not report["requires_large_run_override"]
    assert not executed
    assert not output.exists()


def test_run_blocks_large_workload_until_explicitly_acknowledged(
    tmp_path: Path, monkeypatch: Any
) -> None:
    config = _write_config(
        tmp_path / "large.yaml",
        episodes=40_000,
        seeds=[0],
        horizon=300,
    )
    calls = 0

    def fake_run(self: Experiment, *, persist: bool = True, progress: bool = True) -> Any:
        nonlocal calls
        calls += 1
        return SimpleNamespace(
            metadata={"trial_count": len(self.config.trials())},
            run_directory=tmp_path / "fake-run",
        )

    monkeypatch.setattr(Experiment, "run", fake_run)
    runner = CliRunner()
    blocked = runner.invoke(app, ["run", str(config), "--no-progress"])
    assert blocked.exit_code == 2
    assert "Run blocked by preflight safety limits" in blocked.output
    assert calls == 0

    allowed = runner.invoke(
        app,
        ["run", str(config), "--no-progress", "--allow-large-run"],
    )
    assert allowed.exit_code == 0, allowed.output
    assert "Completed 1 trials" in allowed.output
    assert calls == 1


def test_run_forwards_resume_directory_without_changing_output(
    tmp_path: Path, monkeypatch: Any
) -> None:
    config = _write_config(tmp_path / "resume.yaml", seeds=[0])
    run_directory = tmp_path / "existing-run"
    run_directory.mkdir()
    received: dict[str, Any] = {}

    def fake_run(self: Experiment, **kwargs: Any) -> Any:
        received.update(kwargs)
        return SimpleNamespace(
            metadata={"trial_count": len(self.config.trials())},
            run_directory=run_directory,
        )

    monkeypatch.setattr(Experiment, "run", fake_run)
    result = CliRunner().invoke(
        app,
        ["run", str(config), "--resume", str(run_directory), "--no-progress"],
    )
    assert result.exit_code == 0, result.output
    assert received == {"progress": False, "resume_from": run_directory}


def test_summarize_fails_closed_when_default_group_hides_conditions(tmp_path: Path) -> None:
    path = tmp_path / "episodes.csv"
    _episode_table().to_csv(path, index=False)
    result = CliRunner().invoke(app, ["summarize", str(path), "--last", "1"])
    assert result.exit_code == 2
    assert "multiple experimental conditions" in result.output
    assert "sweep_environment_action_reliability" in result.output


def test_summarize_accepts_all_varying_condition_factors(tmp_path: Path) -> None:
    path = tmp_path / "episodes.csv"
    _episode_table().to_csv(path, index=False)
    result = CliRunner().invoke(
        app,
        [
            "summarize",
            str(path),
            "--last",
            "1",
            "--group",
            "agent",
            "--group",
            "sweep_environment_action_reliability",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["groups"] == ["agent", "sweep_environment_action_reliability"]
    assert len(payload["summaries"]) == 2
    assert [row["n_units"] for row in payload["summaries"]] == [2, 2]
    assert [row["n_seeds"] for row in payload["summaries"]] == [2, 2]


def test_versioned_stochasticity_script_supports_quick_dry_run() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--quick", "--dry-run"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    assert report["trial_count"] == 12
    assert report["training_episode_count"] == 360
    assert report["workers"] == 1
    assert not report["requires_large_run_override"]
