from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest

from rllab.experiments.artifacts import RunStore
from rllab.experiments.config import (
    AgentSpec,
    ArtifactSpec,
    EnvironmentSpec,
    ExecutionSpec,
    ExperimentConfig,
)
from rllab.experiments.runner import Experiment
from rllab.experiments.schema import (
    ARTIFACT_SCHEMA_VERSION,
    CONFIG_SCHEMA_VERSION,
    TABLE_SCHEMA_VERSION,
)


class _Discrete:
    def __init__(self, n: int) -> None:
        self.n = n


class _FiniteHorizonEnv:
    observation_space = _Discrete(2)
    action_space = _Discrete(1)

    def __init__(self, parameters: Mapping[str, Any]) -> None:
        self.horizon = int(parameters.get("horizon", 3))
        self.elapsed = 0

    def reset(self, seed: int | None = None) -> tuple[int, dict[str, Any]]:
        del seed
        self.elapsed = 0
        return 0, {"latent_state_index": 1}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        assert action == 0
        self.elapsed += 1
        terminated = self.elapsed == self.horizon
        return (
            self.elapsed % 2,
            1.0,
            terminated,
            False,
            {
                "latent_state_index": self.elapsed % 2,
                "success": terminated,
            },
        )

    def close(self) -> None:
        return None


class _FailOnceEnv(_FiniteHorizonEnv):
    fail_next_attempt: ClassVar[bool] = True

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        if self.fail_next_attempt and self.elapsed == 1:
            type(self).fail_next_attempt = False
            raise RuntimeError("injected first-attempt failure")
        return super().step(action)


class _ConstantAgent:
    def __init__(self) -> None:
        self.q_values = np.zeros((2, 1), dtype=float)

    def reset(self, seed: int | None = None) -> None:
        del seed
        self.q_values.fill(0.0)

    def act(self, state: int, training: bool = True) -> int:
        del state, training
        return 0

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        terminated: bool,
    ) -> dict[str, float]:
        del state, action, reward, next_state, terminated
        return {"td_error": 0.0}


def _agent_factory(spec: AgentSpec, env: Any, seed: int) -> _ConstantAgent:
    del spec, env, seed
    return _ConstantAgent()


def _config(
    output_dir: Path,
    *,
    episodes: int,
    flush_rows: int,
    failure_policy: str = "fail_fast",
) -> ExperimentConfig:
    return ExperimentConfig(
        name="protocol-v2-runner-persistence",
        episodes=episodes,
        seeds=(41,),
        environments=(
            EnvironmentSpec(name="finite-horizon", kind="test", parameters={"horizon": 3}),
        ),
        agents=(AgentSpec(name="constant", kind="test"),),
        snapshot_interval=2,
        exact_reference=False,
        execution=ExecutionSpec(
            parallel_workers=1,
            failure_policy=failure_policy,  # type: ignore[arg-type]
        ),
        artifacts=ArtifactSpec(
            output_dir=output_dir,
            table_format="csv",
            flush_rows=flush_rows,
            save_q_snapshots=False,
        ),
    )


def test_runner_flushes_bounded_parts_and_result_reads_lazily(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flush_rows = 2
    result = Experiment(
        _config(tmp_path, episodes=3, flush_rows=flush_rows),
        environment_factory=_FiniteHorizonEnv,
        agent_factory=_agent_factory,
    ).run(persist=True, progress=False)
    assert result.run_directory is not None

    store = RunStore.open(result.run_directory)
    for table, expected_rows in {"steps": 9, "training_episodes": 3}.items():
        references = store.artifact_references(table)
        assert len(references) > 1
        assert sum(reference.rows for reference in references) == expected_rows
        assert all(0 < reference.rows <= flush_rows for reference in references)
        assert [reference.part for reference in references] == list(range(len(references)))

    original_read_table = RunStore.read_table
    materialized: list[str] = []

    def tracked_read_table(
        store: RunStore,
        table: str = "episodes",
        **kwargs: Any,
    ) -> pd.DataFrame:
        materialized.append(table)
        return original_read_table(store, table, **kwargs)

    monkeypatch.setattr(RunStore, "read_table", tracked_read_table)

    assert result._tables == {}
    batches = list(result.iter_table("steps", batch_size=1))
    assert len(batches) == 9
    assert all(len(batch) == 1 for batch in batches)
    assert result._tables == {}
    assert materialized == []

    steps = result.steps
    assert materialized == ["steps"]
    assert result.steps is steps
    assert materialized == ["steps"]
    assert len(steps) == 9
    assert not steps[["trial_id", "episode", "step"]].duplicated().any()


def test_runner_resume_retries_failure_and_exposes_only_successful_attempt(
    tmp_path: Path,
) -> None:
    _FailOnceEnv.fail_next_attempt = True
    config = _config(tmp_path, episodes=1, flush_rows=1, failure_policy="continue")
    experiment = Experiment(
        config,
        environment_factory=_FailOnceEnv,
        agent_factory=_agent_factory,
    )

    failed_result = experiment.run(persist=True, progress=False)
    assert failed_result.run_directory is not None
    assert failed_result.metadata["status"] == "failed"
    assert failed_result.metadata["failed_trial_count"] == 1
    assert failed_result.steps.empty

    failed_store = RunStore.open(failed_result.run_directory)
    failed_entry = next(iter(failed_store.manifest.trials.values()))
    assert failed_entry.status == "failed"
    assert failed_entry.attempts == 1
    assert failed_entry.committed_attempt is None
    failed_attempt = failed_result.run_directory / failed_entry.path / "attempts" / "0001"
    assert (failed_attempt / "failure.json").is_file()
    assert not (failed_attempt / "commit.json").exists()
    assert list((failed_attempt / "tables" / "steps").glob("part-*.csv"))

    resumed_result = experiment.run(
        persist=True,
        progress=False,
        resume_from=failed_result.run_directory,
    )
    assert resumed_result.run_directory == failed_result.run_directory
    assert resumed_result.metadata["status"] == "complete"
    assert resumed_result.metadata["completed_trial_count"] == 1
    assert resumed_result.metadata["failed_trial_count"] == 0

    resumed_store = RunStore.open(resumed_result.run_directory)
    resumed_entry = next(iter(resumed_store.manifest.trials.values()))
    assert resumed_entry.status == "succeeded"
    assert resumed_entry.attempts == 2
    assert resumed_entry.committed_attempt == 2
    commits = resumed_store.committed_attempts()
    assert len(commits) == 1
    assert commits[0].attempt == 2

    steps = resumed_result.steps
    assert len(steps) == 3
    assert steps["step"].tolist() == [0, 1, 2]
    assert not steps[["trial_id", "episode", "step"]].duplicated().any()
    assert resumed_store.read_table("steps").equals(steps)


def test_runner_persists_source_dirty_and_schema_metadata(tmp_path: Path) -> None:
    result = Experiment(
        _config(tmp_path, episodes=1, flush_rows=2),
        environment_factory=_FiniteHorizonEnv,
        agent_factory=_agent_factory,
    ).run(persist=True, progress=False)
    assert result.run_directory is not None

    metadata = json.loads((result.run_directory / "metadata.json").read_text(encoding="utf-8"))
    config = json.loads((result.run_directory / "config.json").read_text(encoding="utf-8"))
    provenance = json.loads((result.run_directory / "provenance.json").read_text(encoding="utf-8"))
    store = RunStore.open(result.run_directory)
    commits = store.committed_attempts()

    sha256 = re.compile(r"[0-9a-f]{64}").fullmatch
    assert metadata["artifact_schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert sha256(metadata["source_sha256"])
    assert sha256(metadata["runtime_fingerprint"])
    assert "git_dirty" in metadata
    assert metadata["git_dirty"] is None or isinstance(metadata["git_dirty"], bool)

    assert config["config_schema_version"] == CONFIG_SCHEMA_VERSION
    assert provenance["provenance_schema_version"] == 2
    assert provenance["source_hash_algorithm"] == "sha256-path-length-content-v1"
    assert provenance["source_sha256"] == metadata["source_sha256"]
    assert provenance["source_files"]
    assert all(sha256(source_file["sha256"]) for source_file in provenance["source_files"])
    assert provenance["runtime"]["fingerprint"] == metadata["runtime_fingerprint"]
    assert {"available", "commit", "branch", "dirty", "status", "diff_sha256"} <= set(
        provenance["git"]
    )
    assert provenance["git"]["dirty"] is None or isinstance(provenance["git"]["dirty"], bool)

    assert store.manifest.artifact_schema_version == ARTIFACT_SCHEMA_VERSION
    assert sha256(store.manifest.config_sha256)
    assert sha256(store.manifest.provenance_sha256)
    assert len(commits) == 1
    assert commits[0].artifact_schema_version == ARTIFACT_SCHEMA_VERSION
    assert commits[0].source_hash == provenance["source_sha256"]
    table_references = [reference for reference in commits[0].artifacts if reference.table]
    assert table_references
    assert all(
        reference.table_schema_version == TABLE_SCHEMA_VERSION for reference in table_references
    )
