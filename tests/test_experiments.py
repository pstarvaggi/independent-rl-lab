from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rllab.experiments import AgentSpec, Experiment, ExperimentConfig, RunStore
from rllab.experiments.config import EnvironmentSpec
from rllab.metrics import MetricRecorder
from rllab.metrics.recorder import diagnostic_fields


class _Space:
    def __init__(self, n: int) -> None:
        self.n = n


class DummyEnvironment:
    observation_space = _Space(2)
    action_space = _Space(2)

    def __init__(self, parameters: dict[str, Any]) -> None:
        self.horizon = int(parameters.get("horizon", 3))
        self.reliability = float(parameters.get("movement_reliability", 1.0))
        self.rng = np.random.default_rng()
        self.time = 0
        self.state = 0

    def reset(self, seed: int | None = None) -> tuple[int, dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.time = 0
        self.state = 0
        return self.state, {"regime": "stationary"}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        self.time += 1
        realized = action if self.rng.random() < self.reliability else 1 - action
        self.state = int(realized == 1)
        success = self.state == 1
        terminated = success or self.time >= self.horizon
        return (
            self.state,
            float(success),
            terminated,
            False,
            {
                "success": success,
                "realized_action": realized,
                "regime": "stationary",
                "structural_events": ["none"],
            },
        )

    def exact_solution(self, gamma: float = 0.9) -> dict[str, np.ndarray]:
        q_star = np.array([[gamma, 1.0], [0.0, 0.0]])
        return {"q_values": q_star, "policy": np.array([1, 0])}

    def close(self) -> None:
        pass


class DummyAgent:
    def __init__(self, seed: int) -> None:
        self.q_values = np.zeros((2, 2), dtype=float)
        self.rng = np.random.default_rng(seed)

    @property
    def greedy_policy(self) -> np.ndarray:
        return np.argmax(self.q_values, axis=1)

    def reset(self, seed: int | None = None) -> None:
        self.q_values.fill(0.0)
        self.rng = np.random.default_rng(seed)

    def act(self, state: int, training: bool = True) -> int:
        return int(self.rng.integers(2))

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        terminated: bool,
    ) -> dict[str, float]:
        target = reward if terminated else reward + 0.9 * float(np.max(self.q_values[next_state]))
        delta = target - self.q_values[state, action]
        self.q_values[state, action] += 0.2 * delta
        return {"delta": delta, "learning_rate": 0.2, "exploration_rate": 0.1}


def _environment_factory(parameters: dict[str, Any]) -> DummyEnvironment:
    return DummyEnvironment(parameters)


def _agent_factory(spec: AgentSpec, env: DummyEnvironment, seed: int) -> DummyAgent:
    return DummyAgent(seed)


def _config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        name="dummy",
        episodes=5,
        seeds=(3, 7),
        environments=(EnvironmentSpec(name="dummy", parameters={"horizon": 3}),),
        agents=(AgentSpec(name="learner", kind="dummy"),),
        sweep={"environment.movement_reliability": (1.0, 0.6)},
        max_steps=4,
        snapshot_interval=2,
        output_dir=tmp_path,
    )


def test_configuration_expands_cartesian_sweep_with_unique_ids(tmp_path: Path) -> None:
    trials = _config(tmp_path).trials()
    assert len(trials) == 4
    assert len({trial.trial_id for trial in trials}) == 4
    assert {trial.environment.parameters["movement_reliability"] for trial in trials} == {0.6, 1.0}


def test_integer_seeds_mean_range() -> None:
    config = ExperimentConfig.from_mapping({"seeds": 3, "episodes": 1})
    assert config.seeds == (0, 1, 2)


def test_experiment_logs_steps_episodes_exact_errors_and_environment_state(tmp_path: Path) -> None:
    result = Experiment(
        _config(tmp_path),
        environment_factory=_environment_factory,
        agent_factory=_agent_factory,
    ).run(persist=False, progress=False)
    assert len(result.episodes) == 4 * 5
    assert result.episodes.groupby("trial_id")["episode"].nunique().eq(5).all()
    assert {
        "episode_return",
        "success",
        "failure",
        "policy_entropy",
        "td_error_variance",
        "episode_regret",
    } <= set(result.episodes)
    assert {
        "td_error",
        "absolute_td_error",
        "squared_td_error",
        "state_visit_count",
        "state_action_visit_count",
        "transition_count",
        "empirical_transition_probability",
        "empirical_reward_mean",
        "env_realized_action",
        "env_regime",
        "env_structural_events",
    } <= set(result.steps)
    assert result.snapshots["exact_evaluation_available"].all()
    assert result.snapshots["q_error_inf"].notna().all()


def test_experiment_is_reproducible_under_fixed_seeds(tmp_path: Path) -> None:
    config = replace(_config(tmp_path), seeds=(13,), sweep={})
    runs = [
        Experiment(
            config, environment_factory=_environment_factory, agent_factory=_agent_factory
        ).run(persist=False, progress=False)
        for _ in range(2)
    ]
    columns = ["seed", "episode", "episode_return", "episode_length", "success"]
    pd.testing.assert_frame_equal(runs[0].episodes[columns], runs[1].episodes[columns])
    step_columns = ["episode", "step", "state", "action", "reward", "next_state", "td_error"]
    pd.testing.assert_frame_equal(runs[0].steps[step_columns], runs[1].steps[step_columns])


def test_persistence_writes_metadata_tables_and_numeric_snapshots(tmp_path: Path) -> None:
    config = replace(_config(tmp_path), seeds=(1,), sweep={}, episodes=2)
    result = Experiment(
        config, environment_factory=_environment_factory, agent_factory=_agent_factory
    ).run(persist=True, progress=False)
    assert result.run_directory is not None
    metadata = json.loads((result.run_directory / "metadata.json").read_text())
    assert metadata["experiment_id"] == result.experiment_id
    assert metadata["git_commit"] is None or len(metadata["git_commit"]) == 40
    assert metadata["artifact_schema_version"] == 2
    assert metadata["tables"]["training_episodes"] == 2
    store = RunStore.open(result.run_directory)
    assert store.manifest.status == "complete"
    store.verify()
    pd.testing.assert_frame_equal(result.episodes, store.read_table("training_episodes"))
    snapshot_references = [
        reference
        for commit in store.committed_attempts()
        for reference in commit.artifacts
        if reference.format == "npz"
    ]
    assert len(snapshot_references) == 1
    with np.load(result.run_directory / snapshot_references[0].path, allow_pickle=False) as values:
        assert values.files


def test_metric_recorder_empirical_transition_probability() -> None:
    recorder = MetricRecorder(trial_id="t", seed=0, agent="a", environment="e")
    recorder.start_episode(0, 0)
    recorder.record_step(
        state=0, action=1, reward=2.0, next_state=1, terminated=False, truncated=False
    )
    recorder.record_step(
        state=0, action=1, reward=4.0, next_state=0, terminated=True, truncated=False
    )
    recorder.finish_episode(terminated=True, truncated=False)
    _, steps, _ = recorder.frames()
    assert steps.iloc[-1]["state_action_visit_count"] == 2
    assert steps.iloc[-1]["empirical_transition_probability"] == 0.5
    assert steps.iloc[-1]["empirical_reward_mean"] == 4.0


def test_environment_null_diagnostic_remains_a_missing_value() -> None:
    fields = diagnostic_fields({"realized_action": None, "regime": 2})
    assert fields == {"env_realized_action": None, "env_regime": 2}


def test_step_retention_keeps_real_wall_events_but_not_none_sentinels() -> None:
    recorder = MetricRecorder(
        trial_id="events",
        seed=0,
        agent="a",
        environment="e",
        step_retention_mode="none",
        keep_terminal_steps=False,
        keep_event_steps=True,
    )
    recorder.start_episode(0, 0)
    recorder.record_step(
        state=0,
        action=0,
        reward=0.0,
        next_state=0,
        terminated=False,
        truncated=False,
        info={"wall_events": []},
    )
    recorder.record_step(
        state=0,
        action=0,
        reward=0.0,
        next_state=0,
        terminated=False,
        truncated=False,
        info={"wall_events": [{"mechanism": "scheduled"}]},
    )
    recorder.finish_episode(terminated=False, truncated=True)
    _, steps, _ = recorder.frames()
    assert len(steps) == 1
    assert steps.iloc[0]["retention_reason"] == "event"


def test_default_factories_run_the_real_compact_state_maze(tmp_path: Path) -> None:
    config = ExperimentConfig.from_mapping(
        {
            "name": "real-maze-smoke",
            "episodes": 2,
            "seeds": [5],
            "snapshot_interval": 1,
            "output_dir": str(tmp_path),
            "environment": {
                "name": "blocked",
                "parameters": {
                    "shape": [2, 3],
                    "start": [0, 0],
                    "goals": [[1, 2]],
                    "blocked_cells": [[0, 1]],
                    "action_reliability": 0.8,
                    "max_episode_steps": 8,
                },
            },
            "agent": {
                "kind": "q_learning",
                "parameters": {"learning_rate": 0.2, "gamma": 0.9, "epsilon": 0.1},
            },
        }
    )
    result = Experiment(config).run(persist=False, progress=False)
    assert len(result.episodes) == 2
    assert result.steps["state"].max() < 5  # compact indexing excludes the blocked cell
    assert result.snapshots["exact_evaluation_available"].all()
    assert result.snapshots["q_error_inf"].notna().all()


def test_persistence_normalizes_coordinate_keyed_notebook_configs(tmp_path: Path) -> None:
    config = ExperimentConfig.from_mapping(
        {
            "name": "coordinate-keys",
            "episodes": 1,
            "seeds": [2],
            "snapshot_interval": 1,
            "output_dir": str(tmp_path),
            "environment": {
                "name": "tuple-goal",
                "parameters": {
                    "shape": (1, 2),
                    "start": (0, 0),
                    "goals": {(0, 1): 2.0},
                    "max_episode_steps": 2,
                },
            },
            "agent": {"kind": "q_learning", "parameters": {"epsilon": 0.0}},
        }
    )
    result = Experiment(config).run(persist=True, progress=False)
    assert result.run_directory is not None
    metadata = json.loads((result.run_directory / "metadata.json").read_text())
    parameters = metadata["trials"][0]["environment"]["parameters"]
    assert parameters["goals"] == {"(0, 1)": 2.0}
