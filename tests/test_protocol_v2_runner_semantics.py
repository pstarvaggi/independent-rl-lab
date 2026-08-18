from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest

from rllab.experiments.config import (
    AgentSpec,
    EnvironmentSpec,
    EvaluationScenario,
    ExperimentConfig,
    PolicyEvaluationSpec,
)
from rllab.experiments.observation import ObservationEncodingError
from rllab.experiments.runner import Experiment
from rllab.utils.seeding import spawn_seeds


class _Discrete:
    def __init__(self, n: int) -> None:
        self.n = n


class _ObservationConflictEnv:
    """Returns observations that deliberately disagree with privileged info."""

    observation_space = _Discrete(3)
    action_space = _Discrete(1)

    def __init__(self, _: Mapping[str, Any]) -> None:
        self.done = False

    def reset(self, seed: int | None = None) -> tuple[int, dict[str, int]]:
        del seed
        self.done = False
        return 1, {"latent_state_index": 2, "state_index": 2}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, int]]:
        assert action == 0
        assert not self.done
        self.done = True
        return 0, 1.0, True, False, {"latent_state_index": 2, "state_index": 2}

    def close(self) -> None:
        return None


class _ObservationProbeAgent:
    instances: ClassVar[list[_ObservationProbeAgent]] = []

    def __init__(self) -> None:
        self.q_values = np.zeros((3, 1), dtype=float)
        self.act_states: list[int] = []
        self.transitions: list[tuple[int, int]] = []
        self.instances.append(self)

    def reset(self, seed: int | None = None) -> None:
        del seed

    def act(self, state: int, training: bool = True) -> int:
        assert training
        self.act_states.append(state)
        return 0

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        terminated: bool,
    ) -> dict[str, float]:
        del action, reward, terminated
        self.transitions.append((state, next_state))
        return {"td_error": 0.0}


def _observation_probe_agent_factory(
    spec: AgentSpec,
    env: _ObservationConflictEnv,
    seed: int,
) -> _ObservationProbeAgent:
    del spec, env, seed
    return _ObservationProbeAgent()


def test_runner_uses_returned_observations_even_when_latent_info_conflicts() -> None:
    _ObservationProbeAgent.instances.clear()
    config = ExperimentConfig(
        name="observation-not-latent",
        episodes=1,
        seeds=(17,),
        environments=(EnvironmentSpec(name="conflict", kind="test"),),
        agents=(AgentSpec(name="probe", kind="test"),),
        exact_reference=False,
        snapshot_interval=1,
    )
    result = Experiment(
        config,
        environment_factory=_ObservationConflictEnv,
        agent_factory=_observation_probe_agent_factory,
    ).run(persist=False, progress=False)

    agent = _ObservationProbeAgent.instances[-1]
    assert agent.act_states == [1]
    assert agent.transitions == [(1, 0)]
    assert result.steps[["state", "next_state"]].to_records(index=False).tolist() == [(1, 0)]
    assert result.steps[["latent_state", "next_latent_state"]].to_records(index=False).tolist() == [
        (2, 2)
    ]


class _StructuredSpace:
    shape = (2,)


class _UnsupportedStructuredEnv:
    observation_space = _StructuredSpace()
    action_space = _Discrete(1)

    def __init__(self, _: Mapping[str, Any]) -> None:
        pass

    def reset(self, seed: int | None = None) -> tuple[dict[str, int], dict[str, int]]:
        del seed
        return {"visible": 0}, {"latent_state_index": 1}


def _agent_factory_must_not_run(spec: AgentSpec, env: Any, seed: int) -> Any:
    del spec, env, seed
    raise AssertionError("agent construction must follow observation-contract validation")


def test_runner_rejects_structured_observation_without_explicit_encoder() -> None:
    config = ExperimentConfig(
        name="unsupported-structured-observation",
        episodes=1,
        seeds=(1,),
        environments=(EnvironmentSpec(name="structured", kind="test"),),
        agents=(AgentSpec(name="probe", kind="test"),),
        exact_reference=False,
    )
    with pytest.raises(
        ObservationEncodingError,
        match="Discrete observation space or an explicit observation_to_state",
    ):
        Experiment(
            config,
            environment_factory=_UnsupportedStructuredEnv,
            agent_factory=_agent_factory_must_not_run,
        ).run(persist=False, progress=False)


def test_runner_rejects_local_maze_observation_with_domain_error() -> None:
    config = ExperimentConfig(
        name="unsupported-local-observation",
        episodes=1,
        seeds=(1,),
        environments=(
            EnvironmentSpec(
                name="local-maze",
                parameters={
                    "shape": (2, 2),
                    "start": (0, 0),
                    "goals": {(1, 1): 1.0},
                    "observation_mode": "local",
                },
            ),
        ),
        agents=(AgentSpec(name="q", kind="q_learning"),),
        exact_reference=False,
    )
    with pytest.raises(
        ObservationEncodingError,
        match="Discrete observation space or an explicit observation_to_state",
    ):
        Experiment(config).run(persist=False, progress=False)


class _SeededOneStepEnv:
    observation_space = _Discrete(2)
    action_space = _Discrete(1)
    reset_events: ClassVar[list[tuple[str, int | None]]] = []

    def __init__(self, parameters: Mapping[str, Any]) -> None:
        self.phase = str(parameters.get("phase", "training"))
        self.rng = np.random.default_rng()
        self.done = False

    def reset(self, seed: int | None = None) -> tuple[int, dict[str, int]]:
        self.reset_events.append((self.phase, seed))
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.done = False
        observation = int(self.rng.integers(2))
        return observation, {"latent_state_index": 1 - observation}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        assert action == 0
        assert not self.done
        self.done = True
        observation = int(self.rng.integers(2))
        reward = float(self.rng.normal())
        return (
            observation,
            reward,
            True,
            False,
            {
                "latent_state_index": 1 - observation,
                "success": reward >= 0.0,
                "failure": reward < 0.0,
            },
        )

    def close(self) -> None:
        return None


class _EvaluationSpyAgent:
    act_events: ClassVar[list[tuple[str, bool, bool, int]]] = []
    act_step_events: ClassVar[list[tuple[str, bool, bool, int]]] = []
    update_events: ClassVar[list[tuple[str, bool]]] = []
    lifecycle_events: ClassVar[list[tuple[str, bool, bool]]] = []

    def __init__(self, name: str, *, evaluation_clone: bool = False) -> None:
        self.name = name
        self.evaluation_clone = evaluation_clone
        self.q_values = np.zeros((2, 1), dtype=float)
        self.step = 0

    @classmethod
    def clear_events(cls) -> None:
        cls.act_events.clear()
        cls.act_step_events.clear()
        cls.update_events.clear()
        cls.lifecycle_events.clear()

    def reset(self, seed: int | None = None) -> None:
        del seed

    def clone_for_evaluation(self) -> _EvaluationSpyAgent:
        clone = _EvaluationSpyAgent(self.name, evaluation_clone=True)
        clone.q_values[...] = self.q_values
        clone.step = self.step
        return clone

    def start_episode(
        self,
        state: int,
        episode: int,
        training: bool = True,
    ) -> None:
        del state, episode
        self.lifecycle_events.append((self.name, self.evaluation_clone, training))

    def act(self, state: int, training: bool = True) -> int:
        self.act_events.append((self.name, self.evaluation_clone, training, state))
        self.act_step_events.append((self.name, self.evaluation_clone, training, self.step))
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
        self.update_events.append((self.name, self.evaluation_clone))
        self.step += 1
        return {"td_error": 0.0}

    def end_episode(
        self,
        episode: int,
        terminated: bool,
        truncated: bool,
        training: bool = True,
    ) -> None:
        del episode, terminated, truncated
        self.lifecycle_events.append((self.name, self.evaluation_clone, training))


def _evaluation_spy_agent_factory(
    spec: AgentSpec,
    env: _SeededOneStepEnv,
    seed: int,
) -> _EvaluationSpyAgent:
    del env, seed
    return _EvaluationSpyAgent(spec.name)


def _evaluation_config() -> ExperimentConfig:
    return ExperimentConfig(
        name="held-out-evaluation-contract",
        episodes=2,
        seeds=(29,),
        environments=(
            EnvironmentSpec(
                name="seeded",
                kind="test",
                parameters={"phase": "training"},
            ),
        ),
        agents=(
            AgentSpec(name="agent-a", kind="test"),
            AgentSpec(name="agent-b", kind="test"),
        ),
        exact_reference=False,
        snapshot_interval=1,
        policy_evaluation=PolicyEvaluationSpec(
            enabled=True,
            interval_episodes=1,
            episodes_per_checkpoint=3,
            include_initial=False,
            include_final=True,
            scenarios=(
                EvaluationScenario(
                    name="held-out",
                    environment_overrides={"phase": "evaluation"},
                ),
            ),
        ),
    )


def _run_evaluation_probe() -> tuple[pd.DataFrame, list[tuple[str, int | None]]]:
    _EvaluationSpyAgent.clear_events()
    _SeededOneStepEnv.reset_events.clear()
    result = Experiment(
        _evaluation_config(),
        environment_factory=_SeededOneStepEnv,
        agent_factory=_evaluation_spy_agent_factory,
    ).run(persist=False, progress=False)
    return result.evaluations.copy(), list(_SeededOneStepEnv.reset_events)


def test_held_out_evaluation_is_update_free_deterministic_and_paired() -> None:
    first, first_resets = _run_evaluation_probe()
    first_act_events = list(_EvaluationSpyAgent.act_events)
    first_update_events = list(_EvaluationSpyAgent.update_events)
    first_lifecycle_events = list(_EvaluationSpyAgent.lifecycle_events)

    second, second_resets = _run_evaluation_probe()
    sort_columns = ["trial_id", "checkpoint_episode", "evaluation_episode"]
    pd.testing.assert_frame_equal(
        first.drop(columns="experiment_id").sort_values(sort_columns).reset_index(drop=True),
        second.drop(columns="experiment_id").sort_values(sort_columns).reset_index(drop=True),
    )
    assert first_resets == second_resets

    # Two training episodes for each of two agents, and never an evaluation update.
    assert len(first_update_events) == 4
    assert all(not evaluation_clone for _, evaluation_clone in first_update_events)

    evaluation_actions = [event for event in first_act_events if event[1]]
    training_actions = [event for event in first_act_events if not event[1]]
    assert len(evaluation_actions) == 12
    assert len(training_actions) == 4
    assert all(training is False for _, _, training, _ in evaluation_actions)
    assert all(training is True for _, _, training, _ in training_actions)
    assert all(
        training is False
        for _, evaluation_clone, training in first_lifecycle_events
        if evaluation_clone
    )

    # Every checkpoint and agent reuses the same ordered held-out seed panel.
    panels = {
        tuple(group.sort_values("evaluation_episode")["evaluation_seed"].astype(int).tolist())
        for _, group in first.groupby(["trial_id", "checkpoint_episode"], sort=True)
    }
    assert len(panels) == 1
    evaluation_panel = next(iter(panels))
    assert len(evaluation_panel) == 3
    assert len(set(evaluation_panel)) == 3

    # Evaluation roots are independent of the root/training-environment/agent streams.
    training_streams = spawn_seeds(29)
    assert set(evaluation_panel).isdisjoint(
        {training_streams.root, training_streams.environment, training_streams.agent}
    )
    evaluation_reset_seeds = [seed for phase, seed in first_resets if phase == "evaluation"]
    assert all(seed is not None for seed in evaluation_reset_seeds)
    assert set(evaluation_reset_seeds) == set(evaluation_panel)


class _BudgetProbeEnv:
    observation_space = _Discrete(1)
    action_space = _Discrete(1)

    def __init__(self, parameters: Mapping[str, Any]) -> None:
        self.horizon = int(parameters.get("horizon", 3))
        self.episode_step = 0
        self.completed_naturally = False

    def reset(self, seed: int | None = None) -> tuple[int, dict[str, Any]]:
        del seed
        self.episode_step = 0
        self.completed_naturally = False
        return 0, {}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        assert action == 0
        self.episode_step += 1
        self.completed_naturally = self.episode_step >= self.horizon
        return 0, 1.0, self.completed_naturally, False, {}

    def episode_summary(self) -> dict[str, Any]:
        return {
            "route": "probe-route",
            "probe_episode_steps": self.episode_step,
            "completed_naturally": self.completed_naturally,
        }

    def close(self) -> None:
        return None


def test_total_interaction_budget_is_exact_and_forces_final_artifacts() -> None:
    config = ExperimentConfig(
        name="fixed-interaction-budget",
        episodes=1,
        total_interaction_steps=7,
        seeds=(13,),
        environments=(
            EnvironmentSpec(name="budget-probe", kind="test", parameters={"horizon": 3}),
        ),
        agents=(
            AgentSpec(name="agent-a", kind="test"),
            AgentSpec(name="agent-b", kind="test"),
        ),
        exact_reference=False,
        snapshot_interval=10,
        policy_evaluation=PolicyEvaluationSpec(
            enabled=True,
            interval_episodes=10,
            episodes_per_checkpoint=2,
            include_initial=False,
            include_final=True,
        ),
    )
    result = Experiment(
        config,
        environment_factory=_BudgetProbeEnv,
        agent_factory=_evaluation_spy_agent_factory,
    ).run(persist=False, progress=False)

    assert result.steps.groupby("trial_id")["global_step"].max().eq(7).all()
    assert result.steps.groupby("trial_id").size().eq(7).all()
    assert result.training_episodes.groupby("trial_id").size().eq(3).all()

    final_episodes = (
        result.training_episodes.sort_values("episode").groupby("trial_id", as_index=False).tail(1)
    )
    assert final_episodes["truncated"].all()
    assert final_episodes["episode_length"].eq(1).all()
    assert final_episodes["env_runner_interaction_budget_reached"].all()
    assert final_episodes["env_runner_interaction_budget"].eq(7).all()
    assert final_episodes["env_route"].eq("probe-route").all()
    assert final_episodes["env_probe_episode_steps"].eq(1).all()

    final_snapshots = result.snapshots.groupby("trial_id")["global_step"].max()
    assert final_snapshots.eq(7).all()
    assert result.evaluations.groupby("trial_id")["checkpoint_global_step"].max().eq(7).all()
    assert result.evaluations["env_route"].eq("probe-route").all()


def test_step_indexed_snapshots_are_exact_and_deduplicated(tmp_path: Path) -> None:
    config = ExperimentConfig(
        name="step-indexed-snapshots",
        episodes=1,
        total_interaction_steps=12,
        seeds=(17,),
        environments=(
            EnvironmentSpec(name="snapshot-probe", kind="test", parameters={"horizon": 5}),
        ),
        agents=(AgentSpec(name="agent-a", kind="test"),),
        exact_reference=False,
        snapshot_interval=1,
        snapshot_step_interval=3,
        output_dir=tmp_path,
    )
    result = Experiment(
        config,
        environment_factory=_BudgetProbeEnv,
        agent_factory=_evaluation_spy_agent_factory,
    ).run(persist=True, progress=False)
    snapshots = result.snapshots

    assert snapshots["global_step"].astype(int).tolist() == [0, 3, 5, 6, 9, 10, 12]
    assert not snapshots["global_step"].duplicated().any()
    assert not snapshots["snapshot_key"].duplicated().any()
    assert snapshots.loc[snapshots["global_step"].isin([3, 6, 9, 12]), "snapshot_key"].tolist() == [
        "global_step_000000000003",
        "global_step_000000000006",
        "global_step_000000000009",
        "global_step_000000000012",
    ]
    assert snapshots.iloc[-1]["global_step"] == 12
    assert "episode_00000002" not in set(snapshots["snapshot_key"])
    trial_id = str(snapshots["trial_id"].iloc[0])
    assert set(result.q_snapshots(trial_id)) == set(snapshots["snapshot_key"])


def test_greedy_and_behavior_evaluations_share_seed_group_without_updates() -> None:
    _EvaluationSpyAgent.clear_events()
    config = ExperimentConfig(
        name="paired-policy-modes",
        episodes=1,
        seeds=(31,),
        environments=(
            EnvironmentSpec(name="seeded", kind="test", parameters={"phase": "training"}),
        ),
        agents=(AgentSpec(name="agent-a", kind="test"),),
        exact_reference=False,
        policy_evaluation=PolicyEvaluationSpec(
            enabled=True,
            interval_episodes=1,
            episodes_per_checkpoint=4,
            include_final=True,
            scenarios=(
                EvaluationScenario(
                    name="frozen-greedy",
                    environment_overrides={"phase": "evaluation"},
                    policy_mode="greedy",
                    seed_group="deployment-panel",
                ),
                EvaluationScenario(
                    name="continuing-behavior",
                    environment_overrides={"phase": "evaluation"},
                    policy_mode="behavior",
                    seed_group="deployment-panel",
                ),
            ),
        ),
    )
    evaluations = (
        Experiment(
            config,
            environment_factory=_SeededOneStepEnv,
            agent_factory=_evaluation_spy_agent_factory,
        )
        .run(persist=False, progress=False)
        .evaluations
    )

    assert set(evaluations["evaluation_policy_mode"]) == {"greedy", "behavior"}
    assert set(evaluations["evaluation_seed_group"]) == {"deployment-panel"}
    panels = {
        mode: tuple(sample.sort_values("evaluation_episode")["evaluation_seed"].astype(int))
        for mode, sample in evaluations.groupby("evaluation_policy_mode")
    }
    assert panels["greedy"] == panels["behavior"]

    evaluation_actions = [event for event in _EvaluationSpyAgent.act_events if event[1]]
    assert sum(not training for _, _, training, _ in evaluation_actions) == 4
    assert sum(training for _, _, training, _ in evaluation_actions) == 4
    evaluation_schedule_steps = [
        step
        for _, evaluation_clone, _, step in _EvaluationSpyAgent.act_step_events
        if evaluation_clone
    ]
    assert evaluation_schedule_steps == [1] * 8
    assert all(not evaluation_clone for _, evaluation_clone in _EvaluationSpyAgent.update_events)
