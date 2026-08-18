"""Reproducible execution of tabular RL trials.

The interaction loop is intentionally visible here.  Researchers can inspect and
modify the precise reset/action/update order without navigating a callback graph.
"""

from __future__ import annotations

import inspect
import json
import os
import warnings
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from tqdm.auto import tqdm  # type: ignore[import-untyped]

from rllab.evaluation import compare_to_optimal, exact_solution_status, extract_q_values
from rllab.experiments.artifacts import (
    AttemptReservation,
    RunStore,
    TrialAttemptWriter,
    atomic_write_json,
)
from rllab.experiments.config import (
    AgentSpec,
    EnvironmentSpec,
    ExperimentConfig,
    TrialSpec,
    canonical_json,
    stable_identifier,
)
from rllab.experiments.observation import TabularObservationAdapter, latent_state_from_info
from rllab.experiments.persistence import iter_table as iter_persisted_table
from rllab.experiments.policy_evaluation import evaluate_policy
from rllab.experiments.preflight import estimate_run
from rllab.experiments.provenance import collect_provenance, value_sha256
from rllab.experiments.schema import AttemptCommit, FailureRecord
from rllab.metrics.recorder import MetricRecorder, environment_episode_summary, update_fields
from rllab.utils.seeding import spawn_seeds

EnvironmentFactory = Callable[[Mapping[str, Any]], Any]
ResolvedEnvironmentFactory = Callable[[EnvironmentSpec], Any]
AgentFactory = Callable[[AgentSpec, Any, int], Any]


@dataclass(slots=True)
class _TrialOutput:
    trial: TrialSpec
    tables: dict[str, pd.DataFrame]
    exact_source: str | None
    exact_unavailable_reason: str | None
    commit: AttemptCommit | None = None
    failure: FailureRecord | None = None


@dataclass(slots=True)
class ExperimentResult:
    """A run handle whose persisted tables materialize only when requested."""

    experiment_id: str
    run_directory: Path | None
    metadata: dict[str, Any]
    _tables: dict[str, pd.DataFrame] = field(default_factory=dict, repr=False)

    @staticmethod
    def _canonical_table(name: str) -> str:
        aliases = {"episodes": "training_episodes"}
        canonical = aliases.get(name, name)
        if canonical not in {
            "training_episodes",
            "steps",
            "snapshots",
            "evaluations",
            "state_actions",
        }:
            raise KeyError(name)
        return canonical

    @property
    def training_episodes(self) -> pd.DataFrame:
        return self.table("training_episodes")

    @property
    def episodes(self) -> pd.DataFrame:
        """Compatibility alias for :attr:`training_episodes`."""

        return self.training_episodes

    @property
    def steps(self) -> pd.DataFrame:
        return self.table("steps")

    @property
    def snapshots(self) -> pd.DataFrame:
        return self.table("snapshots")

    @property
    def evaluations(self) -> pd.DataFrame:
        return self.table("evaluations")

    @property
    def state_actions(self) -> pd.DataFrame:
        return self.table("state_actions")

    def table(self, name: str) -> pd.DataFrame:
        canonical = self._canonical_table(name)
        if canonical not in self._tables:
            if self.run_directory is None:
                self._tables[canonical] = pd.DataFrame()
            else:
                self._tables[canonical] = RunStore.open(self.run_directory).read_table(canonical)
        return self._tables[canonical]

    def iter_table(
        self,
        name: str,
        *,
        columns: tuple[str, ...] | None = None,
        filters: Mapping[str, Any] | None = None,
        batch_size: int | None = None,
        verify: bool = False,
    ) -> Iterator[pd.DataFrame]:
        """Yield bounded table batches without materializing a persisted run."""

        canonical = self._canonical_table(name)
        if self.run_directory is not None:
            yield from iter_persisted_table(
                self.run_directory,
                canonical,
                columns=columns,
                filters=filters,
                batch_size=batch_size,
                verify=verify,
            )
            return
        frame = self.table(canonical)
        if columns is not None:
            frame = frame.loc[:, list(columns)]
        if filters:
            for column, expected in filters.items():
                frame = frame.loc[
                    frame[column].isin(expected)
                    if isinstance(expected, (list, tuple, set, frozenset))
                    else frame[column].eq(expected)
                ]
        size = batch_size or max(len(frame), 1)
        for start in range(0, len(frame), size):
            yield frame.iloc[start : start + size].reset_index(drop=True)

    def q_snapshots(
        self,
        trial_id: str,
        *,
        keys: tuple[str, ...] | None = None,
        verify: bool = False,
    ) -> dict[str, np.ndarray]:
        """Load committed Q snapshots for one trial in a persisted run."""

        if self.run_directory is None:
            raise RuntimeError("Q snapshots require a persisted experiment run")
        return RunStore.open(self.run_directory).read_q_snapshots(
            trial_id,
            keys=keys,
            verify=verify,
        )


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load the supported YAML configuration format."""

    return ExperimentConfig.from_yaml(path)


def _resume_config_identity(value: Mapping[str, Any]) -> str:
    """Fingerprint fields that must remain homogeneous across retried attempts."""

    normalized = json.loads(canonical_json(value))
    normalized.pop("execution", None)
    artifacts = normalized.get("artifacts")
    if isinstance(artifacts, dict):
        artifacts.pop("output_dir", None)
    return value_sha256(normalized)


def _n_actions(env: Any) -> int:
    for name in ("n_actions", "num_actions"):
        if hasattr(env, name):
            value = getattr(env, name)
            return int(value() if callable(value) else value)
    space = getattr(env, "action_space", None)
    if space is not None and hasattr(space, "n"):
        return int(space.n)
    raise ValueError("The environment does not expose a finite action count")


def _coordinate(value: Any) -> tuple[int, int]:
    if isinstance(value, str):
        parts = value.split(",")
        if len(parts) != 2:
            raise ValueError(f"Coordinate keys must look like 'row,column', got {value!r}")
        return int(parts[0].strip()), int(parts[1].strip())
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise ValueError(f"Expected a row-column coordinate, got {value!r}")


def _maze_parameters(parameters: Mapping[str, Any], environments: Any) -> dict[str, Any]:
    """Turn YAML-friendly lists/mappings into the maze's typed constructor values."""

    values = dict(parameters)
    for key in ("shape", "start", "reliability_range"):
        if key in values and values[key] is not None:
            values[key] = tuple(values[key])
    if "blocked_cells" in values:
        values["blocked_cells"] = [_coordinate(item) for item in values["blocked_cells"]]
    for key in ("static_walls", "walls"):
        if key in values:
            values[key] = [
                tuple(_coordinate(endpoint) for endpoint in edge) for edge in values[key]
            ]
    for key in ("state_reliability", "state_reward_noise_std", "rare_rewards"):
        if key in values and isinstance(values[key], Mapping):
            values[key] = {_coordinate(state): item for state, item in values[key].items()}
    if "goals" in values:
        goal_class = getattr(environments, "Goal", None)
        raw_goals = values["goals"]
        if isinstance(raw_goals, Mapping):
            values["goals"] = {_coordinate(state): item for state, item in raw_goals.items()}
        elif goal_class is not None:
            values["goals"] = [
                goal_class(**{**item, "position": _coordinate(item["position"])})
                if isinstance(item, Mapping)
                else goal_class(position=_coordinate(item))
                for item in raw_goals
            ]
    for key, class_name in (
        ("nonstationarity", "NonstationarityConfig"),
        ("parameter_randomization", "ParameterRandomization"),
    ):
        if key in values and isinstance(values[key], Mapping) and hasattr(environments, class_name):
            values[key] = getattr(environments, class_name)(**values[key])
    return values


def _default_environment_factory(spec: EnvironmentSpec) -> Any:
    import rllab.environments.stochastic_maze as maze_module
    from rllab import environments

    normalized_kind = spec.kind.lower().replace("-", "_").replace(" ", "_")
    registered_kinds = {
        "stochastic_maze",
        "maze",
        "stochastic_maze_wall_state",
        "risky_corridor",
    }
    if normalized_kind not in registered_kinds:
        raise ValueError(
            f"Unknown environment kind {spec.kind!r}; registered kinds: {sorted(registered_kinds)!r}"
        )

    if normalized_kind == "risky_corridor":
        from rllab.environments import RiskyCorridorEnv

        return RiskyCorridorEnv(**spec.parameters)

    environment_class = getattr(environments, "StochasticMazeEnv", None)
    if environment_class is None:
        try:
            from rllab.environments.stochastic_maze import StochasticMazeEnv as environment_class
        except ImportError as error:
            raise ImportError("Could not import StochasticMazeEnv") from error
    assert environment_class is not None
    config_class = next(
        (
            getattr(environments, name)
            for name in ("StochasticMazeConfig", "MazeConfig")
            if hasattr(environments, name)
        ),
        None,
    )
    if config_class is None:
        config_class = next(
            (
                getattr(maze_module, name)
                for name in ("StochasticMazeConfig", "MazeConfig")
                if hasattr(maze_module, name)
            ),
            None,
        )
    normalized_parameters = _maze_parameters(spec.parameters, maze_module)
    environment = (
        environment_class(config_class(**normalized_parameters))
        if config_class is not None
        else environment_class(**normalized_parameters)
    )
    if normalized_kind == "stochastic_maze_wall_state":
        from rllab.environments import WallStateObservationWrapper

        return WallStateObservationWrapper(environment)
    return environment


def make_environment(spec: EnvironmentSpec | Mapping[str, Any]) -> Any:
    """Instantiate a registered environment from its public specification."""

    resolved = spec if isinstance(spec, EnvironmentSpec) else EnvironmentSpec.from_mapping(spec)
    return _default_environment_factory(resolved)


def _exploration_from(parameters: dict[str, Any], agents: Any) -> None:
    """Translate convenient YAML epsilon/temperature fields to strategy objects."""

    if "exploration" in parameters and isinstance(parameters["exploration"], Mapping):
        raw = dict(parameters["exploration"])
        kind = str(raw.pop("kind", "epsilon_greedy")).lower().replace("-", "_")
        exploration_names = {
            "epsilon_greedy": "EpsilonGreedy",
            "boltzmann": "Boltzmann",
            "softmax": "Boltzmann",
            "ucb": "UCB",
        }
        class_name = exploration_names.get(kind)
        if class_name and hasattr(agents, class_name):
            parameters["exploration"] = getattr(agents, class_name)(**raw)
    if (
        "epsilon" in parameters
        and "exploration" not in parameters
        and hasattr(agents, "EpsilonGreedy")
    ):
        epsilon = parameters.pop("epsilon")
        if isinstance(epsilon, Mapping):
            raw_schedule = dict(epsilon)
            kind = str(raw_schedule.pop("kind", "constant")).lower().replace("-", "_")
            schedule_names = {
                "constant": "ConstantSchedule",
                "linear": "LinearDecaySchedule",
                "linear_decay": "LinearDecaySchedule",
                "exponential": "ExponentialDecaySchedule",
                "exponential_decay": "ExponentialDecaySchedule",
            }
            schedule_class = getattr(agents, schedule_names.get(kind, ""), None)
            epsilon = schedule_class(**raw_schedule) if schedule_class else raw_schedule
        parameters["exploration"] = agents.EpsilonGreedy(epsilon)


def _default_agent_factory(
    spec: AgentSpec,
    env: Any,
    seed: int,
    adapter: TabularObservationAdapter | None = None,
) -> Any:
    from rllab import agents

    normalized = spec.kind.lower().replace("-", "_").replace(" ", "_")
    adapter = adapter or TabularObservationAdapter.from_environment(env)
    class_names = {
        "random": "RandomAgent",
        "random_policy": "RandomAgent",
        "q_learning": "QLearningAgent",
        "qlearning": "QLearningAgent",
        "sarsa": "SarsaAgent",
        "expected_sarsa": "ExpectedSarsaAgent",
        "double_q_learning": "DoubleQLearningAgent",
        "double_qlearning": "DoubleQLearningAgent",
        "planner": "PlannerAgent",
    }
    if hasattr(agents, "make_agent"):
        try:
            return agents.make_agent(
                normalized,
                n_states=adapter.n_observations,
                n_actions=_n_actions(env),
                seed=seed,
                **spec.parameters,
            )
        except (KeyError, TypeError):
            pass
    class_name = class_names.get(normalized)
    if class_name is None or not hasattr(agents, class_name):
        raise ValueError(f"Unknown agent kind {spec.kind!r}")
    parameters = dict(spec.parameters)
    _exploration_from(parameters, agents)
    agent_class = getattr(agents, class_name)
    if class_name == "PlannerAgent":
        model = getattr(env, "exact_mdp", None)
        model = model() if callable(model) else model
        return agent_class(model, seed=seed, **parameters)
    return agent_class(
        n_states=adapter.n_observations,
        n_actions=_n_actions(env),
        seed=seed,
        **parameters,
    )


def _call_supported(function: Callable[..., Any], values: Mapping[str, Any]) -> Any:
    """Call with only supported named arguments, retaining **kwargs behavior."""

    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(**values)
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    kwargs = (
        dict(values)
        if accepts_kwargs
        else {key: value for key, value in values.items() if key in signature.parameters}
    )
    return function(**kwargs)


def _reset_environment(env: Any, *, seed: int | None) -> tuple[Any, Mapping[str, Any]]:
    result = env.reset(seed=seed) if seed is not None else env.reset()
    if isinstance(result, tuple) and len(result) == 2:
        observation, info = result
        return observation, info or {}
    return result, {}


def _step_environment(env: Any, action: int) -> tuple[Any, float, bool, bool, Mapping[str, Any]]:
    result = env.step(action)
    if len(result) == 5:
        observation, reward, terminated, truncated, info = result
    elif len(result) == 4:  # Compatibility with small research environments.
        observation, reward, done, info = result
        terminated, truncated = done, False
    else:
        raise TypeError("env.step must return four or five values")
    return observation, float(reward), bool(terminated), bool(truncated), info or {}


def _environment_diagnostics(env: Any) -> dict[str, Any]:
    """Sample current latent/regime fields even when an env omits them from info."""

    diagnostics: dict[str, Any] = {}
    for name in (
        "current_walls",
        "hazard_positions",
        "elapsed_steps",
        "current_regime",
        "regime",
        "drift_parameters",
        "structural_events",
    ):
        if not hasattr(env, name):
            continue
        value = getattr(env, name)
        if callable(value):
            try:
                value = value()
            except TypeError:
                continue
        diagnostics[name] = value
    return diagnostics


def _reset_agent(agent: Any, seed: int) -> None:
    if hasattr(agent, "reset"):
        _call_supported(agent.reset, {"seed": seed})


def _act(agent: Any, state: int) -> int:
    for name in ("act", "select_action"):
        if hasattr(agent, name):
            return int(
                _call_supported(
                    getattr(agent, name), {"state": state, "observation": state, "training": True}
                )
            )
    raise TypeError("Agent must define act(state) or select_action(state)")


def _update_agent(
    agent: Any,
    *,
    state: int,
    action: int,
    reward: float,
    next_state: int,
    terminated: bool,
    truncated: bool,
) -> Any:
    if not hasattr(agent, "update"):
        return None
    values = {
        "state": state,
        "action": action,
        "reward": reward,
        "next_state": next_state,
        "terminated": terminated,
        "truncated": truncated,
        "done": terminated or truncated,
    }
    return _call_supported(agent.update, values)


def _snapshot_diagnostics(
    agent: Any,
    exact: Any,
    previous_policy: np.ndarray | None,
    state_visits: Counter[int],
    state_index_identity: tuple[Any, ...],
) -> tuple[np.ndarray | None, dict[str, Any], np.ndarray | None]:
    q_values = extract_q_values(agent)
    if q_values is None:
        return None, {"exact_evaluation_available": False}, None
    policy = np.argmax(q_values, axis=1)
    diagnostics: dict[str, Any] = {
        "q_min": float(np.min(q_values)),
        "q_max": float(np.max(q_values)),
        "q_mean": float(np.mean(q_values)),
        "policy_changes": (
            int(np.count_nonzero(policy != previous_policy)) if previous_policy is not None else 0
        ),
        "exact_evaluation_available": exact is not None,
    }
    if exact is not None:
        weights = np.asarray(
            [state_visits.get(state, 0) for state in range(q_values.shape[0])], dtype=float
        )
        diagnostics.update(
            compare_to_optimal(
                q_values,
                exact,
                estimate_state_semantics="observation",
                estimate_state_index_identity=state_index_identity,
            )
        )
        if weights.sum() > 0:
            weighted = compare_to_optimal(
                q_values,
                exact,
                state_weights=weights,
                estimate_state_semantics="observation",
                estimate_state_index_identity=state_index_identity,
            )
            diagnostics.update(
                {f"visitation_weighted_{key}": value for key, value in weighted.items()}
            )
    return q_values, diagnostics, policy


def _success_failure(info: Mapping[str, Any], terminated: bool) -> tuple[bool, bool]:
    success = bool(info.get("success", info.get("is_success", False)))
    reason = str(info.get("termination_reason", info.get("event", ""))).lower()
    if not success:
        success = terminated and any(token in reason for token in ("goal", "success"))
    failure = bool(info.get("failure", info.get("is_failure", False)))
    if not failure:
        failure = (
            terminated
            and not success
            and any(token in reason for token in ("hazard", "failure", "trap"))
        )
    return success, failure


def _write_bounded_table(
    writer: TrialAttemptWriter,
    table: str,
    frame: pd.DataFrame,
    *,
    maximum_rows: int,
) -> None:
    """Write a frame as parts no larger than the configured row bound."""

    for start in range(0, len(frame), maximum_rows):
        writer.write_table(table, frame.iloc[start : start + maximum_rows].reset_index(drop=True))


def _run_trial(
    trial: TrialSpec,
    run_id: str,
    environment_factory: ResolvedEnvironmentFactory,
    agent_factory: Callable[..., Any],
    *,
    writer: TrialAttemptWriter | None,
    flush_rows: int,
    save_q_snapshots: bool,
) -> _TrialOutput:
    seeds = spawn_seeds(trial.seed)
    env = environment_factory(trial.environment)
    adapter = TabularObservationAdapter.from_environment(env)
    agent = _call_supported(
        agent_factory,
        {"spec": trial.agent, "env": env, "seed": seeds.agent, "adapter": adapter},
    )
    _reset_agent(agent, seeds.agent)
    gamma = float(trial.agent.parameters.get("gamma", 0.99))
    exact_status = exact_solution_status(env, gamma=gamma) if trial.exact_reference else None
    exact = exact_status.solution if exact_status is not None else None
    sweep_columns = {
        "sweep_" + key.replace(".", "_"): value for key, value in trial.sweep_values.items()
    }
    recorder = MetricRecorder(
        trial_id=trial.trial_id,
        seed=trial.seed,
        agent=trial.agent.name,
        environment=trial.environment.name,
        step_retention_mode=trial.step_retention.mode,
        step_sample_fraction=trial.step_retention.fraction,
        keep_terminal_steps=trial.step_retention.keep_terminal,
        keep_event_steps=trial.step_retention.keep_events,
        retention_salt=trial.step_retention.salt,
        common={
            "experiment_id": run_id,
            "scenario_id": trial.scenario_id,
            "condition_id": trial.condition_id,
            "observation_indexer_id": adapter.indexer_id,
            "phase": "training",
            **trial.tags,
            **sweep_columns,
        },
    )
    previous_policy: np.ndarray | None = None
    q_values, diagnostics, previous_policy = _snapshot_diagnostics(
        agent,
        exact,
        previous_policy,
        recorder.state_visits,
        adapter.state_index_identity,
    )
    recorder.record_snapshot(-1, q_values, diagnostics)
    snapshot_global_steps = {recorder.global_step}
    evaluation_frames: list[pd.DataFrame] = []

    def flush_recorder_table(table: str, *, force: bool = False) -> None:
        if writer is None:
            return
        buffers = {
            "training_episodes": recorder.episode_rows,
            "steps": recorder.step_rows,
            "snapshots": recorder.snapshot_rows,
            "state_actions": recorder.state_action_rows,
        }
        rows = buffers[table]
        if not rows or (not force and len(rows) < flush_rows):
            return
        _write_bounded_table(
            writer,
            table,
            pd.DataFrame(recorder.drain_rows(table)),
            maximum_rows=flush_rows,
        )

    def flush_q_values() -> None:
        if writer is None:
            return
        if not save_q_snapshots:
            recorder.drain_q_snapshots()
            return
        if len(recorder.q_snapshots) >= min(100, flush_rows):
            writer.write_q_snapshots(recorder.drain_q_snapshots())

    def flush_evaluations(*, force: bool = False) -> None:
        if writer is None or not evaluation_frames:
            return
        row_count = sum(len(frame) for frame in evaluation_frames)
        if not force and row_count < flush_rows:
            return
        _write_bounded_table(
            writer,
            "evaluations",
            pd.concat(evaluation_frames, ignore_index=True),
            maximum_rows=flush_rows,
        )
        evaluation_frames.clear()

    if writer is not None:
        flush_recorder_table("snapshots", force=True)
        flush_q_values()

    def capture_snapshot(episode: int, *, step_scheduled: bool = False) -> bool:
        """Capture one Q/diagnostic state per global interaction count."""

        nonlocal previous_policy
        if recorder.global_step in snapshot_global_steps:
            return False
        q_values, diagnostics, previous_policy = _snapshot_diagnostics(
            agent,
            exact,
            previous_policy,
            recorder.state_visits,
            adapter.state_index_identity,
        )
        snapshot_key = f"global_step_{recorder.global_step:012d}" if step_scheduled else None
        recorder.record_snapshot(
            episode,
            q_values,
            diagnostics,
            snapshot_key=snapshot_key,
        )
        snapshot_global_steps.add(recorder.global_step)
        flush_recorder_table("snapshots")
        flush_recorder_table("state_actions")
        flush_q_values()
        return True

    def run_policy_evaluation(checkpoint_episode: int) -> None:
        for scenario in trial.policy_evaluation.scenarios:
            frame = evaluate_policy(
                agent=agent,
                trial=trial,
                scenario=scenario,
                evaluation_root_seed=seeds.evaluation,
                checkpoint_episode=checkpoint_episode,
                checkpoint_global_step=recorder.global_step,
                environment_factory=environment_factory,
                expected_indexer_id=adapter.indexer_id,
                expected_state_index_identity=adapter.state_index_identity,
                experiment_id=run_id,
            )
            evaluation_frames.append(frame)
            flush_evaluations()

    if trial.policy_evaluation.enabled and trial.policy_evaluation.include_initial:
        run_policy_evaluation(-1)

    first_reset = True
    episode = 0
    try:
        while (
            recorder.global_step < trial.total_interaction_steps
            if trial.total_interaction_steps is not None
            else episode < trial.episodes
        ):
            observation, reset_info = _reset_environment(
                env, seed=seeds.environment if first_reset else None
            )
            first_reset = False
            state = adapter.encode(observation)
            latent_state = latent_state_from_info(reset_info)
            recorder.start_episode(
                episode,
                state,
                reset_info,
                latent_state=latent_state,
            )
            if hasattr(agent, "start_episode"):
                _call_supported(agent.start_episode, {"state": state, "episode": episode})
            terminated = truncated = False
            info: Mapping[str, Any] = reset_info
            action_counts: Counter[int] = Counter()
            episode_regret = 0.0
            episode_steps = 0

            while not (terminated or truncated):
                action = _act(agent, state)
                action_counts[action] += 1
                next_observation, reward, terminated, truncated, info = _step_environment(
                    env, action
                )
                info = {**_environment_diagnostics(env), **info}
                episode_steps += 1
                next_state = adapter.encode(next_observation)
                next_latent_state = latent_state_from_info(info)
                if trial.max_steps is not None and episode_steps >= trial.max_steps:
                    truncated = True
                    info = {**info, "runner_time_limit": True}
                interaction_budget_reached = (
                    trial.total_interaction_steps is not None
                    and recorder.global_step + 1 >= trial.total_interaction_steps
                )
                if interaction_budget_reached:
                    if not terminated:
                        truncated = True
                    info = {
                        **info,
                        "runner_interaction_budget_reached": True,
                        "runner_interaction_budget": trial.total_interaction_steps,
                    }
                update = _update_agent(
                    agent,
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    terminated=terminated,
                    truncated=truncated,
                )
                normalized_update = update_fields(update)
                if (
                    exact is not None
                    and state < exact.q_values.shape[0]
                    and action < exact.q_values.shape[1]
                ):
                    regret = float(np.max(exact.q_values[state]) - exact.q_values[state, action])
                    episode_regret += regret
                    normalized_update["instantaneous_regret"] = regret
                recorder.record_step(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    terminated=terminated,
                    truncated=truncated,
                    info=info,
                    update=normalized_update,
                    latent_state=latent_state,
                    next_latent_state=next_latent_state,
                )
                flush_recorder_table("steps")
                if (
                    trial.snapshot_step_interval is not None
                    and recorder.global_step % trial.snapshot_step_interval == 0
                ):
                    capture_snapshot(episode, step_scheduled=True)
                state = next_state
                latent_state = next_latent_state

            probabilities = np.asarray(list(action_counts.values()), dtype=float)
            probabilities /= probabilities.sum() if probabilities.size else 1.0
            policy_entropy = (
                float(-np.sum(probabilities * np.log(probabilities))) if probabilities.size else 0.0
            )
            final_info = {**info, **environment_episode_summary(env)}
            success, failure = _success_failure(final_info, terminated)
            recorder.finish_episode(
                terminated=terminated,
                truncated=truncated,
                final_info=final_info,
                success=success,
                failure=failure,
                extra={
                    "policy_entropy": policy_entropy,
                    "episode_regret": episode_regret if exact is not None else float("nan"),
                    **{
                        f"action_frequency_{action}": action_counts[action] / episode_steps
                        for action in range(_n_actions(env))
                    },
                },
            )
            flush_recorder_table("training_episodes")
            if hasattr(agent, "end_episode"):
                _call_supported(
                    agent.end_episode,
                    {"episode": episode, "terminated": terminated, "truncated": truncated},
                )
            final_training_episode = (
                recorder.global_step >= trial.total_interaction_steps
                if trial.total_interaction_steps is not None
                else episode == trial.episodes - 1
            )
            if (
                episode == 0
                or (episode + 1) % trial.snapshot_interval == 0
                or final_training_episode
            ):
                capture_snapshot(episode)
            if trial.policy_evaluation.enabled:
                regular_checkpoint = (episode + 1) % trial.policy_evaluation.interval_episodes == 0
                final_checkpoint = trial.policy_evaluation.include_final and final_training_episode
                if regular_checkpoint or final_checkpoint:
                    run_policy_evaluation(episode)
            episode += 1
    finally:
        if hasattr(env, "close"):
            env.close()

    if writer is not None:
        for table in ("training_episodes", "steps", "snapshots", "state_actions"):
            flush_recorder_table(table, force=True)
        flush_evaluations(force=True)
        if save_q_snapshots:
            writer.write_q_snapshots(recorder.drain_q_snapshots())
        else:
            recorder.drain_q_snapshots()
        tables: dict[str, pd.DataFrame] = {}
    else:
        episodes, steps, snapshots = recorder.frames()
        evaluations = (
            pd.concat(evaluation_frames, ignore_index=True) if evaluation_frames else pd.DataFrame()
        )
        tables = {
            "training_episodes": episodes,
            "steps": steps,
            "snapshots": snapshots,
            "evaluations": evaluations,
            "state_actions": recorder.state_action_frame(),
        }
    exact_reason = (
        exact_status.unavailable_reason
        if exact_status is not None and exact_status.solution is None
        else ("disabled by configuration" if exact_status is None else None)
    )
    commit = (
        writer.commit(
            metadata={
                "exact_solution_source": exact.source if exact is not None else None,
                "exact_unavailable_reason": exact_reason,
                "observed_steps": recorder.observed_step_count,
                "retained_steps": recorder.retained_step_count,
                "retention_counts": dict(recorder.retention_counts),
                "observation_indexer_id": adapter.indexer_id,
            }
        )
        if writer is not None
        else None
    )
    return _TrialOutput(
        trial=trial,
        tables=tables,
        exact_source=exact.source if exact is not None else None,
        exact_unavailable_reason=exact_reason,
        commit=commit,
    )


def _execute_trial(
    trial: TrialSpec,
    run_id: str,
    environment_factory: ResolvedEnvironmentFactory,
    agent_factory: Callable[..., Any],
    *,
    reservation: AttemptReservation | None = None,
    source_hash: str | None = None,
    flush_rows: int = 10_000,
    save_q_snapshots: bool = True,
) -> _TrialOutput:
    """Execute one trial and close its attempt with either commit or failure."""

    writer = (
        TrialAttemptWriter(
            reservation,
            source_hash=source_hash,
            metadata={
                "scenario_id": trial.scenario_id,
                "condition_id": trial.condition_id,
                "seed": trial.seed,
            },
        )
        if reservation is not None
        else None
    )
    try:
        return _run_trial(
            trial,
            run_id,
            environment_factory,
            agent_factory,
            writer=writer,
            flush_rows=flush_rows,
            save_q_snapshots=save_q_snapshots,
        )
    except BaseException as error:
        if writer is None:
            raise
        failure = writer.fail(error)
        return _TrialOutput(
            trial=trial,
            tables={},
            exact_source=None,
            exact_unavailable_reason=None,
            failure=failure,
        )


def _execute_trial_default(
    trial: TrialSpec,
    run_id: str,
    reservation: AttemptReservation | None,
    source_hash: str | None,
    flush_rows: int,
    save_q_snapshots: bool,
) -> _TrialOutput:
    return _execute_trial(
        trial,
        run_id,
        _default_environment_factory,
        _default_agent_factory,
        reservation=reservation,
        source_hash=source_hash,
        flush_rows=flush_rows,
        save_q_snapshots=save_q_snapshots,
    )


class Experiment:
    """Expand, execute, aggregate, and persist an :class:`ExperimentConfig`."""

    def __init__(
        self,
        config: ExperimentConfig | None = None,
        *,
        env_config: Mapping[str, Any] | None = None,
        agents: list[Mapping[str, Any] | AgentSpec] | None = None,
        seeds: Any = None,
        episodes: int | None = None,
        total_interaction_steps: int | None = None,
        environment_factory: EnvironmentFactory | None = None,
        agent_factory: AgentFactory | None = None,
    ) -> None:
        """Construct from a config, or from the concise notebook-style arguments."""

        if config is None:
            mapping: dict[str, Any] = {}
            if env_config is not None:
                mapping["environment"] = dict(env_config)
            if agents is not None:
                mapping["agents"] = [
                    asdict(item) if isinstance(item, AgentSpec) else item for item in agents
                ]
            if seeds is not None:
                mapping["seeds"] = list(seeds) if not isinstance(seeds, int) else seeds
            if episodes is not None:
                mapping["episodes"] = episodes
            if total_interaction_steps is not None:
                mapping["total_interaction_steps"] = total_interaction_steps
            config = ExperimentConfig.from_mapping(mapping)
        elif any(
            item is not None
            for item in (env_config, agents, seeds, episodes, total_interaction_steps)
        ):
            raise ValueError("Pass either config or individual experiment arguments, not both")
        self.config = config
        self.environment_factory = environment_factory
        self.agent_factory = agent_factory

    @classmethod
    def from_yaml(cls, path: str | Path) -> Experiment:
        return cls(load_experiment_config(path))

    def _identifier(self) -> str:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        base = stable_identifier(self.config.name, self.config.as_dict(), length=10)
        return f"{base}-{timestamp}"

    def run(
        self,
        *,
        persist: bool = True,
        progress: bool = True,
        resume_from: str | Path | None = None,
    ) -> ExperimentResult:
        """Run pending trials, streaming persisted artifacts and supporting retry."""

        if resume_from is not None and not persist:
            raise ValueError("resume_from requires persist=True")
        trials = self.config.trials()
        config_value = self.config.as_dict()
        repository = Path(__file__).resolve().parents[3]
        provenance_record = collect_provenance(repository=repository, config=config_value)
        source_hash = provenance_record.source_sha256
        estimate = estimate_run(self.config)

        store: RunStore | None = None
        run_directory: Path | None = None
        if resume_from is not None:
            store = RunStore.open(resume_from)
            store.reconcile(verify=False)
            stored_config = json.loads(
                (store.run_directory / "config.json").read_text(encoding="utf-8")
            )
            stored_provenance = json.loads(
                (store.run_directory / "provenance.json").read_text(encoding="utf-8")
            )
            if value_sha256(stored_config) != store.manifest.config_sha256:
                raise ValueError("Cannot resume: the stored run configuration failed integrity")
            if _resume_config_identity(stored_config) != _resume_config_identity(config_value):
                raise ValueError("Cannot resume: current configuration differs from the run plan")
            if stored_provenance.get("source_sha256") != source_hash:
                raise ValueError(
                    "Cannot resume: the source tree differs from the original run; "
                    "start a new run to avoid mixing implementations"
                )
            expected_ids = {trial.trial_id for trial in trials}
            if set(store.manifest.trials) != expected_ids:
                raise ValueError(
                    "Cannot resume: expanded trial identities differ from the run plan"
                )
            experiment_id = store.manifest.run_id
            run_directory = store.run_directory
        else:
            experiment_id = self._identifier()
            if persist:
                run_directory = self.config.output_dir / experiment_id
                specifications = {
                    trial.trial_id: json.loads(
                        canonical_json({**asdict(trial), "trial_id": trial.trial_id})
                    )
                    for trial in trials
                }
                store = RunStore.create(
                    run_directory,
                    run_id=experiment_id,
                    experiment_name=self.config.name,
                    trials=specifications,
                    config=config_value,
                    provenance=provenance_record,
                    table_format=self.config.artifacts.table_format,
                    metadata={"preflight": estimate.as_dict()},
                )

        pending_trials = [
            trial
            for trial in trials
            if store is None or store.manifest.trials[trial.trial_id].status != "succeeded"
        ]
        use_custom_factories = (
            self.environment_factory is not None or self.agent_factory is not None
        )
        workers = min(
            self.config.parallel_workers,
            max(1, len(pending_trials)),
            os.cpu_count() or 1,
        )
        if use_custom_factories and workers > 1:
            warnings.warn(
                "Custom factories run serially because notebook/local callables are often not picklable.",
                stacklevel=2,
            )
            workers = 1

        outputs: list[_TrialOutput] = []
        failures: list[FailureRecord] = []

        def reservation_for(trial: TrialSpec) -> AttemptReservation | None:
            return store.reserve_attempt(trial.trial_id) if store is not None else None

        def accept(output: _TrialOutput) -> None:
            outputs.append(output)
            if output.commit is not None:
                assert store is not None
                store.record_commit(output.commit)
            if output.failure is not None:
                failures.append(output.failure)
                assert store is not None
                store.record_failure(output.failure)

        if workers > 1 and pending_trials:
            try:
                executor: ProcessPoolExecutor | None = ProcessPoolExecutor(max_workers=workers)
            except (OSError, PermissionError, NotImplementedError) as error:
                warnings.warn(
                    f"Process parallelism is unavailable ({error}); running serially.",
                    stacklevel=2,
                )
                executor = None
                workers = 1
            if executor is not None:
                with executor:
                    futures = {
                        executor.submit(
                            _execute_trial_default,
                            trial,
                            experiment_id,
                            reservation_for(trial),
                            source_hash,
                            self.config.artifacts.flush_rows,
                            self.config.save_q_snapshots,
                        ): trial
                        for trial in pending_trials
                    }
                    completed = as_completed(futures)
                    iterator: Any = (
                        tqdm(completed, total=len(futures), desc="Trials", unit="trial")
                        if progress
                        else completed
                    )
                    for future in iterator:
                        trial = futures[future]
                        try:
                            accept(future.result())
                        except Exception as error:
                            raise RuntimeError(f"Trial {trial.trial_id} failed") from error

        if workers == 1 and pending_trials:
            iterator_trials = tqdm(
                pending_trials,
                desc="Trials",
                unit="trial",
                disable=not progress,
            )
            resolved_environment_factory: ResolvedEnvironmentFactory
            if self.environment_factory is None:
                resolved_environment_factory = _default_environment_factory
            else:
                legacy_factory = self.environment_factory

                def from_legacy_factory(spec: EnvironmentSpec) -> Any:
                    return legacy_factory(spec.parameters)

                resolved_environment_factory = from_legacy_factory
            agent_factory = self.agent_factory or _default_agent_factory
            for trial in iterator_trials:
                output = _execute_trial(
                    trial,
                    experiment_id,
                    resolved_environment_factory,
                    agent_factory,
                    reservation=reservation_for(trial),
                    source_hash=source_hash,
                    flush_rows=self.config.artifacts.flush_rows,
                    save_q_snapshots=self.config.save_q_snapshots,
                )
                accept(output)
                if (
                    output.failure is not None
                    and self.config.execution.failure_policy == "fail_fast"
                ):
                    break

        outputs.sort(key=lambda output: output.trial.trial_id)
        table_names = (
            "training_episodes",
            "steps",
            "snapshots",
            "evaluations",
            "state_actions",
        )
        in_memory_tables = {
            table: (
                pd.concat(
                    [output.tables[table] for output in outputs if table in output.tables],
                    ignore_index=True,
                )
                if any(table in output.tables for output in outputs)
                else pd.DataFrame()
            )
            for table in table_names
        }
        output_by_id = {output.trial.trial_id: output for output in outputs}
        commit_by_id = (
            {commit.trial_id: commit for commit in store.committed_attempts()}
            if store is not None
            else {}
        )
        trial_records = []
        for trial in trials:
            trial_output = output_by_id.get(trial.trial_id)
            existing_commit = commit_by_id.get(trial.trial_id)
            entry = store.manifest.trials[trial.trial_id] if store is not None else None
            trial_records.append(
                json.loads(
                    canonical_json(
                        {
                            **asdict(trial),
                            "trial_id": trial.trial_id,
                            "status": entry.status if entry is not None else "succeeded",
                            "exact_solution_source": (
                                trial_output.exact_source
                                if trial_output
                                else (
                                    existing_commit.metadata.get("exact_solution_source")
                                    if existing_commit
                                    else None
                                )
                            ),
                            "exact_unavailable_reason": (
                                trial_output.exact_unavailable_reason
                                if trial_output
                                else (
                                    existing_commit.metadata.get("exact_unavailable_reason")
                                    if existing_commit
                                    else None
                                )
                            ),
                        }
                    )
                )
            )
        status = store.manifest.status if store is not None else "complete"
        metadata: dict[str, Any] = {
            "experiment_id": experiment_id,
            "status": status,
            "trial_count": len(trials),
            "completed_trial_count": (
                sum(entry.status == "succeeded" for entry in store.manifest.trials.values())
                if store is not None
                else len(outputs)
            ),
            "failed_trial_count": (
                sum(entry.status == "failed" for entry in store.manifest.trials.values())
                if store is not None
                else len(failures)
            ),
            "preflight": estimate.as_dict(),
            "trials": trial_records,
            "artifact_schema_version": (
                store.manifest.artifact_schema_version if store is not None else None
            ),
            "git_commit": provenance_record.git.commit,
            "git_dirty": provenance_record.git.dirty,
            "source_sha256": provenance_record.source_sha256,
            "runtime_fingerprint": provenance_record.runtime.fingerprint,
            "package_versions": dict(provenance_record.runtime.package_versions),
        }
        if store is not None:
            metadata["tables"] = {
                table: sum(
                    entry.row_counts.get(table, 0) for entry in store.manifest.trials.values()
                )
                for table in table_names
            }
            atomic_write_json(store.run_directory / "metadata.json", metadata)

        result = ExperimentResult(
            experiment_id=experiment_id,
            run_directory=run_directory,
            metadata=metadata,
            _tables=in_memory_tables if store is None else {},
        )
        if failures and self.config.execution.failure_policy == "fail_fast":
            location = f"; partial artifacts: {run_directory}" if run_directory else ""
            raise RuntimeError(
                f"Trial {failures[0].trial_id} failed: {failures[0].message}{location}"
            )
        return result
