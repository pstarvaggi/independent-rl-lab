"""Held-out, update-free evaluation on paired environment seeds."""

from __future__ import annotations

import copy
import hashlib
import inspect
from collections import Counter
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from rllab.experiments.config import EnvironmentSpec, EvaluationScenario, TrialSpec
from rllab.experiments.observation import TabularObservationAdapter
from rllab.metrics.recorder import diagnostic_fields, environment_episode_summary

EnvironmentFactory = Callable[[EnvironmentSpec], Any]


def paired_evaluation_seeds(root_seed: int, scenario: str, count: int) -> tuple[int, ...]:
    """Derive a fixed seed panel reused at every checkpoint."""

    digest = int.from_bytes(hashlib.sha256(scenario.encode()).digest()[:4], "big")
    sequence = np.random.SeedSequence([int(root_seed), digest])
    return tuple(
        int(child.generate_state(1, dtype=np.uint32)[0]) for child in sequence.spawn(count)
    )


def _call_supported(function: Callable[..., Any], values: Mapping[str, Any]) -> Any:
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


def _evaluation_agent(agent: Any) -> Any:
    clone = getattr(agent, "clone_for_evaluation", None)
    if callable(clone):
        return clone()
    try:
        return copy.deepcopy(agent)
    except Exception as error:  # pragma: no cover - third-party agent boundary
        raise TypeError(
            "Policy evaluation requires a deepcopy-compatible agent or clone_for_evaluation()"
        ) from error


def _seed_evaluation_agent(agent: Any, seed: int) -> None:
    """Give stochastic evaluation policies a stream independent of training."""

    reseed = getattr(agent, "seed_evaluation", None)
    if callable(reseed):
        _call_supported(reseed, {"seed": seed})
        return
    generator = np.random.default_rng(seed)
    if hasattr(agent, "_rng"):
        agent._rng = generator
    elif hasattr(agent, "rng"):
        agent.rng = generator


def _reset(env: Any, seed: int) -> tuple[Any, Mapping[str, Any]]:
    result = env.reset(seed=seed)
    if isinstance(result, tuple) and len(result) == 2:
        observation, info = result
        return observation, info or {}
    return result, {}


def _step(env: Any, action: int) -> tuple[Any, float, bool, bool, Mapping[str, Any]]:
    result = env.step(action)
    if len(result) == 5:
        observation, reward, terminated, truncated, info = result
    elif len(result) == 4:
        observation, reward, done, info = result
        terminated, truncated = done, False
    else:
        raise TypeError("env.step must return four or five values")
    return observation, float(reward), bool(terminated), bool(truncated), info or {}


def _act(agent: Any, state: int, *, training: bool) -> int:
    for name in ("act", "select_action"):
        if hasattr(agent, name):
            return int(
                _call_supported(
                    getattr(agent, name),
                    {"state": state, "observation": state, "training": training},
                )
            )
    raise TypeError("Agent must define act(state) or select_action(state)")


def _outcomes(info: Mapping[str, Any], terminated: bool) -> tuple[bool, bool]:
    success = bool(info.get("success", info.get("is_success", False)))
    reason = str(info.get("termination_reason", info.get("event", ""))).lower()
    if not success:
        success = terminated and any(token in reason for token in ("goal", "success"))
    failure = bool(info.get("failure", info.get("is_failure", False)))
    if not failure:
        failure = (
            terminated
            and not success
            and any(token in reason for token in ("hazard", "failure", "trap", "crash"))
        )
    return success, failure


def evaluate_policy(
    *,
    agent: Any,
    trial: TrialSpec,
    scenario: EvaluationScenario,
    evaluation_root_seed: int,
    checkpoint_episode: int,
    checkpoint_global_step: int,
    environment_factory: EnvironmentFactory,
    expected_indexer_id: str,
    expected_state_index_identity: Sequence[Hashable],
    experiment_id: str,
) -> pd.DataFrame:
    """Evaluate a cloned policy without touching training state or RNG streams."""

    evaluation_agent = _evaluation_agent(agent)
    parameters = {**trial.environment.parameters, **scenario.environment_overrides}
    environment_spec = replace(trial.environment, parameters=parameters)
    env = environment_factory(environment_spec)
    adapter = TabularObservationAdapter.from_environment(env)
    if adapter.indexer_id != expected_indexer_id or adapter.state_index_identity != tuple(
        expected_state_index_identity
    ):
        raise ValueError(
            "Evaluation observation indexing differs from training: "
            f"{adapter.indexer_id!r} / {adapter.state_index_identity!r} != "
            f"{expected_indexer_id!r} / {tuple(expected_state_index_identity)!r}"
        )
    seeds = paired_evaluation_seeds(
        evaluation_root_seed,
        scenario.resolved_seed_group,
        trial.policy_evaluation.episodes_per_checkpoint,
    )
    behavior_policy = scenario.policy_mode == "behavior"
    rows: list[dict[str, Any]] = []
    try:
        for evaluation_episode, seed in enumerate(seeds):
            policy_seed = int(
                np.random.SeedSequence([seed, 0x4556414C]).generate_state(1, dtype=np.uint32)[0]
            )
            _seed_evaluation_agent(evaluation_agent, policy_seed)
            observation, info = _reset(env, seed)
            state = adapter.encode(observation)
            if hasattr(evaluation_agent, "start_episode"):
                _call_supported(
                    evaluation_agent.start_episode,
                    {
                        "state": state,
                        "episode": evaluation_episode,
                        "training": behavior_policy,
                    },
                )
            terminated = truncated = False
            episode_return = 0.0
            episode_length = 0
            action_counts: Counter[int] = Counter()
            while not (terminated or truncated):
                action = _act(evaluation_agent, state, training=behavior_policy)
                action_counts[action] += 1
                observation, reward, terminated, truncated, info = _step(env, action)
                episode_length += 1
                episode_return += reward
                if trial.max_steps is not None and episode_length >= trial.max_steps:
                    truncated = True
                    info = {**info, "runner_time_limit": True}
                state = adapter.encode(observation)
            if hasattr(evaluation_agent, "end_episode"):
                _call_supported(
                    evaluation_agent.end_episode,
                    {
                        "episode": evaluation_episode,
                        "terminated": terminated,
                        "truncated": truncated,
                        "training": behavior_policy,
                    },
                )
            final_diagnostics = {**info, **environment_episode_summary(env)}
            success, failure = _outcomes(final_diagnostics, terminated)
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "scenario_id": trial.scenario_id,
                    "condition_id": trial.condition_id,
                    "trial_id": trial.trial_id,
                    "phase": "evaluation",
                    "agent": trial.agent.name,
                    "environment": trial.environment.name,
                    "seed": trial.seed,
                    "evaluation_seed": seed,
                    "evaluation_scenario": scenario.name,
                    "evaluation_policy_mode": scenario.policy_mode,
                    "evaluation_seed_group": scenario.resolved_seed_group,
                    "checkpoint_episode": checkpoint_episode,
                    "checkpoint_global_step": checkpoint_global_step,
                    "evaluation_episode": evaluation_episode,
                    "episode_return": episode_return,
                    "episode_length": episode_length,
                    "success": success,
                    "failure": failure,
                    "terminated": terminated,
                    "truncated": truncated,
                    **trial.tags,
                    **{
                        "sweep_" + key.replace(".", "_"): value
                        for key, value in trial.sweep_values.items()
                    },
                    **{
                        f"action_frequency_{action}": count / episode_length
                        for action, count in action_counts.items()
                    },
                    **diagnostic_fields(final_diagnostics),
                }
            )
    finally:
        if hasattr(env, "close"):
            env.close()
    return pd.DataFrame(rows)


__all__ = ["evaluate_policy", "paired_evaluation_seeds"]
