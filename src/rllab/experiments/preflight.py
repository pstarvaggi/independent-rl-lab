"""Read-only experiment expansion and resource-risk estimates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from rllab.experiments.config import ExperimentConfig


@dataclass(frozen=True, slots=True)
class RunEstimate:
    trial_count: int
    condition_count: int
    scenario_count: int
    training_budget_unit: str
    training_budget_per_trial: int
    training_episode_count: int | None
    training_interaction_step_count: int | None
    evaluation_episode_count: int | None
    maximum_transition_rows: int | None
    estimated_retained_step_rows: int | None
    step_retention_mode: str
    workers: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def estimate_run(config: ExperimentConfig) -> RunEstimate:
    """Expand a config without creating environments, workers, or output directories."""

    trials = config.trials()
    maximum_transition_rows = 0
    bounded = True
    for trial in trials:
        if trial.total_interaction_steps is not None:
            maximum_transition_rows += trial.total_interaction_steps
            continue
        horizon = trial.max_steps
        if horizon is None:
            raw = trial.environment.parameters.get("max_episode_steps")
            horizon = int(raw) if raw is not None else None
        if horizon is None:
            bounded = False
        else:
            maximum_transition_rows += trial.episodes * horizon
    maximum = maximum_transition_rows if bounded else None

    retention = config.artifacts.step_retention
    if retention.mode == "none" and not retention.keep_terminal and not retention.keep_events:
        retained: int | None = 0
    elif maximum is None:
        retained = None
    elif retention.mode == "sample":
        retained = round(maximum * retention.fraction)
    else:
        retained = maximum

    evaluation_episodes: int | None = 0
    policy = config.policy_evaluation
    if policy.enabled:
        for trial in trials:
            if trial.total_interaction_steps is not None:
                # Checkpoints are episode-indexed, while the number of episodes
                # completed under a transition budget depends on environment
                # outcomes and the learned policy.
                evaluation_episodes = None
                break
            checkpoints = trial.episodes // policy.interval_episodes
            if policy.include_initial:
                checkpoints += 1
            if policy.include_final and trial.episodes % policy.interval_episodes:
                checkpoints += 1
            assert evaluation_episodes is not None
            evaluation_episodes += (
                checkpoints * policy.episodes_per_checkpoint * len(policy.scenarios)
            )

    interaction_budget = config.total_interaction_steps
    uses_interaction_budget = interaction_budget is not None
    if interaction_budget is None:
        training_episode_count = sum(trial.episodes for trial in trials)
        training_interaction_step_count = None
        budget_value = config.episodes
    else:
        training_episode_count = None
        training_interaction_step_count = len(trials) * interaction_budget
        budget_value = interaction_budget

    return RunEstimate(
        trial_count=len(trials),
        condition_count=len({trial.condition_id for trial in trials}),
        scenario_count=len({trial.scenario_id for trial in trials}),
        training_budget_unit="interaction_steps" if uses_interaction_budget else "episodes",
        training_budget_per_trial=budget_value,
        training_episode_count=training_episode_count,
        training_interaction_step_count=training_interaction_step_count,
        evaluation_episode_count=evaluation_episodes,
        maximum_transition_rows=maximum,
        estimated_retained_step_rows=retained,
        step_retention_mode=retention.mode,
        workers=min(config.parallel_workers, max(1, len(trials))),
    )


__all__ = ["RunEstimate", "estimate_run"]
