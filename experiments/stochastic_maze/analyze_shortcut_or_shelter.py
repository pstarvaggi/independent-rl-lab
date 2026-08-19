#!/usr/bin/env python3
"""Reduce the completed shortcut-or-shelter runs to publication artifacts.

This is intentionally a post-processing command, not another experiment runner.
It reads the three immutable Protocol-v2 run directories once, loads only the
Q snapshots needed for the declared estimands, and writes compact CSV, JSON,
SVG, and (optionally) PNG artifacts.  Re-running a notebook should therefore
never be necessary just to inspect the completed study.

The original Notebook 05 publication gate rejected *any* tied or off-route
fork action.  That rule conflicted with the notebook's declared three-outcome
estimand.  This analysis retains ``tie`` and ``other`` as observed outcomes and
reports a condition-level unresolved-action gate at the declared 20% rate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any, cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from rllab.environments import RiskyCorridorEnv
from rllab.experiments import ExperimentConfig, RunStore
from rllab.theory import (
    epsilon_soft_value_iteration,
    value_iteration,
)
from rllab.theory import policy_evaluation as exact_policy_evaluation

GAMMA = 0.98
PERSISTENT_EPSILON = 0.10
UNRESOLVED_CONDITION_LIMIT = 0.20
PROGRESS_TARGETS = (0.25, 0.50, 0.75, 1.00)
METHOD_ORDER = ("q_learning", "sarsa", "expected_sarsa")
METHOD_LABELS = {
    "q_learning": "Q-learning",
    "sarsa": "SARSA",
    "expected_sarsa": "Expected SARSA",
}
METHOD_COLORS = {
    "q_learning": "#1665A8",
    "sarsa": "#C4542D",
    "expected_sarsa": "#16866C",
}
DISAGREEMENT_POINTS = {"recoverable": 0.825, "lethal": 1.0}
AMENDMENT = (
    "Analysis amendment — 2026-08-19. The original executable gate stopped on "
    "any tied or off-route final action. It stopped on one WEST action among "
    "1,920 stationary policies (lethal, SARSA, p=.99, seed 10). This conflicted "
    "with the declared three-outcome estimand. Before inspecting boundary "
    "estimates, the gate was revised to retain and report `other` and to fail "
    "only when unresolved actions exceed 20% within a method by reliability "
    "condition. This post-collection amendment is reported with every output."
)


@dataclass(frozen=True, slots=True)
class EnvironmentParameters:
    """Hashable non-reliability parameters for the exact environment."""

    recoverable_hazard_penalty: float
    lethal_hazard_penalty: float
    goal_reward: float
    step_reward: float
    max_episode_steps: int


@dataclass(slots=True)
class OpenRun:
    """One opened run and its small, reusable metadata tables."""

    label: str
    path: Path
    store: RunStore
    config: ExperimentConfig
    design: pd.DataFrame
    snapshots: pd.DataFrame
    commits: Mapping[str, Any]


class SnapshotReader:
    """Read selected Q arrays without repeatedly rescanning every run commit."""

    def __init__(self, run: OpenRun) -> None:
        self._root = run.path
        self._commits = run.commits

    def load(self, trial_id: str, keys: Sequence[str]) -> dict[str, np.ndarray]:
        requested = tuple(dict.fromkeys(str(key) for key in keys))
        missing = set(requested)
        found: dict[str, np.ndarray] = {}
        commit = self._commits.get(trial_id)
        if commit is None:
            raise KeyError(f"No committed attempt for trial {trial_id!r}")
        references = sorted(
            (
                reference
                for reference in commit.artifacts
                if reference.format == "npz" and reference.table == "q_snapshots"
            ),
            key=lambda reference: -1 if reference.part is None else int(reference.part),
        )
        for reference in references:
            if not missing:
                break
            with np.load(self._root / reference.path, allow_pickle=False) as archive:
                for key in tuple(missing):
                    if key in archive.files:
                        found[key] = np.asarray(archive[key]).copy()
                        missing.remove(key)
        if missing:
            raise KeyError(
                f"Q snapshot keys are absent for trial {trial_id!r}: {sorted(missing)!r}"
            )
        return {key: found[key] for key in requested}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _trial_design(config: ExperimentConfig) -> pd.DataFrame:
    rows = []
    for trial in config.trials():
        parameters = trial.environment.parameters
        rows.append(
            {
                "trial_id": trial.trial_id,
                "seed": int(trial.seed),
                "agent": trial.agent.name,
                "hazard_mode": str(parameters["hazard_mode"]),
                "reliability": float(
                    parameters.get(
                        "corridor_reliability",
                        parameters.get("action_reliability", 1.0),
                    )
                ),
                "interaction_budget": int(trial.total_interaction_steps or 0),
            }
        )
    return pd.DataFrame(rows)


def _open_run(label: str, path: Path) -> OpenRun:
    root = path.expanduser().resolve()
    store = RunStore.open(root)
    if store.manifest.status != "complete":
        raise RuntimeError(
            f"{label} run has status {store.manifest.status!r}, expected 'complete': {root}"
        )
    config = ExperimentConfig.from_mapping(_read_json(root / "config.json"))
    design = _trial_design(config)
    commits = {commit.trial_id: commit for commit in store.committed_attempts()}
    planned = set(design["trial_id"].astype(str))
    committed = set(commits)
    if planned != committed:
        raise RuntimeError(
            f"{label} trial panel mismatch: {len(planned - committed)} missing, "
            f"{len(committed - planned)} unexpected"
        )

    budget_by_trial = design.set_index("trial_id")["interaction_budget"].to_dict()
    budget_mismatches = {
        trial_id: (int(commit.metadata.get("observed_steps", -1)), budget_by_trial[trial_id])
        for trial_id, commit in commits.items()
        if int(commit.metadata.get("observed_steps", -1)) != budget_by_trial[trial_id]
    }
    if budget_mismatches:
        examples = list(budget_mismatches.items())[:5]
        raise RuntimeError(f"{label} interaction-budget mismatches: {examples!r}")

    snapshots = store.read_table(
        "snapshots",
        columns=("trial_id", "seed", "agent", "episode", "global_step", "snapshot_key"),
    )
    expected_snapshot_rows = sum(
        int(commit.row_counts.get("snapshots", 0)) for commit in commits.values()
    )
    if len(snapshots) != expected_snapshot_rows:
        raise RuntimeError(
            f"{label} snapshot row mismatch: read {len(snapshots)}, "
            f"commits declare {expected_snapshot_rows}"
        )
    return OpenRun(label, root, store, config, design, snapshots, commits)


def _environment_parameters(runs: Sequence[OpenRun]) -> EnvironmentParameters:
    values: set[EnvironmentParameters] = set()
    gammas: set[float] = set()
    stationary_epsilons: set[float] = set()
    for run in runs:
        for trial in run.config.trials():
            parameters = trial.environment.parameters
            values.add(
                EnvironmentParameters(
                    recoverable_hazard_penalty=float(parameters["recoverable_hazard_penalty"]),
                    lethal_hazard_penalty=float(parameters["lethal_hazard_penalty"]),
                    goal_reward=float(parameters["goal_reward"]),
                    step_reward=float(parameters["step_reward"]),
                    max_episode_steps=int(parameters["max_episode_steps"]),
                )
            )
            gammas.add(float(trial.agent.parameters["gamma"]))
            epsilon = trial.agent.parameters.get("epsilon")
            if run.label in {"recoverable", "lethal"} and isinstance(epsilon, (float, int)):
                stationary_epsilons.add(float(epsilon))
    if len(values) != 1:
        raise ValueError(f"Runs do not share one reward/environment specification: {values!r}")
    if gammas != {GAMMA}:
        raise ValueError(f"Expected gamma={GAMMA}, found {sorted(gammas)!r}")
    if stationary_epsilons != {PERSISTENT_EPSILON}:
        raise ValueError(
            f"Expected persistent epsilon={PERSISTENT_EPSILON}, "
            f"found {sorted(stationary_epsilons)!r}"
        )
    return next(iter(values))


@cache
def _environment_and_model(
    parameters: EnvironmentParameters,
    hazard_mode: str,
    reliability: float,
) -> tuple[RiskyCorridorEnv, Any]:
    env = RiskyCorridorEnv(
        corridor_reliability=float(reliability),
        hazard_mode=cast(Any, hazard_mode),
        recoverable_hazard_penalty=parameters.recoverable_hazard_penalty,
        lethal_hazard_penalty=parameters.lethal_hazard_penalty,
        goal_reward=parameters.goal_reward,
        step_reward=parameters.step_reward,
        max_episode_steps=parameters.max_episode_steps,
    )
    return env, env.exact_mdp()


@cache
def _oracle_solution(
    parameters: EnvironmentParameters,
    hazard_mode: str,
    reliability: float,
    epsilon: float,
    gamma: float,
) -> tuple[Any, np.ndarray]:
    _env, model = _environment_and_model(parameters, hazard_mode, reliability)
    solution: Any
    if epsilon == 0.0:
        solution = value_iteration(model, gamma=gamma)
        policy = solution.policy
    else:
        solution = epsilon_soft_value_iteration(model, epsilon=epsilon, gamma=gamma)
        policy = solution.greedy_policy
    if not solution.converged:
        raise RuntimeError(
            f"Exact solver did not converge for {hazard_mode}, p={reliability}, epsilon={epsilon}"
        )
    return solution, np.asarray(policy, dtype=int)


def _exact_gap(
    parameters: EnvironmentParameters,
    hazard_mode: str,
    reliability: float,
    epsilon: float,
) -> float:
    env, _model = _environment_and_model(parameters, hazard_mode, reliability)
    solution, _policy = _oracle_solution(parameters, hazard_mode, reliability, epsilon, GAMMA)
    start = env.state_to_index[env.fork_state]
    return float(
        solution.q_values[start, int(env.corridor_action)]
        - solution.q_values[start, int(env.shelter_action)]
    )


def _exact_threshold(
    parameters: EnvironmentParameters,
    hazard_mode: str,
    epsilon: float,
    *,
    low: float = 0.55,
    high: float = 1.0,
) -> float:
    left = float(low)
    right = float(high)
    left_gap = _exact_gap(parameters, hazard_mode, left, epsilon)
    right_gap = _exact_gap(parameters, hazard_mode, right, epsilon)
    if math.isclose(left_gap, 0.0, abs_tol=1e-13):
        return left
    if math.isclose(right_gap, 0.0, abs_tol=1e-13):
        return right
    if left_gap > 0.0 or right_gap < 0.0:
        return float("nan")
    for _ in range(60):
        middle = 0.5 * (left + right)
        middle_gap = _exact_gap(parameters, hazard_mode, middle, epsilon)
        if middle_gap <= 0.0:
            left = middle
        else:
            right = middle
    return 0.5 * (left + right)


def _oracle_tables(
    parameters: EnvironmentParameters,
    points: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    grid = np.linspace(0.55, 1.0, points)
    curve_rows = []
    threshold_rows = []
    for hazard_mode in ("recoverable", "lethal"):
        for epsilon in (0.0, 0.01, PERSISTENT_EPSILON):
            for reliability in grid:
                gap = _exact_gap(parameters, hazard_mode, float(reliability), epsilon)
                curve_rows.append(
                    {
                        "hazard_mode": hazard_mode,
                        "epsilon": epsilon,
                        "reliability": float(reliability),
                        "start_action_gap": gap,
                        "exact_choice": "corridor" if gap > 0.0 else "shelter",
                    }
                )
            threshold_rows.append(
                {
                    "hazard_mode": hazard_mode,
                    "epsilon": epsilon,
                    "threshold": _exact_threshold(parameters, hazard_mode, epsilon),
                    "gap_at_perfect_control": _exact_gap(parameters, hazard_mode, 1.0, epsilon),
                }
            )
    return pd.DataFrame(curve_rows), pd.DataFrame(threshold_rows)


def _classify_action(
    q_row: np.ndarray,
    env: RiskyCorridorEnv,
) -> tuple[str, int]:
    maximizers = np.flatnonzero(np.isclose(q_row, np.max(q_row), rtol=1e-10, atol=1e-12))
    if len(maximizers) != 1:
        return "tie", -1
    action = int(maximizers[0])
    if action == int(env.corridor_action):
        return "corridor", action
    if action == int(env.shelter_action):
        return "shelter", action
    return "other", action


def _selected_snapshot_rows(
    snapshots: pd.DataFrame,
    budget: int,
    targets: Sequence[float],
) -> list[pd.Series]:
    ordered = snapshots.loc[snapshots["episode"].ge(0)].sort_values("global_step")
    if ordered.empty:
        raise RuntimeError("Trial has no post-initialization Q snapshot")
    selected = []
    for target in targets:
        target_step = round(float(target) * budget)
        eligible = ordered.loc[ordered["global_step"].le(target_step)]
        selected.append(eligible.iloc[-1] if len(eligible) else ordered.iloc[0])
    return selected


def _analyze_q_snapshots(
    run: OpenRun,
    parameters: EnvironmentParameters,
    *,
    include_checkpoints: bool,
    annealed: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reader = SnapshotReader(run)
    design = run.design.set_index("trial_id")
    snapshots_by_trial = {
        str(trial_id): sample for trial_id, sample in run.snapshots.groupby("trial_id", sort=False)
    }
    final_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    ordered_trial_ids = sorted(design.index.astype(str))
    for number, trial_id in enumerate(ordered_trial_ids, start=1):
        trial = design.loc[trial_id]
        budget = int(trial["interaction_budget"])
        targets = PROGRESS_TARGETS if include_checkpoints else (1.0,)
        selected = _selected_snapshot_rows(snapshots_by_trial[trial_id], budget, targets)
        keys = [str(row["snapshot_key"]) for row in selected]
        tables = reader.load(trial_id, keys)

        hazard_mode = str(trial["hazard_mode"])
        reliability = float(trial["reliability"])
        agent = str(trial["agent"])
        seed = int(trial["seed"])
        env, model = _environment_and_model(parameters, hazard_mode, reliability)
        start = env.state_to_index[env.fork_state]

        for progress, row in zip(targets, selected, strict=True):
            table = tables[str(row["snapshot_key"])]
            choice, action = _classify_action(table[start], env)
            if include_checkpoints:
                target_step = round(progress * budget)
                checkpoint_rows.append(
                    {
                        "trial_id": trial_id,
                        "seed": seed,
                        "agent": agent,
                        "hazard_mode": hazard_mode,
                        "reliability": reliability,
                        "progress": progress,
                        "target_global_step": target_step,
                        "observed_global_step": int(row["global_step"]),
                        "step_lag": target_step - int(row["global_step"]),
                        "choice": choice,
                        "greedy_action": action,
                        "corridor_selected": float(choice == "corridor"),
                    }
                )

        final_row = selected[-1]
        table = tables[str(final_row["snapshot_key"])]
        choice, action = _classify_action(table[start], env)
        learned_policy = np.argmax(table, axis=1)
        learned_values = exact_policy_evaluation(
            model, learned_policy, gamma=GAMMA, method="direct"
        ).values
        greedy_optimum, _ = _oracle_solution(parameters, hazard_mode, reliability, 0.0, GAMMA)
        learned_gap = float(
            table[start, int(env.corridor_action)] - table[start, int(env.shelter_action)]
        )
        target_epsilon = 0.0 if annealed or agent == "q_learning" else PERSISTENT_EPSILON
        target_gap = _exact_gap(parameters, hazard_mode, reliability, target_epsilon)
        final_rows.append(
            {
                "trial_id": trial_id,
                "seed": seed,
                "agent": agent,
                "hazard_mode": hazard_mode,
                "reliability": reliability,
                "global_step": int(final_row["global_step"]),
                "choice": choice,
                "greedy_action": action,
                "corridor_selected": float(choice == "corridor"),
                "unresolved": choice in {"other", "tie"},
                "start_action_gap": learned_gap,
                "target_epsilon": target_epsilon,
                "target_oracle_gap": target_gap,
                "absolute_gap_error": abs(learned_gap - target_gap),
                "exact_deployment_regret": max(
                    0.0,
                    float(greedy_optimum.values[start] - learned_values[start]),
                ),
            }
        )
        if number % 100 == 0 or number == len(ordered_trial_ids):
            print(
                f"[{run.label}] analyzed {number}/{len(ordered_trial_ids)} trials",
                flush=True,
            )
    return pd.DataFrame(final_rows), pd.DataFrame(checkpoint_rows)


def _wilson_summary(
    frame: pd.DataFrame,
    *,
    value: str,
    groups: Sequence[str],
) -> pd.DataFrame:
    z = 1.959963984540054
    rows = []
    for keys, sample in frame.groupby(list(groups), dropna=False, sort=True):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        values = sample[value].to_numpy(dtype=float)
        if not np.isin(values, (0.0, 1.0)).all():
            raise ValueError(f"{value} must be binary for a Wilson interval")
        n = len(values)
        proportion = float(values.mean())
        denominator = 1.0 + z**2 / n
        center = (proportion + z**2 / (2.0 * n)) / denominator
        radius = (
            z * np.sqrt(proportion * (1.0 - proportion) / n + z**2 / (4.0 * n**2)) / denominator
        )
        rows.append(
            {
                **dict(zip(groups, key_tuple, strict=True)),
                "mean": proportion,
                "ci_low": max(0.0, center - radius),
                "ci_high": min(1.0, center + radius),
                "n_seeds": int(sample["seed"].nunique()),
                "n_observations": n,
            }
        )
    return pd.DataFrame(rows)


def _continuous_summary(
    frame: pd.DataFrame,
    *,
    value: str,
    groups: Sequence[str],
    n_resamples: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    for keys, sample in frame.groupby(list(groups), dropna=False, sort=True):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        values = sample[value].to_numpy(dtype=float)
        draws = values[rng.integers(0, len(values), size=(n_resamples, len(values)))]
        means = draws.mean(axis=1)
        rows.append(
            {
                **dict(zip(groups, key_tuple, strict=True)),
                "mean": float(values.mean()),
                "standard_deviation": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "ci_low": float(np.quantile(means, 0.025)),
                "ci_high": float(np.quantile(means, 0.975)),
                "n_seeds": int(sample["seed"].nunique()),
                "n_observations": len(values),
            }
        )
    return pd.DataFrame(rows)


def _unresolved_conditions(final_choices: pd.DataFrame) -> pd.DataFrame:
    groups = ["hazard_mode", "agent", "reliability"]
    result = (
        final_choices.groupby(groups, as_index=False)
        .agg(
            n_policies=("trial_id", "size"),
            unresolved_count=("unresolved", "sum"),
            unresolved_rate=("unresolved", "mean"),
        )
        .sort_values(groups)
    )
    result["condition_limit"] = UNRESOLVED_CONDITION_LIMIT
    result["passes"] = result["unresolved_rate"].le(UNRESOLVED_CONDITION_LIMIT)
    return result


def _endpoint_calibration(final_choices: pd.DataFrame) -> pd.DataFrame:
    expectations = (
        ("recoverable", "min", METHOD_ORDER, 0.0),
        ("recoverable", "max", METHOD_ORDER, 1.0),
        ("lethal", "min", METHOD_ORDER, 0.0),
        ("lethal", "max", ("q_learning",), 1.0),
        ("lethal", "max", ("sarsa", "expected_sarsa"), 0.0),
    )
    rows = []
    for hazard_mode, endpoint, agents, expected in expectations:
        mode_rows = final_choices.loc[final_choices["hazard_mode"].eq(hazard_mode)]
        reliability = float(
            mode_rows["reliability"].min() if endpoint == "min" else mode_rows["reliability"].max()
        )
        for agent in agents:
            observed = float(
                mode_rows.loc[
                    mode_rows["agent"].eq(agent) & mode_rows["reliability"].eq(reliability),
                    "corridor_selected",
                ].mean()
            )
            rows.append(
                {
                    "hazard_mode": hazard_mode,
                    "endpoint": endpoint,
                    "reliability": reliability,
                    "agent": agent,
                    "expected_corridor_fraction": expected,
                    "observed_corridor_fraction": observed,
                    "passes": observed >= 0.8 if expected == 1.0 else observed <= 0.2,
                }
            )
    return pd.DataFrame(rows)


def _empirical_half_boundary(sample: pd.DataFrame) -> tuple[float, int]:
    ordered = sample.sort_values("reliability")
    return _half_boundary_arrays(
        ordered["reliability"].to_numpy(dtype=float),
        ordered["mean"].to_numpy(dtype=float),
    )


def _half_boundary_arrays(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    violations = int(np.count_nonzero(np.diff(y) < -0.05))
    indices = np.flatnonzero(y >= 0.5)
    if not len(indices):
        return float("nan"), violations
    index = int(indices[0])
    if index == 0:
        return float(x[0]), violations
    if y[index] == y[index - 1]:
        return float(x[index]), violations
    weight = (0.5 - y[index - 1]) / (y[index] - y[index - 1])
    return float(x[index - 1] + weight * (x[index] - x[index - 1])), violations


def _seed_block_boundary_bootstrap(
    frame: pd.DataFrame,
    *,
    n_resamples: int,
    seed: int,
) -> pd.DataFrame:
    seeds = np.sort(frame["seed"].unique())
    rng = np.random.default_rng(seed)
    sampled_indices = rng.integers(0, len(seeds), size=(n_resamples, len(seeds)))
    rows = []
    for (hazard_mode, agent), sample in frame.groupby(["hazard_mode", "agent"], sort=True):
        panel = sample.pivot(
            index="seed", columns="reliability", values="corridor_selected"
        ).reindex(index=seeds)
        if panel.isna().any().any():
            raise ValueError(f"Incomplete matched seed panel for {hazard_mode}/{agent}")
        reliability = panel.columns.to_numpy(dtype=float)
        values = panel.to_numpy(dtype=float)
        curves = values[sampled_indices].mean(axis=1)
        for bootstrap_id, curve in enumerate(curves):
            boundary, violations = _half_boundary_arrays(reliability, curve)
            rows.append(
                {
                    "bootstrap_id": bootstrap_id,
                    "hazard_mode": hazard_mode,
                    "agent": agent,
                    "half_boundary": boundary,
                    "right_censored": not np.isfinite(boundary),
                    "monotonicity_violations": violations,
                }
            )
    return pd.DataFrame(rows)


def _boundary_tables(
    final_choices: pd.DataFrame,
    checkpoint_summary: pd.DataFrame,
    *,
    n_resamples: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    learned_rows = []
    for keys, sample in checkpoint_summary.groupby(
        ["hazard_mode", "agent", "progress"], sort=False
    ):
        boundary, violations = _empirical_half_boundary(sample)
        learned_rows.append(
            {
                "hazard_mode": keys[0],
                "agent": keys[1],
                "progress": keys[2],
                "half_boundary": boundary,
                "monotonicity_violations": violations,
            }
        )
    learned = pd.DataFrame(learned_rows)
    bootstrap = _seed_block_boundary_bootstrap(final_choices, n_resamples=n_resamples, seed=41)
    summary_rows = []
    for (hazard_mode, agent), sample in bootstrap.groupby(["hazard_mode", "agent"], sort=False):
        finite = sample.loc[np.isfinite(sample["half_boundary"]), "half_boundary"]
        summary_rows.append(
            {
                "hazard_mode": hazard_mode,
                "agent": agent,
                "median_half_boundary": float(finite.median()) if len(finite) else np.nan,
                "ci_low": float(finite.quantile(0.025)) if len(finite) else np.nan,
                "ci_high": float(finite.quantile(0.975)) if len(finite) else np.nan,
                "right_censored_fraction": float(sample["right_censored"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    observed = learned.loc[
        learned["progress"].eq(1.0),
        ["hazard_mode", "agent", "half_boundary", "monotonicity_violations"],
    ].rename(columns={"half_boundary": "observed_half_boundary"})
    summary = observed.merge(summary, on=["hazard_mode", "agent"], validate="one_to_one")

    wide = bootstrap.pivot(
        index=["bootstrap_id", "hazard_mode"], columns="agent", values="half_boundary"
    )
    contrast_rows = []
    for hazard_mode, sample in wide.groupby(level="hazard_mode"):
        for comparison in ("sarsa", "expected_sarsa"):
            differences = (sample[comparison] - sample["q_learning"]).dropna()
            contrast_rows.append(
                {
                    "hazard_mode": hazard_mode,
                    "comparison": comparison,
                    "median_boundary_shift_vs_q": float(differences.median())
                    if len(differences)
                    else np.nan,
                    "ci_low": float(differences.quantile(0.025)) if len(differences) else np.nan,
                    "ci_high": float(differences.quantile(0.975)) if len(differences) else np.nan,
                    "complete_bootstrap_pairs": len(differences),
                }
            )
    return learned, bootstrap, summary, pd.DataFrame(contrast_rows)


def _stability_summary(checkpoints: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    late = checkpoints.loc[checkpoints["progress"].isin([0.75, 1.0])]
    choice_wide = late.pivot(
        index=["trial_id", "seed", "agent", "hazard_mode", "reliability"],
        columns="progress",
        values="choice",
    )
    binary_wide = late.pivot(
        index=["trial_id", "seed", "agent", "hazard_mode", "reliability"],
        columns="progress",
        values="corridor_selected",
    )
    choice_wide["changed_choice"] = choice_wide[0.75] != choice_wide[1.0]
    choice_wide["changed_corridor_indicator"] = binary_wide[0.75] != binary_wide[1.0]
    summary = (
        choice_wide.reset_index()
        .groupby(["hazard_mode", "agent", "reliability"], as_index=False)
        .agg(
            changed_choice_fraction=("changed_choice", "mean"),
            changed_corridor_indicator_fraction=("changed_corridor_indicator", "mean"),
            n_seeds=("seed", "nunique"),
        )
    )
    # Compatibility name used by the compact Notebook 05 presentation layer.
    summary["changed_fraction"] = summary["changed_choice_fraction"]
    endpoint_parts = []
    for hazard_mode in ("recoverable", "lethal"):
        sample = summary.loc[summary["hazard_mode"].eq(hazard_mode)]
        for reliability in (
            float(sample["reliability"].min()),
            float(sample["reliability"].max()),
        ):
            endpoint_parts.append(sample.loc[sample["reliability"].eq(reliability)])
    endpoints = pd.concat(endpoint_parts, ignore_index=True)
    endpoints["condition_limit"] = 0.20
    endpoints["passes"] = endpoints["changed_choice_fraction"].le(0.20)
    return summary, endpoints


def _evaluation_trial_summary(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = [
        "trial_id",
        "agent",
        "seed",
        "env_hazard_mode",
        "env_action_reliability",
        "evaluation_policy_mode",
    ]
    result = (
        frame.groupby(grouped, as_index=False)
        .agg(
            episode_return=("episode_return", "mean"),
            success=("success", "mean"),
            failure=("failure", "mean"),
            truncation=("truncated", "mean"),
            truncated_episodes=("truncated", "sum"),
            evaluation_episodes=("truncated", "size"),
            total_episode_steps=("episode_length", "sum"),
            hazard_penalty_steps=("env_hazard_penalty_steps", "sum"),
            realized_corridor=(
                "env_realized_route",
                lambda values: float(np.mean(np.asarray(values) == "corridor")),
            ),
        )
        .rename(
            columns={
                "env_hazard_mode": "hazard_mode",
                "env_action_reliability": "reliability",
            }
        )
    )
    result["penalty_steps_per_1000"] = (
        1_000.0 * result["hazard_penalty_steps"] / result["total_episode_steps"]
    )
    return result


def _truncation_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Report held-out truncations as outcomes rather than hiding them in means."""

    groups = [
        "env_hazard_mode",
        "agent",
        "env_action_reliability",
        "evaluation_policy_mode",
    ]
    result = (
        frame.groupby(groups, as_index=False)
        .agg(
            evaluation_episodes=("truncated", "size"),
            terminated_episodes=("terminated", "sum"),
            truncated_episodes=("truncated", "sum"),
        )
        .rename(
            columns={
                "env_hazard_mode": "hazard_mode",
                "env_action_reliability": "reliability",
            }
        )
    )
    result["truncation_rate"] = result["truncated_episodes"] / result["evaluation_episodes"]
    return result


def _exact_backup_moments(
    model: Any,
    q_values: np.ndarray,
    policy: np.ndarray,
    *,
    state: int,
    action: int,
    integrate_next_action: bool,
) -> tuple[float, float]:
    probabilities = []
    targets = []
    for next_state, transition_probability in enumerate(model.P[state, action]):
        if transition_probability == 0.0:
            continue
        reward = float(model.R[state, action, next_state])
        if model.terminal[next_state]:
            probabilities.append(float(transition_probability))
            targets.append(reward)
        elif integrate_next_action:
            probabilities.append(float(transition_probability))
            targets.append(reward + GAMMA * float(np.dot(policy[next_state], q_values[next_state])))
        else:
            for next_action, action_probability in enumerate(policy[next_state]):
                if action_probability == 0.0:
                    continue
                probabilities.append(float(transition_probability * action_probability))
                targets.append(reward + GAMMA * float(q_values[next_state, next_action]))
    weights = np.asarray(probabilities, dtype=float)
    values = np.asarray(targets, dtype=float)
    weights /= weights.sum()
    mean = float(np.dot(weights, values))
    variance = float(np.dot(weights, np.square(values - mean)))
    return mean, variance


def _backup_variance(parameters: EnvironmentParameters) -> pd.DataFrame:
    rows = []
    for hazard_mode, reliability in DISAGREEMENT_POINTS.items():
        env, model = _environment_and_model(parameters, hazard_mode, reliability)
        solution, _policy = _oracle_solution(
            parameters, hazard_mode, reliability, PERSISTENT_EPSILON, GAMMA
        )
        state = env.state_to_index[env.fork_state]
        action = int(env.corridor_action)
        expected_mean, expected_variance = _exact_backup_moments(
            model,
            solution.q_values,
            solution.policy,
            state=state,
            action=action,
            integrate_next_action=True,
        )
        sampled_mean, sampled_variance = _exact_backup_moments(
            model,
            solution.q_values,
            solution.policy,
            state=state,
            action=action,
            integrate_next_action=False,
        )
        np.testing.assert_allclose(sampled_mean, expected_mean, atol=1e-12)
        rows.append(
            {
                "hazard_mode": hazard_mode,
                "reliability": reliability,
                "target_mean": expected_mean,
                "expected_sarsa_variance": expected_variance,
                "sarsa_variance": sampled_variance,
                "next_action_sampling_component": sampled_variance - expected_variance,
            }
        )
    return pd.DataFrame(rows)


def _choice_counts(final_choices: pd.DataFrame) -> pd.DataFrame:
    return (
        final_choices.groupby(["hazard_mode", "agent", "choice"], as_index=False)
        .size()
        .rename(columns={"size": "training_seeds"})
    )


def _write_frame(frame: pd.DataFrame, output_dir: Path, name: str) -> Path:
    path = output_dir / f"{name}.csv"
    frame.to_csv(path, index=False, float_format="%.12g")
    return path


def _configure_plots() -> None:
    matplotlib.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#252525",
            "axes.labelcolor": "#252525",
            "axes.titleweight": "bold",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "grid.color": "#D7D7D2",
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "svg.hashsalt": "irl-lab-shortcut-or-shelter",
        }
    )


def _save_figure(fig: Any, output_dir: Path, stem: str, *, png: bool) -> list[Path]:
    paths = [output_dir / f"{stem}.svg"]
    fig.savefig(
        paths[0],
        bbox_inches="tight",
        facecolor="white",
        metadata={"Creator": "Independent RL Lab"},
    )
    svg_text = paths[0].read_text(encoding="utf-8")
    paths[0].write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    if png:
        path = output_dir / f"{stem}.png"
        fig.savefig(path, bbox_inches="tight", facecolor="white", dpi=180)
        paths.append(path)
    plt.close(fig)
    return paths


def _plot_exact_boundaries(
    curve: pd.DataFrame,
    output_dir: Path,
    *,
    png: bool,
) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.25))
    styles = {0.0: "-", 0.01: "--", PERSISTENT_EPSILON: ":"}
    for ax, hazard_mode in zip(axes, ("recoverable", "lethal"), strict=True):
        sample = curve.loc[curve["hazard_mode"].eq(hazard_mode)]
        for epsilon, line in sample.groupby("epsilon", sort=True):
            label = "greedy" if epsilon == 0 else f"epsilon-soft, ε={epsilon:g}"
            ax.plot(
                line["reliability"].to_numpy(dtype=float),
                line["start_action_gap"].to_numpy(dtype=float),
                linestyle=styles[float(epsilon)],
                linewidth=2.2,
                label=label,
            )
        ax.axhline(0.0, color="#252525", linewidth=1)
        ax.set(
            xlabel="intended-action reliability p",
            ylabel="exact EAST - SOUTH value",
            title=f"{hazard_mode.title()} hazards",
        )
        if hazard_mode == "lethal":
            ax.set_xlim(0.965, 1.001)
        ax.grid(alpha=0.65)
        ax.legend(fontsize=8)
    fig.suptitle("The exact route boundary depends on the deployed policy class", y=1.01)
    fig.tight_layout()
    return _save_figure(fig, output_dir, "oracle_gaps", png=png)


def _plot_route_selection(
    summary: pd.DataFrame,
    thresholds: pd.DataFrame,
    output_dir: Path,
    *,
    png: bool,
) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))
    for ax, hazard_mode in zip(axes, ("recoverable", "lethal"), strict=True):
        sample = summary.loc[summary["hazard_mode"].eq(hazard_mode)]
        for agent in METHOD_ORDER:
            line = sample.loc[sample["agent"].eq(agent)].sort_values("reliability")
            x = line["reliability"].to_numpy(dtype=float)
            mean = line["mean"].to_numpy(dtype=float)
            low = line["ci_low"].to_numpy(dtype=float)
            high = line["ci_high"].to_numpy(dtype=float)
            ax.plot(
                x,
                mean,
                color=METHOD_COLORS[agent],
                marker="o",
                markersize=4,
                linewidth=1.8,
                label=METHOD_LABELS[agent],
            )
            ax.fill_between(x, low, high, color=METHOD_COLORS[agent], alpha=0.12)
        greedy = thresholds.loc[
            thresholds["hazard_mode"].eq(hazard_mode) & thresholds["epsilon"].eq(0.0),
            "threshold",
        ].iloc[0]
        soft = thresholds.loc[
            thresholds["hazard_mode"].eq(hazard_mode)
            & thresholds["epsilon"].eq(PERSISTENT_EPSILON),
            "threshold",
        ].iloc[0]
        ax.axvline(greedy, color="#252525", linestyle="--", label="greedy oracle")
        if np.isfinite(soft):
            ax.axvline(soft, color="#66645E", linestyle=":", label="ε-soft oracle")
        else:
            ax.text(
                0.02,
                0.04,
                "ε-soft oracle stays with shelter",
                transform=ax.transAxes,
                fontsize=8,
            )
        ax.set(
            title=f"{hazard_mode.title()} hazards",
            xlabel="intended-action reliability p",
            ylabel="fraction selecting corridor",
            ylim=(-0.04, 1.04),
        )
        ax.grid(alpha=0.65)
        ax.legend(fontsize=7.5)
    fig.suptitle("Final greedy route choice across twenty training seeds", y=1.01)
    fig.tight_layout()
    return _save_figure(fig, output_dir, "route_selection", png=png)


def _plot_boundaries_over_training(
    learned: pd.DataFrame,
    thresholds: pd.DataFrame,
    output_dir: Path,
    *,
    png: bool,
) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.3))
    for ax, hazard_mode in zip(axes, ("recoverable", "lethal"), strict=True):
        sample = learned.loc[learned["hazard_mode"].eq(hazard_mode)]
        for agent in METHOD_ORDER:
            line = sample.loc[sample["agent"].eq(agent)].sort_values("progress")
            ax.plot(
                100 * line["progress"].to_numpy(dtype=float),
                line["half_boundary"].to_numpy(dtype=float),
                color=METHOD_COLORS[agent],
                marker="o",
                linewidth=1.9,
                label=METHOD_LABELS[agent],
            )
        greedy = thresholds.loc[
            thresholds["hazard_mode"].eq(hazard_mode) & thresholds["epsilon"].eq(0.0),
            "threshold",
        ].iloc[0]
        soft = thresholds.loc[
            thresholds["hazard_mode"].eq(hazard_mode)
            & thresholds["epsilon"].eq(PERSISTENT_EPSILON),
            "threshold",
        ].iloc[0]
        ax.axhline(greedy, color="#252525", linestyle="--", label="greedy oracle")
        if np.isfinite(soft):
            ax.axhline(soft, color="#66645E", linestyle=":", label="ε-soft oracle")
        ax.set(
            title=f"{hazard_mode.title()} hazards",
            xlabel="interaction budget completed (%)",
            ylabel="50% corridor boundary",
            xticks=(25, 50, 75, 100),
        )
        ax.grid(alpha=0.65)
        ax.legend(fontsize=7.5)
    fig.suptitle("The learned boundary moves over the training budget", y=1.01)
    fig.tight_layout()
    return _save_figure(fig, output_dir, "boundary_training", png=png)


def _plot_consequences(
    regret: pd.DataFrame,
    returns: pd.DataFrame,
    output_dir: Path,
    *,
    png: bool,
) -> list[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.6), constrained_layout=True)
    for row_index, hazard_mode in enumerate(("recoverable", "lethal")):
        for agent in METHOD_ORDER:
            line = regret.loc[
                regret["hazard_mode"].eq(hazard_mode) & regret["agent"].eq(agent)
            ].sort_values("reliability")
            x = line["reliability"].to_numpy(dtype=float)
            axes[row_index, 0].plot(
                x,
                line["mean"].to_numpy(dtype=float),
                color=METHOD_COLORS[agent],
                marker="o",
                markersize=3.5,
                label=METHOD_LABELS[agent],
            )
            axes[row_index, 0].fill_between(
                x,
                line["ci_low"].to_numpy(dtype=float),
                line["ci_high"].to_numpy(dtype=float),
                color=METHOD_COLORS[agent],
                alpha=0.10,
            )
            for policy_mode, linestyle in (("greedy", "-"), ("behavior", "--")):
                held_out = returns.loc[
                    returns["hazard_mode"].eq(hazard_mode)
                    & returns["agent"].eq(agent)
                    & returns["evaluation_policy_mode"].eq(policy_mode)
                ].sort_values("reliability")
                axes[row_index, 1].plot(
                    held_out["reliability"].to_numpy(dtype=float),
                    held_out["mean"].to_numpy(dtype=float),
                    color=METHOD_COLORS[agent],
                    linestyle=linestyle,
                    marker="o" if policy_mode == "greedy" else None,
                    markersize=3.5,
                    label=f"{METHOD_LABELS[agent]} — {policy_mode}",
                )
        axes[row_index, 0].set(
            title=f"{hazard_mode.title()}: exact deployment regret",
            xlabel="reliability p",
            ylabel="start-state regret",
        )
        axes[row_index, 1].set(
            title=f"{hazard_mode.title()}: paired held-out return",
            xlabel="reliability p",
            ylabel="episode return",
        )
        for ax in axes[row_index]:
            ax.grid(alpha=0.65)
            ax.legend(fontsize=6.8, ncol=2 if ax is axes[row_index, 1] else 1)
    return _save_figure(fig, output_dir, "consequences", png=png)


def _plot_mechanism_checks(
    annealed: pd.DataFrame,
    variance: pd.DataFrame,
    output_dir: Path,
    *,
    png: bool,
) -> list[Path]:
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.75), constrained_layout=True)
    for ax, hazard_mode in zip(axes[:2], ("recoverable", "lethal"), strict=True):
        sample = annealed.loc[annealed["hazard_mode"].eq(hazard_mode)].copy()
        sample["method_order"] = sample["agent"].map(
            {method: index for index, method in enumerate(METHOD_ORDER)}
        )
        sample = sample.sort_values("method_order")
        positions = np.arange(len(sample))
        means = sample["mean"].to_numpy(dtype=float)
        ax.bar(
            positions,
            means,
            color=[METHOD_COLORS[str(agent)] for agent in sample["agent"]],
        )
        ax.errorbar(
            positions,
            means,
            yerr=np.maximum(
                0.0,
                np.vstack(
                    [
                        means - sample["ci_low"].to_numpy(dtype=float),
                        sample["ci_high"].to_numpy(dtype=float) - means,
                    ]
                ),
            ),
            fmt="none",
            color="#252525",
            capsize=3,
            linewidth=1,
        )
        reliability = float(sample["reliability"].iloc[0])
        ax.set(
            title=f"Annealed: {hazard_mode}\np={reliability:g}",
            ylabel="fraction selecting corridor",
            xticks=positions,
            xticklabels=[METHOD_LABELS[str(agent)] for agent in sample["agent"]],
            ylim=(0.0, 1.04),
        )
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.65)

    ax = axes[2]
    positions = np.arange(len(variance))
    width = 0.36
    ax.bar(
        positions - width / 2,
        variance["expected_sarsa_variance"].to_numpy(dtype=float),
        width,
        color=METHOD_COLORS["expected_sarsa"],
        label="Expected SARSA",
    )
    ax.bar(
        positions + width / 2,
        variance["sarsa_variance"].to_numpy(dtype=float),
        width,
        color=METHOD_COLORS["sarsa"],
        label="SARSA",
    )
    ax.set(
        title="Exact target variance",
        ylabel="one-step target variance",
        xticks=positions,
        xticklabels=[str(mode).title() for mode in variance["hazard_mode"]],
    )
    ax.grid(axis="y", alpha=0.65)
    ax.legend(fontsize=7.5)
    return _save_figure(fig, output_dir, "mechanism_checks", png=png)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _json_number(value: Any) -> Any:
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_number(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_number(item) for item in value]
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [_json_number(record) for record in frame.to_dict(orient="records")]


def analyze(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_plots()

    print("Opening completed runs and validating their committed budgets…", flush=True)
    runs = {
        "recoverable": _open_run("recoverable", Path(args.recoverable_run)),
        "lethal": _open_run("lethal", Path(args.lethal_run)),
        "annealed": _open_run("annealed", Path(args.annealed_run)),
    }
    parameters = _environment_parameters(tuple(runs.values()))

    stationary_final_parts = []
    checkpoint_parts = []
    for label in ("recoverable", "lethal"):
        final, checkpoints = _analyze_q_snapshots(runs[label], parameters, include_checkpoints=True)
        stationary_final_parts.append(final)
        checkpoint_parts.append(checkpoints)
    final_choices = pd.concat(stationary_final_parts, ignore_index=True)
    checkpoint_choices = pd.concat(checkpoint_parts, ignore_index=True)
    annealed_choices, _unused = _analyze_q_snapshots(
        runs["annealed"],
        parameters,
        include_checkpoints=False,
        annealed=True,
    )

    choice_counts = _choice_counts(final_choices)
    unresolved_conditions = _unresolved_conditions(final_choices)
    print("\nFinal stationary route counts:", flush=True)
    print(choice_counts.to_string(index=False), flush=True)
    print(
        "Maximum unresolved condition rate: "
        f"{unresolved_conditions['unresolved_rate'].max():.1%} "
        f"(limit {UNRESOLVED_CONDITION_LIMIT:.0%})",
        flush=True,
    )

    print("Computing exact boundaries and seed-block uncertainty…", flush=True)
    oracle_curve, exact_thresholds = _oracle_tables(parameters, int(args.oracle_points))
    corridor_summary = _wilson_summary(
        final_choices,
        value="corridor_selected",
        groups=("hazard_mode", "agent", "reliability"),
    )
    checkpoint_summary = _wilson_summary(
        checkpoint_choices,
        value="corridor_selected",
        groups=("hazard_mode", "agent", "progress", "reliability"),
    )
    learned_boundaries, boundary_bootstrap, final_boundary_summary, boundary_contrasts = (
        _boundary_tables(
            final_choices,
            checkpoint_summary,
            n_resamples=int(args.bootstrap_resamples),
        )
    )
    print("\nFinal learned boundaries:", flush=True)
    print(final_boundary_summary.to_string(index=False), flush=True)

    stability_summary, endpoint_stability = _stability_summary(checkpoint_choices)
    endpoint_calibration = _endpoint_calibration(final_choices)
    annealed_summary = _wilson_summary(
        annealed_choices,
        value="corridor_selected",
        groups=("hazard_mode", "agent", "reliability"),
    )

    evaluation_frames = []
    evaluation_columns = (
        "trial_id",
        "agent",
        "seed",
        "evaluation_seed",
        "evaluation_episode",
        "evaluation_policy_mode",
        "episode_return",
        "episode_length",
        "success",
        "failure",
        "terminated",
        "truncated",
        "env_hazard_mode",
        "env_action_reliability",
        "env_hazard_penalty_steps",
        "env_realized_route",
    )
    for label in ("recoverable", "lethal"):
        evaluation_frames.append(
            runs[label].store.read_table("evaluations", columns=evaluation_columns)
        )
    evaluations = pd.concat(evaluation_frames, ignore_index=True)
    evaluation_seed_pairs = evaluations.pivot_table(
        index=("trial_id", "evaluation_episode"),
        columns="evaluation_policy_mode",
        values="evaluation_seed",
        aggfunc="first",
    )
    evaluation_pairing_pass = bool(
        len(evaluation_seed_pairs) * 2 == len(evaluations)
        and {"greedy", "behavior"}.issubset(evaluation_seed_pairs.columns)
        and evaluation_seed_pairs[["greedy", "behavior"]].notna().all().all()
        and evaluation_seed_pairs["greedy"].eq(evaluation_seed_pairs["behavior"]).all()
    )
    evaluation_trials = _evaluation_trial_summary(evaluations)
    truncation_summary = _truncation_summary(evaluations)
    n_resamples = int(args.bootstrap_resamples)
    regret_summary = _continuous_summary(
        final_choices,
        value="exact_deployment_regret",
        groups=("hazard_mode", "agent", "reliability"),
        n_resamples=n_resamples,
        seed=53,
    )
    return_summary = _continuous_summary(
        evaluation_trials,
        value="episode_return",
        groups=("hazard_mode", "agent", "reliability", "evaluation_policy_mode"),
        n_resamples=n_resamples,
        seed=59,
    )
    exposure_summary = _continuous_summary(
        evaluation_trials,
        value="penalty_steps_per_1000",
        groups=("hazard_mode", "agent", "reliability", "evaluation_policy_mode"),
        n_resamples=n_resamples,
        seed=61,
    )
    failure_summary = _continuous_summary(
        evaluation_trials,
        value="failure",
        groups=("hazard_mode", "agent", "reliability", "evaluation_policy_mode"),
        n_resamples=n_resamples,
        seed=67,
    )
    backup_variance = _backup_variance(parameters)

    frames = {
        "exact_oracle_curve": oracle_curve,
        "exact_thresholds": exact_thresholds,
        "final_choices": final_choices,
        "choice_counts": choice_counts,
        "unresolved_conditions": unresolved_conditions,
        "endpoint_calibration": endpoint_calibration,
        "corridor_summary": corridor_summary,
        "checkpoint_choices": checkpoint_choices,
        "checkpoint_summary": checkpoint_summary,
        "stability_summary": stability_summary,
        "endpoint_stability": endpoint_stability,
        "learned_boundaries": learned_boundaries,
        "boundary_bootstrap": boundary_bootstrap,
        "final_boundary_summary": final_boundary_summary,
        "boundary_contrasts": boundary_contrasts,
        "evaluation_trial_summary": evaluation_trials,
        "truncation_summary": truncation_summary,
        "regret_summary": regret_summary,
        "return_summary": return_summary,
        "exposure_summary": exposure_summary,
        "failure_summary": failure_summary,
        "annealed_choices": annealed_choices,
        "annealed_summary": annealed_summary,
        "backup_variance": backup_variance,
    }
    written = [_write_frame(frame, output_dir, name) for name, frame in frames.items()]

    png = not bool(args.no_png)
    written.extend(_plot_exact_boundaries(oracle_curve, output_dir, png=png))
    written.extend(_plot_route_selection(corridor_summary, exact_thresholds, output_dir, png=png))
    written.extend(
        _plot_boundaries_over_training(learned_boundaries, exact_thresholds, output_dir, png=png)
    )
    written.extend(_plot_consequences(regret_summary, return_summary, output_dir, png=png))
    written.extend(
        _plot_mechanism_checks(
            annealed_summary,
            backup_variance,
            output_dir,
            png=png,
        )
    )

    unresolved_pass = bool(unresolved_conditions["passes"].all())
    endpoint_pass = bool(endpoint_calibration["passes"].all())
    stability_pass = bool(endpoint_stability["passes"].all())
    truncated_episodes = int(truncation_summary["truncated_episodes"].sum())
    evaluation_episodes = int(truncation_summary["evaluation_episodes"].sum())
    truncation_pass = truncated_episodes == 0
    primary_gates_pass = (
        unresolved_pass and endpoint_pass and stability_pass and evaluation_pairing_pass
    )
    source_runs = {}
    for label, run in runs.items():
        source_runs[label] = {
            "run_id": run.store.manifest.run_id,
            "experiment_name": run.store.manifest.experiment_name,
            "directory_name": run.path.name,
            "manifest_sha256": _sha256(run.path / "manifest.json"),
            "config_sha256": _sha256(run.path / "config.json"),
            "trial_count": len(run.design),
            "status": run.store.manifest.status,
            "observed_steps_min": min(
                int(commit.metadata["observed_steps"]) for commit in run.commits.values()
            ),
            "observed_steps_max": max(
                int(commit.metadata["observed_steps"]) for commit in run.commits.values()
            ),
            "all_interaction_budgets_match": True,
        }
    quality_gates = {
        "complete_interaction_budgets": {
            "passes": True,
            "required_steps_per_trial": 100_000,
            "trials_checked": sum(len(run.design) for run in runs.values()),
        },
        "paired_evaluation_seeds": {
            "passes": evaluation_pairing_pass,
            "paired_panels_checked": len(evaluation_seed_pairs),
        },
        "unresolved_condition": {
            "passes": unresolved_pass,
            "maximum_observed_rate": float(unresolved_conditions["unresolved_rate"].max()),
            "limit": UNRESOLVED_CONDITION_LIMIT,
        },
        "endpoint_calibration": {"passes": endpoint_pass},
        "endpoint_last_quarter_stability": {"passes": stability_pass},
        "held_out_truncation": {
            "passes": truncation_pass,
            "blocking": False,
            "role": "strict descriptive diagnostic, not a pre-specified publication gate",
            "truncated_episodes": truncated_episodes,
            "evaluation_episodes": evaluation_episodes,
            "truncation_rate": truncated_episodes / evaluation_episodes,
            "rule": "zero held-out episodes may reach the time limit",
        },
        "all_primary_pass": primary_gates_pass,
        "all_diagnostics_clear": truncation_pass,
        # Compatibility summary: only declared/amended blocking gates contribute.
        "all_pass": primary_gates_pass,
    }
    manifest = {
        "analysis": "shortcut_or_shelter",
        "analysis_version": 1,
        "analysis_schema_version": 1,
        "analysis_source_sha256": _sha256(Path(__file__).resolve()),
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "amendment": AMENDMENT,
        "run_ids": {label: run.store.manifest.run_id for label, run in runs.items()},
        "primary_trial_count": len(final_choices),
        "sensitivity_trial_count": len(annealed_choices),
        "source_runs": source_runs,
        "settings": {
            "gamma": GAMMA,
            "persistent_epsilon": PERSISTENT_EPSILON,
            "unresolved_condition_limit": UNRESOLVED_CONDITION_LIMIT,
            "bootstrap_resamples": n_resamples,
            "oracle_points": int(args.oracle_points),
            "environment": _json_number(
                {
                    "recoverable_hazard_penalty": parameters.recoverable_hazard_penalty,
                    "lethal_hazard_penalty": parameters.lethal_hazard_penalty,
                    "goal_reward": parameters.goal_reward,
                    "step_reward": parameters.step_reward,
                    "max_episode_steps": parameters.max_episode_steps,
                }
            ),
        },
        "counts": {
            "stationary_policies": len(final_choices),
            "annealed_policies": len(annealed_choices),
            "unresolved_stationary_policies": int(final_choices["unresolved"].sum()),
            "held_out_trial_summaries": len(evaluation_trials),
        },
        "quality_gates": quality_gates,
        # Compatibility alias for the first compact-notebook draft.
        "gates": quality_gates,
        "exact_thresholds": _records(exact_thresholds),
        "final_boundary_summary": _records(final_boundary_summary),
        "boundary_contrasts": _records(boundary_contrasts),
        "choice_counts": _records(choice_counts),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "artifacts": {},
    }
    manifest["artifacts"] = {
        path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in sorted(written)
    }
    manifest_path = output_dir / "analysis_manifest.json"
    manifest_path.write_text(
        json.dumps(_json_number(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote {len(written) + 1} artifacts to {output_dir}", flush=True)
    print(
        "Publication gates: "
        f"unresolved={unresolved_pass}, endpoints={endpoint_pass}, "
        f"stability={stability_pass}; strict nonblocking truncation diagnostic="
        f"{truncation_pass} ({truncated_episodes}/{evaluation_episodes})",
        flush=True,
    )
    return manifest_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recoverable-run", required=True, type=Path)
    parser.add_argument("--lethal-run", required=True, type=Path)
    parser.add_argument("--annealed-run", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/shortcut_or_shelter"),
        help="Compact output directory (default: reports/shortcut_or_shelter).",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=2_000,
        help="Matched seed-block and mean-bootstrap resamples (default: 2000).",
    )
    parser.add_argument(
        "--oracle-points",
        type=int,
        default=301,
        help="Points in each exact plotting curve; roots use bisection (default: 301).",
    )
    parser.add_argument(
        "--no-png",
        action="store_true",
        help="Write vector SVG figures only.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.bootstrap_resamples < 100:
        parser.error("--bootstrap-resamples must be at least 100")
    if args.oracle_points < 51:
        parser.error("--oracle-points must be at least 51")
    try:
        analyze(args)
    except KeyboardInterrupt:
        raise
    except Exception as error:
        print(f"analysis failed: {error}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
