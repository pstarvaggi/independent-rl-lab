from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rllab.evaluation import evaluation_checkpoint_summary, final_performance
from rllab.metrics import (
    UnsafeAggregationError,
    aggregate_learning_curves,
    autocorrelation,
    bootstrap_confidence_interval,
    distribution_summary,
    paired_seed_contrast,
    rolling_td_statistics,
    td_error_summary,
    td_error_trial_summary,
)


def test_bootstrap_interval_is_reproducible_and_contains_mean() -> None:
    values = np.arange(1.0, 11.0)
    first = bootstrap_confidence_interval(values, n_resamples=500, seed=8)
    second = bootstrap_confidence_interval(values, n_resamples=500, seed=8)
    assert first == second
    assert first[0] < np.mean(values) < first[1]


def test_distribution_summary_exposes_robust_and_tail_statistics() -> None:
    summary = distribution_summary([1, 2, 3, 100, np.nan])
    assert summary["count"] == 4
    assert summary["median"] == 2.5
    assert summary["mean"] > summary["median"]
    assert summary["q95"] > summary["q75"]
    assert summary["mad"] == 1.0


def test_learning_curve_aggregation_uses_seed_replicates() -> None:
    frame = pd.DataFrame(
        {
            "trial_id": ["q-0", "q-0", "q-1", "q-1"],
            "agent": ["q"] * 4,
            "seed": [0, 0, 1, 1],
            "episode": [0, 1, 0, 1],
            "episode_return": [1, 3, 5, 7],
        }
    )
    summary = aggregate_learning_curves(frame, n_resamples=100)
    assert summary["n_seeds"].tolist() == [2, 2]
    assert summary["n_units"].tolist() == [2, 2]
    assert summary["mean"].tolist() == pytest.approx([3.0, 5.0])


def test_learning_curve_rejects_duplicate_unit_episode_rows() -> None:
    frame = pd.DataFrame(
        {
            "trial_id": ["q-0", "q-0"],
            "agent": ["q", "q"],
            "seed": [0, 0],
            "episode": [0, 0],
            "episode_return": [1.0, 2.0],
        }
    )
    with pytest.raises(UnsafeAggregationError, match="one row per"):
        aggregate_learning_curves(frame)


def test_learning_curve_rejects_hidden_sweep_conditions() -> None:
    rows = []
    for reliability in (0.8, 0.9):
        for seed in (0, 1):
            rows.append(
                {
                    "trial_id": f"p{reliability}-{seed}",
                    "condition_id": f"p{reliability}",
                    "agent": "q",
                    "seed": seed,
                    "episode": 0,
                    "episode_return": reliability + seed,
                    "sweep_environment_action_reliability": reliability,
                }
            )
    frame = pd.DataFrame(rows)
    with pytest.raises(UnsafeAggregationError, match="multiple experimental conditions"):
        aggregate_learning_curves(frame, groups=("agent",))

    summary = aggregate_learning_curves(
        frame,
        groups=("sweep_environment_action_reliability",),
        n_resamples=100,
    )
    assert len(summary) == 2
    assert summary["n_units"].tolist() == [2, 2]


def test_paired_seed_contrast_resamples_matched_differences() -> None:
    frame = pd.DataFrame(
        {
            "trial_id": ["q0", "q1", "q2", "s0", "s1", "s2"],
            "condition_id": ["q"] * 3 + ["s"] * 3,
            "scenario_id": ["maze"] * 6,
            "agent": ["q"] * 3 + ["sarsa"] * 3,
            "seed": [0, 1, 2, 0, 1, 2],
            "episode_return": [1.0, 2.0, 3.0, 2.0, 4.0, 7.0],
        }
    )
    result = paired_seed_contrast(
        frame.sample(frac=1.0, random_state=3),
        metric="episode_return",
        factor="agent",
        baseline="q",
        comparison="sarsa",
        strata=("scenario_id",),
        n_resamples=500,
        random_seed=12,
    )
    assert result.pairs["difference"].tolist() == pytest.approx([1.0, 2.0, 4.0])
    row = result.summary.iloc[0]
    assert row["n_pairs"] == 3
    assert row["mean_difference"] == pytest.approx(7 / 3)
    assert row["win_rate"] == 1.0
    assert row["ci_low"] <= row["mean_difference"] <= row["ci_high"]


def test_paired_seed_contrast_rejects_incomplete_pairs() -> None:
    frame = pd.DataFrame(
        {
            "condition_id": ["q", "q", "s"],
            "agent": ["q", "q", "sarsa"],
            "seed": [0, 1, 0],
            "score": [1.0, 2.0, 3.0],
        }
    )
    with pytest.raises(UnsafeAggregationError, match="incomplete matched pairs"):
        paired_seed_contrast(
            frame,
            metric="score",
            factor="agent",
            baseline="q",
            comparison="sarsa",
        )


def test_final_performance_reduces_trials_without_hiding_conditions() -> None:
    rows = []
    for reliability in (0.8, 0.9):
        for seed in (0, 1):
            for episode in (0, 1):
                rows.append(
                    {
                        "trial_id": f"p{reliability}-{seed}",
                        "condition_id": f"p{reliability}",
                        "agent": "q",
                        "seed": seed,
                        "episode": episode,
                        "episode_return": 10 * reliability + seed + episode,
                        "sweep_environment_action_reliability": reliability,
                    }
                )
    frame = pd.DataFrame(rows)
    with pytest.raises(UnsafeAggregationError, match="multiple experimental conditions"):
        final_performance(frame, metrics=("episode_return",), last_episodes=1)

    result = final_performance(
        frame,
        metrics=("episode_return",),
        last_episodes=1,
        groups=("sweep_environment_action_reliability", "seed"),
    )
    assert len(result) == 4
    assert set(result["trial_id"]) == {"p0.8-0", "p0.8-1", "p0.9-0", "p0.9-1"}
    assert "condition_id" in result


def test_evaluation_checkpoint_summary_reduces_repeated_evaluation_episodes() -> None:
    rows = []
    for trial_id, agent in (("q-0", "q"), ("s-0", "sarsa")):
        for checkpoint in (-1, 9):
            for evaluation_episode, score in enumerate((1.0, 3.0)):
                rows.append(
                    {
                        "trial_id": trial_id,
                        "condition_id": agent,
                        "scenario_id": "maze",
                        "agent": agent,
                        "environment": "maze",
                        "seed": 0,
                        "evaluation_scenario": "matched",
                        "checkpoint_episode": checkpoint,
                        "checkpoint_global_step": max(0, checkpoint),
                        "evaluation_episode": evaluation_episode,
                        "evaluation_seed": 100 + evaluation_episode,
                        "episode_return": score + (agent == "sarsa"),
                        "success": score > 1,
                        "episode_length": 5 + evaluation_episode,
                    }
                )
    result = evaluation_checkpoint_summary(pd.DataFrame(rows))
    assert len(result) == 4
    assert result["evaluation_episodes"].eq(2).all()
    assert result["evaluation_seed_count"].eq(2).all()
    assert result.query("agent == 'q'")["episode_return"].eq(2.0).all()


def test_evaluation_checkpoint_summary_keeps_policy_modes_distinct() -> None:
    rows = []
    for mode, reward in (("greedy", 4.0), ("behavior", 1.0)):
        for evaluation_episode in range(2):
            rows.append(
                {
                    "trial_id": "trial-0",
                    "evaluation_scenario": f"deployment-{mode}",
                    "evaluation_policy_mode": mode,
                    "evaluation_seed_group": "deployment-panel",
                    "checkpoint_episode": 9,
                    "evaluation_episode": evaluation_episode,
                    "evaluation_seed": 100 + evaluation_episode,
                    "episode_return": reward,
                    "success": mode == "greedy",
                    "episode_length": 3,
                }
            )

    result = evaluation_checkpoint_summary(pd.DataFrame(rows))

    assert len(result) == 2
    assert set(result["evaluation_policy_mode"]) == {"greedy", "behavior"}
    assert set(result["evaluation_seed_group"]) == {"deployment-panel"}
    assert dict(zip(result["evaluation_policy_mode"], result["episode_return"], strict=True)) == {
        "greedy": 4.0,
        "behavior": 1.0,
    }


def test_td_summary_reports_sign_changes_tails_and_autocorrelation() -> None:
    frame = pd.DataFrame(
        {
            "state": [0] * 6 + [1] * 3,
            "action": [1] * 6 + [0] * 3,
            "td_error": [1, -1, 2, -2, 8, -8, 1, 1, 1],
        }
    )
    summary = td_error_summary(frame, autocorrelation_lags=(1,))
    volatile = summary.query("state == 0 and action == 1").iloc[0]
    assert volatile["count"] == 6
    assert volatile["mean_td_error"] == pytest.approx(0.0)
    assert volatile["sign_change_rate"] == pytest.approx(1.0)
    assert volatile["tail_mean_absolute_td_error"] == pytest.approx(8.0)
    assert "autocorrelation_lag_1" in summary


def test_td_temporal_metrics_are_computed_within_trials() -> None:
    frame = pd.DataFrame(
        {
            "trial_id": ["a"] * 4 + ["b"] * 4,
            "condition_id": ["one"] * 8,
            "state": [0] * 8,
            "action": [1] * 8,
            "global_step": [1, 2, 3, 4] * 2,
            "td_error": [1, -1, 1, -1, -1, 1, -1, 1],
        }
    )
    per_trial = td_error_trial_summary(frame, autocorrelation_lags=(1,))
    assert len(per_trial) == 2
    assert per_trial["autocorrelation_lag_1"].tolist() == pytest.approx([-1.0, -1.0])

    summary = td_error_summary(frame, autocorrelation_lags=(1,))
    assert summary.iloc[0]["count"] == 8
    assert summary.iloc[0]["n_trials"] == 2
    assert summary.iloc[0]["autocorrelation_lag_1"] == pytest.approx(-1.0)


def test_autocorrelation_handles_degenerate_and_invalid_lags() -> None:
    assert autocorrelation(np.ones(5), lag=1) == 0.0
    assert np.isnan(autocorrelation([1.0], lag=1))
    with pytest.raises(ValueError):
        autocorrelation([1, 2], lag=0)


def test_rolling_td_statistics_are_conditional_on_state_action() -> None:
    frame = pd.DataFrame(
        {
            "trial_id": ["x"] * 6,
            "state": [0] * 3 + [1] * 3,
            "action": [0] * 6,
            "global_step": np.arange(6),
            "td_error": [1.0, 2.0, 3.0, 10.0, 12.0, 14.0],
        }
    )
    result = rolling_td_statistics(frame, window=2, min_periods=2)
    state_zero = result[result["state"] == 0]
    assert state_zero["td_error_rolling_mean"].iloc[-1] == pytest.approx(2.5)
    state_one = result[result["state"] == 1]
    assert np.isnan(state_one["td_error_rolling_mean"].iloc[0])


def test_rolling_td_statistics_reject_groups_crossing_trials() -> None:
    frame = pd.DataFrame(
        {
            "trial_id": ["a", "a", "b", "b"],
            "state": [0] * 4,
            "action": [0] * 4,
            "global_step": [1, 2, 1, 2],
            "td_error": [1.0, 2.0, 3.0, 4.0],
        }
    )
    with pytest.raises(UnsafeAggregationError, match="cross trial boundaries"):
        rolling_td_statistics(
            frame,
            window=2,
            groups=("state", "action"),
        )
