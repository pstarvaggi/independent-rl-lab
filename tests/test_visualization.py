from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from rllab.environments import Goal, StochasticMazeEnv
from rllab.metrics import UnsafeAggregationError
from rllab.visualization import (
    plot_final_distribution,
    plot_learning_curves,
    plot_maze,
    plot_paired_contrasts,
    plot_policy,
    plot_q_values,
    plot_state_action_heatmaps,
    plot_state_heatmap,
    plot_sweep_response,
    plot_td_error_diagnostics,
    plot_td_error_heatmap,
    plot_transition_noise,
)

MAZE = {
    "shape": (2, 3),
    "start_states": [0],
    "goal_states": [5],
    "hazard_states": [2],
    "walls": [(1, 1)],
    "movement_reliability": {0: 1.0, 1: 0.8, 2: 0.6, 3: 1.0, 4: 0.9, 5: 1.0},
}


def test_maze_and_heatmap_render_expected_artists() -> None:
    figure, ax = plot_maze(MAZE, trajectory=[0, 1, 4, 5])
    assert ax.lines
    assert ax.collections
    plt.close(figure)

    figure, ax = plot_state_heatmap(np.arange(6), MAZE, annotate=True)
    assert len(ax.images) == 1
    assert len(ax.texts) == 6
    plt.close(figure)


def test_policy_q_and_state_action_visualizations() -> None:
    q_values = np.arange(24, dtype=float).reshape(6, 4)
    figure, ax = plot_policy(
        np.argmax(q_values, axis=1),
        MAZE,
        values=np.linspace(0.0, 1.0, 6),
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        colorbar_label="seed agreement",
    )
    assert ax.patches
    assert figure.axes[-1].get_ylabel() == "seed agreement"
    plt.close(figure)
    figure, ax = plot_q_values(q_values, MAZE)
    assert len(ax.texts) >= 20
    plt.close(figure)
    figure, axes = plot_state_action_heatmaps(q_values, MAZE)
    assert axes.shape == (2, 2)
    plt.close(figure)


def test_noise_and_td_heatmaps_accept_mapping_and_tidy_data() -> None:
    figure, ax = plot_transition_noise(MAZE)
    assert len(ax.images) == 1
    plt.close(figure)
    td = pd.DataFrame({"state": [0, 0, 1, 1], "action": [0, 1, 0, 1], "td_error": [1, -1, 2, -2]})
    figure, ax = plot_td_error_heatmap(td, MAZE, statistic="variance")
    assert len(ax.images) == 1
    plt.close(figure)


def _episodes() -> pd.DataFrame:
    rows = []
    for agent in ("q_learning", "sarsa"):
        for seed in range(3):
            for episode in range(8):
                rows.append(
                    {
                        "trial_id": f"{agent}-{seed}",
                        "condition_id": f"{agent}-{0.8 + 0.1 * (seed % 2):.1f}",
                        "scenario_id": f"reliability-{0.8 + 0.1 * (seed % 2):.1f}",
                        "agent": agent,
                        "seed": seed,
                        "episode": episode,
                        "episode_return": episode + seed + (agent == "q_learning"),
                        "sweep_environment_movement_reliability": 0.8 + 0.1 * (seed % 2),
                    }
                )
    return pd.DataFrame(rows)


def test_learning_distribution_and_sweep_plots_return_analysis_tables() -> None:
    episodes = _episodes()
    with pytest.raises(UnsafeAggregationError, match="multiple experimental conditions"):
        plot_learning_curves(episodes, smooth=2, n_resamples=100)

    one_reliability = episodes.query("sweep_environment_movement_reliability == 0.8")
    figure, _ax, summary = plot_learning_curves(one_reliability, smooth=2, n_resamples=100)
    assert {"mean", "median", "sem", "ci_low", "ci_high", "n_seeds"} <= set(summary)
    plt.close(figure)
    figure, _ax, per_seed = plot_final_distribution(one_reliability, last_episodes=3)
    assert len(per_seed) == 4
    plt.close(figure)
    figure, _ax, per_seed = plot_sweep_response(
        episodes, parameter="environment.movement_reliability", last_episodes=3
    )
    assert not per_seed.empty
    plt.close(figure)


def test_td_diagnostic_plot_accepts_multiple_trials_without_joining_time_series() -> None:
    rows = []
    for seed, values in enumerate(((1.0, -1.0, 1.0), (-1.0, 1.0, -1.0))):
        for step, value in enumerate(values):
            rows.append(
                {
                    "trial_id": f"q-{seed}",
                    "condition_id": "q-one-maze",
                    "agent": "q",
                    "seed": seed,
                    "episode": 0,
                    "global_step": step,
                    "state": 0,
                    "action": 1,
                    "td_error": value,
                }
            )
    figure, axes = plot_td_error_diagnostics(pd.DataFrame(rows), max_lag=2)
    assert axes.shape == (2, 2)
    assert axes[1, 0].lines
    plt.close(figure)


def test_paired_contrast_plot_renders_asymmetric_intervals() -> None:
    summary = pd.DataFrame(
        {
            "comparison": ["sarsa", "double_q"],
            "mean_difference": [1.0, -0.5],
            "ci_low": [0.25, -1.25],
            "ci_high": [1.75, 0.10],
        }
    )
    figure, ax = plot_paired_contrasts(summary)
    assert len(ax.lines) >= 3
    assert [tick.get_text() for tick in ax.get_yticklabels()] == ["sarsa", "double_q"]
    plt.close(figure)


def test_compact_maze_state_arrays_map_around_blocked_cells() -> None:
    env = StochasticMazeEnv(
        shape=(2, 3),
        start=(0, 0),
        goals=[Goal((1, 2))],
        blocked_cells=[(0, 1)],
        action_reliability=0.9,
        state_reliability={(1, 1): 0.5},
    )
    env.reset(seed=2)
    assert env.n_states == 5
    q_values = np.arange(env.n_states * env.n_actions, dtype=float).reshape(
        env.n_states, env.n_actions
    )
    figure, ax = plot_q_values(q_values, env)
    assert len(ax.images) == 1
    plt.close(figure)
    figure, axes = plot_state_action_heatmaps(q_values, env)
    assert axes[0, 0].images[0].get_array().shape == env.shape
    plt.close(figure)
    figure, ax = plot_transition_noise(env)
    reliability_grid = np.asarray(ax.images[0].get_array())
    assert reliability_grid[1, 1] == 0.5
    plt.close(figure)
    env.close()
