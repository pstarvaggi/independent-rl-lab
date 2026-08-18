"""Scientifically useful plots for maze structure and learning dynamics."""

from rllab.visualization.animation import animate_learning, animate_topology, animate_trajectory
from rllab.visualization.learning import (
    plot_final_distribution,
    plot_learning_curves,
    plot_paired_contrasts,
    plot_sweep_response,
    plot_td_error_diagnostics,
)
from rllab.visualization.maze import (
    plot_maze,
    plot_policy,
    plot_q_values,
    plot_state_action_heatmaps,
    plot_state_heatmap,
    plot_td_error_heatmap,
    plot_transition_noise,
)

__all__ = [
    "animate_learning",
    "animate_topology",
    "animate_trajectory",
    "plot_final_distribution",
    "plot_learning_curves",
    "plot_maze",
    "plot_paired_contrasts",
    "plot_policy",
    "plot_q_values",
    "plot_state_action_heatmaps",
    "plot_state_heatmap",
    "plot_sweep_response",
    "plot_td_error_diagnostics",
    "plot_td_error_heatmap",
    "plot_transition_noise",
]
