"""Matplotlib animation helpers that return editable ``FuncAnimation`` objects."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from rllab.visualization.maze import (
    _draw_topology,
    _setup_axes,
    _state_position,
    maze_shape,
)


def animate_trajectory(
    source: Any,
    trajectory: Sequence[Any],
    *,
    interval: int = 150,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, FuncAnimation]:
    """Animate a state trajectory over the current topology."""

    if not trajectory:
        raise ValueError("trajectory cannot be empty")
    if ax is None:
        figure, ax = plt.subplots(figsize=(6, 6))
    else:
        figure = ax.figure  # type: ignore[assignment]
    shape = maze_shape(source)
    _setup_axes(ax, shape)
    _draw_topology(ax, source)
    coordinates = [_state_position(source, value, shape) for value in trajectory]
    (line,) = ax.plot([], [], color="#f58518", linewidth=2)
    point = ax.scatter([], [], color="#f58518", s=55, zorder=8)

    def update(frame: int) -> tuple[Any, ...]:
        partial = coordinates[: frame + 1]
        rows, columns = zip(*partial, strict=True)
        x, y = np.asarray(columns) + 0.5, np.asarray(rows) + 0.5
        line.set_data(x, y)
        point.set_offsets([[x[-1], y[-1]]])
        ax.set_title(f"Trajectory step {frame}")
        return line, point

    animation = FuncAnimation(
        figure, update, frames=len(coordinates), interval=interval, blit=False
    )
    return figure, ax, animation


def animate_topology(
    source: Any,
    wall_realizations: Sequence[Any],
    *,
    interval: int = 250,
) -> tuple[Figure, Axes, FuncAnimation]:
    """Animate externally recorded dynamic-wall realizations."""

    if not wall_realizations:
        raise ValueError("wall_realizations cannot be empty")
    figure, ax = plt.subplots(figsize=(6, 6))
    shape = maze_shape(source)

    def update(frame: int) -> tuple[Any, ...]:
        ax.clear()
        _setup_axes(ax, shape, title=f"Topology step {frame}")
        _draw_topology(ax, source, walls=wall_realizations[frame])
        return tuple(ax.patches) + tuple(ax.lines)

    animation = FuncAnimation(
        figure, update, frames=len(wall_realizations), interval=interval, blit=False
    )
    return figure, ax, animation


def animate_learning(
    q_snapshots: Sequence[np.ndarray],
    source: Any,
    *,
    interval: int = 250,
) -> tuple[Figure, Axes, FuncAnimation]:
    """Animate value and greedy-policy evolution from numeric Q snapshots."""

    if not q_snapshots:
        raise ValueError("q_snapshots cannot be empty")
    arrays = [np.asarray(snapshot, dtype=float) for snapshot in q_snapshots]
    shape = maze_shape(source)
    index_to_state = getattr(source, "index_to_state", range(shape[0] * shape[1]))
    if any(array.ndim != 2 or array.shape[0] != len(index_to_state) for array in arrays):
        raise ValueError("Every Q snapshot must have shape (number of states, number of actions)")
    figure, ax = plt.subplots(figsize=(6, 5))
    lower = min(float(np.nanmin(array)) for array in arrays)
    upper = max(float(np.nanmax(array)) for array in arrays)
    vectors = ((0, -0.3), (0.3, 0), (0, 0.3), (-0.3, 0))

    def update(frame: int) -> tuple[Any, ...]:
        ax.clear()
        q_values = arrays[frame]
        value_grid = np.full(shape, np.nan)
        for state, value in enumerate(np.max(q_values, axis=1)):
            row, column = _state_position(source, state, shape)
            value_grid[row, column] = value
        image = ax.imshow(
            value_grid,
            extent=(0, shape[1], shape[0], 0),
            vmin=lower,
            vmax=upper,
            cmap="viridis",
            interpolation="none",
        )
        _setup_axes(ax, shape, title=f"Learning snapshot {frame}")
        _draw_topology(ax, source)
        for state, action in enumerate(np.argmax(q_values, axis=1)):
            if int(action) < len(vectors):
                row, column = _state_position(source, state, shape)
                dx, dy = vectors[int(action)]
                ax.arrow(
                    column + 0.5,
                    row + 0.5,
                    dx,
                    dy,
                    width=0.02,
                    color="white",
                    length_includes_head=True,
                )
        return (image, *ax.patches)

    animation = FuncAnimation(figure, update, frames=len(arrays), interval=interval, blit=False)
    return figure, ax, animation
