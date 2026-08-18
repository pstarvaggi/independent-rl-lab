"""Topology, value, policy, visitation, noise, and TD-error maze plots."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

DEFAULT_ACTION_VECTORS: dict[int, tuple[float, float]] = {
    0: (0.0, -0.30),
    1: (0.30, 0.0),
    2: (0.0, 0.30),
    3: (-0.30, 0.0),
}


def _value(source: Any, names: Sequence[str], default: Any = None) -> Any:
    objects = (source, getattr(source, "config", None))
    for obj in objects:
        if obj is None:
            continue
        for name in names:
            if isinstance(obj, Mapping) and name in obj:
                return obj[name]
            if hasattr(obj, name):
                item = getattr(obj, name)
                return item() if callable(item) and name.startswith("current_") else item
    return default


def maze_shape(source: Any) -> tuple[int, int]:
    """Infer ``(rows, columns)`` from an environment or config mapping."""

    shape = _value(source, ("shape", "grid_shape"))
    if shape is not None and len(shape) == 2:
        return int(shape[0]), int(shape[1])
    rows = _value(source, ("rows", "height", "n_rows"))
    columns = _value(source, ("cols", "columns", "width", "n_cols"))
    if rows is None or columns is None:
        raise ValueError("Could not infer maze rows and columns")
    return int(rows), int(columns)


def _position(value: Any, shape: tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
        return int(value[0]), int(value[1])
    return divmod(int(value), shape[1])


def _state_position(source: Any, value: Any, shape: tuple[int, int]) -> tuple[int, int]:
    """Map a compact state index through ``index_to_state`` when available."""

    if isinstance(value, (int, np.integer)):
        index_to_state = _value(source, ("index_to_state",))
        if index_to_state is not None:
            index = int(value)
            try:
                coordinate = index_to_state[index]
            except (IndexError, KeyError):
                pass
            else:
                return _position(coordinate, shape)
    return _position(value, shape)


def _positions(values: Any, shape: tuple[int, int], *, source: Any = None) -> set[tuple[int, int]]:
    if values is None:
        return set()
    if isinstance(values, (int, np.integer)):
        return {_state_position(source, values, shape)}
    if (
        isinstance(values, tuple)
        and len(values) == 2
        and all(isinstance(item, (int, np.integer)) for item in values)
    ):
        return {_position(values, shape)}
    return {_state_position(source, item, shape) for item in values}


def _wall_data(
    source: Any, walls: Any = None
) -> tuple[set[tuple[int, int]], list[tuple[tuple[int, int], tuple[int, int]]]]:
    shape = maze_shape(source)
    raw = walls
    if raw is None:
        collections = []
        for names in (
            ("blocked_cells",),
            ("walls", "static_walls"),
            ("dynamic_walls", "active_walls", "current_walls"),
        ):
            item = _value(source, names)
            if item is not None:
                collections.extend(list(item))
        raw = collections
    cells: set[tuple[int, int]] = set()
    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for wall in raw or []:
        value = list(wall) if isinstance(wall, (set, frozenset)) else wall
        if isinstance(value, Sequence) and len(value) == 2:
            first, second = value
            first_is_position = isinstance(first, Sequence) and not isinstance(first, (str, bytes))
            second_is_position = isinstance(second, Sequence) and not isinstance(
                second, (str, bytes)
            )
            if first_is_position and second_is_position:
                edges.append((_position(first, shape), _position(second, shape)))
            elif all(isinstance(item, (int, np.integer)) for item in value):
                cells.add(_position(value, shape))
        elif isinstance(value, (int, np.integer)):
            cells.add(_position(value, shape))
    return cells, edges


def _setup_axes(ax: Axes, shape: tuple[int, int], *, title: str | None = None) -> None:
    rows, columns = shape
    ax.set_xlim(0, columns)
    ax.set_ylim(rows, 0)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(columns + 1), minor=True)
    ax.set_yticks(np.arange(rows + 1), minor=True)
    ax.grid(which="minor", color="0.82", linewidth=0.8)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    if title:
        ax.set_title(title)


def _draw_topology(ax: Axes, source: Any, *, walls: Any = None) -> None:
    cells, edges = _wall_data(source, walls)
    for row, column in cells:
        ax.add_patch(Rectangle((column, row), 1, 1, facecolor="0.16", edgecolor="none", zorder=4))
    for (row_a, column_a), (row_b, column_b) in edges:
        if row_a == row_b:
            x = max(column_a, column_b)
            ax.plot([x, x], [row_a, row_a + 1], color="0.08", linewidth=3.0, zorder=5)
        elif column_a == column_b:
            y = max(row_a, row_b)
            ax.plot([column_a, column_a + 1], [y, y], color="0.08", linewidth=3.0, zorder=5)


def plot_maze(
    source: Any,
    *,
    ax: Axes | None = None,
    trajectory: Sequence[Any] | None = None,
    walls: Any = None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Render static/current topology and optional agent trajectory."""

    if ax is None:
        figure, ax = plt.subplots(figsize=(6, 6))
    else:
        figure = ax.figure  # type: ignore[assignment]
    shape = maze_shape(source)
    _setup_axes(ax, shape, title=title)
    _draw_topology(ax, source, walls=walls)
    markers = (
        (("start_states", "_start_states", "starts", "start"), "o", "#4c78a8", "start"),
        (("goal_states", "goals", "goal"), "*", "#54a24b", "goal"),
        (("hazard_states", "hazards", "hazard_positions"), "X", "#e45756", "hazard"),
    )
    for names, marker, color, label in markers:
        positions = _positions(_value(source, names), shape, source=source)
        if positions:
            rows, columns = zip(*sorted(positions), strict=True)
            ax.scatter(
                np.asarray(columns) + 0.5,
                np.asarray(rows) + 0.5,
                marker=marker,
                s=90,
                color=color,
                label=label,
                zorder=7,
            )
    if trajectory:
        coordinates = [_state_position(source, item, shape) for item in trajectory]
        rows, columns = zip(*coordinates, strict=True)
        ax.plot(
            np.asarray(columns) + 0.5,
            np.asarray(rows) + 0.5,
            color="#f58518",
            linewidth=2,
            alpha=0.9,
            zorder=6,
        )
        ax.scatter(columns[-1] + 0.5, rows[-1] + 0.5, color="#f58518", s=45, zorder=8)
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    return figure, ax


def _grid(values: Any, shape: tuple[int, int], *, source: Any = None) -> np.ndarray:
    if isinstance(values, Mapping):
        array = np.full(shape[0] * shape[1], np.nan)
        for key, value in values.items():
            row, column = _state_position(source, key, shape)
            array[row * shape[1] + column] = value
        return array.reshape(shape)
    array = np.asarray(values, dtype=float)
    if array.shape == shape:
        return array
    if array.size == shape[0] * shape[1]:
        return array.reshape(shape)
    index_to_state = _value(source, ("index_to_state",))
    if index_to_state is not None and array.ndim == 1 and array.size == len(index_to_state):
        grid = np.full(shape, np.nan)
        for state, value in enumerate(array):
            row, column = _state_position(source, state, shape)
            grid[row, column] = value
        return grid
    raise ValueError(f"Expected {shape[0] * shape[1]} state values, got shape {array.shape}")


def plot_state_heatmap(
    values: Any,
    source: Any,
    *,
    ax: Axes | None = None,
    cmap: str = "viridis",
    center: float | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar_label: str | None = None,
    annotate: bool = False,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot one scalar per state while retaining maze topology."""

    if ax is None:
        figure, ax = plt.subplots(figsize=(6, 5))
    else:
        figure = ax.figure  # type: ignore[assignment]
    shape = maze_shape(source)
    grid = _grid(values, shape, source=source)
    finite = grid[np.isfinite(grid)]
    norm = None
    if center is not None and (vmin is not None or vmax is not None):
        raise ValueError("center cannot be combined with vmin or vmax")
    if center is not None and finite.size:
        limit = max(abs(float(np.min(finite)) - center), abs(float(np.max(finite)) - center))
        norm = Normalize(vmin=center - limit, vmax=center + limit)
    image = ax.imshow(
        grid,
        cmap=cmap,
        norm=norm,
        vmin=None if norm is not None else vmin,
        vmax=None if norm is not None else vmax,
        extent=(0, shape[1], shape[0], 0),
        interpolation="none",
    )
    _setup_axes(ax, shape, title=title)
    _draw_topology(ax, source)
    if annotate:
        for row in range(shape[0]):
            for column in range(shape[1]):
                if np.isfinite(grid[row, column]):
                    ax.text(
                        column + 0.5,
                        row + 0.5,
                        f"{grid[row, column]:.2g}",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )
    colorbar = figure.colorbar(image, ax=ax, shrink=0.82)
    if colorbar_label:
        colorbar.set_label(colorbar_label)
    return figure, ax


def plot_policy(
    policy: Sequence[int] | np.ndarray,
    source: Any,
    *,
    values: Sequence[float] | np.ndarray | None = None,
    ax: Axes | None = None,
    action_vectors: Mapping[int, tuple[float, float]] = DEFAULT_ACTION_VECTORS,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar_label: str = "value",
    title: str = "Greedy policy",
) -> tuple[Figure, Axes]:
    """Plot a deterministic greedy policy, optionally over its state values."""

    shape = maze_shape(source)
    if values is None:
        if ax is None:
            figure, ax = plt.subplots(figsize=(6, 5))
        else:
            figure = ax.figure  # type: ignore[assignment]
        _setup_axes(ax, shape, title=title)
        _draw_topology(ax, source)
    else:
        figure, ax = plot_state_heatmap(
            values,
            source,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            colorbar_label=colorbar_label,
            title=title,
        )
    policy_array = np.asarray(policy).reshape(-1)
    cells, _ = _wall_data(source)
    terminals = _positions(_value(source, ("goal_states", "goals", "goal")), shape, source=source)
    for state, action in enumerate(policy_array):
        row, column = _state_position(source, state, shape)
        if (row, column) in cells | terminals or int(action) not in action_vectors:
            continue
        dx, dy = action_vectors[int(action)]
        ax.arrow(
            column + 0.5,
            row + 0.5,
            dx,
            dy,
            width=0.025,
            head_width=0.16,
            length_includes_head=True,
            color="white" if values is not None else "#2f2f2f",
            zorder=8,
        )
    return figure, ax


def plot_q_values(
    q_values: np.ndarray,
    source: Any,
    *,
    ax: Axes | None = None,
    action_vectors: Mapping[int, tuple[float, float]] = DEFAULT_ACTION_VECTORS,
    title: str = "Action values and greedy policy",
) -> tuple[Figure, Axes]:
    """Show max Q as color, greedy actions as arrows, and all Q values as text."""

    q_array = np.asarray(q_values, dtype=float)
    shape = maze_shape(source)
    expected_states = len(_value(source, ("index_to_state",), range(shape[0] * shape[1])))
    if q_array.ndim != 2 or q_array.shape[0] != expected_states:
        raise ValueError("q_values must have shape (number of grid states, number of actions)")
    figure, ax = plot_policy(
        np.argmax(q_array, axis=1),
        source,
        values=np.max(q_array, axis=1),
        ax=ax,
        action_vectors=action_vectors,
        title=title,
    )
    offsets = ((0, -0.28), (0.27, 0), (0, 0.29), (-0.27, 0))
    for state in range(q_array.shape[0]):
        row, column = _state_position(source, state, shape)
        for action in range(min(q_array.shape[1], len(offsets))):
            dx, dy = offsets[action]
            ax.text(
                column + 0.5 + dx,
                row + 0.5 + dy,
                f"{q_array[state, action]:.1f}",
                color="white",
                fontsize=5.5,
                ha="center",
                va="center",
                zorder=9,
            )
    return figure, ax


def plot_state_action_heatmaps(
    values: np.ndarray,
    source: Any,
    *,
    action_names: Sequence[str] = ("north", "east", "south", "west"),
    cmap: str = "magma",
    title: str | None = None,
) -> tuple[Figure, np.ndarray]:
    """One comparable heatmap per action for visitation or uncertainty arrays."""

    array = np.asarray(values, dtype=float)
    shape = maze_shape(source)
    expected_states = len(_value(source, ("index_to_state",), range(shape[0] * shape[1])))
    if array.ndim != 2 or array.shape[0] != expected_states:
        raise ValueError("values must have shape (number of states, number of actions)")
    columns = min(2, array.shape[1])
    rows = int(np.ceil(array.shape[1] / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(5 * columns, 4 * rows), squeeze=False, constrained_layout=True
    )
    vmin, vmax = float(np.nanmin(array)), float(np.nanmax(array))
    for action, ax in enumerate(axes.flat):
        if action >= array.shape[1]:
            ax.set_visible(False)
            continue
        grid = _grid(array[:, action], shape, source=source)
        image = ax.imshow(
            grid,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=(0, shape[1], shape[0], 0),
            interpolation="none",
        )
        label = action_names[action] if action < len(action_names) else str(action)
        _setup_axes(ax, shape, title=label)
        _draw_topology(ax, source)
    figure.colorbar(image, ax=list(axes.flat), shrink=0.75, label="value")
    if title:
        figure.suptitle(title)
    return figure, axes


def plot_transition_noise(
    source: Any,
    *,
    reliability: Any = None,
    ax: Axes | None = None,
    title: str = "Intended-action probability",
) -> tuple[Figure, Axes]:
    """Map spatially heterogeneous transition reliability."""

    state_overrides = None
    if reliability is None:
        reliability = _value(
            source,
            (
                "current_reliability",
                "movement_reliability",
                "action_reliability",
                "action_success_probability",
                "reliability",
                "transition_noise",
            ),
        )
        state_overrides = _value(source, ("state_reliability",))
        state_action = _value(source, ("state_action_reliability",))
        if state_action:
            state_action_grouped: dict[Any, list[float]] = {}
            for key, value in state_action.items():
                state_key = key[0] if isinstance(key, tuple) and len(key) == 2 else key
                state_action_grouped.setdefault(state_key, []).append(float(value))
            state_overrides = {
                **dict(state_overrides or {}),
                **{
                    state_key: float(np.mean(values))
                    for state_key, values in state_action_grouped.items()
                },
            }
    if reliability is None:
        raise ValueError("No transition-reliability field was found")
    shape = maze_shape(source)
    if np.isscalar(reliability):
        reliability = np.full(shape, float(reliability))  # type: ignore[arg-type]
    elif (
        isinstance(reliability, Mapping)
        and reliability
        and all(
            isinstance(key, tuple) and len(key) == 2 and all(isinstance(item, int) for item in key)
            for key in reliability
        )
    ):
        # A state-action map is collapsed to mean intended-action reliability.
        keys = list(reliability)
        if any(key[0] >= shape[0] or key[1] >= shape[1] for key in keys):
            grouped: dict[int, list[float]] = {}
            for (state, _action), value in reliability.items():
                grouped.setdefault(int(state), []).append(float(np.asarray(value).item()))
            reliability = {state: float(np.mean(values)) for state, values in grouped.items()}
    if state_overrides:
        reliability = _grid(reliability, shape, source=source)
        for raw_key, override_value in state_overrides.items():
            parsed_key: Any = raw_key
            if isinstance(parsed_key, str) and "," in parsed_key:
                parsed_key = tuple(int(part.strip()) for part in parsed_key.split(",", maxsplit=1))
            row, column = _position(parsed_key, shape)
            reliability[row, column] = (
                float(np.mean(override_value))
                if isinstance(override_value, Sequence)
                else float(override_value)
            )
    return plot_state_heatmap(
        reliability, source, ax=ax, cmap="viridis", colorbar_label="reliability", title=title
    )


def plot_td_error_heatmap(
    td_data: pd.DataFrame | np.ndarray,
    source: Any,
    *,
    statistic: str = "variance",
    ax: Axes | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Locate persistent TD volatility (or another aggregate) in the maze."""

    if isinstance(td_data, pd.DataFrame):
        if "state" not in td_data:
            raise KeyError("TD data needs a state column")
        if statistic in td_data:
            column = statistic
        elif statistic == "variance" and "variance_td_error" in td_data:
            column = "variance_td_error"
        elif statistic == "mean_absolute" and "mean_absolute_td_error" in td_data:
            column = "mean_absolute_td_error"
        elif "td_error" in td_data:
            functions = {
                "variance": "var",
                "mean": "mean",
                "mean_absolute": lambda x: np.mean(np.abs(x)),
            }
            if statistic not in functions:
                raise ValueError(f"Unknown TD statistic {statistic!r}")
            values = td_data.groupby("state")["td_error"].agg(functions[statistic])
            td_data = values.rename("_value").reset_index()
            column = "_value"
        else:
            raise KeyError(f"Could not find TD statistic {statistic!r}")
        values = dict(zip(td_data["state"].astype(int), td_data[column].astype(float), strict=True))
    else:
        values = td_data
    return plot_state_heatmap(
        values,
        source,
        ax=ax,
        cmap="magma",
        colorbar_label=f"TD {statistic}",
        title=title or f"TD-error {statistic}",
    )
