"""Sample-efficiency summaries derived from tidy episode/snapshot tables."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from rllab.metrics.validation import (
    UnsafeAggregationError,
    assert_condition_safe,
    assert_unique_rows,
    require_columns,
)


def _columns(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, str):
        return (values,)
    return tuple(dict.fromkeys(values))


def evaluation_checkpoint_summary(
    frame: pd.DataFrame,
    *,
    metrics: Sequence[str] = ("episode_return", "success", "episode_length"),
) -> pd.DataFrame:
    """Reduce held-out episodes to one independent trial/checkpoint estimate.

    Evaluation episodes share one frozen learned policy, so they are repeated
    measurements rather than independent training replicates. This helper
    averages them before any across-trial bootstrap or paired comparison.
    """

    metric_columns = _columns(metrics)
    mode_keys = ("evaluation_policy_mode",) if "evaluation_policy_mode" in frame.columns else ()
    keys = ("trial_id", "evaluation_scenario", *mode_keys, "checkpoint_episode")
    require_columns(
        frame,
        {*keys, "evaluation_episode", *metric_columns},
        context="evaluation checkpoint summary",
    )
    assert_unique_rows(
        frame,
        (*keys, "evaluation_episode"),
        context="evaluation checkpoint summary",
    )
    context_columns = tuple(
        column
        for column in (
            "experiment_id",
            "scenario_id",
            "condition_id",
            "agent",
            "environment",
            "seed",
            "checkpoint_global_step",
            "evaluation_seed_group",
            *sorted(frame.filter(regex=r"^sweep_").columns),
        )
        if column in frame.columns
    )
    for column in context_columns:
        counts = frame.groupby(list(keys), dropna=False)[column].nunique(dropna=False)
        if bool(counts.gt(1).any()):
            raise UnsafeAggregationError(
                f"evaluation checkpoint summary found {column!r} changing within a trial/checkpoint"
            )

    grouped = frame.groupby(list(keys), dropna=False, as_index=False)
    result = grouped[list(metric_columns)].mean()
    counts = grouped.agg(
        evaluation_episodes=("evaluation_episode", "size"),
        evaluation_seed_count=("evaluation_seed", "nunique")
        if "evaluation_seed" in frame.columns
        else ("evaluation_episode", "size"),
    )
    result = result.merge(counts, on=list(keys), validate="one_to_one")
    if context_columns:
        context = grouped[list(context_columns)].first()
        result = result.merge(context, on=list(keys), validate="one_to_one")
    ordered = [
        *keys,
        *context_columns,
        *metric_columns,
        "evaluation_episodes",
        "evaluation_seed_count",
    ]
    return result.loc[:, list(dict.fromkeys(ordered))]


def episodes_to_threshold(
    frame: pd.DataFrame,
    *,
    metric: str = "policy_disagreement",
    threshold: float = 0.05,
    direction: str = "below",
    sustain: int = 1,
    groups: Sequence[str] = ("trial_id",),
) -> pd.DataFrame:
    """First episode reaching and sustaining a requested accuracy threshold."""

    required = {metric, "episode", *groups}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Missing threshold columns: {sorted(missing)}")
    if direction not in {"below", "above"}:
        raise ValueError("direction must be 'below' or 'above'")
    if sustain < 1:
        raise ValueError("sustain must be positive")
    rows: list[dict[str, object]] = []
    grouper: str | list[str] = groups[0] if len(groups) == 1 else list(groups)
    for keys, sample in frame.groupby(grouper, dropna=False, sort=False):
        sample = sample.sort_values("episode")
        values = sample[metric].to_numpy(dtype=float)
        reached = values <= threshold if direction == "below" else values >= threshold
        windows = np.convolve(reached.astype(int), np.ones(sustain, dtype=int), mode="valid")
        indices = np.flatnonzero(windows == sustain)
        episode = float("nan") if not indices.size else int(sample.iloc[int(indices[0])]["episode"])
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(groups, key_tuple, strict=True))
        row.update(
            {
                "episodes_to_threshold": episode,
                "reached": bool(indices.size),
                "threshold": threshold,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def final_performance(
    frame: pd.DataFrame,
    *,
    metrics: Sequence[str] = ("episode_return", "success"),
    last_episodes: int = 100,
    groups: Sequence[str] = ("agent", "seed"),
    unit_column: str = "trial_id",
    retain: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Per-trial mean performance over the final training window.

    ``groups`` describe the comparison cell, not the independent unit. A
    protocol-v2 ``trial_id`` is always reduced separately and retained in the
    output. Legacy tables without trial identifiers use the requested groups
    (plus ``seed`` when available) as their run key.
    """

    metric_columns = _columns(metrics)
    group_columns = _columns(groups)
    require_columns(
        frame,
        {"episode", *metric_columns, *group_columns},
        context="final-performance",
    )
    if last_episodes < 1:
        raise ValueError("last_episodes must be positive")
    assert_condition_safe(frame, group_columns, context="final-performance aggregation")

    unit_keys: tuple[str, ...]
    if unit_column in frame.columns:
        unit_keys = (unit_column,)
    else:
        unit_keys = group_columns
        if "seed" in frame.columns and "seed" not in unit_keys:
            unit_keys = (*unit_keys, "seed")
        if not unit_keys:
            raise KeyError(
                "final-performance needs trial_id or explicit groups identifying each run"
            )
    assert_unique_rows(frame, (*unit_keys, "episode"), context="final-performance")

    automatic_context = [
        column
        for column in (
            *group_columns,
            "condition_id",
            "scenario_id",
            "seed",
            "agent",
            "environment",
            *sorted(frame.filter(regex=r"^sweep_").columns),
        )
        if column in frame.columns and column not in unit_keys
    ]
    requested_context = automatic_context if retain is None else [*group_columns, *retain]
    retained = tuple(dict.fromkeys(requested_context))
    require_columns(frame, retained, context="final-performance")
    for column in retained:
        counts = frame.groupby(list(unit_keys), dropna=False)[column].nunique(dropna=False)
        unsafe = counts[counts > 1]
        if not unsafe.empty:
            raise UnsafeAggregationError(
                f"final-performance found {column!r} changing within a trial; "
                f"affected units include {list(unsafe.index[:3])!r}"
            )

    ordered = frame.sort_values([*unit_keys, "episode"])
    tail = ordered.groupby(list(unit_keys), dropna=False, group_keys=False).tail(last_episodes)
    performance = tail.groupby(list(unit_keys), dropna=False, as_index=False)[
        list(metric_columns)
    ].mean()
    if retained:
        context = ordered.groupby(list(unit_keys), dropna=False, as_index=False)[
            list(retained)
        ].first()
        performance = performance.merge(context, on=list(unit_keys), validate="one_to_one")
    ordered_columns = [
        *unit_keys,
        *[column for column in retained if column not in unit_keys],
        *metric_columns,
    ]
    return performance.loc[:, list(dict.fromkeys(ordered_columns))]
