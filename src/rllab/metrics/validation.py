"""Guards against statistically ambiguous aggregation of experiment tables."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd  # type: ignore[import-untyped]


class UnsafeAggregationError(ValueError):
    """Raised when an analysis would silently combine distinct conditions or units."""


def require_columns(
    frame: pd.DataFrame,
    columns: Sequence[str] | set[str],
    *,
    context: str,
) -> None:
    """Require named columns and report errors in terms of the calling analysis."""

    missing = set(columns) - set(frame.columns)
    if missing:
        raise KeyError(f"Missing {context} columns: {sorted(missing)}")


def inferred_condition_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    """Return the strongest available columns describing a scientific condition.

    Protocol-v2 tables carry ``condition_id``. Older tidy tables can still be
    checked conservatively from their environment, agent, and sweep-factor
    columns. If none of those columns exists there is no evidence with which to
    distinguish conditions, so callers must rely on unit/row uniqueness checks.
    """

    if "condition_id" in frame.columns:
        return ("condition_id",)
    columns = [
        column for column in ("scenario_id", "environment", "agent") if column in frame.columns
    ]
    columns.extend(
        sorted(
            column
            for column in frame.columns
            if column.startswith("sweep_") and column not in columns
        )
    )
    return tuple(columns)


def _condition_count(
    frame: pd.DataFrame,
    *,
    groups: tuple[str, ...],
    condition_columns: tuple[str, ...],
) -> pd.Series:
    selected_columns = list(dict.fromkeys((*groups, *condition_columns)))
    distinct = frame.loc[:, selected_columns].drop_duplicates()
    if not groups:
        return pd.Series([len(distinct)], index=["all"])
    return distinct.groupby(list(groups), dropna=False, sort=False).size()


def assert_condition_safe(
    frame: pd.DataFrame,
    groups: Sequence[str],
    *,
    context: str,
    condition_columns: Sequence[str] | None = None,
) -> None:
    """Reject grouping keys that hide more than one scientific condition.

    A group such as ``("agent",)`` is safe in a single environment but unsafe
    when that agent appears at several reliability settings. Callers can filter
    first or add the varying factor to their grouping keys.
    """

    group_columns = tuple(dict.fromkeys(groups))
    require_columns(frame, group_columns, context=context)
    conditions = tuple(condition_columns or inferred_condition_columns(frame))
    if not conditions or frame.empty:
        return
    require_columns(frame, conditions, context=context)
    counts = _condition_count(frame, groups=group_columns, condition_columns=conditions)
    unsafe = counts[counts > 1]
    if unsafe.empty:
        return

    human_factors = [
        column
        for column in ("environment", "agent", *sorted(frame.filter(regex=r"^sweep_").columns))
        if column in frame.columns and column not in group_columns
    ]
    varying = [column for column in human_factors if frame[column].nunique(dropna=False) > 1]
    suggestion = (
        f" Add the varying columns {varying!r} to the groups or filter to one condition."
        if varying
        else " Add condition_id to the groups or filter to one condition."
    )
    examples = list(unsafe.index[:3])
    raise UnsafeAggregationError(
        f"{context} would combine multiple experimental conditions for group(s) "
        f"{examples!r}.{suggestion}"
    )


def assert_unique_rows(
    frame: pd.DataFrame,
    keys: Sequence[str],
    *,
    context: str,
) -> None:
    """Reject duplicate observations instead of silently averaging them."""

    key_columns = tuple(dict.fromkeys(keys))
    require_columns(frame, key_columns, context=context)
    duplicated = frame.duplicated(list(key_columns), keep=False)
    if not bool(duplicated.any()):
        return
    examples = frame.loc[duplicated, list(key_columns)].drop_duplicates().head(3).to_dict("records")
    raise UnsafeAggregationError(
        f"{context} requires one row per {key_columns!r}; duplicate keys include {examples!r}. "
        "Reduce repeated observations explicitly before aggregation."
    )


def independent_unit_column(
    frame: pd.DataFrame,
    *,
    preferred: str = "trial_id",
    fallback: str = "seed",
    context: str,
) -> str:
    """Choose the trial identifier, falling back to a seed for legacy tables."""

    if preferred in frame.columns:
        return preferred
    if fallback in frame.columns:
        return fallback
    raise KeyError(
        f"{context} needs an independent-unit column; neither {preferred!r} nor "
        f"legacy fallback {fallback!r} is present"
    )


def assert_columns_constant_within_unit(
    frame: pd.DataFrame,
    *,
    unit: str,
    columns: Sequence[str],
    context: str,
) -> None:
    """Ensure identifiers and factors do not change inside an independent trial."""

    check_columns = tuple(column for column in dict.fromkeys(columns) if column != unit)
    require_columns(frame, (unit, *check_columns), context=context)
    for column in check_columns:
        counts = frame.groupby(unit, dropna=False)[column].nunique(dropna=False)
        unsafe = counts[counts > 1]
        if not unsafe.empty:
            raise UnsafeAggregationError(
                f"{context} found {column!r} changing within {unit!r}; affected units include "
                f"{list(unsafe.index[:3])!r}."
            )


def assert_trial_local_groups(
    frame: pd.DataFrame,
    groups: Sequence[str],
    *,
    trial_column: str = "trial_id",
    context: str,
) -> None:
    """Require temporal groups to contain the trial boundary when several exist."""

    if trial_column not in frame.columns or trial_column in groups:
        return
    if frame[trial_column].nunique(dropna=False) > 1:
        raise UnsafeAggregationError(
            f"{context} would cross trial boundaries; include {trial_column!r} in groups."
        )
