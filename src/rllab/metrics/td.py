"""Diagnostics for temporal-difference residual sequences."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from rllab.metrics.validation import (
    assert_condition_safe,
    assert_trial_local_groups,
    assert_unique_rows,
    require_columns,
)


def _columns(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, str):
        return (values,)
    return tuple(dict.fromkeys(values))


def _grouper(columns: Sequence[str]) -> str | list[str]:
    return columns[0] if len(columns) == 1 else list(columns)


def autocorrelation(values: Sequence[float] | np.ndarray, lag: int = 1) -> float:
    """Finite-sample autocorrelation after pairwise NaN removal."""

    if lag < 1:
        raise ValueError("lag must be positive")
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size <= lag:
        return float("nan")
    left, right = array[:-lag], array[lag:]
    valid = np.isfinite(left) & np.isfinite(right)
    if np.count_nonzero(valid) < 2:
        return float("nan")
    left, right = left[valid], right[valid]
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _sign_change_rate(values: np.ndarray) -> float:
    nonzero = np.sign(values[np.isfinite(values)])
    nonzero = nonzero[nonzero != 0]
    if nonzero.size < 2:
        return 0.0
    return float(np.mean(nonzero[1:] != nonzero[:-1]))


def td_error_trial_summary(
    frame: pd.DataFrame,
    *,
    td_column: str = "td_error",
    groups: Sequence[str] = ("state", "action"),
    autocorrelation_lags: Sequence[int] = (1, 5, 10),
    tail_quantile: float = 0.95,
    trial_column: str = "trial_id",
    order_column: str = "global_step",
) -> pd.DataFrame:
    """Compute TD diagnostics separately inside every independent trial.

    Temporal order is never allowed to cross a trial boundary. For older tables,
    seeds define trials when present; otherwise the input is treated as one
    explicitly legacy sequence.
    """

    group_columns = _columns(groups)
    require_columns(frame, {td_column, *group_columns}, context="TD diagnostic")
    if not 0.5 < tail_quantile < 1.0:
        raise ValueError("tail_quantile must be in (0.5, 1)")
    if any(lag < 1 for lag in autocorrelation_lags):
        raise ValueError("autocorrelation lags must be positive")
    assert_condition_safe(frame, group_columns, context="TD diagnostic aggregation")

    data = frame.copy()
    if trial_column not in data.columns:
        if "seed" in data.columns:
            data[trial_column] = data["seed"].map(lambda seed: f"legacy-seed-{seed}")
        else:
            data[trial_column] = "legacy-single-trial"
    grouping = (trial_column, *group_columns)
    if order_column in data.columns:
        assert_unique_rows(data, (*grouping, order_column), context="TD temporal ordering")
        data = data.sort_values([*grouping, order_column])
    else:
        data = data.assign(_td_row_order=np.arange(len(data))).sort_values(
            [*grouping, "_td_row_order"]
        )

    rows: list[dict[str, Any]] = []
    for keys, sample in data.groupby(_grouper(grouping), dropna=False, sort=True):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        values = sample[td_column].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        row: dict[str, Any] = dict(zip(grouping, key_tuple, strict=True))
        if not values.size:
            continue
        absolute = np.abs(values)
        threshold = float(np.quantile(absolute, tail_quantile))
        row.update(
            {
                "count": int(values.size),
                "mean_td_error": float(np.mean(values)),
                "variance_td_error": float(np.var(values, ddof=1)) if values.size > 1 else 0.0,
                "mean_absolute_td_error": float(np.mean(absolute)),
                "rms_td_error": float(np.sqrt(np.mean(values**2))),
                "median_td_error": float(np.median(values)),
                "q05_td_error": float(np.quantile(values, 0.05)),
                "q95_td_error": float(np.quantile(values, 0.95)),
                "tail_threshold": threshold,
                "tail_mean_absolute_td_error": float(np.mean(absolute[absolute >= threshold])),
                "positive_fraction": float(np.mean(values > 0)),
                "negative_fraction": float(np.mean(values < 0)),
                "sign_change_rate": _sign_change_rate(values),
            }
        )
        row.update(
            {
                f"autocorrelation_lag_{lag}": autocorrelation(values, lag)
                for lag in autocorrelation_lags
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def td_error_summary(
    frame: pd.DataFrame,
    *,
    td_column: str = "td_error",
    groups: Sequence[str] = ("state", "action"),
    autocorrelation_lags: Sequence[int] = (1, 5, 10),
    tail_quantile: float = 0.95,
    trial_column: str = "trial_id",
    order_column: str = "global_step",
) -> pd.DataFrame:
    """Aggregate equally weighted trial-local TD diagnostics.

    ``count`` remains the total number of transitions for compatibility. Every
    other statistic is the mean of a statistic first computed within each trial,
    preventing seed boundaries from creating artificial sign changes or serial
    correlation.
    """

    group_columns = _columns(groups)
    per_trial = td_error_trial_summary(
        frame,
        td_column=td_column,
        groups=group_columns,
        autocorrelation_lags=autocorrelation_lags,
        tail_quantile=tail_quantile,
        trial_column=trial_column,
        order_column=order_column,
    )
    if per_trial.empty:
        return per_trial.drop(columns=[trial_column], errors="ignore")
    metric_columns = [
        column for column in per_trial.columns if column not in {trial_column, *group_columns}
    ]
    mean_columns = [column for column in metric_columns if column != "count"]

    rows: list[dict[str, Any]] = []
    if group_columns:
        grouped: Any = per_trial.groupby(_grouper(group_columns), dropna=False, sort=True)
    else:
        grouped = [((), per_trial)]
    for keys, sample in grouped:
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        row: dict[str, Any] = dict(zip(group_columns, key_tuple, strict=True))
        row["count"] = int(sample["count"].sum())
        row["n_trials"] = int(sample[trial_column].nunique(dropna=False))
        row.update({column: float(sample[column].mean()) for column in mean_columns})
        rows.append(row)
    return pd.DataFrame(rows)


def rolling_td_statistics(
    frame: pd.DataFrame,
    *,
    window: int = 100,
    min_periods: int | None = None,
    td_column: str = "td_error",
    groups: Sequence[str] = ("trial_id", "state", "action"),
    order_column: str = "global_step",
) -> pd.DataFrame:
    """Append rolling conditional TD mean, variance, RMS, and sign-change rate."""

    group_columns = _columns(groups)
    if window < 2:
        raise ValueError("window must be at least two")
    min_periods = min_periods or max(2, window // 5)
    if not 1 <= min_periods <= window:
        raise ValueError("min_periods must lie between one and window")
    require_columns(
        frame,
        {td_column, order_column, *group_columns},
        context="rolling TD",
    )
    assert_trial_local_groups(
        frame,
        group_columns,
        context="rolling TD statistics",
    )
    assert_unique_rows(
        frame,
        (*group_columns, order_column),
        context="rolling TD temporal ordering",
    )
    result = frame.copy().sort_values([*group_columns, order_column]).reset_index(drop=True)
    columns = (
        "td_error_rolling_mean",
        "td_error_rolling_variance",
        "td_error_rolling_rms",
        "td_error_rolling_sign_change_rate",
    )
    for column in columns:
        result[column] = np.nan

    # Work with original row indices rather than groupby.apply: pandas 2.2's
    # include_groups=False intentionally removes the conditioning columns.
    for _, indices in result.groupby(list(group_columns), dropna=False, sort=False).groups.items():
        sample = result.loc[indices].sort_values(order_column)
        values = sample[td_column].astype(float)
        rolling = values.rolling(window=window, min_periods=min_periods)
        result.loc[sample.index, "td_error_rolling_mean"] = rolling.mean()
        result.loc[sample.index, "td_error_rolling_variance"] = rolling.var(ddof=1)
        result.loc[sample.index, "td_error_rolling_rms"] = (
            values.pow(2).rolling(window, min_periods=min_periods).mean().pow(0.5)
        )
        signs = np.sign(values).replace(0, np.nan).ffill()
        changes = signs.ne(signs.shift()).astype(float)
        if not changes.empty:
            changes.iloc[0] = 0.0
        result.loc[sample.index, "td_error_rolling_sign_change_rate"] = changes.rolling(
            window, min_periods=min_periods
        ).mean()
    return result
