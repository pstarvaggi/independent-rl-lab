"""Seed-aware statistical summaries for noisy reinforcement-learning results."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from rllab.metrics.validation import (
    UnsafeAggregationError,
    assert_columns_constant_within_unit,
    assert_condition_safe,
    assert_unique_rows,
    independent_unit_column,
    require_columns,
)

ArrayStatistic = Callable[[np.ndarray], float]


@dataclass(frozen=True, slots=True)
class PairedContrastResult:
    """Matched seed-level differences and their across-pair summary."""

    pairs: pd.DataFrame
    summary: pd.DataFrame


def _columns(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, str):
        return (values,)
    return tuple(dict.fromkeys(values))


def _grouper(columns: Sequence[str]) -> str | list[str]:
    return columns[0] if len(columns) == 1 else list(columns)


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values))


def _median(values: np.ndarray) -> float:
    return float(np.median(values))


def _clean(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float).reshape(-1)
    return array[np.isfinite(array)]


def standard_error(values: Iterable[float]) -> float:
    """Sample standard error, returning NaN when fewer than two values exist."""

    array = _clean(values)
    return float(np.std(array, ddof=1) / np.sqrt(array.size)) if array.size > 1 else float("nan")


def bootstrap_confidence_interval(
    values: Iterable[float],
    *,
    statistic: str | ArrayStatistic = "mean",
    confidence: float = 0.95,
    n_resamples: int = 2_000,
    seed: int | np.random.Generator | None = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval for a one-dimensional sample.

    NaNs are removed.  Degenerate one-observation samples produce a zero-width
    interval instead of an exception, which is useful during exploratory runs.
    """

    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    array = _clean(values)
    if not array.size:
        return float("nan"), float("nan")
    function: ArrayStatistic
    if statistic == "mean":
        function = _mean
    elif statistic == "median":
        function = _median
    elif callable(statistic):
        function = statistic
    else:
        raise ValueError("statistic must be 'mean', 'median', or a callable")
    if array.size == 1:
        value = function(array)
        return value, value

    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(n_resamples, array.size))
    resampled = array[indices]
    if statistic == "mean":
        estimates = np.mean(resampled, axis=1)
    elif statistic == "median":
        estimates = np.median(resampled, axis=1)
    else:
        estimates = np.fromiter((function(sample) for sample in resampled), dtype=float)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(estimates, [alpha, 1.0 - alpha])
    return float(low), float(high)


def quantile_summary(
    values: Iterable[float],
    *,
    quantiles: Sequence[float] = (0.05, 0.25, 0.5, 0.75, 0.95),
) -> dict[str, float]:
    """Named empirical quantiles without hiding the sample size."""

    array = _clean(values)
    result = {"count": float(array.size)}
    if not array.size:
        result.update({f"q{quantile:g}": float("nan") for quantile in quantiles})
        return result
    if any(not 0 <= quantile <= 1 for quantile in quantiles):
        raise ValueError("quantiles must lie in [0, 1]")
    result.update(
        {
            f"q{quantile:g}": float(value)
            for quantile, value in zip(quantiles, np.quantile(array, quantiles), strict=True)
        }
    )
    return result


def distribution_summary(values: Iterable[float]) -> dict[str, float]:
    """A robust scalar summary retaining central and tail information."""

    array = _clean(values)
    if not array.size:
        result = {
            key: float("nan")
            for key in (
                "mean",
                "std",
                "sem",
                "min",
                "q05",
                "q25",
                "median",
                "q75",
                "q95",
                "max",
                "iqr",
                "mad",
            )
        }
        return {"count": 0.0, **result}
    q05, q25, median, q75, q95 = np.quantile(array, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "count": float(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else float("nan"),
        "sem": standard_error(array),
        "min": float(np.min(array)),
        "q05": float(q05),
        "q25": float(q25),
        "median": float(median),
        "q75": float(q75),
        "q95": float(q95),
        "max": float(np.max(array)),
        "iqr": float(q75 - q25),
        "mad": float(np.median(np.abs(array - median))),
    }


def aggregate_learning_curves(
    frame: pd.DataFrame,
    *,
    metric: str = "episode_return",
    x: str = "episode",
    groups: Sequence[str] = ("agent",),
    seed_column: str = "seed",
    unit_column: str = "trial_id",
    confidence: float = 0.95,
    n_resamples: int = 1_000,
    random_seed: int = 0,
) -> pd.DataFrame:
    """Aggregate a learning curve across independent trial/seed replicates.

    Protocol-v2 tables use ``trial_id`` as the independent unit and ``seed`` as
    its pairing label. Legacy tables without trial identifiers fall back to the
    seed itself. Duplicate unit/x observations and groupings that conceal more
    than one condition fail explicitly rather than being averaged.
    """

    group_columns = _columns(groups)
    require_columns(
        frame,
        {metric, x, seed_column, *group_columns},
        context="learning-curve",
    )
    assert_condition_safe(frame, group_columns, context="learning-curve aggregation")
    unit = independent_unit_column(
        frame,
        preferred=unit_column,
        fallback=seed_column,
        context="learning-curve aggregation",
    )
    assert_columns_constant_within_unit(
        frame,
        unit=unit,
        columns=(*group_columns, seed_column),
        context="learning-curve aggregation",
    )
    assert_unique_rows(frame, (unit, x), context="learning-curve aggregation")

    columns = list(dict.fromkeys((*group_columns, unit, seed_column, x, metric)))
    clean = frame.loc[:, columns].dropna(subset=[metric]).copy()
    if unit != seed_column and not clean.empty:
        replicate_counts = clean.groupby([*group_columns, seed_column], dropna=False)[unit].nunique(
            dropna=False
        )
        repeated = replicate_counts[replicate_counts > 1]
        if not repeated.empty:
            raise UnsafeAggregationError(
                "learning-curve aggregation found multiple trial_ids for the same "
                f"condition/seed; examples include {list(repeated.index[:3])!r}. "
                "Repeated runs with the same seed are not independent replicates."
            )

    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, Any]] = []
    curve_groups = (*group_columns, x)
    for keys, sample in clean.groupby(_grouper(curve_groups), dropna=False, sort=True):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        values = sample[metric].to_numpy(dtype=float)
        summary = distribution_summary(values)
        low, high = bootstrap_confidence_interval(
            values,
            confidence=confidence,
            n_resamples=n_resamples,
            seed=rng,
        )
        row: dict[str, Any] = dict(zip(curve_groups, key_tuple, strict=True))
        row.update(summary)
        row.update(
            {
                "ci_low": low,
                "ci_high": high,
                "n_units": int(sample[unit].nunique()),
                "n_seeds": int(sample[seed_column].nunique()),
            }
        )
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(list(curve_groups)).reset_index(drop=True)


def paired_seed_contrast(
    frame: pd.DataFrame,
    *,
    metric: str,
    factor: str,
    baseline: Any,
    comparison: Any,
    pair_by: Sequence[str] = ("seed",),
    strata: Sequence[str] = (),
    confidence: float = 0.95,
    n_resamples: int = 2_000,
    random_seed: int = 0,
    require_complete: bool = True,
) -> PairedContrastResult:
    """Contrast two conditions after matching their independent seed units.

    Differences are always ``comparison - baseline``. The confidence interval
    resamples matched difference rows, preserving the experimental pairing.
    Input should contain one scalar estimate per factor/seed cell, typically the
    output of :func:`rllab.evaluation.final_performance`.
    """

    pair_columns = _columns(pair_by)
    stratum_columns = _columns(strata)
    if not pair_columns:
        raise ValueError("pair_by must name at least one pairing column")
    if baseline == comparison:
        raise ValueError("baseline and comparison must be different factor levels")
    required = {metric, factor, *pair_columns, *stratum_columns}
    require_columns(frame, required, context="paired contrast")

    selected = frame.loc[frame[factor].isin([baseline, comparison])].copy()
    present = set(selected[factor].dropna().unique())
    missing_levels = [level for level in (baseline, comparison) if level not in present]
    if missing_levels:
        raise ValueError(f"Paired contrast factor {factor!r} is missing levels {missing_levels!r}")
    assert_condition_safe(
        selected,
        (*stratum_columns, factor),
        context="paired contrast",
    )
    cell_keys = (*stratum_columns, *pair_columns, factor)
    assert_unique_rows(selected, cell_keys, context="paired contrast")

    index_columns = (*stratum_columns, *pair_columns)
    wide = selected.pivot(index=list(index_columns), columns=factor, values=metric)
    incomplete = wide[[baseline, comparison]].isna().any(axis=1)
    if bool(incomplete.any()) and require_complete:
        examples = wide.loc[incomplete].head(3).index.tolist()
        raise UnsafeAggregationError(
            f"Paired contrast has {int(incomplete.sum())} incomplete matched pairs; "
            f"examples include {examples!r}. Use require_complete=False only when "
            "dropping unmatched seeds is part of the analysis plan."
        )
    wide = (
        wide.loc[~incomplete, [baseline, comparison]]
        .rename(columns={baseline: "baseline_value", comparison: "comparison_value"})
        .reset_index()
    )
    if wide.empty:
        raise UnsafeAggregationError("Paired contrast has no complete matched pairs")
    wide["difference"] = wide["comparison_value"] - wide["baseline_value"]
    wide["baseline"] = baseline
    wide["comparison"] = comparison
    wide["factor"] = factor

    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, Any]] = []
    grouped: Any
    if stratum_columns:
        grouped = wide.groupby(_grouper(stratum_columns), dropna=False, sort=True)
    else:
        grouped = [((), wide)]
    for keys, sample in grouped:
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        differences = sample["difference"].to_numpy(dtype=float)
        low, high = bootstrap_confidence_interval(
            differences,
            confidence=confidence,
            n_resamples=n_resamples,
            seed=rng,
        )
        row: dict[str, Any] = dict(zip(stratum_columns, key_tuple, strict=True))
        row.update(
            {
                "factor": factor,
                "baseline": baseline,
                "comparison": comparison,
                "n_pairs": len(sample),
                "baseline_mean": float(sample["baseline_value"].mean()),
                "comparison_mean": float(sample["comparison_value"].mean()),
                "mean_difference": float(np.mean(differences)),
                "median_difference": float(np.median(differences)),
                "ci_low": low,
                "ci_high": high,
                "win_rate": float(np.mean(differences > 0.0)),
            }
        )
        rows.append(row)
    return PairedContrastResult(
        pairs=wide.reset_index(drop=True),
        summary=pd.DataFrame(rows),
    )
