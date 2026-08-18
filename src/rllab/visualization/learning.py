"""Seed-level learning curves, distributions, sweeps, and TD diagnostics."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from rllab.evaluation.sample_efficiency import final_performance
from rllab.metrics import (
    aggregate_learning_curves,
    td_error_summary,
    td_error_trial_summary,
)
from rllab.metrics.validation import assert_condition_safe


def plot_learning_curves(
    frame: pd.DataFrame,
    *,
    metric: str = "episode_return",
    x: str = "episode",
    group: str = "agent",
    ax: Axes | None = None,
    confidence: float = 0.95,
    n_resamples: int = 1_000,
    individual: bool = False,
    smooth: int | None = None,
    log_x: bool = False,
    log_y: bool = False,
) -> tuple[Figure, Axes, pd.DataFrame]:
    """Mean seed curve with bootstrap bands and optional individual runs."""

    if ax is None:
        figure, ax = plt.subplots(figsize=(8, 4.8))
    else:
        figure = ax.figure  # type: ignore[assignment]
    data = frame.copy()
    if smooth is not None:
        if smooth < 1:
            raise ValueError("smooth must be positive")
        keys = [group, "seed"]
        if "trial_id" in data:
            keys = ["trial_id"]
        data = data.sort_values([*keys, x])
        data[metric] = data.groupby(keys, dropna=False)[metric].transform(
            lambda values: values.rolling(smooth, min_periods=1).mean()
        )
    summary = aggregate_learning_curves(
        data,
        metric=metric,
        x=x,
        groups=(group,),
        confidence=confidence,
        n_resamples=n_resamples,
    )
    colors = plt.get_cmap("tab10")
    for index, (label, sample) in enumerate(summary.groupby(group, sort=False)):
        color = colors(index % 10)
        ax.plot(sample[x], sample["mean"], label=str(label), color=color, linewidth=2)
        ax.fill_between(
            sample[x], sample["ci_low"], sample["ci_high"], color=color, alpha=0.2, linewidth=0
        )
        if individual:
            raw = data[data[group] == label]
            run_column = "trial_id" if "trial_id" in raw else "seed"
            for _, run in raw.groupby(run_column):
                ax.plot(run[x], run[metric], color=color, alpha=0.12, linewidth=0.7)
    ax.set_xlabel(x.replace("_", " ").title())
    ax.set_ylabel(metric.replace("_", " ").title())
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    return figure, ax, summary


def plot_final_distribution(
    frame: pd.DataFrame,
    *,
    metric: str = "episode_return",
    group: str = "agent",
    last_episodes: int = 100,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, pd.DataFrame]:
    """Show the seed distribution of final-window performance as box + points."""

    if last_episodes < 1:
        raise ValueError("last_episodes must be positive")
    if ax is None:
        figure, ax = plt.subplots(figsize=(7, 4.5))
    else:
        figure = ax.figure  # type: ignore[assignment]
    assert_condition_safe(frame, (group,), context="final-distribution plot")
    per_seed = final_performance(
        frame,
        metrics=(metric,),
        last_episodes=last_episodes,
        groups=(group,),
    )
    labels = list(pd.unique(per_seed[group]))
    samples = [
        per_seed.loc[per_seed[group] == label, metric].dropna().to_numpy() for label in labels
    ]
    ax.boxplot(samples, tick_labels=[str(label) for label in labels], showfliers=False)
    rng = np.random.default_rng(0)
    for index, values in enumerate(samples, start=1):
        ax.scatter(index + rng.uniform(-0.08, 0.08, values.size), values, alpha=0.65, s=22)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.grid(axis="y", alpha=0.22)
    return figure, ax, per_seed


def plot_sweep_response(
    frame: pd.DataFrame,
    *,
    parameter: str,
    metric: str = "episode_return",
    group: str = "agent",
    last_episodes: int = 100,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, pd.DataFrame]:
    """Performance distribution versus a stochasticity or dynamics parameter."""

    if parameter not in frame:
        prefixed = f"sweep_{parameter.replace('.', '_')}"
        if prefixed in frame:
            parameter = prefixed
        else:
            raise KeyError(parameter)
    if ax is None:
        figure, ax = plt.subplots(figsize=(7.5, 4.8))
    else:
        figure = ax.figure  # type: ignore[assignment]
    assert_condition_safe(frame, (group, parameter), context="sweep-response plot")
    per_seed = final_performance(
        frame,
        metrics=(metric,),
        last_episodes=last_episodes,
        groups=(group, parameter),
    )
    colors = plt.get_cmap("tab10")
    for index, (label, sample) in enumerate(per_seed.groupby(group, sort=False)):
        aggregate = sample.groupby(parameter)[metric].agg(["mean", "sem"]).reset_index()
        ax.errorbar(
            aggregate[parameter],
            aggregate["mean"],
            yerr=aggregate["sem"],
            marker="o",
            capsize=3,
            label=str(label),
            color=colors(index % 10),
        )
    ax.set_xlabel(parameter.removeprefix("sweep_").replace("_", " ").title())
    ax.set_ylabel(f"Final {metric.replace('_', ' ')}")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    return figure, ax, per_seed


def plot_paired_contrasts(
    summary: pd.DataFrame,
    *,
    label: str = "comparison",
    ax: Axes | None = None,
    difference_label: str = "Paired mean difference (comparison - baseline)",
    title: str = "Matched-seed contrasts",
) -> tuple[Figure, Axes]:
    """Plot paired mean differences with bootstrap confidence intervals.

    ``summary`` is normally one or more concatenated ``.summary`` frames from
    :func:`rllab.metrics.paired_seed_contrast`. Positive values favor the
    comparison when the underlying metric is better when larger.
    """

    required = {label, "mean_difference", "ci_low", "ci_high"}
    missing = required - set(summary.columns)
    if missing:
        raise KeyError(f"Missing paired-contrast columns: {sorted(missing)}")
    if summary.empty:
        raise ValueError("paired-contrast summary is empty")
    if ax is None:
        height = max(3.2, 0.55 * len(summary) + 1.4)
        figure, ax = plt.subplots(figsize=(8, height))
    else:
        figure = ax.figure  # type: ignore[assignment]

    estimates = summary["mean_difference"].to_numpy(dtype=float)
    lows = summary["ci_low"].to_numpy(dtype=float)
    highs = summary["ci_high"].to_numpy(dtype=float)
    positions = np.arange(len(summary))
    errors = np.vstack((estimates - lows, highs - estimates))
    colors = ["#2a9d8f" if value >= 0 else "#e76f51" for value in estimates]
    for position, estimate, error, color in zip(
        positions, estimates, errors.T, colors, strict=True
    ):
        ax.errorbar(
            estimate,
            position,
            xerr=error.reshape(2, 1),
            fmt="o",
            color=color,
            capsize=4,
            markersize=6,
        )
    ax.axvline(0.0, color="#333333", linewidth=1, linestyle="--")
    ax.set_yticks(positions, summary[label].astype(str))
    ax.set_xlabel(difference_label)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.22)
    ax.invert_yaxis()
    return figure, ax


def plot_td_error_diagnostics(
    steps: pd.DataFrame,
    *,
    group: str = "agent",
    max_lag: int = 30,
) -> tuple[Figure, np.ndarray]:
    """Compact distribution/time/state-action view of TD residual behavior."""

    if "td_error" not in steps:
        raise KeyError("steps table has no td_error column")
    if max_lag < 1:
        raise ValueError("max_lag must be positive")
    assert_condition_safe(steps, (group,), context="TD-diagnostic plot")
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for label, sample in steps.groupby(group, dropna=False, sort=False):
        values = sample["td_error"].dropna().to_numpy(dtype=float)
        axes[0, 0].hist(
            values, bins=60, density=True, histtype="step", linewidth=1.5, label=str(label)
        )
        temporal = sample.copy()
        unit = "trial_id" if "trial_id" in temporal else "seed"
        if unit not in temporal:
            unit = "_trial_id"
            temporal[unit] = "legacy-single-trial"
        episode = (
            temporal.groupby([unit, "episode"], dropna=False)["td_error"]
            .agg(lambda x: np.var(x, ddof=1) if len(x) > 1 else 0.0)
            .groupby("episode")
            .mean()
        )
        axes[0, 1].plot(episode.index, episode.values, label=str(label))

        maximum_length = int(temporal.groupby(unit).size().max())
        lags = np.arange(1, min(max_lag, max(1, maximum_length - 1)) + 1)
        trial_summary = td_error_trial_summary(
            temporal,
            groups=(),
            autocorrelation_lags=tuple(int(lag) for lag in lags),
            trial_column=unit,
        )
        correlations = [trial_summary[f"autocorrelation_lag_{int(lag)}"].mean() for lag in lags]
        axes[1, 0].plot(lags, correlations, label=str(label))
    summary = td_error_summary(steps, groups=(group, "state", "action"))
    if not summary.empty:
        top = summary.nlargest(min(20, len(summary)), "variance_td_error")
        labels = [
            f"{label}:{int(state)},{int(action)}"
            for label, state, action in zip(top[group], top["state"], top["action"], strict=True)
        ]
        axes[1, 1].bar(np.arange(len(top)), top["variance_td_error"])
        axes[1, 1].set_xticks(np.arange(len(top)), labels, rotation=70, fontsize=7)
    axes[0, 0].set(title="TD-error distribution", xlabel="TD error", ylabel="density")
    axes[0, 1].set(title="Within-episode variance", xlabel="episode", ylabel="variance")
    axes[1, 0].set(title="TD-error autocorrelation", xlabel="lag", ylabel="correlation")
    axes[1, 1].set(
        title="Most volatile state-action pairs", xlabel="state, action", ylabel="variance"
    )
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False)
    return figure, axes
