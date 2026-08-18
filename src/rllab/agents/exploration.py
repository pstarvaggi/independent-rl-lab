"""Composable scalar schedules and tabular exploration strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class Schedule(Protocol):
    """A scalar schedule indexed by a nonnegative integer step."""

    def __call__(self, step: int) -> float:
        """Return the scalar value at ``step``."""


type ScheduleLike = float | Schedule | Callable[[int], float]


def _step(step: int) -> int:
    if not isinstance(step, (int, np.integer)) or step < 0:
        raise ValueError("schedule step must be a nonnegative integer")
    return int(step)


@dataclass(frozen=True)
class ConstantSchedule:
    """A schedule with a fixed value."""

    value: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.value):
            raise ValueError("schedule value must be finite")

    def __call__(self, step: int) -> float:
        _step(step)
        return float(self.value)


@dataclass(frozen=True)
class LinearDecaySchedule:
    """Linearly interpolate from ``start`` to ``end`` over ``duration`` steps."""

    start: float
    end: float
    duration: int

    def __post_init__(self) -> None:
        if not np.isfinite(self.start) or not np.isfinite(self.end):
            raise ValueError("schedule endpoints must be finite")
        if not isinstance(self.duration, (int, np.integer)) or self.duration <= 0:
            raise ValueError("duration must be a positive integer")

    def __call__(self, step: int) -> float:
        position = min(_step(step) / int(self.duration), 1.0)
        return float(self.start + position * (self.end - self.start))


@dataclass(frozen=True)
class ExponentialDecaySchedule:
    """Exponentially decay ``initial`` by ``decay_rate`` to ``minimum``."""

    initial: float
    decay_rate: float
    minimum: float = 0.0

    def __post_init__(self) -> None:
        if not all(np.isfinite(value) for value in (self.initial, self.decay_rate, self.minimum)):
            raise ValueError("schedule parameters must be finite")
        if self.initial < self.minimum:
            raise ValueError("initial must be at least minimum")
        if not 0.0 < self.decay_rate <= 1.0:
            raise ValueError("decay_rate must lie in (0, 1]")

    def __call__(self, step: int) -> float:
        return float(max(self.minimum, self.initial * self.decay_rate ** _step(step)))


def as_schedule(value: ScheduleLike) -> Schedule:
    """Turn a constant or callable into the common schedule protocol."""

    if isinstance(value, (int, float, np.integer, np.floating)):
        return ConstantSchedule(float(value))
    if not callable(value):
        raise TypeError("a schedule must be a finite scalar or callable")
    return cast(Schedule, value)


def schedule_value(
    schedule: Schedule,
    step: int,
    *,
    name: str,
    lower: float | None = None,
    upper: float | None = None,
) -> float:
    """Evaluate a schedule and enforce a finite optional range."""

    value = float(schedule(_step(step)))
    if not np.isfinite(value):
        raise ValueError(f"{name} schedule returned a non-finite value at step {step}")
    if lower is not None and value < lower:
        raise ValueError(f"{name} schedule returned {value}, below {lower}")
    if upper is not None and value > upper:
        raise ValueError(f"{name} schedule returned {value}, above {upper}")
    return value


def _validate_action_vectors(
    q_values: ArrayLike, action_counts: ArrayLike
) -> tuple[FloatArray, IntArray]:
    q = np.asarray(q_values, dtype=np.float64)
    counts = np.asarray(action_counts, dtype=np.int64)
    if q.ndim != 1 or q.size == 0:
        raise ValueError("q_values must be a nonempty one-dimensional vector")
    if counts.shape != q.shape:
        raise ValueError(f"action_counts must have shape {q.shape}, got {counts.shape}")
    if not np.all(np.isfinite(q)):
        raise ValueError("q_values contains a non-finite entry")
    if np.any(counts < 0):
        raise ValueError("action_counts cannot be negative")
    return q, counts


def _greedy_probabilities(q_values: FloatArray) -> FloatArray:
    maximizers = np.flatnonzero(np.isclose(q_values, np.max(q_values), rtol=1e-12, atol=1e-14))
    probabilities = np.zeros(q_values.size, dtype=np.float64)
    probabilities[maximizers] = 1.0 / maximizers.size
    return probabilities


class ExplorationStrategy(ABC):
    """An action distribution derived from Q-values and visitation counts."""

    @abstractmethod
    def probabilities(
        self,
        q_values: ArrayLike,
        action_counts: ArrayLike,
        step: int,
    ) -> FloatArray:
        """Return a probability vector over actions."""

    def select(
        self,
        q_values: ArrayLike,
        action_counts: ArrayLike,
        step: int,
        rng: np.random.Generator,
    ) -> int:
        """Sample an action using this strategy."""

        probabilities = self.probabilities(q_values, action_counts, step)
        return int(rng.choice(probabilities.size, p=probabilities))

    def exploration_rate(self, step: int) -> float | None:
        """Return epsilon when the strategy has an epsilon interpretation."""

        _step(step)
        return None


class EpsilonGreedy(ExplorationStrategy):
    """Uniform exploration mixed with random tie-breaking among greedy actions."""

    def __init__(self, epsilon: ScheduleLike = 0.1) -> None:
        self.epsilon_schedule = as_schedule(epsilon)
        # Validate constants immediately while still allowing arbitrary schedules.
        schedule_value(self.epsilon_schedule, 0, name="epsilon", lower=0.0, upper=1.0)

    def exploration_rate(self, step: int) -> float:
        return schedule_value(self.epsilon_schedule, step, name="epsilon", lower=0.0, upper=1.0)

    def probabilities(
        self,
        q_values: ArrayLike,
        action_counts: ArrayLike,
        step: int,
    ) -> FloatArray:
        q, _ = _validate_action_vectors(q_values, action_counts)
        epsilon = self.exploration_rate(step)
        return epsilon / q.size + (1.0 - epsilon) * _greedy_probabilities(q)


class Boltzmann(ExplorationStrategy):
    """Softmax exploration with a fixed or scheduled temperature."""

    def __init__(self, temperature: ScheduleLike = 1.0) -> None:
        self.temperature_schedule = as_schedule(temperature)
        schedule_value(self.temperature_schedule, 0, name="temperature", lower=0.0)

    def probabilities(
        self,
        q_values: ArrayLike,
        action_counts: ArrayLike,
        step: int,
    ) -> FloatArray:
        q, _ = _validate_action_vectors(q_values, action_counts)
        temperature = schedule_value(self.temperature_schedule, step, name="temperature", lower=0.0)
        if temperature == 0.0:
            return _greedy_probabilities(q)
        logits = (q - np.max(q)) / temperature
        weights = np.exp(logits)
        return weights / weights.sum()


class UCB(ExplorationStrategy):
    """Upper-confidence-bound action selection for tabular state-action counts."""

    def __init__(self, coefficient: ScheduleLike = 1.0) -> None:
        self.coefficient_schedule = as_schedule(coefficient)
        schedule_value(self.coefficient_schedule, 0, name="UCB coefficient", lower=0.0)

    def probabilities(
        self,
        q_values: ArrayLike,
        action_counts: ArrayLike,
        step: int,
    ) -> FloatArray:
        q, counts = _validate_action_vectors(q_values, action_counts)
        coefficient = schedule_value(
            self.coefficient_schedule, step, name="UCB coefficient", lower=0.0
        )
        unvisited = counts == 0
        if np.any(unvisited):
            probabilities = np.zeros(q.size, dtype=np.float64)
            probabilities[unvisited] = 1.0 / np.count_nonzero(unvisited)
            return probabilities
        total = max(int(counts.sum()), 1)
        bonus = coefficient * np.sqrt(np.log(total + 1.0) / counts)
        return _greedy_probabilities(q + bonus)


# Descriptive aliases make configuration and notebook prose read naturally.
EpsilonGreedyExploration = EpsilonGreedy
BoltzmannExploration = Boltzmann
UCBExploration = UCB


__all__ = [
    "UCB",
    "Boltzmann",
    "BoltzmannExploration",
    "ConstantSchedule",
    "EpsilonGreedy",
    "EpsilonGreedyExploration",
    "ExplorationStrategy",
    "ExponentialDecaySchedule",
    "LinearDecaySchedule",
    "Schedule",
    "ScheduleLike",
    "UCBExploration",
    "as_schedule",
]
