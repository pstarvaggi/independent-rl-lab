"""Small common interface shared by tabular agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class UpdateRecord:
    """A single, instrumentation-friendly agent update.

    ``td_error``, ``alpha``, and value fields are ``None`` for non-learning
    policies.  The redundant derived properties are intentionally cheap: they
    keep metric recorders independent of the particular update rule.
    """

    algorithm: str
    step: int
    state: int
    action: int
    reward: float
    next_state: int
    terminated: bool
    td_error: float | None
    target: float | None
    old_value: float | None
    new_value: float | None
    alpha: float | None
    epsilon: float | None
    next_action: int | None = None

    @property
    def learning_rate(self) -> float | None:
        """Descriptive alias for ``alpha``."""

        return self.alpha

    @property
    def exploration_rate(self) -> float | None:
        """Descriptive alias for ``epsilon``."""

        return self.epsilon

    @property
    def absolute_td_error(self) -> float | None:
        """Absolute TD error when the update has one."""

        return None if self.td_error is None else abs(self.td_error)

    @property
    def squared_td_error(self) -> float | None:
        """Squared TD error when the update has one."""

        return None if self.td_error is None else self.td_error**2

    def as_dict(self) -> dict[str, Any]:
        """Return a flat dictionary suitable for a row-oriented metric table."""

        result = asdict(self)
        result["learning_rate"] = self.learning_rate
        result["exploration_rate"] = self.exploration_rate
        result["absolute_td_error"] = self.absolute_td_error
        result["squared_td_error"] = self.squared_td_error
        return result


class Agent(ABC):
    """Minimal protocol for integer-state, fixed-action tabular agents."""

    algorithm = "agent"

    def __init__(self, n_states: int, n_actions: int, *, seed: int | None = None) -> None:
        if not isinstance(n_states, (int, np.integer)) or n_states <= 0:
            raise ValueError("n_states must be a positive integer")
        if not isinstance(n_actions, (int, np.integer)) or n_actions <= 0:
            raise ValueError("n_actions must be a positive integer")
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self._initial_seed = seed
        self._rng = np.random.default_rng(seed)
        self._step = 0
        self._action_counts = np.zeros((self.n_states, self.n_actions), dtype=np.int64)
        self._update_counts = np.zeros((self.n_states, self.n_actions), dtype=np.int64)

    @property
    def step(self) -> int:
        """Number of completed update calls."""

        return self._step

    @property
    def action_counts(self) -> IntArray:
        """Counts of training-mode action selections, indexed by state and action."""

        return self._action_counts

    @property
    def update_counts(self) -> IntArray:
        """Counts of updates, indexed by state and action."""

        return self._update_counts

    @property
    @abstractmethod
    def q_values(self) -> FloatArray:
        """Current action-value estimates with shape ``[n_states, n_actions]``."""

    @property
    def greedy_policy(self) -> IntArray:
        """Deterministic lowest-index argmax policy."""

        return np.argmax(self.q_values, axis=1).astype(np.int64)

    @property
    def values(self) -> FloatArray:
        """Greedy state-value estimates ``max_a Q(s,a)``."""

        return np.max(self.q_values, axis=1)

    def reset(self, seed: int | None = None) -> None:
        """Clear learned state and counters, then deterministically reseed.

        Passing ``None`` reuses the constructor seed.  This makes repeated
        experimental runs explicit and reproducible.
        """

        selected_seed = self._initial_seed if seed is None else seed
        self._rng = np.random.default_rng(selected_seed)
        self._step = 0
        self._action_counts.fill(0)
        self._update_counts.fill(0)
        self._reset_state()

    def seed_evaluation(self, seed: int) -> None:
        """Reseed action sampling on an evaluation clone without clearing learning."""

        self._rng = np.random.default_rng(seed)

    def _reset_state(self) -> None:
        """Hook for subclasses to clear learned parameters and cached actions."""

        return None

    def _validate_state(self, state: int, *, name: str = "state") -> int:
        if not isinstance(state, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
        state = int(state)
        if not 0 <= state < self.n_states:
            raise IndexError(f"{name} must lie in [0, {self.n_states}), got {state}")
        return state

    def _validate_action(self, action: int) -> int:
        if not isinstance(action, (int, np.integer)):
            raise TypeError("action must be an integer")
        action = int(action)
        if not 0 <= action < self.n_actions:
            raise IndexError(f"action must lie in [0, {self.n_actions}), got {action}")
        return action

    def _validate_transition(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
    ) -> tuple[int, int, float, int]:
        state = self._validate_state(state)
        action = self._validate_action(action)
        next_state = self._validate_state(next_state, name="next_state")
        reward = float(reward)
        if not np.isfinite(reward):
            raise ValueError("reward must be finite")
        return state, action, reward, next_state

    @abstractmethod
    def act(self, state: int, training: bool = True) -> int:
        """Choose an action for an integer state."""

    @abstractmethod
    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        terminated: bool,
    ) -> UpdateRecord:
        """Observe one transition and return its transparent update record."""


# A descriptive alias for readers who prefer the explicit name.
TabularAgent = Agent


__all__ = ["Agent", "TabularAgent", "UpdateRecord"]
