"""Readable one-step tabular temporal-difference control algorithms."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from rllab.agents.base import Agent, FloatArray, UpdateRecord
from rllab.agents.exploration import (
    EpsilonGreedy,
    ExplorationStrategy,
    Schedule,
    ScheduleLike,
    as_schedule,
    schedule_value,
)
from rllab.theory.mdp import validate_discount

type InitialQ = float | ArrayLike


class TDControlAgent(Agent):
    """Common mechanics for one-step tabular TD control.

    Subclasses expose their mathematical difference through ``_bootstrap``.
    A learning-rate schedule is indexed by the number of previous updates to
    the particular state-action pair, while exploration schedules are indexed
    by the global number of completed updates.
    """

    algorithm = "td_control"

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        *,
        learning_rate: ScheduleLike = 0.1,
        gamma: float = 0.99,
        exploration: ExplorationStrategy | None = None,
        initial_q: InitialQ = 0.0,
        seed: int | None = None,
        alpha: ScheduleLike | None = None,
        epsilon: ScheduleLike | None = None,
    ) -> None:
        super().__init__(n_states, n_actions, seed=seed)
        if alpha is not None:
            if learning_rate != 0.1:
                raise ValueError("specify only one of learning_rate and alpha")
            learning_rate = alpha
        if epsilon is not None and exploration is not None:
            raise ValueError("specify epsilon or exploration, not both")
        if exploration is None:
            exploration = EpsilonGreedy(0.1 if epsilon is None else epsilon)

        self.gamma = validate_discount(gamma)
        self.learning_rate_schedule: Schedule = as_schedule(learning_rate)
        schedule_value(
            self.learning_rate_schedule,
            0,
            name="learning rate",
            lower=0.0,
            upper=1.0,
        )
        self.exploration = exploration
        self._initial_q = self._validate_initial_q(initial_q)
        self._q_values = self._initial_q.copy()
        self._pending_action: tuple[int, int] | None = None

    def _validate_initial_q(self, initial_q: InitialQ) -> FloatArray:
        values = np.asarray(initial_q, dtype=np.float64)
        if values.ndim == 0:
            value = float(values.item())
            if not np.isfinite(value):
                raise ValueError("initial_q must be finite")
            return np.full((self.n_states, self.n_actions), value, dtype=np.float64)
        if values.shape != (self.n_states, self.n_actions):
            raise ValueError(
                "initial_q must be a scalar or have shape "
                f"({self.n_states}, {self.n_actions}), got {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("initial_q contains a non-finite entry")
        return values.copy()

    @property
    def q_values(self) -> FloatArray:
        return self._q_values

    def _reset_state(self) -> None:
        self._q_values[...] = self._initial_q
        self._pending_action = None

    def end_episode(self) -> None:
        """Discard an on-policy action cached beyond an episode boundary."""

        self._pending_action = None

    def _learning_rate(self, state: int, action: int) -> float:
        """Return alpha; override here for visitation/variance-adaptive rules."""

        visits = int(self._update_counts[state, action])
        return schedule_value(
            self.learning_rate_schedule,
            visits,
            name="learning rate",
            lower=0.0,
            upper=1.0,
        )

    def _exploration_rate(self) -> float | None:
        rate_method = getattr(self.exploration, "exploration_rate", None)
        if rate_method is None:
            return None
        rate = rate_method(self._step)
        return None if rate is None else float(rate)

    def _sample_training_action(self, state: int) -> int:
        return int(
            self.exploration.select(
                self._q_values[state],
                self._action_counts[state],
                self._step,
                self._rng,
            )
        )

    def act(self, state: int, training: bool = True) -> int:
        state = self._validate_state(state)
        if not training:
            return int(np.argmax(self._q_values[state]))

        if self._pending_action is not None:
            pending_state, pending_action = self._pending_action
            self._pending_action = None
            if pending_state == state:
                self._action_counts[state, pending_action] += 1
                return pending_action

        action = self._sample_training_action(state)
        self._action_counts[state, action] += 1
        return action

    def _bootstrap(self, next_state: int, terminated: bool) -> tuple[float, int | None]:
        """Return next-state value and optional selected action."""

        raise NotImplementedError

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        terminated: bool,
    ) -> UpdateRecord:
        state, action, reward, next_state = self._validate_transition(
            state, action, reward, next_state
        )
        terminated = bool(terminated)
        alpha = self._learning_rate(state, action)
        epsilon = self._exploration_rate()
        bootstrap, next_action = self._bootstrap(next_state, terminated)
        target = reward if terminated else reward + self.gamma * bootstrap
        old_value = float(self._q_values[state, action])
        td_error = float(target - old_value)
        new_value = float(old_value + alpha * td_error)
        self._q_values[state, action] = new_value
        self._update_counts[state, action] += 1

        record = UpdateRecord(
            algorithm=self.algorithm,
            step=self._step,
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            terminated=terminated,
            td_error=td_error,
            target=float(target),
            old_value=old_value,
            new_value=new_value,
            alpha=alpha,
            epsilon=epsilon,
            next_action=next_action,
        )
        self._step += 1
        return record


class SarsaAgent(TDControlAgent):
    """On-policy SARSA: ``r + gamma Q(s', a')``."""

    algorithm = "sarsa"

    def _bootstrap(self, next_state: int, terminated: bool) -> tuple[float, int | None]:
        if terminated:
            self._pending_action = None
            return 0.0, None
        next_action = self._sample_training_action(next_state)
        self._pending_action = (next_state, next_action)
        return float(self._q_values[next_state, next_action]), next_action


class ExpectedSarsaAgent(TDControlAgent):
    """Expected SARSA under the current exploration policy."""

    algorithm = "expected_sarsa"

    def _bootstrap(self, next_state: int, terminated: bool) -> tuple[float, int | None]:
        if terminated:
            return 0.0, None
        probabilities = self.exploration.probabilities(
            self._q_values[next_state],
            self._action_counts[next_state],
            self._step,
        )
        return float(np.dot(probabilities, self._q_values[next_state])), None


class QLearningAgent(TDControlAgent):
    """Off-policy Q-learning: ``r + gamma max_a Q(s', a)``."""

    algorithm = "q_learning"

    def _bootstrap(self, next_state: int, terminated: bool) -> tuple[float, int | None]:
        if terminated:
            return 0.0, None
        next_action = int(np.argmax(self._q_values[next_state]))
        return float(self._q_values[next_state, next_action]), next_action


class DoubleQLearningAgent(TDControlAgent):
    """Double Q-learning with unbiased cross-table action evaluation."""

    algorithm = "double_q_learning"

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        *,
        learning_rate: ScheduleLike = 0.1,
        gamma: float = 0.99,
        exploration: ExplorationStrategy | None = None,
        initial_q: InitialQ = 0.0,
        seed: int | None = None,
        alpha: ScheduleLike | None = None,
        epsilon: ScheduleLike | None = None,
    ) -> None:
        super().__init__(
            n_states,
            n_actions,
            learning_rate=learning_rate,
            gamma=gamma,
            exploration=exploration,
            initial_q=initial_q,
            seed=seed,
            alpha=alpha,
            epsilon=epsilon,
        )
        # The superclass table becomes Q1.  Q2 begins at the same prior.
        self._q1_values = self._q_values
        self._q2_values = self._initial_q.copy()

    @property
    def q1_values(self) -> FloatArray:
        """First estimator table."""

        return self._q1_values

    @property
    def q2_values(self) -> FloatArray:
        """Second estimator table."""

        return self._q2_values

    @property
    def q_values(self) -> FloatArray:
        """Mean of the two estimators, on the same scale as Q*."""

        return (self._q1_values + self._q2_values) / 2.0

    def _reset_state(self) -> None:
        self._q1_values[...] = self._initial_q
        self._q2_values[...] = self._initial_q
        # Exploration in the base class reads _q_values; retain Q1 as that
        # private storage but override action selection to use the mean.
        self._pending_action = None

    def _sample_training_action(self, state: int) -> int:
        return int(
            self.exploration.select(
                self.q_values[state],
                self._action_counts[state],
                self._step,
                self._rng,
            )
        )

    def act(self, state: int, training: bool = True) -> int:
        state = self._validate_state(state)
        if not training:
            return int(np.argmax(self.q_values[state]))
        action = self._sample_training_action(state)
        self._action_counts[state, action] += 1
        return action

    def _bootstrap(self, next_state: int, terminated: bool) -> tuple[float, int | None]:
        # Double Q-learning has a coupled update, implemented below.
        raise NotImplementedError

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        terminated: bool,
    ) -> UpdateRecord:
        state, action, reward, next_state = self._validate_transition(
            state, action, reward, next_state
        )
        terminated = bool(terminated)
        alpha = self._learning_rate(state, action)
        epsilon = self._exploration_rate()
        update_first = bool(self._rng.integers(2))

        if update_first:
            update_table, evaluation_table = self._q1_values, self._q2_values
        else:
            update_table, evaluation_table = self._q2_values, self._q1_values

        if terminated:
            next_action = None
            target = reward
        else:
            next_action = int(np.argmax(update_table[next_state]))
            target = reward + self.gamma * float(evaluation_table[next_state, next_action])

        old_value = float(update_table[state, action])
        td_error = float(target - old_value)
        new_value = float(old_value + alpha * td_error)
        update_table[state, action] = new_value
        self._update_counts[state, action] += 1

        record = UpdateRecord(
            algorithm=self.algorithm,
            step=self._step,
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            terminated=terminated,
            td_error=td_error,
            target=float(target),
            old_value=old_value,
            new_value=new_value,
            alpha=alpha,
            epsilon=epsilon,
            next_action=next_action,
        )
        self._step += 1
        return record


# Conventional acronym-preserving spellings are useful in papers and notebooks.
SARSAAgent = SarsaAgent
ExpectedSARSAAgent = ExpectedSarsaAgent


__all__ = [
    "DoubleQLearningAgent",
    "ExpectedSARSAAgent",
    "ExpectedSarsaAgent",
    "QLearningAgent",
    "SARSAAgent",
    "SarsaAgent",
    "TDControlAgent",
]
