"""Random and exact-planning tabular policies."""

from __future__ import annotations

import numpy as np

from rllab.agents.base import Agent, FloatArray, UpdateRecord
from rllab.theory import FiniteMDP, ValueIterationResult, value_iteration


class RandomAgent(Agent):
    """A seeded uniform-random policy with the standard agent interface."""

    algorithm = "random"

    def __init__(self, n_states: int, n_actions: int, *, seed: int | None = None) -> None:
        super().__init__(n_states, n_actions, seed=seed)
        self._q_values = np.zeros((self.n_states, self.n_actions), dtype=np.float64)

    @property
    def q_values(self) -> FloatArray:
        return self._q_values

    def act(self, state: int, training: bool = True) -> int:
        state = self._validate_state(state)
        action = int(self._rng.integers(self.n_actions))
        if training:
            self._action_counts[state, action] += 1
        return action

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
        self._update_counts[state, action] += 1
        record = UpdateRecord(
            algorithm=self.algorithm,
            step=self._step,
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            terminated=bool(terminated),
            td_error=None,
            target=None,
            old_value=None,
            new_value=None,
            alpha=None,
            epsilon=None,
        )
        self._step += 1
        return record


class PlannerAgent(Agent):
    """A non-learning optimal policy computed once by value iteration."""

    algorithm = "planner"

    def __init__(
        self,
        mdp: FiniteMDP,
        *,
        gamma: float = 0.99,
        tolerance: float = 1e-10,
        max_iterations: int = 10_000,
        seed: int | None = None,
    ) -> None:
        super().__init__(mdp.n_states, mdp.n_actions, seed=seed)
        self.mdp = mdp
        self.gamma = float(gamma)
        self.solution: ValueIterationResult = value_iteration(
            mdp, gamma=self.gamma, tolerance=tolerance, max_iterations=max_iterations
        )
        if not self.solution.converged:
            raise RuntimeError(
                f"planner value iteration did not converge in {max_iterations} iterations"
            )
        self._q_values = self.solution.q_values.copy()
        self._q_values.setflags(write=False)

    @property
    def q_values(self) -> FloatArray:
        return self._q_values

    @property
    def greedy_policy(self) -> np.ndarray:
        return self.solution.policy.copy()

    def act(self, state: int, training: bool = True) -> int:
        state = self._validate_state(state)
        action = int(self.solution.policy[state])
        if training:
            self._action_counts[state, action] += 1
        return action

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
        self._update_counts[state, action] += 1
        record = UpdateRecord(
            algorithm=self.algorithm,
            step=self._step,
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            terminated=bool(terminated),
            td_error=None,
            target=None,
            old_value=float(self._q_values[state, action]),
            new_value=float(self._q_values[state, action]),
            alpha=None,
            epsilon=None,
        )
        self._step += 1
        return record


__all__ = ["PlannerAgent", "RandomAgent"]
