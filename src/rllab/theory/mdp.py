"""Finite Markov decision processes with explicit transition and reward tensors.

The representation deliberately keeps the mathematical objects visible: ``P[s,
a, s_next]`` is a transition probability and ``R[s, a, s_next]`` is the reward
received on that transition.  Terminal states have zero continuation value; a
reward for entering a terminal state therefore belongs in ``R`` as usual.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def validate_discount(gamma: float) -> float:
    """Validate and return a discount factor in ``[0, 1]``."""

    gamma = float(gamma)
    if not np.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be finite and in [0, 1], got {gamma!r}")
    return gamma


@dataclass(frozen=True)
class FiniteMDP:
    """A finite MDP represented by dense transition and reward tensors.

    Parameters
    ----------
    P:
        Transition kernel with shape ``(n_states, n_actions, n_states)``.
        Every state-action row must be a probability distribution, including
        rows belonging to terminal states.  Solvers ignore outgoing terminal
        transitions, so an absorbing self-loop is a natural convention.
    R:
        Transition-conditioned rewards with exactly the same shape as ``P``.
    terminal:
        Boolean vector identifying terminal states.  Continuation value is
        masked whenever a transition enters one of these states.
    state_labels:
        Optional unique, hashable labels, useful when state indices encode maze
        coordinates or augmented state.
    """

    P: FloatArray
    R: FloatArray
    terminal: BoolArray
    state_labels: tuple[Hashable, ...] | None = None

    def __init__(
        self,
        P: ArrayLike,
        R: ArrayLike,
        terminal: ArrayLike,
        state_labels: Sequence[Hashable] | None = None,
    ) -> None:
        transitions = np.asarray(P, dtype=np.float64)
        rewards = np.asarray(R, dtype=np.float64)
        raw_terminal = np.asarray(terminal)

        if transitions.ndim != 3:
            raise ValueError(f"P must have three dimensions [S, A, S], got {transitions.shape}")
        n_states, n_actions, n_next_states = transitions.shape
        if n_states == 0 or n_actions == 0 or n_next_states != n_states:
            raise ValueError(
                "P must have nonempty shape [S, A, S] with equal state dimensions, "
                f"got {transitions.shape}"
            )
        if rewards.shape != transitions.shape:
            raise ValueError(f"R must have shape {transitions.shape}, got {rewards.shape}")
        if raw_terminal.shape != (n_states,):
            raise ValueError(f"terminal must have shape ({n_states},), got {raw_terminal.shape}")
        if not np.issubdtype(raw_terminal.dtype, np.bool_) and not np.all(
            np.isin(raw_terminal, (0, 1))
        ):
            raise ValueError("terminal must contain only booleans (or numeric 0/1 values)")
        terminals = raw_terminal.astype(np.bool_, copy=True)

        if not np.all(np.isfinite(transitions)):
            raise ValueError("P contains a non-finite transition probability")
        if not np.all(np.isfinite(rewards)):
            raise ValueError("R contains a non-finite reward")
        if np.any(transitions < 0.0) or np.any(transitions > 1.0):
            raise ValueError("P probabilities must lie in [0, 1]")
        row_sums = transitions.sum(axis=2)
        if not np.allclose(row_sums, 1.0, rtol=1e-10, atol=1e-12):
            bad = np.argwhere(~np.isclose(row_sums, 1.0, rtol=1e-10, atol=1e-12))[0]
            state, action = (int(bad[0]), int(bad[1]))
            raise ValueError(
                "every P[s, a] row must sum to one; "
                f"P[{state}, {action}] sums to {row_sums[state, action]:.16g}"
            )

        labels: tuple[Hashable, ...] | None
        if state_labels is None:
            labels = None
        else:
            labels = tuple(state_labels)
            if len(labels) != n_states:
                raise ValueError(f"state_labels must have length {n_states}, got {len(labels)}")
            try:
                unique_labels = set(labels)
            except TypeError as exc:
                raise ValueError("state_labels must be hashable") from exc
            if len(unique_labels) != n_states:
                raise ValueError("state_labels must be unique")

        # Copy at the model boundary so later mutation of caller-owned arrays
        # cannot silently invalidate an exact solution.
        transitions = transitions.copy()
        rewards = rewards.copy()
        transitions.setflags(write=False)
        rewards.setflags(write=False)
        terminals.setflags(write=False)
        object.__setattr__(self, "P", transitions)
        object.__setattr__(self, "R", rewards)
        object.__setattr__(self, "terminal", terminals)
        object.__setattr__(self, "state_labels", labels)

    @property
    def n_states(self) -> int:
        """Number of states."""

        return int(self.P.shape[0])

    @property
    def n_actions(self) -> int:
        """Number of actions available in every state."""

        return int(self.P.shape[1])

    @property
    def transitions(self) -> FloatArray:
        """Descriptive alias for :attr:`P`."""

        return self.P

    @property
    def rewards(self) -> FloatArray:
        """Descriptive alias for :attr:`R`."""

        return self.R

    @property
    def expected_rewards(self) -> FloatArray:
        """Return ``E[R | s, a]`` with shape ``(n_states, n_actions)``."""

        return np.sum(self.P * self.R, axis=2)

    def state_index(self, label: Hashable) -> int:
        """Resolve a state label to its integer index."""

        if self.state_labels is None:
            raise ValueError("this MDP has no state labels")
        try:
            return self.state_labels.index(label)
        except ValueError as exc:
            raise KeyError(label) from exc

    def q_from_v(self, values: ArrayLike, gamma: float = 0.99) -> FloatArray:
        """Apply the one-step Bellman lookahead to a value vector.

        ``Q(s,a) = sum_s' P(s'|s,a) [R(s,a,s') + gamma V(s')]``.
        Continuation value is zero on entry to a terminal state, and Q-values in
        terminal source states are set to zero because no action is taken there.
        """

        discount = validate_discount(gamma)
        value_array = np.asarray(values, dtype=np.float64)
        if value_array.shape != (self.n_states,):
            raise ValueError(f"values must have shape ({self.n_states},), got {value_array.shape}")
        if not np.all(np.isfinite(value_array)):
            raise ValueError("values contains a non-finite entry")

        continuation = np.where(self.terminal, 0.0, value_array)
        q_values = np.sum(
            self.P * (self.R + discount * continuation[np.newaxis, np.newaxis, :]),
            axis=2,
        )
        q_values[self.terminal, :] = 0.0
        return q_values


def as_policy_matrix(mdp: FiniteMDP, policy: ArrayLike) -> FloatArray:
    """Validate a deterministic or stochastic policy and return ``[S, A]`` form."""

    raw_policy = np.asarray(policy)
    if raw_policy.shape == (mdp.n_states,):
        if not np.issubdtype(raw_policy.dtype, np.integer) and not np.all(
            np.equal(raw_policy, np.floor(raw_policy))
        ):
            raise ValueError("a deterministic policy must contain integer action indices")
        actions = raw_policy.astype(np.int64)
        if np.any(actions < 0) or np.any(actions >= mdp.n_actions):
            raise ValueError(f"policy actions must lie in [0, {mdp.n_actions})")
        matrix = np.zeros((mdp.n_states, mdp.n_actions), dtype=np.float64)
        matrix[np.arange(mdp.n_states), actions] = 1.0
        return matrix

    if raw_policy.shape != (mdp.n_states, mdp.n_actions):
        raise ValueError(
            "policy must have shape "
            f"({mdp.n_states},) or ({mdp.n_states}, {mdp.n_actions}), got {raw_policy.shape}"
        )
    matrix = raw_policy.astype(np.float64, copy=True)
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError("policy probabilities must be finite and nonnegative")
    row_sums = matrix.sum(axis=1)
    nonterminal = ~mdp.terminal
    if not np.allclose(row_sums[nonterminal], 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError("each nonterminal stochastic-policy row must sum to one")
    # A zero row is convenient for terminal states.  A normalized row is also
    # accepted because its actions are never evaluated.
    terminal_sums = row_sums[mdp.terminal]
    valid_terminal = np.isclose(terminal_sums, 0.0) | np.isclose(terminal_sums, 1.0)
    if not np.all(valid_terminal):
        raise ValueError("each terminal policy row must sum to zero or one")
    matrix[mdp.terminal, :] = 0.0
    return matrix


__all__ = [
    "BoolArray",
    "FiniteMDP",
    "FloatArray",
    "as_policy_matrix",
    "validate_discount",
]
