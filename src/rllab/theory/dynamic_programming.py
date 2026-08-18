"""Exact dynamic-programming solvers for dense finite MDPs.

The iteration histories are returned rather than hidden so convergence can be
inspected, plotted, and tested.  All updates are synchronous Bellman updates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rllab.theory.mdp import FiniteMDP, FloatArray, as_policy_matrix, validate_discount

IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class PolicyEvaluationResult:
    """Result and diagnostics from policy evaluation."""

    values: FloatArray
    q_values: FloatArray
    policy: FloatArray
    iterations: int
    converged: bool
    residual_history: tuple[float, ...]
    method: str

    @property
    def residual(self) -> float:
        """Final Bellman residual (zero only when the history is empty)."""

        return self.residual_history[-1] if self.residual_history else 0.0


@dataclass(frozen=True)
class ValueIterationResult:
    """Optimal values, greedy policy, and value-iteration diagnostics."""

    values: FloatArray
    q_values: FloatArray
    policy: IntArray
    iterations: int
    converged: bool
    residual_history: tuple[float, ...]

    @property
    def residual(self) -> float:
        """Final Bellman residual (zero only when the history is empty)."""

        return self.residual_history[-1] if self.residual_history else 0.0


@dataclass(frozen=True)
class EpsilonSoftValueIterationResult:
    """Optimal epsilon-soft values, policy, and iteration diagnostics.

    ``policy`` is the stochastic epsilon-soft policy that assigns
    ``1 - epsilon + epsilon / n_actions`` to one greedy action and
    ``epsilon / n_actions`` to every other action.  ``greedy_policy`` records
    that selected action explicitly; ties are resolved by the lowest action
    index, matching :func:`numpy.argmax` and :func:`value_iteration`.
    """

    values: FloatArray
    q_values: FloatArray
    policy: FloatArray
    greedy_policy: IntArray
    epsilon: float
    iterations: int
    converged: bool
    residual_history: tuple[float, ...]

    @property
    def residual(self) -> float:
        """Final Bellman residual (zero only when the history is empty)."""

        return self.residual_history[-1] if self.residual_history else 0.0


@dataclass(frozen=True)
class PolicyIterationResult:
    """Optimal values, greedy policy, and policy-iteration diagnostics."""

    values: FloatArray
    q_values: FloatArray
    policy: IntArray
    iterations: int
    converged: bool
    policy_changes: tuple[int, ...]
    evaluation_iterations: tuple[int, ...]
    evaluation_residuals: tuple[float, ...]


def _validate_iteration_controls(tolerance: float, max_iterations: int) -> tuple[float, int]:
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError(f"tolerance must be finite and positive, got {tolerance!r}")
    if not isinstance(max_iterations, (int, np.integer)) or max_iterations <= 0:
        raise ValueError("max_iterations must be a positive integer")
    return tolerance, int(max_iterations)


def _validate_epsilon(epsilon: float) -> float:
    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or not 0.0 <= epsilon <= 1.0:
        raise ValueError(f"epsilon must be finite and in [0, 1], got {epsilon!r}")
    return epsilon


def _initial_values(mdp: FiniteMDP, initial_values: ArrayLike | None) -> FloatArray:
    if initial_values is None:
        values = np.zeros(mdp.n_states, dtype=np.float64)
    else:
        values = np.asarray(initial_values, dtype=np.float64).copy()
        if values.shape != (mdp.n_states,):
            raise ValueError(
                f"initial_values must have shape ({mdp.n_states},), got {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("initial_values contains a non-finite entry")
    values[mdp.terminal] = 0.0
    return values


def bellman_expectation_operator(
    mdp: FiniteMDP,
    policy: ArrayLike,
    values: ArrayLike,
    gamma: float = 0.99,
) -> FloatArray:
    """Apply one Bellman expectation update ``T_pi V``."""

    policy_matrix = as_policy_matrix(mdp, policy)
    updated = np.sum(policy_matrix * mdp.q_from_v(values, gamma), axis=1)
    updated[mdp.terminal] = 0.0
    return updated


def bellman_optimality_operator(
    mdp: FiniteMDP,
    values: ArrayLike,
    gamma: float = 0.99,
) -> FloatArray:
    """Apply one Bellman optimality update ``T_* V``."""

    updated = np.max(mdp.q_from_v(values, gamma), axis=1)
    updated[mdp.terminal] = 0.0
    return updated


def bellman_epsilon_soft_optimality_operator(
    mdp: FiniteMDP,
    values: ArrayLike,
    epsilon: float,
    gamma: float = 0.99,
) -> FloatArray:
    r"""Apply one epsilon-soft optimality update.

    For each nonterminal state, this operator computes

    .. math::

       (T_\epsilon V)(s) = (1 - \epsilon) \max_a Q_V(s,a)
                           + \epsilon \frac{1}{|A|}\sum_a Q_V(s,a).

    This is the best policy in the class formed by mixing one greedy action
    with a uniform random action.  In particular, ``epsilon=0`` recovers the
    ordinary Bellman optimality operator and ``epsilon=1`` evaluates the
    uniform policy.
    """

    exploration = _validate_epsilon(epsilon)
    q_values = mdp.q_from_v(values, gamma)
    updated = (1.0 - exploration) * np.max(q_values, axis=1)
    updated += exploration * np.mean(q_values, axis=1)
    updated[mdp.terminal] = 0.0
    return updated


def policy_evaluation(
    mdp: FiniteMDP,
    policy: ArrayLike,
    gamma: float = 0.99,
    tolerance: float = 1e-10,
    max_iterations: int = 10_000,
    *,
    initial_values: ArrayLike | None = None,
    method: Literal["iterative", "direct"] = "iterative",
) -> PolicyEvaluationResult:
    """Evaluate a deterministic or stochastic policy.

    ``method="iterative"`` performs transparent synchronous Bellman updates.
    ``method="direct"`` solves ``(I - gamma P_pi) V = r_pi`` and is useful as a
    numerical cross-check.  The direct method can be singular for an improper
    undiscounted continuing policy.
    """

    discount = validate_discount(gamma)
    tolerance, max_iterations = _validate_iteration_controls(tolerance, max_iterations)
    policy_matrix = as_policy_matrix(mdp, policy)

    if method == "direct":
        continuation = (~mdp.terminal).astype(np.float64)
        transition_policy = np.einsum("sa,san->sn", policy_matrix, mdp.P)
        transition_policy *= continuation[np.newaxis, :]
        reward_policy = np.einsum("sa,san,san->s", policy_matrix, mdp.P, mdp.R)
        system = np.eye(mdp.n_states, dtype=np.float64) - discount * transition_policy
        system[mdp.terminal, :] = 0.0
        system[mdp.terminal, mdp.terminal] = 1.0
        reward_policy[mdp.terminal] = 0.0
        try:
            values = np.linalg.solve(system, reward_policy)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "policy evaluation linear system is singular; use a discounted or proper policy"
            ) from exc
        values[mdp.terminal] = 0.0
        q_values = mdp.q_from_v(values, discount)
        residual = float(np.max(np.abs(np.sum(policy_matrix * q_values, axis=1) - values)))
        return PolicyEvaluationResult(
            values=values,
            q_values=q_values,
            policy=policy_matrix,
            iterations=1,
            converged=residual <= tolerance,
            residual_history=(residual,),
            method=method,
        )

    if method != "iterative":
        raise ValueError(f"unknown policy-evaluation method {method!r}")

    values = _initial_values(mdp, initial_values)
    residuals: list[float] = []
    converged = False
    for _ in range(max_iterations):
        updated = bellman_expectation_operator(mdp, policy_matrix, values, discount)
        residual = float(np.max(np.abs(updated - values)))
        residuals.append(residual)
        values = updated
        if residual <= tolerance:
            converged = True
            break

    q_values = mdp.q_from_v(values, discount)
    return PolicyEvaluationResult(
        values=values,
        q_values=q_values,
        policy=policy_matrix,
        iterations=len(residuals),
        converged=converged,
        residual_history=tuple(residuals),
        method=method,
    )


def value_iteration(
    mdp: FiniteMDP,
    gamma: float = 0.99,
    tolerance: float = 1e-10,
    max_iterations: int = 10_000,
    *,
    initial_values: ArrayLike | None = None,
) -> ValueIterationResult:
    """Compute an optimal deterministic policy by value iteration."""

    discount = validate_discount(gamma)
    tolerance, max_iterations = _validate_iteration_controls(tolerance, max_iterations)
    values = _initial_values(mdp, initial_values)
    residuals: list[float] = []
    converged = False

    for _ in range(max_iterations):
        updated = bellman_optimality_operator(mdp, values, discount)
        residual = float(np.max(np.abs(updated - values)))
        residuals.append(residual)
        values = updated
        if residual <= tolerance:
            converged = True
            break

    q_values = mdp.q_from_v(values, discount)
    policy = np.argmax(q_values, axis=1).astype(np.int64)
    policy[mdp.terminal] = 0
    return ValueIterationResult(
        values=values,
        q_values=q_values,
        policy=policy,
        iterations=len(residuals),
        converged=converged,
        residual_history=tuple(residuals),
    )


def epsilon_soft_value_iteration(
    mdp: FiniteMDP,
    epsilon: float,
    gamma: float = 0.99,
    tolerance: float = 1e-10,
    max_iterations: int = 10_000,
    *,
    initial_values: ArrayLike | None = None,
) -> EpsilonSoftValueIterationResult:
    r"""Compute the optimal epsilon-soft policy by value iteration.

    The policy class uses the common epsilon-greedy convention

    .. math::

       \pi(a \mid s) = \frac{\epsilon}{|A|}
                       + (1 - \epsilon)\,\mathbf{1}[a = a^*(s)].

    Thus the greedy action remains eligible for the uniform exploration mass.
    The returned ``policy`` is a stochastic ``[state, action]`` matrix.  Rows
    for terminal states are zero because no action is taken there.
    """

    exploration = _validate_epsilon(epsilon)
    discount = validate_discount(gamma)
    tolerance, max_iterations = _validate_iteration_controls(tolerance, max_iterations)
    values = _initial_values(mdp, initial_values)
    residuals: list[float] = []
    converged = False

    for _ in range(max_iterations):
        updated = bellman_epsilon_soft_optimality_operator(
            mdp,
            values,
            exploration,
            discount,
        )
        residual = float(np.max(np.abs(updated - values)))
        residuals.append(residual)
        values = updated
        if residual <= tolerance:
            converged = True
            break

    q_values = mdp.q_from_v(values, discount)
    greedy_policy = np.argmax(q_values, axis=1).astype(np.int64)
    greedy_policy[mdp.terminal] = 0
    policy = np.full(
        (mdp.n_states, mdp.n_actions),
        exploration / mdp.n_actions,
        dtype=np.float64,
    )
    policy[np.arange(mdp.n_states), greedy_policy] += 1.0 - exploration
    policy[mdp.terminal, :] = 0.0
    return EpsilonSoftValueIterationResult(
        values=values,
        q_values=q_values,
        policy=policy,
        greedy_policy=greedy_policy,
        epsilon=exploration,
        iterations=len(residuals),
        converged=converged,
        residual_history=tuple(residuals),
    )


def policy_iteration(
    mdp: FiniteMDP,
    gamma: float = 0.99,
    tolerance: float = 1e-10,
    max_iterations: int = 1_000,
    *,
    evaluation_max_iterations: int = 10_000,
    initial_policy: ArrayLike | None = None,
) -> PolicyIterationResult:
    """Compute an optimal deterministic policy by evaluate/improve iteration."""

    discount = validate_discount(gamma)
    tolerance, max_iterations = _validate_iteration_controls(tolerance, max_iterations)
    _, evaluation_max_iterations = _validate_iteration_controls(
        tolerance, evaluation_max_iterations
    )

    if initial_policy is None:
        policy = np.zeros(mdp.n_states, dtype=np.int64)
    else:
        policy_matrix = as_policy_matrix(mdp, initial_policy)
        policy = np.argmax(policy_matrix, axis=1).astype(np.int64)
    policy[mdp.terminal] = 0

    changes: list[int] = []
    evaluation_iterations: list[int] = []
    evaluation_residuals: list[float] = []
    values = np.zeros(mdp.n_states, dtype=np.float64)
    q_values = mdp.q_from_v(values, discount)
    converged = False

    for _ in range(max_iterations):
        evaluation = policy_evaluation(
            mdp,
            policy,
            discount,
            tolerance,
            evaluation_max_iterations,
            initial_values=values,
        )
        values = evaluation.values
        q_values = mdp.q_from_v(values, discount)
        improved = np.argmax(q_values, axis=1).astype(np.int64)
        improved[mdp.terminal] = 0
        changed = int(np.count_nonzero(improved[~mdp.terminal] != policy[~mdp.terminal]))
        changes.append(changed)
        evaluation_iterations.append(evaluation.iterations)
        evaluation_residuals.append(evaluation.residual)
        policy = improved

        if not evaluation.converged:
            break
        if changed == 0:
            converged = True
            break

    return PolicyIterationResult(
        values=values,
        q_values=q_values,
        policy=policy,
        iterations=len(changes),
        converged=converged,
        policy_changes=tuple(changes),
        evaluation_iterations=tuple(evaluation_iterations),
        evaluation_residuals=tuple(evaluation_residuals),
    )


__all__ = [
    "EpsilonSoftValueIterationResult",
    "PolicyEvaluationResult",
    "PolicyIterationResult",
    "ValueIterationResult",
    "bellman_epsilon_soft_optimality_operator",
    "bellman_expectation_operator",
    "bellman_optimality_operator",
    "epsilon_soft_value_iteration",
    "policy_evaluation",
    "policy_iteration",
    "value_iteration",
]
