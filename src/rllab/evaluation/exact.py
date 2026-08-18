"""Semantically checked adapters for exact finite-MDP evaluation."""

from __future__ import annotations

import inspect
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

type StateSemantics = Literal["observation", "latent"]
type StateIndexIdentity = tuple[Hashable, ...]


def _normalize_index_identity(
    value: Sequence[Hashable] | None,
    *,
    n_states: int,
    name: str,
) -> StateIndexIdentity | None:
    if value is None:
        return None
    identity = tuple(value)
    if len(identity) != n_states:
        raise ValueError(f"{name} must have length {n_states}, got {len(identity)}")
    try:
        unique = set(identity)
    except TypeError as error:
        raise ValueError(f"{name} entries must be hashable") from error
    if len(unique) != n_states:
        raise ValueError(f"{name} entries must be unique")
    return identity


@dataclass(frozen=True, slots=True)
class ExactSolution:
    """Ground-truth values with an explicit state-index contract.

    ``state_semantics`` says whether rows describe agent observations or
    privileged latent state. ``state_index_identity[i]`` names row ``i``; two
    tables with equal shapes are not comparable when those ordered identities
    differ.
    """

    q_values: np.ndarray
    values: np.ndarray
    policy: np.ndarray
    source: str
    state_semantics: StateSemantics = "observation"
    state_index_identity: StateIndexIdentity | None = None

    def __post_init__(self) -> None:
        q_values = np.asarray(self.q_values, dtype=np.float64)
        values = np.asarray(self.values, dtype=np.float64)
        raw_policy = np.asarray(self.policy)
        if q_values.ndim != 2 or 0 in q_values.shape:
            raise ValueError(f"q_values must have nonempty shape [S, A], got {q_values.shape}")
        n_states, n_actions = q_values.shape
        if values.shape != (n_states,):
            raise ValueError(f"values must have shape ({n_states},), got {values.shape}")
        if raw_policy.shape != (n_states,):
            raise ValueError(f"policy must have shape ({n_states},), got {raw_policy.shape}")
        if not np.all(np.isfinite(q_values)) or not np.all(np.isfinite(values)):
            raise ValueError("exact values must be finite")
        if not np.issubdtype(raw_policy.dtype, np.integer) and not np.all(
            np.equal(raw_policy, np.floor(raw_policy))
        ):
            raise ValueError("exact policy must contain integer action indices")
        policy = raw_policy.astype(np.int64, copy=True)
        if np.any(policy < 0) or np.any(policy >= n_actions):
            raise ValueError(f"exact policy actions must lie in [0, {n_actions})")
        if self.state_semantics not in {"observation", "latent"}:
            raise ValueError(
                "state_semantics must be either 'observation' or 'latent', "
                f"got {self.state_semantics!r}"
            )
        if not self.source:
            raise ValueError("source must be nonempty")
        identity = _normalize_index_identity(
            self.state_index_identity,
            n_states=n_states,
            name="state_index_identity",
        )
        if identity is None:
            identity = tuple(range(n_states))

        q_values = q_values.copy()
        values = values.copy()
        q_values.setflags(write=False)
        values.setflags(write=False)
        policy.setflags(write=False)
        object.__setattr__(self, "q_values", q_values)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "state_index_identity", identity)

    @property
    def n_states(self) -> int:
        return int(self.q_values.shape[0])

    @property
    def n_actions(self) -> int:
        return int(self.q_values.shape[1])


@dataclass(frozen=True, slots=True)
class ExactSolutionStatus:
    """An exact solution or a precise, machine-readable unavailability result."""

    solution: ExactSolution | None
    unavailable_reason: str | None
    attempted_sources: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (self.solution is None) == (self.unavailable_reason is None):
            raise ValueError("status must contain exactly one of solution or unavailable_reason")

    @property
    def available(self) -> bool:
        return self.solution is not None


def extract_q_values(agent: Any) -> np.ndarray | None:
    """Read Q-values from the small set of conventional agent surfaces."""

    for name in ("q_values", "q", "Q", "action_values"):
        if not hasattr(agent, name):
            continue
        value = getattr(agent, name)
        value = value() if callable(value) else value
        if value is not None:
            array = np.asarray(value, dtype=float)
            if array.ndim == 2:
                return array
    return None


def _field(value: Any, *names: str) -> Any:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
        return None
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return None


def _environment_observation_identity(env: Any, n_states: int) -> StateIndexIdentity | None:
    """Return the environment's declared ordered observation-state labels."""

    if hasattr(env, "observation_state_index_identity"):
        value = env.observation_state_index_identity
        value = value() if callable(value) else value
        return _normalize_index_identity(
            value,
            n_states=n_states,
            name="environment observation_state_index_identity",
        )
    observation_space = getattr(env, "observation_space", None)
    if observation_space is not None and hasattr(observation_space, "n"):
        count = int(observation_space.n)
        if count != n_states:
            raise ValueError(
                f"exact solution has {n_states} states but observation_space.n is {count}"
            )
        return tuple(range(n_states))
    return None


def _solution_from(
    value: Any,
    source: str,
    *,
    env: Any,
    default_identity: Sequence[Hashable] | None = None,
) -> ExactSolution | None:
    if value is None:
        return None
    q_values = _field(value, "q_values", "q_star", "Q")
    values = _field(value, "values", "v_values", "v_star")
    policy = _field(value, "policy", "greedy_policy")
    if q_values is None:
        if isinstance(value, np.ndarray) and value.ndim == 2:
            q_values = value
        else:
            return None
    q_array = np.asarray(q_values, dtype=np.float64)
    if q_array.ndim != 2:
        raise ValueError(f"{source} q_values must be two-dimensional, got {q_array.shape}")
    n_states = int(q_array.shape[0])
    values = np.max(q_array, axis=1) if values is None else values
    policy = np.argmax(q_array, axis=1) if policy is None else policy
    raw_semantics = str(_field(value, "state_semantics") or "observation")
    if raw_semantics not in {"observation", "latent"}:
        raise ValueError(
            f"{source} state_semantics must be 'observation' or 'latent', got {raw_semantics!r}"
        )
    semantics: StateSemantics = "latent" if raw_semantics == "latent" else "observation"
    declared_identity = _field(value, "state_index_identity", "state_labels")
    if declared_identity is None:
        declared_identity = default_identity
    if declared_identity is None:
        declared_identity = _environment_observation_identity(env, n_states)
    return ExactSolution(
        q_values=q_array,
        values=np.asarray(values, dtype=np.float64),
        policy=np.asarray(policy),
        source=source,
        state_semantics=semantics,
        state_index_identity=declared_identity,
    )


def _semantic_unavailability(solution: ExactSolution, env: Any) -> str | None:
    if solution.state_semantics != "observation":
        return (
            f"{solution.source} declares state_semantics={solution.state_semantics!r}; "
            "latent-state values are not agent-observation ground truth"
        )
    try:
        observation_identity = _environment_observation_identity(env, solution.n_states)
    except (TypeError, ValueError) as error:
        return str(error)
    if observation_identity is None:
        return (
            "the environment does not expose a finite observation-state index identity; "
            "an exact latent model cannot be compared with an agent Q-table"
        )
    if solution.state_index_identity != observation_identity:
        return (
            f"{solution.source} state-index identity does not match the environment's "
            "ordered observation-state identity"
        )
    return None


def _unavailable_error(error: Exception) -> bool:
    """Recognize the environment's deliberate exact-model guard without coupling."""

    return error.__class__.__name__ == "ExactModelUnavailable"


def exact_solution_status(env: Any, *, gamma: float = 0.99) -> ExactSolutionStatus:
    """Resolve exact observation-state values or explain exactly why none exist."""

    attempted: list[str] = []
    failures: list[str] = []

    for name in ("exact_solution", "optimal_solution"):
        if not hasattr(env, name):
            continue
        source = f"environment.{name}"
        attempted.append(source)
        try:
            function = getattr(env, name)
            value = function(gamma=gamma) if callable(function) else function
            solution = _solution_from(value, source, env=env)
        except (NotImplementedError, TypeError, ValueError, RuntimeError) as error:
            reason = str(error) or error.__class__.__name__
            if _unavailable_error(error):
                return ExactSolutionStatus(None, reason, tuple(attempted))
            failures.append(f"{source}: {reason}")
            continue
        if solution is None:
            failures.append(f"{source}: no two-dimensional Q-values were exposed")
            continue
        semantic_reason = _semantic_unavailability(solution, env)
        if semantic_reason is not None:
            return ExactSolutionStatus(None, semantic_reason, tuple(attempted))
        return ExactSolutionStatus(solution, None, tuple(attempted))

    direct_value = {
        "q_values": next(
            (getattr(env, name) for name in ("q_star", "optimal_q_values") if hasattr(env, name)),
            None,
        ),
        "values": getattr(env, "v_star", None),
        "policy": getattr(env, "optimal_policy", None),
    }
    if direct_value["q_values"] is not None:
        source = "environment.attributes"
        attempted.append(source)
        try:
            solution = _solution_from(direct_value, source, env=env)
            assert solution is not None
        except (TypeError, ValueError) as error:
            failures.append(f"{source}: {error}")
        else:
            semantic_reason = _semantic_unavailability(solution, env)
            if semantic_reason is not None:
                return ExactSolutionStatus(None, semantic_reason, tuple(attempted))
            return ExactSolutionStatus(solution, None, tuple(attempted))

    model: Any = None
    model_source = ""
    for name in ("exact_mdp", "build_exact_mdp", "to_finite_mdp", "model"):
        if not hasattr(env, name):
            continue
        source = f"environment.{name}"
        attempted.append(source)
        try:
            accessor = getattr(env, name)
            candidate = accessor() if callable(accessor) else accessor
        except (NotImplementedError, TypeError, ValueError, RuntimeError) as error:
            reason = str(error) or error.__class__.__name__
            if _unavailable_error(error):
                return ExactSolutionStatus(None, reason, tuple(attempted))
            failures.append(f"{source}: {reason}")
            continue
        if candidate is not None:
            model = candidate
            model_source = source
            break
        failures.append(f"{source}: returned no model")

    if model is None:
        for name in ("transition_kernel", "get_transition_kernel"):
            if not hasattr(env, name):
                continue
            source = f"environment.{name}"
            attempted.append(source)
            try:
                accessor = getattr(env, name)
                transition = accessor() if callable(accessor) else accessor
                reward_accessor = getattr(
                    env,
                    "expected_rewards",
                    getattr(env, "reward_kernel", None),
                )
                rewards = reward_accessor() if callable(reward_accessor) else reward_accessor
                model = (transition, rewards)
                model_source = source
                break
            except (NotImplementedError, TypeError, ValueError, RuntimeError) as error:
                reason = str(error) or error.__class__.__name__
                if _unavailable_error(error):
                    return ExactSolutionStatus(None, reason, tuple(attempted))
                failures.append(f"{source}: {reason}")

    if model is None:
        reason = (
            "; ".join(dict.fromkeys(failures))
            if failures
            else "environment exposes no exact solution or finite-model interface"
        )
        return ExactSolutionStatus(None, reason, tuple(attempted))

    try:
        from rllab.theory import value_iteration
    except (ImportError, AttributeError):
        try:
            from rllab.theory.dynamic_programming import value_iteration
        except (ImportError, AttributeError):
            return ExactSolutionStatus(
                None,
                "the value-iteration solver is unavailable",
                tuple(attempted),
            )

    source = f"value_iteration({model_source})"
    attempted.append(source)
    try:
        parameters = inspect.signature(value_iteration).parameters
        result = value_iteration(model, gamma) if "gamma" in parameters else value_iteration(model)
        if hasattr(result, "converged") and not bool(result.converged):
            return ExactSolutionStatus(
                None,
                f"{source} did not converge",
                tuple(attempted),
            )
        model_identity = getattr(model, "state_labels", None)
        solution = _solution_from(
            result,
            source,
            env=env,
            default_identity=model_identity,
        )
    except (NotImplementedError, TypeError, ValueError, RuntimeError) as error:
        return ExactSolutionStatus(
            None,
            f"{source}: {str(error) or error.__class__.__name__}",
            tuple(attempted),
        )
    if solution is None:
        return ExactSolutionStatus(
            None,
            f"{source} returned no two-dimensional Q-values",
            tuple(attempted),
        )
    semantic_reason = _semantic_unavailability(solution, env)
    if semantic_reason is not None:
        return ExactSolutionStatus(None, semantic_reason, tuple(attempted))
    return ExactSolutionStatus(solution, None, tuple(attempted))


def solve_environment_exactly(env: Any, *, gamma: float = 0.99) -> ExactSolution | None:
    """Return exact agent-observation values, or ``None`` when unavailable.

    Use :func:`exact_solution_status` when the structured unavailability reason is
    needed for metadata or user-facing diagnostics.
    """

    return exact_solution_status(env, gamma=gamma).solution


def compare_to_optimal(
    q_values: np.ndarray,
    optimal: ExactSolution | np.ndarray,
    *,
    state_weights: np.ndarray | None = None,
    estimate_state_semantics: StateSemantics = "observation",
    estimate_state_index_identity: Sequence[Hashable] | None = None,
) -> dict[str, float]:
    """Compute norm, value, and tie-aware policy diagnostics against exact Q*.

    When ``optimal`` is an :class:`ExactSolution`, semantic mismatches are rejected
    before any numeric diagnostic is computed. Callers crossing an indexing
    boundary should provide ``estimate_state_index_identity``; omission preserves
    compatibility for Q-tables already guaranteed to use the runner's observation
    indexing.
    """

    estimate = np.asarray(q_values, dtype=float)
    if estimate.ndim != 2:
        raise ValueError(f"q_values must be two-dimensional, got {estimate.shape}")
    if isinstance(optimal, ExactSolution):
        if optimal.state_semantics != estimate_state_semantics:
            raise ValueError(
                "State-semantics mismatch: estimate uses "
                f"{estimate_state_semantics!r}, truth uses {optimal.state_semantics!r}"
            )
        if estimate_state_index_identity is not None:
            estimate_identity = _normalize_index_identity(
                estimate_state_index_identity,
                n_states=estimate.shape[0],
                name="estimate_state_index_identity",
            )
            if estimate_identity != optimal.state_index_identity:
                raise ValueError(
                    "State-index identity mismatch: estimate and exact solution assign "
                    "different meanings or orderings to Q-table rows"
                )
        truth = optimal.q_values
    else:
        truth = np.asarray(optimal, dtype=float)
    if estimate.shape != truth.shape:
        raise ValueError(f"Q-value shape mismatch: estimate {estimate.shape}, truth {truth.shape}")
    finite_states = np.all(np.isfinite(estimate) & np.isfinite(truth), axis=1)
    if not np.any(finite_states):
        return {
            key: float("nan")
            for key in (
                "q_error_inf",
                "q_error_l2",
                "value_error_inf",
                "value_error_l2",
                "policy_disagreement",
                "optimal_action_fraction",
            )
        }
    estimate = estimate[finite_states]
    truth = truth[finite_states]
    residual = estimate - truth
    value_residual = np.max(estimate, axis=1) - np.max(truth, axis=1)

    weights: np.ndarray | None = None
    if state_weights is not None:
        raw_weights = np.asarray(state_weights, dtype=float)
        if raw_weights.shape != (finite_states.size,):
            raise ValueError(
                f"state_weights must have shape ({finite_states.size},), got {raw_weights.shape}"
            )
        weights = raw_weights[finite_states]
        if not np.all(np.isfinite(weights)):
            raise ValueError("state_weights must be finite")
        if np.any(weights < 0) or not np.sum(weights) > 0:
            raise ValueError("state_weights must be nonnegative with positive total mass")
        weights = weights / np.sum(weights)
    q_l2 = (
        float(np.sqrt(np.sum(weights[:, None] * residual**2) / residual.shape[1]))
        if weights is not None
        else float(np.sqrt(np.mean(residual**2)))
    )
    value_l2 = (
        float(np.sqrt(np.sum(weights * value_residual**2)))
        if weights is not None
        else float(np.sqrt(np.mean(value_residual**2)))
    )
    learned_actions = np.argmax(estimate, axis=1)
    optimal_value = np.max(truth, axis=1)
    learned_is_optimal = np.isclose(
        truth[np.arange(truth.shape[0]), learned_actions], optimal_value, rtol=1e-10, atol=1e-12
    )
    disagreement = 1.0 - learned_is_optimal.astype(float)
    disagreement_rate = (
        float(np.sum(weights * disagreement))
        if weights is not None
        else float(np.mean(disagreement))
    )
    return {
        "q_error_inf": float(np.max(np.abs(residual))),
        "q_error_l2": q_l2,
        "value_error_inf": float(np.max(np.abs(value_residual))),
        "value_error_l2": value_l2,
        "policy_disagreement": disagreement_rate,
        "optimal_action_fraction": 1.0 - disagreement_rate,
    }


__all__ = [
    "ExactSolution",
    "ExactSolutionStatus",
    "StateIndexIdentity",
    "StateSemantics",
    "compare_to_optimal",
    "exact_solution_status",
    "extract_q_values",
    "solve_environment_exactly",
]
