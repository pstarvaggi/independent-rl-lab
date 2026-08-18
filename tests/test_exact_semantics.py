from __future__ import annotations

import numpy as np
import pytest

from rllab.environments import StochasticMazeEnv
from rllab.evaluation.exact import (
    ExactSolution,
    compare_to_optimal,
    exact_solution_status,
    solve_environment_exactly,
)


def _solution(
    *,
    semantics: str = "observation",
    identity: tuple[str, str] = ("left", "right"),
) -> ExactSolution:
    q_values = np.array([[0.0, 1.0], [2.0, 0.0]])
    return ExactSolution(
        q_values=q_values,
        values=np.max(q_values, axis=1),
        policy=np.argmax(q_values, axis=1),
        source="test",
        state_semantics=semantics,  # type: ignore[arg-type]
        state_index_identity=identity,
    )


def test_maze_exact_solution_carries_observation_semantics_and_row_identity() -> None:
    env = StochasticMazeEnv(shape=(1, 3), start=(0, 0), goals={(0, 2): 1.0})
    solution = solve_environment_exactly(env, gamma=0.95)
    assert solution is not None
    assert solution.state_semantics == "observation"
    assert solution.state_index_identity == env.index_to_state
    assert not solution.q_values.flags.writeable
    assert not solution.values.flags.writeable
    assert not solution.policy.flags.writeable


def test_partial_observation_has_a_precise_exact_unavailability_reason() -> None:
    env = StochasticMazeEnv(
        shape=(1, 2),
        goals={(0, 1): 1.0},
        observation_mode="noisy_state",
        state_observation_noise=0.25,
    )
    status = exact_solution_status(env)
    assert not status.available
    assert status.solution is None
    assert status.unavailable_reason is not None
    assert "partially observed" in status.unavailable_reason
    assert "latent positions, not noisy observations" in status.unavailable_reason
    assert status.attempted_sources == ("environment.exact_mdp",)


def test_exact_status_explains_an_environment_without_a_model() -> None:
    status = exact_solution_status(object())
    assert not status.available
    assert status.unavailable_reason == (
        "environment exposes no exact solution or finite-model interface"
    )
    assert status.attempted_sources == ()


def test_comparison_rejects_latent_truth_for_an_observation_q_table() -> None:
    truth = _solution(semantics="latent")
    with pytest.raises(ValueError, match="State-semantics mismatch"):
        compare_to_optimal(truth.q_values, truth)


def test_comparison_rejects_equal_shape_but_reordered_state_identity() -> None:
    truth = _solution()
    with pytest.raises(ValueError, match="State-index identity mismatch"):
        compare_to_optimal(
            truth.q_values,
            truth,
            estimate_state_index_identity=("right", "left"),
        )


def test_comparison_accepts_matching_semantics_and_state_identity() -> None:
    truth = _solution()
    diagnostics = compare_to_optimal(
        truth.q_values,
        truth,
        estimate_state_index_identity=("left", "right"),
    )
    assert diagnostics["q_error_inf"] == 0.0
    assert diagnostics["policy_disagreement"] == 0.0
