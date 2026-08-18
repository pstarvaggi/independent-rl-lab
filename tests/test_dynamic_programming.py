"""Exact checks for finite-MDP validation and Bellman solvers."""

from __future__ import annotations

import numpy as np
import pytest

from rllab.theory import (
    FiniteMDP,
    bellman_epsilon_soft_optimality_operator,
    bellman_expectation_operator,
    bellman_optimality_operator,
    epsilon_soft_value_iteration,
    policy_evaluation,
    policy_iteration,
    value_iteration,
)


@pytest.fixture
def stop_or_wait_mdp() -> FiniteMDP:
    """State 0 can stop for reward one or wait; state 1 is terminal."""

    transitions = np.zeros((2, 2, 2), dtype=float)
    rewards = np.zeros_like(transitions)
    transitions[0, 0, 1] = 1.0
    rewards[0, 0, 1] = 1.0
    transitions[0, 1, 0] = 1.0
    transitions[1, :, 1] = 1.0
    return FiniteMDP(
        transitions,
        rewards,
        terminal=np.array([False, True]),
        state_labels=("decision", "done"),
    )


def test_finite_mdp_validates_shapes_probabilities_and_labels() -> None:
    transitions = np.ones((2, 1, 2), dtype=float) / 2.0
    rewards = np.zeros_like(transitions)
    mdp = FiniteMDP(transitions, rewards, [False, True], state_labels=("a", "b"))

    assert mdp.n_states == 2
    assert mdp.n_actions == 1
    assert mdp.state_index("b") == 1
    np.testing.assert_allclose(mdp.expected_rewards, 0.0)
    assert not mdp.P.flags.writeable

    with pytest.raises(ValueError, match="sum to one"):
        FiniteMDP(transitions * 0.5, rewards, [False, True])
    with pytest.raises(ValueError, match="R must have shape"):
        FiniteMDP(transitions, np.zeros((2, 1, 1)), [False, True])
    with pytest.raises(ValueError, match="unique"):
        FiniteMDP(transitions, rewards, [False, True], state_labels=("a", "a"))


def test_q_lookahead_masks_terminal_continuation_but_keeps_entry_reward(
    stop_or_wait_mdp: FiniteMDP,
) -> None:
    q_values = stop_or_wait_mdp.q_from_v(np.array([4.0, 100.0]), gamma=0.5)

    np.testing.assert_allclose(q_values[0], [1.0, 2.0])
    np.testing.assert_allclose(q_values[1], [0.0, 0.0])


def test_bellman_operators_are_exact_one_step_updates(stop_or_wait_mdp: FiniteMDP) -> None:
    values = np.array([0.4, 0.0])
    stochastic_policy = np.array([[0.25, 0.75], [0.0, 0.0]])

    expectation = bellman_expectation_operator(
        stop_or_wait_mdp, stochastic_policy, values, gamma=0.5
    )
    optimality = bellman_optimality_operator(stop_or_wait_mdp, values, gamma=0.5)

    # 0.25 * 1 + 0.75 * (0 + .5 * .4)
    np.testing.assert_allclose(expectation, [0.4, 0.0])
    np.testing.assert_allclose(optimality, [1.0, 0.0])


def test_epsilon_soft_bellman_operator_is_hand_checkable(
    stop_or_wait_mdp: FiniteMDP,
) -> None:
    values = np.array([0.4, 100.0])

    updated = bellman_epsilon_soft_optimality_operator(
        stop_or_wait_mdp,
        values,
        epsilon=0.2,
        gamma=0.5,
    )

    # Q(stop)=1 and Q(wait)=.2, so .8 * 1 + .2 * mean(1, .2) = .92.
    np.testing.assert_allclose(updated, [0.92, 0.0])


@pytest.mark.parametrize("method", ["iterative", "direct"])
def test_policy_evaluation_matches_closed_form(stop_or_wait_mdp: FiniteMDP, method: str) -> None:
    policy = np.array([[0.25, 0.75], [0.0, 0.0]])
    result = policy_evaluation(
        stop_or_wait_mdp,
        policy,
        gamma=0.5,
        tolerance=1e-12,
        method=method,  # type: ignore[arg-type]
    )
    expected = 0.25 / (1.0 - 0.75 * 0.5)

    assert result.converged
    assert result.residual <= 1e-12
    np.testing.assert_allclose(result.values, [expected, 0.0], atol=2e-12)
    np.testing.assert_allclose(result.q_values[0], [1.0, 0.5 * expected], atol=2e-12)


def test_value_iteration_converges_to_inspectable_solution(
    stop_or_wait_mdp: FiniteMDP,
) -> None:
    result = value_iteration(stop_or_wait_mdp, gamma=0.9, tolerance=1e-13)

    assert result.converged
    assert result.iterations == len(result.residual_history)
    np.testing.assert_allclose(result.values, [1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(result.q_values[0], [1.0, 0.9], atol=1e-12)
    np.testing.assert_array_equal(result.policy, [0, 0])


def test_epsilon_zero_exactly_matches_value_iteration(stop_or_wait_mdp: FiniteMDP) -> None:
    optimal = value_iteration(stop_or_wait_mdp, gamma=0.9, tolerance=1e-13)
    epsilon_soft = epsilon_soft_value_iteration(
        stop_or_wait_mdp,
        epsilon=0.0,
        gamma=0.9,
        tolerance=1e-13,
    )

    assert epsilon_soft.converged
    assert epsilon_soft.epsilon == 0.0
    assert epsilon_soft.iterations == optimal.iterations
    assert epsilon_soft.residual_history == optimal.residual_history
    np.testing.assert_array_equal(epsilon_soft.values, optimal.values)
    np.testing.assert_array_equal(epsilon_soft.q_values, optimal.q_values)
    np.testing.assert_array_equal(epsilon_soft.greedy_policy, optimal.policy)
    np.testing.assert_array_equal(epsilon_soft.policy[0], [1.0, 0.0])
    np.testing.assert_array_equal(epsilon_soft.policy[1], [0.0, 0.0])


def test_epsilon_soft_value_iteration_matches_closed_form(
    stop_or_wait_mdp: FiniteMDP,
) -> None:
    epsilon = 0.2
    gamma = 0.5
    result = epsilon_soft_value_iteration(
        stop_or_wait_mdp,
        epsilon=epsilon,
        gamma=gamma,
        tolerance=1e-13,
    )
    # The epsilon-soft policy chooses stop with .9 and wait with .1, yielding
    # V = .9 / (1 - .1 * gamma).
    expected_value = 0.9 / (1.0 - 0.1 * gamma)

    assert result.converged
    assert result.residual <= 1e-13
    np.testing.assert_allclose(result.values, [expected_value, 0.0], atol=2e-13)
    np.testing.assert_allclose(result.q_values[0], [1.0, gamma * expected_value], atol=2e-13)
    np.testing.assert_allclose(result.policy, [[0.9, 0.1], [0.0, 0.0]])
    np.testing.assert_array_equal(result.greedy_policy, [0, 0])


@pytest.mark.parametrize("epsilon", [-0.01, 1.01, np.nan, np.inf])
def test_epsilon_soft_solver_rejects_invalid_epsilon(
    stop_or_wait_mdp: FiniteMDP,
    epsilon: float,
) -> None:
    with pytest.raises(ValueError, match="epsilon must be finite and in \\[0, 1\\]"):
        epsilon_soft_value_iteration(stop_or_wait_mdp, epsilon=epsilon)


def test_policy_iteration_improves_a_deliberately_bad_policy(
    stop_or_wait_mdp: FiniteMDP,
) -> None:
    result = policy_iteration(
        stop_or_wait_mdp,
        gamma=0.9,
        tolerance=1e-13,
        initial_policy=np.array([1, 0]),
    )

    assert result.converged
    assert result.policy_changes[0] == 1
    assert result.policy_changes[-1] == 0
    np.testing.assert_array_equal(result.policy, [0, 0])
    np.testing.assert_allclose(result.values, [1.0, 0.0], atol=1e-12)


def test_discounted_continuing_mdp_has_geometric_value() -> None:
    mdp = FiniteMDP(
        P=np.ones((1, 1, 1)),
        R=np.full((1, 1, 1), 2.0),
        terminal=np.array([False]),
    )
    result = value_iteration(mdp, gamma=0.75, tolerance=1e-12)

    assert result.converged
    np.testing.assert_allclose(result.values, [8.0], atol=4e-12)


def test_solver_reports_nonconvergence_instead_of_hiding_it() -> None:
    mdp = FiniteMDP(np.ones((1, 1, 1)), np.ones((1, 1, 1)), [False])
    result = value_iteration(mdp, gamma=0.99, tolerance=1e-15, max_iterations=2)

    assert not result.converged
    assert result.iterations == 2
    assert len(result.residual_history) == 2
