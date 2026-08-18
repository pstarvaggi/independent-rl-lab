"""Exact TD updates, exploration behavior, reset semantics, and reproducibility."""

from __future__ import annotations

import numpy as np
import pytest

from rllab.agents import (
    UCB,
    Boltzmann,
    DoubleQLearningAgent,
    EpsilonGreedy,
    ExpectedSarsaAgent,
    ExponentialDecaySchedule,
    LinearDecaySchedule,
    PlannerAgent,
    QLearningAgent,
    RandomAgent,
    SarsaAgent,
)
from rllab.theory import FiniteMDP


def test_q_learning_update_matches_bellman_equation() -> None:
    agent = QLearningAgent(2, 2, alpha=0.5, gamma=0.9, epsilon=0.0, seed=4)
    agent.q_values[0, 0] = 1.0
    agent.q_values[1] = [2.0, 4.0]

    record = agent.update(0, 0, reward=3.0, next_state=1, terminated=False)

    assert record.target == pytest.approx(3.0 + 0.9 * 4.0)
    assert record.td_error == pytest.approx(5.6)
    assert record.alpha == pytest.approx(0.5)
    assert record.epsilon == pytest.approx(0.0)
    assert record.old_value == pytest.approx(1.0)
    assert record.new_value == pytest.approx(3.8)
    assert record.next_action == 1
    assert record.absolute_td_error == pytest.approx(5.6)
    assert record.squared_td_error == pytest.approx(5.6**2)
    assert agent.q_values[0, 0] == pytest.approx(3.8)
    assert agent.update_counts[0, 0] == 1


def test_terminal_q_learning_target_has_no_bootstrap() -> None:
    agent = QLearningAgent(2, 2, alpha=1.0, gamma=0.99, epsilon=0.0)
    agent.q_values[1] = [1_000.0, 2_000.0]

    record = agent.update(0, 1, reward=-7.0, next_state=1, terminated=True)

    assert record.target == -7.0
    assert record.td_error == -7.0
    assert agent.q_values[0, 1] == -7.0


def test_sarsa_uses_and_reuses_sampled_next_action() -> None:
    agent = SarsaAgent(2, 2, alpha=0.25, gamma=0.5, epsilon=0.0, seed=8)
    agent.q_values[1] = [2.0, 5.0]

    record = agent.update(0, 0, reward=1.0, next_state=1, terminated=False)

    assert record.next_action == 1
    assert record.target == pytest.approx(3.5)
    assert record.td_error == pytest.approx(3.5)
    assert agent.q_values[0, 0] == pytest.approx(0.875)
    assert agent.act(1, training=True) == 1
    assert agent.action_counts[1, 1] == 1


def test_sarsa_discards_cached_action_at_episode_boundary() -> None:
    agent = SarsaAgent(2, 2, alpha=0.25, gamma=0.5, epsilon=0.0, seed=8)
    agent.q_values[1] = [0.0, 5.0]
    record = agent.update(0, 0, reward=0.0, next_state=1, terminated=False)
    assert record.next_action == 1

    agent.end_episode()
    agent.q_values[1] = [5.0, 0.0]

    assert agent.act(1, training=True) == 0


def test_expected_sarsa_uses_full_epsilon_greedy_distribution() -> None:
    agent = ExpectedSarsaAgent(2, 2, alpha=1.0, gamma=0.5, epsilon=0.2)
    agent.q_values[1] = [2.0, 4.0]

    record = agent.update(0, 0, reward=1.0, next_state=1, terminated=False)

    # epsilon-greedy probabilities are [.1, .9], hence E[Q]=3.8.
    assert record.target == pytest.approx(1.0 + 0.5 * 3.8)
    assert agent.q_values[0, 0] == pytest.approx(2.9)


def test_double_q_learning_cross_evaluates_the_selected_action() -> None:
    seed = 17
    update_first = bool(np.random.default_rng(seed).integers(2))
    agent = DoubleQLearningAgent(2, 2, alpha=0.5, gamma=0.5, epsilon=0.0, seed=seed)
    agent.q1_values[1] = [5.0, 1.0]
    agent.q2_values[1] = [2.0, 4.0]

    record = agent.update(0, 0, reward=1.0, next_state=1, terminated=False)

    if update_first:
        # argmax Q1 is action 0, evaluated by Q2.
        assert record.next_action == 0
        assert record.target == pytest.approx(2.0)
        assert agent.q1_values[0, 0] == pytest.approx(1.0)
        assert agent.q2_values[0, 0] == 0.0
    else:
        # argmax Q2 is action 1, evaluated by Q1.
        assert record.next_action == 1
        assert record.target == pytest.approx(1.5)
        assert agent.q2_values[0, 0] == pytest.approx(0.75)
        assert agent.q1_values[0, 0] == 0.0
    assert agent.q_values[0, 0] == pytest.approx(record.new_value / 2.0)


def test_schedules_have_explicit_endpoint_semantics() -> None:
    linear = LinearDecaySchedule(start=1.0, end=0.1, duration=10)
    exponential = ExponentialDecaySchedule(initial=1.0, decay_rate=0.5, minimum=0.2)

    assert linear(0) == 1.0
    assert linear(5) == pytest.approx(0.55)
    assert linear(10) == pytest.approx(0.1)
    assert linear(100) == pytest.approx(0.1)
    assert exponential(0) == 1.0
    assert exponential(2) == 0.25
    assert exponential(3) == 0.2


def test_exploration_distributions_are_normalized_and_inspectable() -> None:
    q_values = np.array([0.0, 2.0, 1.0])
    counts = np.array([5, 5, 5])

    epsilon_probabilities = EpsilonGreedy(0.3).probabilities(q_values, counts, step=0)
    np.testing.assert_allclose(epsilon_probabilities, [0.1, 0.8, 0.1])

    softmax_probabilities = Boltzmann(temperature=0.5).probabilities(q_values, counts, 0)
    assert softmax_probabilities.sum() == pytest.approx(1.0)
    assert np.argmax(softmax_probabilities) == 1

    ucb_probabilities = UCB(coefficient=2.0).probabilities(q_values, [0, 3, 0], 10)
    np.testing.assert_allclose(ucb_probabilities, [0.5, 0.0, 0.5])


def test_random_agent_and_epsilon_agent_are_reproducible_after_reset() -> None:
    random_agent = RandomAgent(3, 4, seed=123)
    first_random = [random_agent.act(state=1) for _ in range(20)]
    random_agent.reset(123)
    second_random = [random_agent.act(state=1) for _ in range(20)]
    assert first_random == second_random

    learner = QLearningAgent(3, 4, epsilon=1.0, seed=321)
    first_actions = [learner.act(state=2) for _ in range(20)]
    learner.q_values[0, 0] = 99.0
    learner.reset(321)
    second_actions = [learner.act(state=2) for _ in range(20)]
    assert first_actions == second_actions
    np.testing.assert_allclose(learner.q_values, 0.0)
    np.testing.assert_array_equal(learner.action_counts[2].sum(), 20)


def test_learning_rate_schedule_is_indexed_by_state_action_visits() -> None:
    agent = QLearningAgent(
        2,
        1,
        learning_rate=LinearDecaySchedule(1.0, 0.0, duration=2),
        gamma=0.0,
        epsilon=0.0,
    )

    first = agent.update(0, 0, 1.0, 1, True)
    agent.update(1, 0, 1.0, 0, True)
    second_for_pair = agent.update(0, 0, 3.0, 1, True)

    assert first.alpha == 1.0
    assert second_for_pair.alpha == 0.5
    assert agent.q_values[0, 0] == pytest.approx(2.0)


def test_planner_agent_exposes_exact_q_and_standard_noop_update() -> None:
    transitions = np.zeros((2, 2, 2), dtype=float)
    rewards = np.zeros_like(transitions)
    transitions[0, :, 1] = 1.0
    rewards[0, 1, 1] = 2.0
    transitions[1, :, 1] = 1.0
    mdp = FiniteMDP(transitions, rewards, [False, True])
    agent = PlannerAgent(mdp, gamma=0.9)

    assert agent.act(0, training=False) == 1
    np.testing.assert_allclose(agent.q_values[0], [0.0, 2.0])
    record = agent.update(0, 1, 2.0, 1, True)
    assert record.td_error is None
    assert record.old_value == record.new_value == 2.0


def test_agent_rejects_out_of_range_integer_state() -> None:
    agent = QLearningAgent(2, 2)
    with pytest.raises(IndexError, match="state"):
        agent.act(2)
    with pytest.raises(IndexError, match="action"):
        agent.update(0, 2, 0.0, 1, False)
