from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from rllab.environments import (
    EventWall,
    ExactModelUnavailable,
    Goal,
    Hazard,
    IndependentWall,
    MarkovWall,
    MazeAction,
    MovingHazard,
    NonstationarityConfig,
    ParameterRandomization,
    ScheduledWall,
    StochasticMazeEnv,
    WallStateObservationWrapper,
)


def test_deterministic_motion_static_walls_and_terminal_reward() -> None:
    wall = ((1, 0), (1, 1))
    env = StochasticMazeEnv(
        shape=(2, 2),
        start=(1, 0),
        goals={(0, 1): 2.0},
        static_walls={wall},
        step_reward=-0.1,
    )
    observation, info = env.reset(seed=7)
    assert observation == env.state_to_index[(1, 0)]
    assert info["walls"] == (wall,)

    observation, reward, terminated, truncated, info = env.step(MazeAction.EAST)
    assert observation == env.state_to_index[(1, 0)]
    assert reward == pytest.approx(-0.1)
    assert info["movement_blocked"]
    assert not terminated and not truncated

    env.step(MazeAction.NORTH)
    observation, reward, terminated, truncated, info = env.step(MazeAction.EAST)
    assert observation == env.state_to_index[(0, 1)]
    assert reward == pytest.approx(1.9)
    assert terminated and not truncated
    assert info["success"] and not info["failure"]
    with pytest.raises(RuntimeError, match="after episode completion"):
        env.step(MazeAction.WEST)


def test_seeded_latent_process_is_reproducible_across_observation_modes() -> None:
    edge_a = ((1, 0), (1, 1))
    edge_b = ((0, 1), (1, 1))
    common = dict(
        shape=(3, 3),
        start=(1, 1),
        goals=[Goal((2, 2), reward=0.3, terminal_probability=0.0)],
        action_reliability=0.63,
        independent_walls=[IndependentWall(edge_a, 0.35)],
        markov_walls=[MarkovWall(edge_b, p01=0.2, p11=0.8)],
        reward_noise_std=0.4,
        rare_reward_probability=0.15,
        rare_reward=3.0,
        max_episode_steps=50,
    )
    state_env = StochasticMazeEnv(**common, observation_mode="state")
    noisy_env = StochasticMazeEnv(
        **common,
        observation_mode="local",
        observation_radius=2,
        wall_observation_noise=0.5,
    )
    state_env.reset(seed=184)
    noisy_env.reset(seed=184)

    actions = [MazeAction.NORTH, MazeAction.EAST, MazeAction.SOUTH, MazeAction.WEST] * 6
    for action in actions:
        _, reward_a, terminated_a, truncated_a, info_a = state_env.step(action)
        _, reward_b, terminated_b, truncated_b, info_b = noisy_env.step(action)
        assert reward_a == reward_b
        assert terminated_a == terminated_b
        assert truncated_a == truncated_b
        assert info_a["position"] == info_b["position"]
        assert info_a["walls"] == info_b["walls"]
        assert info_a["realized_action"] == info_b["realized_action"]


def test_state_action_reliability_has_exact_interpretable_kernel() -> None:
    center = (1, 1)
    env = StochasticMazeEnv(
        shape=(3, 3),
        start=center,
        goals=[Goal((2, 2), terminal_probability=0.0)],
        action_reliability=0.9,
        state_reliability={center: 0.7},
        state_action_reliability={(center, MazeAction.EAST): 0.5},
        slip_weights={"left": 0.2, "right": 0.3, "backward": 0.1, "stay": 0.4},
    )
    model = env.exact_mdp()
    row = model.P[env.state_to_index[center], MazeAction.EAST]
    expected = {
        (1, 2): 0.50,  # intended east
        (0, 1): 0.10,  # relative left
        (2, 1): 0.15,  # relative right
        (1, 0): 0.05,  # backward
        (1, 1): 0.20,  # stay
    }
    for state, probability in expected.items():
        assert row[env.state_to_index[state]] == pytest.approx(probability)
    assert row.sum() == pytest.approx(1.0)


def test_independent_wall_is_marginalized_in_stationary_model() -> None:
    edge = ((0, 0), (0, 1))
    env = StochasticMazeEnv(
        shape=(1, 2),
        start=(0, 0),
        goals={(0, 1): 4.0},
        independent_walls={edge: 0.3},
        action_reliability=1.0,
        step_reward=-1.0,
    )
    model = env.exact_mdp()
    source = env.state_to_index[(0, 0)]
    goal = env.state_to_index[(0, 1)]
    assert model.P[source, MazeAction.EAST, source] == pytest.approx(0.3)
    assert model.P[source, MazeAction.EAST, goal] == pytest.approx(0.7)
    assert model.R[source, MazeAction.EAST, source] == pytest.approx(-1.0)
    assert model.R[source, MazeAction.EAST, goal] == pytest.approx(3.0)
    assert model.terminal.tolist() == [False, True]
    np.testing.assert_allclose(model.P.sum(axis=2), 1.0)


def test_markov_wall_augmented_model_matches_bit_dynamics() -> None:
    edge = ((0, 0), (0, 1))
    env = StochasticMazeEnv(
        shape=(1, 2),
        start=(0, 0),
        goals={(0, 1): 1.0},
        markov_walls=[MarkovWall(edge, p01=0.25, p11=0.75, initial_present=False)],
    )
    with pytest.raises(ExactModelUnavailable, match="augment_walls=True"):
        env.exact_mdp()

    model = env.exact_mdp(augment_walls=True)
    assert model.P.shape == (4, 4, 4)
    assert env.augmented_wall_edges == (edge,)
    assert env.decode_augmented_state(0) == ((0, 0), (False,))
    assert env.decode_augmented_state(1) == ((0, 0), (True,))
    # Absent wall: cross, then next wall is absent/present with .75/.25.
    np.testing.assert_allclose(model.P[0, MazeAction.EAST], [0.0, 0.0, 0.75, 0.25])
    # Present wall: remain, then it disappears/remains with .25/.75.
    np.testing.assert_allclose(model.P[1, MazeAction.EAST], [0.25, 0.75, 0.0, 0.0])
    np.testing.assert_allclose(model.P.sum(axis=2), 1.0)


def test_markov_wall_runtime_uses_current_then_next_configuration() -> None:
    edge = ((0, 0), (0, 1))
    env = StochasticMazeEnv(
        shape=(1, 3),
        start=(0, 0),
        goals=[Goal((0, 2), terminal_probability=0.0)],
        markov_walls=[MarkovWall(edge, p01=1.0, p11=0.0, initial_present=False)],
    )
    _, reset_info = env.reset(seed=2)
    assert edge not in reset_info["walls"]
    _, _, _, _, first_info = env.step(MazeAction.EAST)
    assert first_info["position"] == (0, 1)
    assert edge in first_info["walls"]
    _, _, _, _, second_info = env.step(MazeAction.WEST)
    assert second_info["position"] == (0, 1)
    assert second_info["movement_blocked"]
    assert edge not in second_info["walls"]


def test_scheduled_and_event_walls_change_next_decision_topology() -> None:
    edge = ((0, 0), (0, 1))
    scheduled = StochasticMazeEnv(
        shape=(2, 2),
        start=(1, 0),
        goals=[Goal((0, 1), terminal_probability=0.0)],
        scheduled_walls=[ScheduledWall(edge, changes={1: True, 2: False})],
    )
    scheduled.reset(seed=1)
    _, _, _, _, info = scheduled.step(MazeAction.NORTH)
    assert edge in info["walls"]
    _, _, _, _, info = scheduled.step(MazeAction.EAST)
    assert info["position"] == (0, 0)
    assert info["movement_blocked"]
    assert edge not in info["walls"]
    _, _, _, _, info = scheduled.step(MazeAction.EAST)
    assert info["position"] == (0, 1)

    triggered = StochasticMazeEnv(
        shape=(2, 2),
        start=(1, 0),
        goals=[Goal((0, 1), terminal_probability=0.0)],
        event_walls=[EventWall(edge, trigger_states={(0, 0)})],
    )
    triggered.reset(seed=1)
    _, _, _, _, info = triggered.step(MazeAction.NORTH)
    assert edge in info["walls"]
    assert info["wall_events"][0]["mechanism"] == "event"
    _, _, _, _, info = triggered.step(MazeAction.EAST)
    assert info["position"] == (0, 0)


def test_nonstationary_abrupt_reliability_and_reward_regime() -> None:
    env = StochasticMazeEnv(
        shape=(1, 3),
        start=(0, 1),
        goals=[Goal((0, 2), reward=0.0, terminal_probability=0.0)],
        action_reliability=1.0,
        slip_weights={"stay": 1.0},
        step_reward=-1.0,
        nonstationarity=NonstationarityConfig(
            mode="abrupt",
            reliability_multipliers=(1.0, 0.0),
            reward_multipliers=(1.0, 2.0),
            change_step=1,
        ),
    )
    env.reset(seed=5)
    _, first_reward, _, _, first = env.step(MazeAction.EAST)
    assert first_reward == pytest.approx(-1.0)
    assert first["position"] == (0, 2)
    assert env.current_reliability == 0.0
    _, second_reward, _, _, second = env.step(MazeAction.WEST)
    assert second_reward == pytest.approx(-2.0)
    assert second["position"] == (0, 2)  # all failed-action mass is configured as stay
    assert second["action_reliability"] == 0.0


def test_expected_rewards_cover_rare_rewards_and_nonterminal_hazards() -> None:
    env = StochasticMazeEnv(
        shape=(1, 3),
        start=(0, 0),
        goals=[Goal((0, 2), terminal_probability=0.0)],
        hazards=[
            Hazard(
                (0, 1),
                penalty=-4.0,
                terminal=False,
                activation_probability=0.25,
            )
        ],
        step_reward=-1.0,
        rare_reward_probability=0.1,
        rare_reward=10.0,
        rare_rewards={(0, 1): (0.5, 2.0)},
    )
    model = env.exact_mdp()
    source = env.state_to_index[(0, 0)]
    hazard = env.state_to_index[(0, 1)]
    # -1 step -1 expected hazard +1 global rare +1 state rare = 0.
    assert model.R[source, MazeAction.EAST, hazard] == pytest.approx(0.0)
    assert not model.terminal[hazard]

    terminal_env = StochasticMazeEnv(
        shape=(1, 3),
        start=(0, 0),
        goals={(0, 2): 1.0},
        hazards=[Hazard((0, 1), penalty=-5.0, terminal=True)],
        step_reward=0.0,
    )
    terminal_env.reset(seed=1)
    _, reward, terminated, _, info = terminal_env.step(MazeAction.EAST)
    assert reward == pytest.approx(-5.0)
    assert terminated and info["failure"]


def test_moving_hazard_moves_deterministically_after_reward_evaluation() -> None:
    env = StochasticMazeEnv(
        shape=(2, 3),
        start=(1, 0),
        goals=[Goal((1, 2), terminal_probability=0.0)],
        moving_hazards=[
            MovingHazard(
                (0, 0),
                terminal=False,
                movement_probability=1.0,
                motion_weights=(0.0, 1.0, 0.0, 0.0, 0.0),
            )
        ],
    )
    _, reset_info = env.reset(seed=3)
    assert reset_info["hazard_positions"] == ((0, 0),)
    _, _, _, _, info = env.step(MazeAction.EAST)
    assert info["hazard_positions"] == ((0, 1),)


@pytest.mark.parametrize("mode", ["state", "noisy_state", "local", "full"])
def test_all_observation_modes_satisfy_gymnasium_contract(mode: str) -> None:
    env = StochasticMazeEnv(
        shape=(3, 3),
        goals={(2, 2): 1.0},
        static_walls={((0, 0), (0, 1))},
        hazards=[Hazard((1, 1), terminal=False)],
        observation_mode=mode,  # type: ignore[arg-type]
        state_observation_noise=0.25,
        wall_observation_noise=0.25,
        render_mode="rgb_array",
    )
    check_env(env, skip_render_check=True)
    observation, _ = env.reset(seed=13)
    assert env.observation_space.contains(observation)
    rendered = env.render()
    assert isinstance(rendered, np.ndarray)
    assert rendered.dtype == np.uint8
    assert rendered.ndim == 3 and rendered.shape[2] == 3


def test_noisy_state_and_wall_observations_are_explicit() -> None:
    edge = ((0, 0), (0, 1))
    noisy_state = StochasticMazeEnv(
        shape=(1, 2),
        start=(0, 0),
        goals=[Goal((0, 1), terminal_probability=0.0)],
        observation_mode="noisy_state",
        state_observation_noise=1.0,
    )
    observation, info = noisy_state.reset(seed=9)
    assert observation != info["latent_state_index"]
    assert info["state_index"] == info["latent_state_index"]  # compatibility alias
    assert info["position"] == info["latent_position"]
    assert noisy_state.unwrapped_state_index == noisy_state.latent_state_index

    full = StochasticMazeEnv(
        shape=(1, 2),
        goals={(0, 1): 1.0},
        static_walls={edge},
        observation_mode="full",
        wall_observation_noise=1.0,
    )
    observation, info = full.reset(seed=9)
    assert edge in info["latent_walls"]
    assert info["walls"] == info["latent_walls"]  # compatibility alias
    assert observation["walls"].tolist() == [0]


@pytest.mark.parametrize(
    ("mode", "state_noise", "message"),
    [
        ("noisy_state", 0.4, "partially observed"),
        ("local", 0.0, "does not use the integer position-state index"),
        ("full", 0.0, "does not use the integer position-state index"),
    ],
)
def test_exact_model_rejects_observation_incompatible_state_indexing(
    mode: str,
    state_noise: float,
    message: str,
) -> None:
    env = StochasticMazeEnv(
        shape=(1, 2),
        goals={(0, 1): 1.0},
        observation_mode=mode,  # type: ignore[arg-type]
        state_observation_noise=state_noise,
    )
    assert env.observation_state_index_identity is None
    with pytest.raises(ExactModelUnavailable, match=message):
        env.exact_mdp()


def test_zero_noise_integer_observation_retains_exact_index_identity() -> None:
    env = StochasticMazeEnv(
        shape=(1, 2),
        goals={(0, 1): 1.0},
        observation_mode="noisy_state",
        state_observation_noise=0.0,
    )
    assert env.observation_state_index_identity == env.index_to_state
    assert env.exact_mdp().state_labels == env.index_to_state


def test_start_and_parameter_randomization_are_seed_deterministic() -> None:
    env = StochasticMazeEnv(
        shape=(2, 2),
        start_distribution={(0, 0): 0.2, (1, 0): 0.8},
        goals={(1, 1): 1.0},
        parameter_randomization=ParameterRandomization(
            action_reliability=(0.4, 0.9), reward_scale=(0.5, 1.5)
        ),
    )
    _, first = env.reset(seed=911)
    _, second = env.reset(seed=911)
    assert first["position"] == second["position"]
    assert first["episode_action_reliability"] == second["episode_action_reliability"]
    assert first["reward_multiplier"] == second["reward_multiplier"]
    assert 0.4 <= first["episode_action_reliability"] <= 0.9


def test_episode_limit_truncates_and_exact_limitations_are_loud() -> None:
    truncated_env = StochasticMazeEnv(
        shape=(1, 2),
        start=(0, 0),
        goals=[Goal((0, 1), terminal_probability=0.0)],
        max_episode_steps=1,
    )
    truncated_env.reset(seed=0)
    _, _, terminated, truncated, _ = truncated_env.step(MazeAction.WEST)
    assert not terminated and truncated

    probabilistic_terminal = StochasticMazeEnv(
        shape=(1, 2), goals=[Goal((0, 1), terminal_probability=0.5)]
    )
    with pytest.raises(ExactModelUnavailable, match="probabilistic goal termination"):
        probabilistic_terminal.exact_mdp()

    scheduled = StochasticMazeEnv(
        shape=(1, 2),
        goals={(0, 1): 1.0},
        scheduled_walls=[ScheduledWall(((0, 0), (0, 1)), {2: True})],
    )
    with pytest.raises(ExactModelUnavailable, match="scheduled walls"):
        scheduled.exact_mdp()

    two_walls = StochasticMazeEnv(
        shape=(2, 2),
        goals={(1, 1): 1.0},
        markov_walls=[
            MarkovWall(((0, 0), (0, 1)), 0.1, 0.9),
            MarkovWall(((0, 0), (1, 0)), 0.1, 0.9),
        ],
    )
    with pytest.raises(ExactModelUnavailable, match="max_augmented_states"):
        two_walls.exact_mdp(augment_walls=True, max_augmented_states=8)


def test_wall_state_wrapper_matches_augmented_model_identity_and_timing() -> None:
    edge = ((0, 1), (0, 2))
    base = StochasticMazeEnv(
        shape=(2, 3),
        start=(0, 0),
        goals={(1, 2): 1.0},
        markov_walls=[MarkovWall(edge, p01=0.0, p11=1.0, initial_present=True)],
    )
    env = WallStateObservationWrapper(base)
    observation, _ = env.reset(seed=4)
    assert observation == base.state_to_index[(0, 0)] * 2 + 1
    assert env.n_states == base.n_states * 2

    next_observation, _, _, _, info = env.step(MazeAction.EAST)
    assert info["decision_wall_mask"] == 1
    assert info["decision_walls"] == (edge,)
    assert next_observation == base.latent_state_index * 2 + 1

    model = env.exact_mdp()
    assert model.n_states == env.n_states
    assert tuple(model.state_labels) == env.observation_state_index_identity
