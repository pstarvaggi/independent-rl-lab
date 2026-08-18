from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from rllab.environments import MazeAction, RiskyCorridorEnv
from rllab.experiments import make_environment
from rllab.theory import value_iteration


def test_geometry_has_one_exposed_route_and_one_protected_route() -> None:
    env = RiskyCorridorEnv()

    assert env.shape == (4, 9)
    assert env.fork_state == (1, 0)
    assert env.goal_state == (1, 8)
    assert env.corridor_action == MazeAction.EAST
    assert env.shelter_action == MazeAction.SOUTH
    assert env.hazard_positions == frozenset(
        (row, column) for row in (0, 2) for column in range(2, 7)
    )
    assert env.static_walls == frozenset(((2, column), (3, column)) for column in range(1, 8))

    # The northern row cannot form a third hazard-free lane: every eastward
    # passage from its two-cell start-side spur to its goal-side spur is a hazard.
    assert all((0, column) in env.hazard_positions for column in range(2, 7))
    # The shelter lane has no hazards and every lateral north slip is walled off.
    assert env.shelter_states.isdisjoint(env.hazard_positions)
    assert all(((2, column), (3, column)) in env.static_walls for column in range(1, 8))


@pytest.mark.parametrize("hazard_mode", ["recoverable", "lethal"])
def test_exact_model_matches_observation_identity_and_hazard_semantics(
    hazard_mode: str,
) -> None:
    env = RiskyCorridorEnv(
        corridor_reliability=1.0,
        hazard_mode=hazard_mode,  # type: ignore[arg-type]
        recoverable_hazard_penalty=-2.5,
        lethal_hazard_penalty=-8.0,
    )
    model = env.exact_mdp()

    assert env.observation_state_index_identity == env.index_to_state
    assert tuple(model.state_labels) == env.index_to_state
    hazard_index = env.state_to_index[(0, 2)]
    assert bool(model.terminal[hazard_index]) is (hazard_mode == "lethal")

    source = env.state_to_index[(1, 2)]
    expected_penalty = -8.0 if hazard_mode == "lethal" else -2.5
    assert model.P[source, MazeAction.NORTH, hazard_index] == pytest.approx(1.0)
    assert model.R[source, MazeAction.NORTH, hazard_index] == pytest.approx(
        -0.06 + expected_penalty
    )
    np.testing.assert_allclose(model.P.sum(axis=2), 1.0)


def test_yaml_friendly_registered_kind_and_reliability_aliases() -> None:
    env = make_environment(
        {
            "name": "recoverable-shortcut",
            "kind": "risky_corridor",
            "parameters": {
                "corridor_reliability": 0.83,
                "hazard_mode": "recoverable",
                "recoverable_hazard_penalty": -3.0,
            },
        }
    )

    assert isinstance(env, RiskyCorridorEnv)
    assert env.action_reliability == pytest.approx(0.83)
    assert env.corridor_reliability == pytest.approx(0.83)
    assert env.hazard_mode == "recoverable"
    assert env.hazard_penalty == pytest.approx(-3.0)

    alias = RiskyCorridorEnv(action_reliability=0.72)
    assert alias.corridor_reliability == pytest.approx(0.72)
    with pytest.raises(ValueError, match="aliases and must match"):
        RiskyCorridorEnv(corridor_reliability=0.8, action_reliability=0.7)


def test_route_and_hazard_episode_diagnostics() -> None:
    corridor = RiskyCorridorEnv(action_reliability=1.0, hazard_mode="recoverable")
    corridor.reset(seed=1)
    _, _, _, _, info = corridor.step(MazeAction.EAST)
    assert info["route_intention"] == "corridor"
    assert info["realized_route"] == "corridor"
    assert corridor.episode_summary()["hazard_penalty_steps"] == 0

    shelter = RiskyCorridorEnv(action_reliability=1.0, hazard_mode="recoverable")
    shelter.reset(seed=1)
    shelter.step(MazeAction.SOUTH)
    _, _, _, _, info = shelter.step(MazeAction.SOUTH)
    assert info["route_intention"] == "shelter"
    assert info["realized_route"] == "shelter"

    hazard = RiskyCorridorEnv(action_reliability=1.0, hazard_mode="lethal")
    hazard.reset(seed=1)
    hazard.step(MazeAction.EAST)
    hazard.step(MazeAction.EAST)
    _, reward, terminated, truncated, info = hazard.step(MazeAction.NORTH)
    assert reward == pytest.approx(-8.06)
    assert terminated and not truncated
    assert info["failure"]
    assert info["hazard_penalty_steps"] == 1
    assert hazard.episode_summary() == {
        "route_intention": "corridor",
        "realized_route": "corridor",
        "hazard_penalty_steps": 1,
        "hazard_encounter_count": 1,
        "hazard_mode": "lethal",
        "action_reliability": 1.0,
    }


@pytest.mark.parametrize(
    ("hazard_mode", "reliability", "expected_action"),
    [
        ("recoverable", 0.85, MazeAction.SOUTH),
        ("recoverable", 0.90, MazeAction.EAST),
        ("lethal", 0.985, MazeAction.SOUTH),
        ("lethal", 0.995, MazeAction.EAST),
    ],
)
def test_exact_policy_brackets_the_two_intended_route_boundaries(
    hazard_mode: str,
    reliability: float,
    expected_action: MazeAction,
) -> None:
    env = RiskyCorridorEnv(
        action_reliability=reliability,
        hazard_mode=hazard_mode,  # type: ignore[arg-type]
    )
    solution = value_iteration(env.exact_mdp(), gamma=0.98, tolerance=1e-12)
    fork_index = env.state_to_index[env.fork_state]

    assert solution.converged
    assert solution.policy[fork_index] == expected_action


def test_environment_satisfies_gymnasium_contract() -> None:
    check_env(RiskyCorridorEnv())
