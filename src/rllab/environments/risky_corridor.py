"""A fixed risky-shortcut versus protected-detour research environment.

The geometry is deliberately opinionated.  The central lane is the shortest
route from start to goal, but action slips can enter a hazard band on either
side.  A longer southern lane is protected by a wall and the grid boundary, so
the same actuator noise usually produces delay rather than hazard contact.

This named environment keeps the comparison reproducible in YAML experiments:
the maze layout is part of the environment implementation rather than notebook
setup code.  ``hazard_mode="lethal"`` terminates on contact, while
``hazard_mode="recoverable"`` charges a penalty and leaves the episode active.
Both variants expose an exact position-state MDP.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from rllab.environments.stochastic_maze import (
    Coordinate,
    Goal,
    Hazard,
    MazeAction,
    StochasticMazeEnv,
    WallEdge,
)

type HazardMode = Literal["recoverable", "lethal"]
type RouteLabel = Literal["corridor", "shelter", "other"]


class RiskyCorridorEnv(StochasticMazeEnv):
    """The canonical shortcut-or-shelter maze used by Notebook 05.

    ``corridor_reliability`` and ``action_reliability`` are aliases for one
    *global* intended-action reliability.  Noise is not privileged by route:
    the shelter is safer because its wall and the southern grid boundary turn
    lateral slips into blocked moves.  Supplying both aliases with different
    values is rejected.

    The four-row layout intentionally has no navigable lane above the northern
    hazard band.  Consequently there is no third hazard-free northern route.
    The only hazard-free progress choices are the exposed central corridor and
    the longer, wall-protected southern lane.
    """

    shape = (4, 9)
    fork_state: Coordinate = (1, 0)
    start_state: Coordinate = fork_state
    goal_state: Coordinate = (1, 8)
    corridor_action: MazeAction = MazeAction.EAST
    shelter_action: MazeAction = MazeAction.SOUTH
    corridor_states = frozenset((1, column) for column in range(1, 8))
    shelter_states = frozenset((3, column) for column in range(9))
    hazard_band = frozenset((row, column) for row in (0, 2) for column in range(2, 7))
    shelter_walls: frozenset[WallEdge] = frozenset(
        ((2, column), (3, column)) for column in range(1, 8)
    )

    def __init__(
        self,
        *,
        corridor_reliability: float | None = None,
        action_reliability: float | None = None,
        hazard_mode: HazardMode = "lethal",
        recoverable_hazard_penalty: float = -1.0,
        lethal_hazard_penalty: float = -8.0,
        goal_reward: float = 8.0,
        step_reward: float = -0.06,
        max_episode_steps: int = 250,
        render_mode: str | None = None,
    ) -> None:
        reliability = self._resolve_reliability(
            corridor_reliability=corridor_reliability,
            action_reliability=action_reliability,
        )
        normalized_mode = str(hazard_mode).lower().replace("-", "_")
        if normalized_mode not in {"recoverable", "lethal"}:
            raise ValueError(f"hazard_mode must be 'recoverable' or 'lethal', got {hazard_mode!r}")
        self.hazard_mode: HazardMode = normalized_mode  # type: ignore[assignment]
        self.recoverable_hazard_penalty = float(recoverable_hazard_penalty)
        self.lethal_hazard_penalty = float(lethal_hazard_penalty)
        if self.recoverable_hazard_penalty > 0.0 or self.lethal_hazard_penalty > 0.0:
            raise ValueError("hazard penalties must be nonpositive")
        self.hazard_penalty = (
            self.lethal_hazard_penalty
            if self.hazard_mode == "lethal"
            else self.recoverable_hazard_penalty
        )
        hazards = [
            Hazard(
                position,
                penalty=self.hazard_penalty,
                terminal=self.hazard_mode == "lethal",
                activation_probability=1.0,
                terminal_probability=1.0,
            )
            for position in sorted(self.hazard_band)
        ]
        super().__init__(
            shape=self.shape,
            start=self.start_state,
            goals=[Goal(self.goal_state, reward=float(goal_reward))],
            static_walls=self.shelter_walls,
            action_reliability=reliability,
            slip_weights={"left": 0.5, "right": 0.5},
            step_reward=float(step_reward),
            hazards=hazards,
            observation_mode="state",
            max_episode_steps=max_episode_steps,
            render_mode=render_mode,
        )

        self._route_intention: RouteLabel | None = None
        self._realized_route: RouteLabel | None = None
        self._hazard_penalty_steps = 0

    @staticmethod
    def _resolve_reliability(
        *,
        corridor_reliability: float | None,
        action_reliability: float | None,
    ) -> float:
        if corridor_reliability is None and action_reliability is None:
            return 1.0
        if corridor_reliability is None:
            assert action_reliability is not None
            return float(action_reliability)
        if action_reliability is None:
            return float(corridor_reliability)
        if float(corridor_reliability) != float(action_reliability):
            raise ValueError(
                "corridor_reliability and action_reliability are aliases and must match"
            )
        return float(action_reliability)

    @property
    def corridor_reliability(self) -> float:
        """Alias for the global actuator reliability used by notebook sweeps."""

        return self.action_reliability

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Reset the maze and its episode-level route diagnostics."""

        self._route_intention = None
        self._realized_route = None
        self._hazard_penalty_steps = 0
        observation, info = super().reset(seed=seed, options=options)
        return observation, {**info, **self._route_diagnostics(final=False)}

    def step(self, action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Advance the maze and retain route choice and hazard-contact summaries."""

        observation, reward, terminated, truncated, info = super().step(action)
        previous = info["previous_position"]
        position = info["latent_position"]

        if self._route_intention is None and previous == self.fork_state:
            self._route_intention = self._route_for_action(int(info["intended_action"]))
        if self._realized_route is None:
            if position in self.corridor_states:
                self._realized_route = "corridor"
            elif position in self.shelter_states:
                self._realized_route = "shelter"

        if position in self.hazard_band:
            self._hazard_penalty_steps += 1

        final = bool(terminated or truncated)
        if final:
            self._route_intention = self._route_intention or "other"
            self._realized_route = self._realized_route or "other"
        info.update(self._route_diagnostics(final=final))
        return observation, reward, terminated, truncated, info

    def episode_summary(self) -> dict[str, Any]:
        """Return scalar route, exposure, and outcome diagnostics for this episode."""

        return {
            **self._route_diagnostics(final=True),
            "hazard_penalty_steps": int(self._hazard_penalty_steps),
            # Backwards-compatible alias. In the recoverable environment an
            # agent can occupy a hazard for multiple penalty-bearing steps, so
            # this is not necessarily a count of distinct entries.
            "hazard_encounter_count": int(self._hazard_penalty_steps),
            "hazard_mode": self.hazard_mode,
            "action_reliability": float(self.action_reliability),
        }

    def _route_diagnostics(self, *, final: bool) -> dict[str, Any]:
        fallback: RouteLabel | None = "other" if final else None
        return {
            "route_intention": self._route_intention or fallback,
            "realized_route": self._realized_route or fallback,
            "hazard_penalty_steps": int(self._hazard_penalty_steps),
            "hazard_encounter_count": int(self._hazard_penalty_steps),
        }

    def _route_for_action(self, action: int) -> RouteLabel:
        if action == int(self.corridor_action):
            return "corridor"
        if action == int(self.shelter_action):
            return "shelter"
        return "other"


__all__ = ["HazardMode", "RiskyCorridorEnv", "RouteLabel"]
