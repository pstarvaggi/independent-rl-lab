"""A transparent, configurable stochastic grid-maze research environment.

The environment deliberately keeps the generative mechanisms visible.  Agent
motion, wall evolution, rewards, hazards, and observations each draw from an
independent random stream derived from Gymnasium's reset seed.  Consequently an
experiment does not acquire different dynamics merely by changing how much
observation noise is requested.

Timing convention
-----------------
At decision time ``t`` the action is realized using the reliability and wall
configuration reported by the previous observation.  The agent then moves and
receives a reward.  Event-triggered changes are applied, time becomes ``t + 1``,
independent and Markov walls evolve, scheduled changes for ``t + 1`` are
applied, and moving hazards move.  The returned observation describes this next
decision-time configuration.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces

type Coordinate = tuple[int, int]
type WallEdge = tuple[Coordinate, Coordinate]
type ObservationMode = Literal["state", "full", "local", "noisy_state"]


class MazeAction(IntEnum):
    """Cardinal actions in clockwise order."""

    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3


ACTION_DELTAS: dict[int, Coordinate] = {
    MazeAction.NORTH: (-1, 0),
    MazeAction.EAST: (0, 1),
    MazeAction.SOUTH: (1, 0),
    MazeAction.WEST: (0, -1),
}


class ExactModelUnavailable(ValueError):
    """Raised when the requested finite model would be incorrect or intractable."""


@dataclass(frozen=True, slots=True)
class Goal:
    """A goal reward and its termination law.

    ``reward_std`` adds zero-mean Gaussian noise whenever the goal is entered.
    Values strictly between zero and one for ``terminal_probability`` are valid
    in simulation, but require terminal-state augmentation and are therefore
    rejected by the current exact-model builder.
    """

    position: Coordinate
    reward: float = 1.0
    reward_std: float = 0.0
    terminal_probability: float = 1.0


@dataclass(frozen=True, slots=True)
class Hazard:
    """A fixed hazard.

    A hazard first activates with ``activation_probability``.  If active its
    penalty is paid.  A terminal hazard then ends the episode with
    ``terminal_probability``.
    """

    position: Coordinate
    penalty: float = -1.0
    terminal: bool = True
    activation_probability: float = 1.0
    terminal_probability: float = 1.0


@dataclass(frozen=True, slots=True)
class MovingHazard(Hazard):
    """A hazard which moves between decisions.

    ``motion_weights`` correspond to north, east, south, west, and stay.  Invalid
    moves are converted to stay.  Motion occurs after the transition reward is
    evaluated, so a moving hazard affects the agent on the following step.
    """

    movement_probability: float = 1.0
    motion_weights: tuple[float, float, float, float, float] = (0.2, 0.2, 0.2, 0.2, 0.2)


@dataclass(frozen=True, slots=True)
class IndependentWall:
    """A wall whose next presence bit is an independent Bernoulli draw."""

    edge: WallEdge
    presence_probability: float


@dataclass(frozen=True, slots=True)
class MarkovWall:
    """A two-state Markov wall.

    ``p11`` is the probability that a present wall remains present; ``p01`` is
    the probability that an absent wall appears.  ``initial_probability`` is
    used at reset unless ``initial_present`` is supplied.
    """

    edge: WallEdge
    p01: float
    p11: float
    initial_probability: float = 0.5
    initial_present: bool | None = None


@dataclass(frozen=True, slots=True)
class ScheduledWall:
    """A wall with deterministic changes at specified decision times."""

    edge: WallEdge
    changes: Mapping[int, bool] | Sequence[tuple[int, bool]]
    initial_present: bool = False


@dataclass(frozen=True, slots=True)
class EventWall:
    """A wall changed when the agent enters any trigger state."""

    edge: WallEdge
    trigger_states: frozenset[Coordinate] | set[Coordinate] | Sequence[Coordinate]
    present_after_trigger: bool = True
    initial_present: bool = False
    once: bool = True


type NonstationarityMode = Literal["stationary", "gradual", "abrupt", "periodic", "random"]


@dataclass(frozen=True, slots=True)
class NonstationarityConfig:
    """Time variation in action reliability and reward scale.

    Multipliers are applied to the spatial reliability and all deterministic
    reward means (noise remains zero mean).  ``gradual`` interpolates between
    the first and last values over ``horizon``.  ``abrupt`` changes from the
    first to the last at ``change_step``.  ``periodic`` advances one regime
    every ``period`` steps.  ``random`` changes regime after each transition
    with ``switch_probability``.
    """

    mode: NonstationarityMode = "stationary"
    reliability_multipliers: tuple[float, ...] = (1.0,)
    reward_multipliers: tuple[float, ...] = (1.0,)
    horizon: int = 100
    change_step: int = 50
    period: int = 25
    switch_probability: float = 0.01


@dataclass(frozen=True, slots=True)
class ParameterRandomization:
    """Per-episode parameter randomization ranges, sampled uniformly at reset."""

    action_reliability: tuple[float, float] | None = None
    reward_scale: tuple[float, float] | None = None


@dataclass(frozen=True, slots=True)
class _FallbackFiniteMDP:
    """Minimal fallback matching :class:`rllab.theory.mdp.FiniteMDP`."""

    P: np.ndarray
    R: np.ndarray
    terminal: np.ndarray
    state_labels: Sequence[Any] | None = None

    @property
    def n_states(self) -> int:
        return int(self.P.shape[0])

    @property
    def n_actions(self) -> int:
        return int(self.P.shape[1])


def canonical_edge(edge: WallEdge) -> WallEdge:
    """Return a deterministic representation of an undirected adjacent edge."""

    if len(edge) != 2:
        raise ValueError(f"A wall edge must have two endpoints, got {edge!r}")
    first = tuple(edge[0])
    second = tuple(edge[1])
    if len(first) != 2 or len(second) != 2:
        raise ValueError(f"Wall endpoints must be grid coordinates, got {edge!r}")
    a = (int(first[0]), int(first[1]))
    b = (int(second[0]), int(second[1]))
    if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
        raise ValueError(f"Wall endpoints must be cardinally adjacent, got {edge!r}")
    return (a, b) if a <= b else (b, a)


def _probability(value: float, name: str) -> float:
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1], got {result}")
    return result


def _finite_mdp_type() -> type[Any]:
    try:
        from rllab.theory.mdp import FiniteMDP
    except ImportError:  # pragma: no cover - only during partial source-tree use
        return _FallbackFiniteMDP
    return FiniteMDP


class StochasticMazeEnv(gym.Env[Any, int]):
    """Gymnasium environment for finite stochastic-maze experiments.

    Parameters use row-column coordinates.  Walls are undirected edges between
    adjacent, non-blocked cells; grid boundaries and ``blocked_cells`` are
    implicit walls.  The default ``"state"`` observation is a compact row-major
    integer index over non-blocked cells and is suited to tabular agents.
    """

    metadata: dict[str, Any] = {  # noqa: RUF012 - Gymnasium defines this on Env
        "render_modes": ["ansi", "human", "rgb_array"],
        "render_fps": 4,
    }

    def __init__(
        self,
        *,
        shape: tuple[int, int] = (5, 5),
        start: Coordinate | None = (0, 0),
        start_distribution: Mapping[Coordinate, float] | Sequence[Coordinate] | None = None,
        goals: Mapping[Coordinate, float | Goal] | Sequence[Goal] | None = None,
        blocked_cells: Iterable[Coordinate] = (),
        static_walls: Iterable[WallEdge] = (),
        walls: Iterable[WallEdge] | None = None,
        action_reliability: float = 1.0,
        state_reliability: Mapping[Coordinate, float] | None = None,
        state_action_reliability: Mapping[tuple[Coordinate, int], float] | None = None,
        slip_weights: Mapping[str, float] | None = None,
        independent_walls: Mapping[WallEdge, float] | Sequence[IndependentWall] | None = None,
        markov_walls: Mapping[WallEdge, tuple[float, float]] | Sequence[MarkovWall] | None = None,
        scheduled_walls: Mapping[Any, Any] | Sequence[ScheduledWall] | None = None,
        event_walls: Sequence[EventWall] | None = None,
        nonstationarity: NonstationarityConfig | None = None,
        step_reward: float = -0.01,
        reward_noise_std: float = 0.0,
        state_reward_noise_std: Mapping[Coordinate, float] | None = None,
        rare_reward_probability: float = 0.0,
        rare_reward: float = 0.0,
        rare_rewards: Mapping[Coordinate, tuple[float, float]] | None = None,
        hazards: Sequence[Hazard] | Mapping[Coordinate, float] | None = None,
        moving_hazards: Sequence[MovingHazard] | None = None,
        observation_mode: ObservationMode = "state",
        observation_radius: int = 1,
        wall_observation_noise: float = 0.0,
        state_observation_noise: float = 0.0,
        max_episode_steps: int = 200,
        parameter_randomization: ParameterRandomization | None = None,
        reliability_range: tuple[float, float] | None = None,
        randomize_initial_walls: bool = True,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if len(shape) != 2 or int(shape[0]) <= 0 or int(shape[1]) <= 0:
            raise ValueError(f"shape must contain two positive integers, got {shape!r}")
        self.shape = (int(shape[0]), int(shape[1]))
        self.rows, self.cols = self.shape
        self.render_mode = render_mode
        if render_mode not in {None, *self.metadata["render_modes"]}:
            raise ValueError(f"Unsupported render_mode {render_mode!r}")

        self.blocked_cells = frozenset(
            self._coordinate(cell, "blocked cell") for cell in blocked_cells
        )
        self.index_to_state = tuple(
            (row, col)
            for row in range(self.rows)
            for col in range(self.cols)
            if (row, col) not in self.blocked_cells
        )
        if not self.index_to_state:
            raise ValueError("blocked_cells cannot cover the entire grid")
        self.state_to_index = {state: index for index, state in enumerate(self.index_to_state)}
        self.n_states = len(self.index_to_state)
        self.n_actions = 4

        if walls is not None:
            if tuple(static_walls):
                raise ValueError("Use either walls or static_walls, not both")
            static_walls = walls
        self.static_walls = frozenset(self._wall(edge) for edge in static_walls)
        self._all_edges = tuple(
            canonical_edge((cell, neighbor))
            for cell in self.index_to_state
            for neighbor in self._cardinal_neighbors(cell)
            if cell < neighbor
        )
        self._edge_to_index = {edge: index for index, edge in enumerate(self._all_edges)}

        self.goals = self._normalize_goals(goals)
        self._start_states, self._start_probabilities = self._normalize_starts(
            start, start_distribution
        )
        self.action_reliability = _probability(action_reliability, "action_reliability")
        self.state_reliability = {
            self._coordinate(state, "state_reliability key"): _probability(value, "reliability")
            for state, value in (state_reliability or {}).items()
        }
        self.state_action_reliability: dict[tuple[Coordinate, int], float] = {}
        for (state, action), value in (state_action_reliability or {}).items():
            coordinate = self._coordinate(state, "state_action_reliability key")
            action_int = int(action)
            if action_int not in ACTION_DELTAS:
                raise ValueError(f"Invalid action {action!r} in state_action_reliability")
            self.state_action_reliability[(coordinate, action_int)] = _probability(
                value, "reliability"
            )
        self.slip_weights = self._normalize_slip_weights(slip_weights)

        self.independent_walls = self._normalize_independent_walls(independent_walls)
        self.markov_walls = self._normalize_markov_walls(markov_walls)
        self.scheduled_walls = self._normalize_scheduled_walls(scheduled_walls)
        self.event_walls = self._normalize_event_walls(event_walls)
        self._validate_wall_mechanisms()

        self.nonstationarity = nonstationarity or NonstationarityConfig()
        self._validate_nonstationarity(self.nonstationarity)
        self.step_reward = float(step_reward)
        if reward_noise_std < 0:
            raise ValueError("reward_noise_std must be nonnegative")
        self.reward_noise_std = float(reward_noise_std)
        self.state_reward_noise_std = {
            self._coordinate(state, "state_reward_noise_std key"): float(value)
            for state, value in (state_reward_noise_std or {}).items()
        }
        if any(value < 0 for value in self.state_reward_noise_std.values()):
            raise ValueError("state-dependent reward standard deviations must be nonnegative")
        self.rare_reward_probability = _probability(
            rare_reward_probability, "rare_reward_probability"
        )
        self.rare_reward = float(rare_reward)
        self.rare_rewards = {
            self._coordinate(state, "rare_rewards key"): (
                _probability(spec[0], "rare reward probability"),
                float(spec[1]),
            )
            for state, spec in (rare_rewards or {}).items()
        }
        self.hazards = self._normalize_hazards(hazards)
        self.moving_hazards = self._normalize_moving_hazards(moving_hazards)

        if observation_mode not in {"state", "full", "local", "noisy_state"}:
            raise ValueError(f"Unknown observation_mode {observation_mode!r}")
        self.observation_mode = observation_mode
        if int(observation_radius) < 0:
            raise ValueError("observation_radius must be nonnegative")
        self.observation_radius = int(observation_radius)
        self.wall_observation_noise = _probability(wall_observation_noise, "wall_observation_noise")
        self.state_observation_noise = _probability(
            state_observation_noise, "state_observation_noise"
        )
        if int(max_episode_steps) <= 0:
            raise ValueError("max_episode_steps must be positive")
        self.max_episode_steps = int(max_episode_steps)

        if (
            reliability_range is not None
            and parameter_randomization is not None
            and parameter_randomization.action_reliability is not None
        ):
            raise ValueError("Specify action-reliability randomization through only one argument")
        self.parameter_randomization = parameter_randomization or ParameterRandomization(
            action_reliability=reliability_range
        )
        self._validate_randomization(self.parameter_randomization)
        self.randomize_initial_walls = bool(randomize_initial_walls)

        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = self._make_observation_space()

        self.agent_position: Coordinate | None = None
        self.elapsed_steps = 0
        self._episode_reliability = self.action_reliability
        self._episode_reward_scale = 1.0
        self._regime = 0
        self._independent_present: set[WallEdge] = set()
        self._markov_present: set[WallEdge] = set()
        self._scheduled_present: set[WallEdge] = set()
        self._event_present: set[WallEdge] = set()
        self._fired_events: set[int] = set()
        self._moving_hazard_positions: list[Coordinate] = []
        self._has_reset = False

    # ------------------------------------------------------------------
    # Configuration normalization

    def _coordinate(self, value: Coordinate, name: str) -> Coordinate:
        try:
            row, col = value
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a (row, column) pair, got {value!r}") from exc
        coordinate = (int(row), int(col))
        if not (0 <= coordinate[0] < self.rows and 0 <= coordinate[1] < self.cols):
            raise ValueError(f"{name} {coordinate!r} lies outside shape {self.shape}")
        if coordinate in getattr(self, "blocked_cells", frozenset()):
            raise ValueError(f"{name} {coordinate!r} is a blocked cell")
        return coordinate

    def _wall(self, edge: WallEdge) -> WallEdge:
        canonical = canonical_edge(edge)
        self._coordinate(canonical[0], "wall endpoint")
        self._coordinate(canonical[1], "wall endpoint")
        return canonical

    def _cardinal_neighbors(self, state: Coordinate) -> tuple[Coordinate, ...]:
        neighbors: list[Coordinate] = []
        for delta in ACTION_DELTAS.values():
            candidate = (state[0] + delta[0], state[1] + delta[1])
            if (
                0 <= candidate[0] < self.rows
                and 0 <= candidate[1] < self.cols
                and candidate not in self.blocked_cells
            ):
                neighbors.append(candidate)
        return tuple(neighbors)

    def _normalize_goals(
        self, goals: Mapping[Coordinate, float | Goal] | Sequence[Goal] | None
    ) -> dict[Coordinate, Goal]:
        if goals is None:
            default = self.index_to_state[-1]
            entries: list[Goal] = [Goal(default)]
        elif isinstance(goals, Mapping):
            entries = []
            for coordinate, value in goals.items():
                position = self._coordinate(coordinate, "goal")
                if isinstance(value, Goal):
                    if self._coordinate(value.position, "goal") != position:
                        raise ValueError("A Goal's position must agree with its mapping key")
                    entries.append(value)
                else:
                    entries.append(Goal(position=position, reward=float(value)))
        else:
            entries = list(goals)
        result: dict[Coordinate, Goal] = {}
        for goal in entries:
            if not isinstance(goal, Goal):
                raise TypeError("goals sequences must contain Goal objects")
            position = self._coordinate(goal.position, "goal")
            if goal.reward_std < 0:
                raise ValueError("Goal.reward_std must be nonnegative")
            _probability(goal.terminal_probability, "Goal.terminal_probability")
            if position in result:
                raise ValueError(f"Duplicate goal at {position}")
            result[position] = Goal(
                position=position,
                reward=float(goal.reward),
                reward_std=float(goal.reward_std),
                terminal_probability=float(goal.terminal_probability),
            )
        if not result:
            raise ValueError("At least one goal is required")
        return result

    def _normalize_starts(
        self,
        start: Coordinate | None,
        distribution: Mapping[Coordinate, float] | Sequence[Coordinate] | None,
    ) -> tuple[tuple[Coordinate, ...], np.ndarray]:
        states: tuple[Coordinate, ...]
        if distribution is None:
            if start is None:
                start = self.index_to_state[0]
            states = (self._coordinate(start, "start"),)
            probabilities = np.array([1.0], dtype=np.float64)
        elif isinstance(distribution, Mapping):
            if start not in {None, (0, 0)}:
                raise ValueError("Specify either start or start_distribution, not both")
            states = tuple(self._coordinate(state, "start state") for state in distribution)
            probabilities = np.asarray(tuple(distribution.values()), dtype=np.float64)
        else:
            if start not in {None, (0, 0)}:
                raise ValueError("Specify either start or start_distribution, not both")
            states = tuple(self._coordinate(state, "start state") for state in distribution)
            probabilities = np.ones(len(states), dtype=np.float64)
        if not states:
            raise ValueError("start_distribution cannot be empty")
        if len(set(states)) != len(states):
            raise ValueError("start_distribution contains duplicate states")
        if np.any(~np.isfinite(probabilities)) or np.any(probabilities < 0):
            raise ValueError("Start probabilities must be finite and nonnegative")
        total = float(probabilities.sum())
        if total <= 0:
            raise ValueError("At least one start probability must be positive")
        return states, probabilities / total

    @staticmethod
    def _normalize_slip_weights(weights: Mapping[str, float] | None) -> dict[str, float]:
        raw = dict(weights or {"left": 0.4, "right": 0.4, "backward": 0.0, "stay": 0.2})
        allowed = {"left", "right", "backward", "stay"}
        unknown = set(raw).difference(allowed)
        if unknown:
            raise ValueError(f"Unknown slip outcomes: {sorted(unknown)}")
        complete = {name: float(raw.get(name, 0.0)) for name in allowed}
        if any(not np.isfinite(value) or value < 0 for value in complete.values()):
            raise ValueError("Slip weights must be finite and nonnegative")
        total = sum(complete.values())
        if total <= 0:
            raise ValueError("At least one slip weight must be positive")
        return {name: value / total for name, value in complete.items()}

    def _normalize_independent_walls(
        self, walls: Mapping[WallEdge, float] | Sequence[IndependentWall] | None
    ) -> tuple[IndependentWall, ...]:
        if walls is None:
            return ()
        entries = (
            [
                IndependentWall(edge=edge, presence_probability=probability)
                for edge, probability in walls.items()
            ]
            if isinstance(walls, Mapping)
            else list(walls)
        )
        normalized: list[IndependentWall] = []
        for wall in entries:
            if not isinstance(wall, IndependentWall):
                raise TypeError("independent_walls must contain IndependentWall objects")
            normalized.append(
                IndependentWall(
                    edge=self._wall(wall.edge),
                    presence_probability=_probability(
                        wall.presence_probability, "IndependentWall.presence_probability"
                    ),
                )
            )
        return tuple(normalized)

    def _normalize_markov_walls(
        self,
        walls: Mapping[WallEdge, tuple[float, float]] | Sequence[MarkovWall] | None,
    ) -> tuple[MarkovWall, ...]:
        if walls is None:
            return ()
        entries = (
            [MarkovWall(edge=edge, p01=spec[0], p11=spec[1]) for edge, spec in walls.items()]
            if isinstance(walls, Mapping)
            else list(walls)
        )
        normalized: list[MarkovWall] = []
        for wall in entries:
            if not isinstance(wall, MarkovWall):
                raise TypeError("markov_walls must contain MarkovWall objects")
            normalized.append(
                MarkovWall(
                    edge=self._wall(wall.edge),
                    p01=_probability(wall.p01, "MarkovWall.p01"),
                    p11=_probability(wall.p11, "MarkovWall.p11"),
                    initial_probability=_probability(
                        wall.initial_probability, "MarkovWall.initial_probability"
                    ),
                    initial_present=wall.initial_present,
                )
            )
        return tuple(normalized)

    def _normalize_scheduled_walls(
        self, walls: Mapping[Any, Any] | Sequence[ScheduledWall] | None
    ) -> tuple[ScheduledWall, ...]:
        if walls is None:
            return ()
        if not isinstance(walls, Mapping):
            entries = list(walls)
        elif walls and all(isinstance(key, int) for key in walls):
            # Time-centric shorthand: {time: {edge: present}}.
            changes_by_edge: dict[WallEdge, dict[int, bool]] = defaultdict(dict)
            for time, edge_changes in walls.items():
                if not isinstance(edge_changes, Mapping):
                    raise TypeError("Time-centric scheduled_walls values must be mappings")
                for edge, present in edge_changes.items():
                    changes_by_edge[canonical_edge(edge)][int(time)] = bool(present)
            entries = [
                ScheduledWall(edge=edge, changes=value) for edge, value in changes_by_edge.items()
            ]
        else:
            # Edge-centric shorthand: {edge: {time: present}}.
            entries = [ScheduledWall(edge=edge, changes=value) for edge, value in walls.items()]

        normalized: list[ScheduledWall] = []
        for wall in entries:
            if not isinstance(wall, ScheduledWall):
                raise TypeError("scheduled_walls must contain ScheduledWall objects")
            raw_changes = dict(wall.changes)
            normalized_changes: dict[int, bool] = {}
            for time, present in raw_changes.items():
                time_int = int(time)
                if time_int < 0:
                    raise ValueError("Scheduled wall times must be nonnegative")
                normalized_changes[time_int] = bool(present)
            normalized.append(
                ScheduledWall(
                    edge=self._wall(wall.edge),
                    changes=dict(sorted(normalized_changes.items())),
                    initial_present=bool(wall.initial_present),
                )
            )
        return tuple(normalized)

    def _normalize_event_walls(self, walls: Sequence[EventWall] | None) -> tuple[EventWall, ...]:
        normalized: list[EventWall] = []
        for wall in walls or ():
            if not isinstance(wall, EventWall):
                raise TypeError("event_walls must contain EventWall objects")
            triggers = frozenset(
                self._coordinate(state, "event trigger state") for state in wall.trigger_states
            )
            if not triggers:
                raise ValueError("EventWall.trigger_states cannot be empty")
            normalized.append(
                EventWall(
                    edge=self._wall(wall.edge),
                    trigger_states=triggers,
                    present_after_trigger=bool(wall.present_after_trigger),
                    initial_present=bool(wall.initial_present),
                    once=bool(wall.once),
                )
            )
        return tuple(normalized)

    def _validate_wall_mechanisms(self) -> None:
        groups: list[tuple[str, Iterable[WallEdge]]] = [
            ("static", self.static_walls),
            ("independent", (wall.edge for wall in self.independent_walls)),
            ("Markov", (wall.edge for wall in self.markov_walls)),
            ("scheduled", (wall.edge for wall in self.scheduled_walls)),
            ("event", (wall.edge for wall in self.event_walls)),
        ]
        owner: dict[WallEdge, str] = {}
        for name, edges in groups:
            for edge in edges:
                if edge in owner:
                    raise ValueError(
                        f"Wall {edge} is configured as both {owner[edge]} and {name}; "
                        "mechanisms must own disjoint edges"
                    )
                owner[edge] = name

    @staticmethod
    def _validate_nonstationarity(config: NonstationarityConfig) -> None:
        if config.mode not in {"stationary", "gradual", "abrupt", "periodic", "random"}:
            raise ValueError(f"Unknown nonstationarity mode {config.mode!r}")
        if not config.reliability_multipliers or not config.reward_multipliers:
            raise ValueError("Nonstationarity multiplier sequences cannot be empty")
        if any(value < 0 or not np.isfinite(value) for value in config.reliability_multipliers):
            raise ValueError("Reliability multipliers must be finite and nonnegative")
        if any(not np.isfinite(value) for value in config.reward_multipliers):
            raise ValueError("Reward multipliers must be finite")
        if config.horizon <= 0 or config.period <= 0 or config.change_step < 0:
            raise ValueError("horizon/period must be positive and change_step nonnegative")
        _probability(config.switch_probability, "switch_probability")

    @staticmethod
    def _validate_randomization(config: ParameterRandomization) -> None:
        if config.action_reliability is not None:
            low, high = map(float, config.action_reliability)
            _probability(low, "random reliability lower bound")
            _probability(high, "random reliability upper bound")
            if low > high:
                raise ValueError("Random reliability bounds must be ordered")
        if config.reward_scale is not None:
            low, high = map(float, config.reward_scale)
            if not np.isfinite(low) or not np.isfinite(high) or low > high:
                raise ValueError("Random reward-scale bounds must be finite and ordered")

    def _normalize_hazards(
        self, hazards: Sequence[Hazard] | Mapping[Coordinate, float] | None
    ) -> dict[Coordinate, Hazard]:
        if hazards is None:
            entries: list[Hazard] = []
        elif isinstance(hazards, Mapping):
            entries = [
                Hazard(position=state, penalty=penalty) for state, penalty in hazards.items()
            ]
        else:
            entries = list(hazards)
        result: dict[Coordinate, Hazard] = {}
        for hazard in entries:
            if not isinstance(hazard, Hazard) or isinstance(hazard, MovingHazard):
                raise TypeError("hazards must contain Hazard objects (moving ones go separately)")
            position = self._coordinate(hazard.position, "hazard")
            _probability(hazard.activation_probability, "Hazard.activation_probability")
            _probability(hazard.terminal_probability, "Hazard.terminal_probability")
            if position in self.goals:
                raise ValueError(f"A fixed hazard and goal cannot share {position}")
            if position in result:
                raise ValueError(f"Duplicate fixed hazard at {position}")
            result[position] = Hazard(
                position=position,
                penalty=float(hazard.penalty),
                terminal=bool(hazard.terminal),
                activation_probability=float(hazard.activation_probability),
                terminal_probability=float(hazard.terminal_probability),
            )
        return result

    def _normalize_moving_hazards(
        self, hazards: Sequence[MovingHazard] | None
    ) -> tuple[MovingHazard, ...]:
        result: list[MovingHazard] = []
        for hazard in hazards or ():
            if not isinstance(hazard, MovingHazard):
                raise TypeError("moving_hazards must contain MovingHazard objects")
            position = self._coordinate(hazard.position, "moving hazard")
            _probability(hazard.activation_probability, "MovingHazard.activation_probability")
            _probability(hazard.terminal_probability, "MovingHazard.terminal_probability")
            _probability(hazard.movement_probability, "MovingHazard.movement_probability")
            weights = np.asarray(hazard.motion_weights, dtype=np.float64)
            if weights.shape != (5,) or np.any(weights < 0) or not np.all(np.isfinite(weights)):
                raise ValueError(
                    "MovingHazard.motion_weights must be five nonnegative finite values"
                )
            if float(weights.sum()) <= 0:
                raise ValueError("MovingHazard.motion_weights must contain positive mass")
            normalized_weights = tuple(float(value) for value in weights / weights.sum())
            result.append(
                MovingHazard(
                    position=position,
                    penalty=float(hazard.penalty),
                    terminal=bool(hazard.terminal),
                    activation_probability=float(hazard.activation_probability),
                    terminal_probability=float(hazard.terminal_probability),
                    movement_probability=float(hazard.movement_probability),
                    motion_weights=normalized_weights,  # type: ignore[arg-type]
                )
            )
        return tuple(result)

    def _make_observation_space(self) -> spaces.Space[Any]:
        if self.observation_mode in {"state", "noisy_state"}:
            return spaces.Discrete(self.n_states)
        if self.observation_mode == "local":
            diameter = 2 * self.observation_radius + 1
            return spaces.MultiBinary((diameter, diameter, 8))
        return spaces.Dict(
            {
                "position": spaces.MultiDiscrete(np.array([self.rows, self.cols], dtype=np.int64)),
                "walls": spaces.MultiBinary(len(self._all_edges)),
                "blocked": spaces.MultiBinary(self.rows * self.cols),
                "goals": spaces.MultiBinary(self.rows * self.cols),
                "hazards": spaces.MultiBinary(self.rows * self.cols),
                "time": spaces.Discrete(self.max_episode_steps + 1),
            }
        )

    # ------------------------------------------------------------------
    # Public state and Gymnasium API

    @property
    def state(self) -> Coordinate | None:
        """Compatibility alias for :attr:`latent_position`.

        The value is simulator state, not necessarily part of the agent's
        observation.  New code should use ``latent_position`` when recording
        diagnostics and the value returned by ``reset``/``step`` for decisions.
        """

        return self.agent_position

    @property
    def latent_position(self) -> Coordinate | None:
        """The simulator's true grid position, irrespective of observation mode."""

        return self.agent_position

    @property
    def current_walls(self) -> frozenset[WallEdge]:
        """The complete wall realization used by the next transition."""

        return frozenset(
            self.static_walls
            | self._independent_present
            | self._markov_present
            | self._scheduled_present
            | self._event_present
        )

    @property
    def hazard_positions(self) -> frozenset[Coordinate]:
        """Current fixed and moving hazard positions."""

        return frozenset((*self.hazards, *self._moving_hazard_positions))

    @property
    def current_reliability(self) -> float:
        """Current global/spatial reliability, before selecting a specific action."""

        if self.agent_position is None:
            base = self._episode_reliability
        else:
            base = self.state_reliability.get(self.agent_position, self._episode_reliability)
        reliability_multiplier, _ = self._nonstationary_multipliers(self.elapsed_steps)
        return float(np.clip(base * reliability_multiplier, 0.0, 1.0))

    @property
    def current_reward_multiplier(self) -> float:
        """Current per-episode and nonstationary deterministic reward scale."""

        _, multiplier = self._nonstationary_multipliers(self.elapsed_steps)
        return float(self._episode_reward_scale * multiplier)

    @property
    def unwrapped_state_index(self) -> int:
        """Compatibility alias for :attr:`latent_state_index`."""

        return self.latent_state_index

    @property
    def latent_state_index(self) -> int:
        """True position-state index, irrespective of what the agent observes."""

        if self.agent_position is None:
            raise RuntimeError("reset() must be called before reading the state index")
        return self.state_to_index[self.agent_position]

    @property
    def observation_state_index_identity(self) -> tuple[Coordinate, ...] | None:
        """Ordered labels for the integer observation states used by exact control.

        ``None`` means that the current observation is not the fully observed,
        Markov position index represented by the ordinary exact MDP.  The tuple's
        order is the semantic identity of a Q-table row index; equal cardinality
        alone is not enough to establish comparability.
        """

        if self._exact_observation_incompatibility() is not None:
            return None
        return self.index_to_state

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Start an episode and independently seed each stochastic mechanism.

        Useful ``options`` are ``start_state``, ``action_reliability``,
        ``reward_scale``, and ``wall_configuration`` (an edge-to-bool mapping).
        Explicit options override parameter randomization for that reset.
        """

        super().reset(seed=seed)
        options = dict(options or {})
        entropy = int(self.np_random.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))
        streams = np.random.SeedSequence(entropy).spawn(4)
        self._dynamics_rng = np.random.default_rng(streams[0])
        self._reward_rng = np.random.default_rng(streams[1])
        self._observation_rng = np.random.default_rng(streams[2])
        self._parameter_rng = np.random.default_rng(streams[3])
        self.action_space.seed(int(streams[0].generate_state(1, dtype=np.uint32)[0]))

        if "start_state" in options:
            self.agent_position = self._coordinate(options["start_state"], "start_state option")
        else:
            start_index = int(
                self._parameter_rng.choice(len(self._start_states), p=self._start_probabilities)
            )
            self.agent_position = self._start_states[start_index]

        reliability_range = self.parameter_randomization.action_reliability
        if "action_reliability" in options:
            self._episode_reliability = _probability(
                options["action_reliability"], "action_reliability option"
            )
        elif reliability_range is not None:
            self._episode_reliability = float(
                self._parameter_rng.uniform(reliability_range[0], reliability_range[1])
            )
        else:
            self._episode_reliability = self.action_reliability

        reward_range = self.parameter_randomization.reward_scale
        if "reward_scale" in options:
            self._episode_reward_scale = float(options["reward_scale"])
        elif reward_range is not None:
            self._episode_reward_scale = float(
                self._parameter_rng.uniform(reward_range[0], reward_range[1])
            )
        else:
            self._episode_reward_scale = 1.0

        self.elapsed_steps = 0
        self._regime = 0
        self._fired_events.clear()
        self._initialize_walls(options.get("wall_configuration"))
        self._moving_hazard_positions = [hazard.position for hazard in self.moving_hazards]
        self._episode_done = False
        self._has_reset = True

        observation = self._observation()
        info = self._base_info()
        info["reset_seed"] = seed
        if self.render_mode == "human":
            self.render()
        return observation, info

    def step(self, action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Advance one decision using the timing convention in the module docstring."""

        if not self._has_reset:
            raise RuntimeError("reset() must be called before step()")
        if self._episode_done:
            raise RuntimeError("step() called after episode completion; call reset()")
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action!r} is outside {self.action_space}")
        assert self.agent_position is not None
        intended_action = int(action)
        reliability = self._effective_reliability(self.agent_position, intended_action)
        decision_walls = self.current_walls
        decision_wall_mask = sum(
            1 << bit for bit, edge in enumerate(self.augmented_wall_edges) if edge in decision_walls
        )
        outcomes, probabilities = self._realized_action_distribution(intended_action, reliability)
        outcome_index = int(self._dynamics_rng.choice(len(outcomes), p=probabilities))
        realized_action = outcomes[outcome_index]
        previous_position = self.agent_position
        next_position, blocked = self._move_from(
            previous_position, realized_action, self.current_walls
        )
        self.agent_position = next_position

        reward, terminated, reward_info = self._sample_reward(next_position)
        structure_events = self._trigger_events(next_position)
        self.elapsed_steps += 1
        structure_events.extend(self._advance_walls())
        self._advance_random_regime()
        self._advance_moving_hazards()

        truncated = self.elapsed_steps >= self.max_episode_steps and not terminated
        self._episode_done = bool(terminated or truncated)
        observation = self._observation()
        info = self._base_info()
        info.update(
            {
                "previous_position": previous_position,
                "intended_action": intended_action,
                "realized_action": realized_action,
                "slipped": realized_action != intended_action,
                "movement_blocked": blocked,
                "action_reliability": reliability,
                "decision_wall_mask": decision_wall_mask,
                "decision_walls": tuple(sorted(decision_walls)),
                "wall_events": structure_events,
                "reward_components": reward_info,
                "success": bool(terminated and next_position in self.goals),
                "failure": bool(terminated and reward_info["hazard_terminated"]),
            }
        )
        if self.render_mode == "human":
            self.render()
        return observation, float(reward), bool(terminated), bool(truncated), info

    # ------------------------------------------------------------------
    # Generative mechanisms

    def _initialize_walls(self, override: Any) -> None:
        self._independent_present = set()
        for independent_wall in self.independent_walls:
            draw = (
                self._dynamics_rng.random() < independent_wall.presence_probability
                if self.randomize_initial_walls
                else independent_wall.presence_probability >= 0.5
            )
            if draw:
                self._independent_present.add(independent_wall.edge)

        self._markov_present = set()
        for markov_wall in self.markov_walls:
            if markov_wall.initial_present is not None:
                present = markov_wall.initial_present
            elif self.randomize_initial_walls:
                present = self._dynamics_rng.random() < markov_wall.initial_probability
            else:
                present = markov_wall.initial_probability >= 0.5
            if present:
                self._markov_present.add(markov_wall.edge)

        self._scheduled_present = {
            wall.edge for wall in self.scheduled_walls if wall.initial_present
        }
        self._apply_schedule(time=0)
        self._event_present = {wall.edge for wall in self.event_walls if wall.initial_present}

        if override is not None:
            if isinstance(override, Mapping):
                configured = {self._wall(edge): bool(present) for edge, present in override.items()}
            else:
                present_edges = {self._wall(edge) for edge in override}
                configured = {
                    edge: edge in present_edges
                    for edge in (
                        *(wall.edge for wall in self.independent_walls),
                        *(wall.edge for wall in self.markov_walls),
                        *(wall.edge for wall in self.scheduled_walls),
                        *(wall.edge for wall in self.event_walls),
                    )
                }
            owners = {
                **{wall.edge: self._independent_present for wall in self.independent_walls},
                **{wall.edge: self._markov_present for wall in self.markov_walls},
                **{wall.edge: self._scheduled_present for wall in self.scheduled_walls},
                **{wall.edge: self._event_present for wall in self.event_walls},
            }
            unknown = set(configured).difference(owners)
            if unknown:
                raise ValueError(f"wall_configuration contains unmanaged dynamic walls: {unknown}")
            for edge, present in configured.items():
                if present:
                    owners[edge].add(edge)
                else:
                    owners[edge].discard(edge)

    def _effective_reliability(self, state: Coordinate, action: int) -> float:
        base = self.state_action_reliability.get(
            (state, action), self.state_reliability.get(state, self._episode_reliability)
        )
        multiplier, _ = self._nonstationary_multipliers(self.elapsed_steps)
        return float(np.clip(base * multiplier, 0.0, 1.0))

    def _realized_action_distribution(
        self, intended_action: int, reliability: float
    ) -> tuple[tuple[int | None, ...], np.ndarray]:
        relative: dict[str, int | None] = {
            "left": (intended_action - 1) % 4,
            "right": (intended_action + 1) % 4,
            "backward": (intended_action + 2) % 4,
            "stay": None,
        }
        outcomes: list[int | None] = [intended_action]
        probabilities: list[float] = [reliability]
        for label in ("left", "right", "backward", "stay"):
            probability = (1.0 - reliability) * self.slip_weights[label]
            if probability > 0:
                outcomes.append(relative[label])
                probabilities.append(probability)
        result = np.asarray(probabilities, dtype=np.float64)
        result /= result.sum()
        return tuple(outcomes), result

    def _move_from(
        self,
        state: Coordinate,
        action: int | None,
        walls: Iterable[WallEdge],
    ) -> tuple[Coordinate, bool]:
        if action is None:
            return state, False
        delta = ACTION_DELTAS[int(action)]
        candidate = (state[0] + delta[0], state[1] + delta[1])
        if candidate not in self.state_to_index:
            return state, True
        if canonical_edge((state, candidate)) in walls:
            return state, True
        return candidate, False

    def _sample_reward(self, state: Coordinate) -> tuple[float, bool, dict[str, Any]]:
        scale = self.current_reward_multiplier
        components: dict[str, Any] = {
            "step": self.step_reward * scale,
            "goal": 0.0,
            "hazard": 0.0,
            "rare": 0.0,
            "noise": 0.0,
            "goal_terminated": False,
            "hazard_terminated": False,
        }
        reward = float(components["step"])
        terminated = False

        goal = self.goals.get(state)
        if goal is not None:
            goal_reward = goal.reward * scale
            if goal.reward_std > 0:
                goal_reward += float(self._reward_rng.normal(0.0, goal.reward_std))
            reward += goal_reward
            components["goal"] = goal_reward
            if self._reward_rng.random() < goal.terminal_probability:
                terminated = True
                components["goal_terminated"] = True

        encountered: list[Hazard] = []
        fixed = self.hazards.get(state)
        if fixed is not None:
            encountered.append(fixed)
        encountered.extend(
            hazard
            for hazard, position in zip(
                self.moving_hazards, self._moving_hazard_positions, strict=True
            )
            if position == state
        )
        for hazard in encountered:
            if self._reward_rng.random() < hazard.activation_probability:
                penalty = hazard.penalty * scale
                reward += penalty
                components["hazard"] += penalty
                if hazard.terminal and self._reward_rng.random() < hazard.terminal_probability:
                    terminated = True
                    components["hazard_terminated"] = True

        if self._reward_rng.random() < self.rare_reward_probability:
            rare = self.rare_reward * scale
            reward += rare
            components["rare"] += rare
        if state in self.rare_rewards:
            probability, amount = self.rare_rewards[state]
            if self._reward_rng.random() < probability:
                rare = amount * scale
                reward += rare
                components["rare"] += rare

        noise_std = float(
            np.hypot(self.reward_noise_std, self.state_reward_noise_std.get(state, 0.0))
        )
        if noise_std > 0:
            noise = float(self._reward_rng.normal(0.0, noise_std))
            reward += noise
            components["noise"] = noise
        return reward, terminated, components

    def _trigger_events(self, state: Coordinate) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for index, wall in enumerate(self.event_walls):
            if state not in wall.trigger_states or (wall.once and index in self._fired_events):
                continue
            before = wall.edge in self._event_present
            if wall.present_after_trigger:
                self._event_present.add(wall.edge)
            else:
                self._event_present.discard(wall.edge)
            self._fired_events.add(index)
            events.append(
                {
                    "mechanism": "event",
                    "edge": wall.edge,
                    "before": before,
                    "after": wall.present_after_trigger,
                    "trigger_state": state,
                }
            )
        return events

    def _advance_walls(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for independent_wall in self.independent_walls:
            before = independent_wall.edge in self._independent_present
            after = bool(self._dynamics_rng.random() < independent_wall.presence_probability)
            if after:
                self._independent_present.add(independent_wall.edge)
            else:
                self._independent_present.discard(independent_wall.edge)
            if before != after:
                events.append(
                    {
                        "mechanism": "independent",
                        "edge": independent_wall.edge,
                        "before": before,
                        "after": after,
                    }
                )
        for markov_wall in self.markov_walls:
            before = markov_wall.edge in self._markov_present
            probability = markov_wall.p11 if before else markov_wall.p01
            after = bool(self._dynamics_rng.random() < probability)
            if after:
                self._markov_present.add(markov_wall.edge)
            else:
                self._markov_present.discard(markov_wall.edge)
            if before != after:
                events.append(
                    {
                        "mechanism": "markov",
                        "edge": markov_wall.edge,
                        "before": before,
                        "after": after,
                    }
                )
        events.extend(self._apply_schedule(self.elapsed_steps))
        return events

    def _apply_schedule(self, time: int) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for wall in self.scheduled_walls:
            changes = dict(wall.changes)
            if time not in changes:
                continue
            before = wall.edge in self._scheduled_present
            after = changes[time]
            if after:
                self._scheduled_present.add(wall.edge)
            else:
                self._scheduled_present.discard(wall.edge)
            if before != after:
                events.append(
                    {
                        "mechanism": "schedule",
                        "edge": wall.edge,
                        "before": before,
                        "after": after,
                        "time": time,
                    }
                )
        return events

    def _advance_moving_hazards(self) -> None:
        for index, (hazard, position) in enumerate(
            zip(self.moving_hazards, self._moving_hazard_positions, strict=True)
        ):
            if self._dynamics_rng.random() >= hazard.movement_probability:
                continue
            action_index = int(self._dynamics_rng.choice(5, p=hazard.motion_weights))
            action = None if action_index == 4 else action_index
            next_position, _ = self._move_from(position, action, self.current_walls)
            self._moving_hazard_positions[index] = next_position

    def _advance_random_regime(self) -> None:
        config = self.nonstationarity
        if config.mode != "random" or self._dynamics_rng.random() >= config.switch_probability:
            return
        number = max(len(config.reliability_multipliers), len(config.reward_multipliers))
        if number <= 1:
            return
        choices = [index for index in range(number) if index != self._regime]
        self._regime = int(self._dynamics_rng.choice(choices))

    def _nonstationary_multipliers(self, time: int) -> tuple[float, float]:
        config = self.nonstationarity

        def value(values: tuple[float, ...], index: int) -> float:
            return float(values[index % len(values)])

        if config.mode == "gradual":
            alpha = min(max(time / config.horizon, 0.0), 1.0)
            reliability = (1 - alpha) * config.reliability_multipliers[
                0
            ] + alpha * config.reliability_multipliers[-1]
            reward = (1 - alpha) * config.reward_multipliers[0] + alpha * config.reward_multipliers[
                -1
            ]
            return float(reliability), float(reward)
        if config.mode == "abrupt":
            index = 0 if time < config.change_step else -1
            return float(config.reliability_multipliers[index]), float(
                config.reward_multipliers[index]
            )
        if config.mode == "periodic":
            index = time // config.period
            return value(config.reliability_multipliers, index), value(
                config.reward_multipliers, index
            )
        if config.mode == "random":
            return value(config.reliability_multipliers, self._regime), value(
                config.reward_multipliers, self._regime
            )
        return float(config.reliability_multipliers[0]), float(config.reward_multipliers[0])

    def _reported_regime(self, time: int) -> int | float:
        """A compact regime diagnostic consistent with the configured mode."""

        config = self.nonstationarity
        if config.mode == "gradual":
            return float(min(max(time / config.horizon, 0.0), 1.0))
        if config.mode == "abrupt":
            return int(time >= config.change_step)
        if config.mode == "periodic":
            number = max(len(config.reliability_multipliers), len(config.reward_multipliers))
            return int((time // config.period) % number)
        if config.mode == "random":
            return self._regime
        return 0

    # ------------------------------------------------------------------
    # Observations and diagnostics

    def _observation(self) -> Any:
        assert self.agent_position is not None
        state_index = self.state_to_index[self.agent_position]
        if self.observation_mode == "state":
            return int(state_index)
        if self.observation_mode == "noisy_state":
            if self.n_states > 1 and self._observation_rng.random() < self.state_observation_noise:
                draw = int(self._observation_rng.integers(self.n_states - 1))
                state_index = draw if draw < state_index else draw + 1
            return int(state_index)
        if self.observation_mode == "local":
            return self._local_observation()
        return self._full_observation()

    def _full_observation(self) -> dict[str, np.ndarray | int]:
        assert self.agent_position is not None
        walls = np.fromiter(
            (edge in self.current_walls for edge in self._all_edges),
            dtype=np.int8,
            count=len(self._all_edges),
        )
        if self.wall_observation_noise > 0 and walls.size:
            flips = self._observation_rng.random(walls.size) < self.wall_observation_noise
            walls = np.logical_xor(walls, flips).astype(np.int8)
        blocked = np.zeros(self.rows * self.cols, dtype=np.int8)
        goals = np.zeros_like(blocked)
        hazards = np.zeros_like(blocked)
        for row, col in self.blocked_cells:
            blocked[row * self.cols + col] = 1
        for row, col in self.goals:
            goals[row * self.cols + col] = 1
        for row, col in self.hazard_positions:
            hazards[row * self.cols + col] = 1
        return {
            "position": np.asarray(self.agent_position, dtype=np.int64),
            "walls": walls,
            "blocked": blocked,
            "goals": goals,
            "hazards": hazards,
            "time": min(self.elapsed_steps, self.max_episode_steps),
        }

    def _local_observation(self) -> np.ndarray:
        assert self.agent_position is not None
        radius = self.observation_radius
        diameter = 2 * radius + 1
        observation = np.zeros((diameter, diameter, 8), dtype=np.int8)
        walls = self.current_walls
        for local_row, delta_row in enumerate(range(-radius, radius + 1)):
            for local_col, delta_col in enumerate(range(-radius, radius + 1)):
                state = (self.agent_position[0] + delta_row, self.agent_position[1] + delta_col)
                if state not in self.state_to_index:
                    continue
                observation[local_row, local_col, 0] = 1
                observation[local_row, local_col, 1] = int(state == self.agent_position)
                observation[local_row, local_col, 2] = int(state in self.goals)
                observation[local_row, local_col, 3] = int(state in self.hazard_positions)
                for action, channel in zip(range(4), range(4, 8), strict=True):
                    delta = ACTION_DELTAS[action]
                    candidate = (state[0] + delta[0], state[1] + delta[1])
                    present = candidate not in self.state_to_index
                    if not present:
                        present = canonical_edge((state, candidate)) in walls
                    if self.wall_observation_noise > 0:
                        present = bool(
                            present ^ (self._observation_rng.random() < self.wall_observation_noise)
                        )
                    observation[local_row, local_col, channel] = int(present)
        return observation

    def _base_info(self) -> dict[str, Any]:
        assert self.agent_position is not None
        reliability_multiplier, reward_multiplier = self._nonstationary_multipliers(
            self.elapsed_steps
        )
        latent_walls = tuple(sorted(self.current_walls))
        latent_hazards = tuple(sorted(self.hazard_positions))
        latent_state_index = self.state_to_index[self.agent_position]
        return {
            # Protocol-v2 names make privileged simulator state unmistakable.
            "latent_position": self.agent_position,
            "latent_state_index": latent_state_index,
            "latent_walls": latent_walls,
            "latent_hazard_positions": latent_hazards,
            # Compatibility aliases are diagnostics only.  The experiment runner
            # must derive an agent state from the returned observation, never info.
            "position": self.agent_position,
            "state_index": latent_state_index,
            "elapsed_steps": self.elapsed_steps,
            "walls": latent_walls,
            "hazard_positions": latent_hazards,
            "transition_regime": self._reported_regime(self.elapsed_steps),
            "reliability_multiplier": reliability_multiplier,
            "reward_multiplier": self._episode_reward_scale * reward_multiplier,
            "episode_action_reliability": self._episode_reliability,
        }

    # ------------------------------------------------------------------
    # Exact finite models

    def exact_mdp(
        self,
        *,
        augment_walls: bool = False,
        max_augmented_states: int = 5_000,
        max_model_bytes: int = 512 * 1024 * 1024,
    ) -> Any:
        """Construct the exact stationary expected-reward MDP.

        Independent walls can be marginalized in the ordinary position-state
        model because their decision-time realization is iid and independent of
        position.  Markov walls retain memory and therefore require
        ``augment_walls=True``.  In the augmented model a state label is
        ``(position, wall_bits)`` and each bit is ordered as reported by
        :attr:`augmented_wall_edges`.

        Reward noise, random terminal-reward noise, and rare rewards are
        represented by their conditional expectations.  Cases whose termination
        flag cannot be represented by a state-based ``terminal`` vector raise
        :class:`ExactModelUnavailable` rather than silently changing the process.
        """

        self._validate_exact_model_request(augment_walls=augment_walls)
        if augment_walls and (self.independent_walls or self.markov_walls):
            return self._build_augmented_mdp(
                max_augmented_states=max_augmented_states,
                max_model_bytes=max_model_bytes,
            )
        return self._build_position_mdp()

    def exact_model(self, **kwargs: Any) -> Any:
        """Alias for :meth:`exact_mdp`."""

        return self.exact_mdp(**kwargs)

    def build_exact_mdp(self, **kwargs: Any) -> Any:
        """Alias for :meth:`exact_mdp`."""

        return self.exact_mdp(**kwargs)

    def to_finite_mdp(self, **kwargs: Any) -> Any:
        """Alias for :meth:`exact_mdp`."""

        return self.exact_mdp(**kwargs)

    def transition_reward_kernels(self, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
        """Return copies of exact ``P[s,a,s']`` and ``R[s,a,s']`` arrays."""

        model = self.exact_mdp(**kwargs)
        return np.array(model.P, copy=True), np.array(model.R, copy=True)

    def transition_kernel(self, **kwargs: Any) -> np.ndarray:
        """Return a copy of exact ``P[s,a,s']``."""

        return self.transition_reward_kernels(**kwargs)[0]

    def reward_kernel(self, **kwargs: Any) -> np.ndarray:
        """Return conditional expected rewards ``R[s,a,s']``."""

        return self.transition_reward_kernels(**kwargs)[1]

    def expected_rewards(self, **kwargs: Any) -> np.ndarray:
        """Return ``sum_s' P[s,a,s'] R[s,a,s']`` with shape ``[S,A]``."""

        transition, rewards = self.transition_reward_kernels(**kwargs)
        return np.sum(transition * rewards, axis=2)

    @property
    def augmented_wall_edges(self) -> tuple[WallEdge, ...]:
        """Bit ordering used by the augmented-state model."""

        return tuple(
            [wall.edge for wall in self.independent_walls]
            + [wall.edge for wall in self.markov_walls]
        )

    def decode_augmented_state(self, index: int) -> tuple[Coordinate, tuple[bool, ...]]:
        """Decode an augmented model index into position and wall-presence bits."""

        wall_count = len(self.augmented_wall_edges)
        number_masks = 1 << wall_count
        number_states = self.n_states * number_masks
        if not 0 <= int(index) < number_states:
            raise IndexError(f"Augmented state index must lie in [0, {number_states})")
        position_index, mask = divmod(int(index), number_masks)
        bits = tuple(bool(mask & (1 << bit)) for bit in range(wall_count))
        return self.index_to_state[position_index], bits

    def _validate_exact_model_request(self, *, augment_walls: bool) -> None:
        blockers: list[str] = []
        observation_incompatibility = self._exact_observation_incompatibility()
        if observation_incompatibility is not None:
            blockers.append(observation_incompatibility)
        if self.scheduled_walls:
            blockers.append("scheduled walls (time must be augmented)")
        if self.event_walls:
            blockers.append("event walls (trigger memory must be augmented)")
        if self.moving_hazards:
            blockers.append("moving hazards (joint positions must be augmented)")
        if self.nonstationarity.mode != "stationary":
            blockers.append(f"{self.nonstationarity.mode!r} nonstationarity")
        if self.markov_walls and not augment_walls:
            blockers.append("Markov walls (pass augment_walls=True)")
        if self.parameter_randomization.action_reliability is not None and not self._has_reset:
            blockers.append("randomized action reliability before a realization is reset")
        if self.parameter_randomization.reward_scale is not None and not self._has_reset:
            blockers.append("randomized reward scale before a realization is reset")
        if self.independent_walls and not self.randomize_initial_walls and not augment_walls:
            blockers.append(
                "deterministic initial independent-wall realization (the first-step kernel is exceptional)"
            )

        for goal in self.goals.values():
            if 0.0 < goal.terminal_probability < 1.0:
                blockers.append(
                    f"probabilistic goal termination at {goal.position} (outcome state must be augmented)"
                )
        for hazard in self.hazards.values():
            termination_probability = (
                hazard.activation_probability * hazard.terminal_probability
                if hazard.terminal
                else 0.0
            )
            if 0.0 < termination_probability < 1.0:
                blockers.append(
                    f"probabilistic terminal hazard at {hazard.position} "
                    "(outcome state must be augmented)"
                )
        if blockers:
            raise ExactModelUnavailable(
                "No exact state-based stationary MDP is exposed for: " + "; ".join(blockers)
            )

    def _exact_observation_incompatibility(self) -> str | None:
        """Explain why position-index Q* would not index the agent observation."""

        if self.observation_mode == "state":
            return None
        if self.observation_mode == "noisy_state" and self.state_observation_noise == 0.0:
            return None
        if self.observation_mode == "noisy_state":
            return (
                f"observation_mode='noisy_state' with state_observation_noise="
                f"{self.state_observation_noise:g} is partially observed; exact_mdp() rows index "
                "latent positions, not noisy observations"
            )
        return (
            f"observation_mode={self.observation_mode!r} does not use the integer position-state "
            "index represented by exact_mdp()"
        )

    def _terminal_positions(self) -> frozenset[Coordinate]:
        positions = {
            goal.position for goal in self.goals.values() if goal.terminal_probability == 1.0
        }
        positions.update(
            hazard.position
            for hazard in self.hazards.values()
            if hazard.terminal
            and hazard.activation_probability == 1.0
            and hazard.terminal_probability == 1.0
        )
        return frozenset(positions)

    def _model_reliability(self, state: Coordinate, action: int) -> float:
        # Exact requests with randomized parameters are admitted only after reset.
        base = self.state_action_reliability.get(
            (state, action), self.state_reliability.get(state, self._episode_reliability)
        )
        multiplier = self.nonstationarity.reliability_multipliers[0]
        return float(np.clip(base * multiplier, 0.0, 1.0))

    def _expected_reward(self, destination: Coordinate) -> float:
        reward_multiplier = self._episode_reward_scale * self.nonstationarity.reward_multipliers[0]
        reward = self.step_reward * reward_multiplier
        goal = self.goals.get(destination)
        if goal is not None:
            reward += goal.reward * reward_multiplier
        hazard = self.hazards.get(destination)
        if hazard is not None:
            reward += hazard.activation_probability * hazard.penalty * reward_multiplier
        reward += self.rare_reward_probability * self.rare_reward * reward_multiplier
        if destination in self.rare_rewards:
            probability, amount = self.rare_rewards[destination]
            reward += probability * amount * reward_multiplier
        return float(reward)

    def _build_position_mdp(self) -> Any:
        number_states = self.n_states
        transition = np.zeros((number_states, self.n_actions, number_states), dtype=np.float64)
        rewards = np.zeros_like(transition)
        terminal_positions = self._terminal_positions()
        terminal = np.fromiter(
            (state in terminal_positions for state in self.index_to_state),
            dtype=bool,
            count=number_states,
        )
        independent_probabilities = {
            wall.edge: wall.presence_probability for wall in self.independent_walls
        }

        for state_index, state in enumerate(self.index_to_state):
            if terminal[state_index]:
                transition[state_index, :, state_index] = 1.0
                continue
            for action in range(self.n_actions):
                reliability = self._model_reliability(state, action)
                outcomes, probabilities = self._realized_action_distribution(action, reliability)
                for realized, action_probability in zip(outcomes, probabilities, strict=True):
                    if realized is None:
                        transition[state_index, action, state_index] += action_probability
                        continue
                    delta = ACTION_DELTAS[realized]
                    candidate = (state[0] + delta[0], state[1] + delta[1])
                    if candidate not in self.state_to_index:
                        transition[state_index, action, state_index] += action_probability
                        continue
                    edge = canonical_edge((state, candidate))
                    if edge in self.static_walls:
                        transition[state_index, action, state_index] += action_probability
                        continue
                    wall_probability = independent_probabilities.get(edge, 0.0)
                    candidate_index = self.state_to_index[candidate]
                    transition[state_index, action, state_index] += (
                        action_probability * wall_probability
                    )
                    transition[state_index, action, candidate_index] += action_probability * (
                        1.0 - wall_probability
                    )
                reachable = np.flatnonzero(transition[state_index, action] > 0)
                for destination_index in reachable:
                    rewards[state_index, action, destination_index] = self._expected_reward(
                        self.index_to_state[int(destination_index)]
                    )

        model_type = _finite_mdp_type()
        return model_type(
            P=transition,
            R=rewards,
            terminal=terminal,
            state_labels=self.index_to_state,
        )

    def _build_augmented_mdp(self, *, max_augmented_states: int, max_model_bytes: int) -> Any:
        walls = self.augmented_wall_edges
        number_masks = 1 << len(walls)
        number_states = self.n_states * number_masks
        if number_states > max_augmented_states:
            raise ExactModelUnavailable(
                f"Augmenting {len(walls)} wall bits produces {number_states:,} states, "
                f"above max_augmented_states={max_augmented_states:,}"
            )
        estimated_bytes = 2 * number_states * self.n_actions * number_states * 8
        if estimated_bytes > max_model_bytes:
            raise ExactModelUnavailable(
                f"Dense augmented P/R arrays require about {estimated_bytes / 2**20:.1f} MiB, "
                f"above max_model_bytes={max_model_bytes / 2**20:.1f} MiB"
            )

        wall_transitions = self._wall_mask_transition_matrix()
        transition = np.zeros((number_states, self.n_actions, number_states), dtype=np.float64)
        rewards = np.zeros_like(transition)
        terminal_positions = self._terminal_positions()
        terminal = np.zeros(number_states, dtype=bool)

        for position_index, state in enumerate(self.index_to_state):
            for mask in range(number_masks):
                state_index = position_index * number_masks + mask
                if state in terminal_positions:
                    terminal[state_index] = True
                    transition[state_index, :, state_index] = 1.0
                    continue
                present_walls = self.static_walls | {
                    edge for bit, edge in enumerate(walls) if mask & (1 << bit)
                }
                for action in range(self.n_actions):
                    reliability = self._model_reliability(state, action)
                    outcomes, action_probabilities = self._realized_action_distribution(
                        action, reliability
                    )
                    destination_probabilities: dict[int, float] = defaultdict(float)
                    for realized, action_probability in zip(
                        outcomes, action_probabilities, strict=True
                    ):
                        destination, _ = self._move_from(state, realized, present_walls)
                        destination_probabilities[self.state_to_index[destination]] += float(
                            action_probability
                        )
                    next_masks = np.flatnonzero(wall_transitions[mask] > 0)
                    for (
                        destination_index,
                        movement_probability,
                    ) in destination_probabilities.items():
                        expected_reward = self._expected_reward(
                            self.index_to_state[destination_index]
                        )
                        for next_mask in next_masks:
                            next_index = destination_index * number_masks + int(next_mask)
                            probability = movement_probability * wall_transitions[mask, next_mask]
                            transition[state_index, action, next_index] += probability
                            rewards[state_index, action, next_index] = expected_reward

        labels = tuple(
            (position, tuple(bool(mask & (1 << bit)) for bit in range(len(walls))))
            for position in self.index_to_state
            for mask in range(number_masks)
        )
        model_type = _finite_mdp_type()
        return model_type(P=transition, R=rewards, terminal=terminal, state_labels=labels)

    def _wall_mask_transition_matrix(self) -> np.ndarray:
        independent_count = len(self.independent_walls)
        wall_count = independent_count + len(self.markov_walls)
        number_masks = 1 << wall_count
        matrix = np.ones((number_masks, number_masks), dtype=np.float64)
        for current_mask in range(number_masks):
            for next_mask in range(number_masks):
                probability = 1.0
                for bit, independent_wall in enumerate(self.independent_walls):
                    next_present = bool(next_mask & (1 << bit))
                    probability *= (
                        independent_wall.presence_probability
                        if next_present
                        else 1.0 - independent_wall.presence_probability
                    )
                for offset, markov_wall in enumerate(self.markov_walls, start=independent_count):
                    current_present = bool(current_mask & (1 << offset))
                    next_present = bool(next_mask & (1 << offset))
                    present_probability = markov_wall.p11 if current_present else markov_wall.p01
                    probability *= (
                        present_probability if next_present else 1.0 - present_probability
                    )
                matrix[current_mask, next_mask] = probability
        return matrix

    # ------------------------------------------------------------------
    # Rendering

    def render(self) -> str | np.ndarray | None:
        """Render the current latent maze realization."""

        if not self._has_reset:
            raise RuntimeError("reset() must be called before render()")
        if self.render_mode == "rgb_array":
            return self._rgb_render()
        text = self._ansi_render()
        if self.render_mode == "human":
            print(text)
            return None
        return text

    def _barrier(self, state: Coordinate, neighbor: Coordinate) -> bool:
        if state in self.blocked_cells or neighbor not in self.state_to_index:
            return True
        return canonical_edge((state, neighbor)) in self.current_walls

    def _ansi_render(self) -> str:
        assert self.agent_position is not None
        lines: list[str] = []
        for row in range(self.rows):
            top = "+"
            middle = ""
            for col in range(self.cols):
                cell = (row, col)
                north = (row - 1, col)
                top += ("---" if self._barrier(cell, north) else "   ") + "+"
                west = (row, col - 1)
                middle += "|" if self._barrier(cell, west) else " "
                if cell == self.agent_position:
                    marker = "A"
                elif cell in self.blocked_cells:
                    marker = "#"
                elif cell in self.hazard_positions:
                    marker = "H"
                elif cell in self.goals:
                    marker = "G"
                elif cell in self._start_states:
                    marker = "S"
                else:
                    marker = " "
                middle += f" {marker} "
            last = (row, self.cols - 1)
            middle += "|" if self._barrier(last, (row, self.cols)) else " "
            lines.extend((top, middle))
        bottom = "+"
        for col in range(self.cols):
            cell = (self.rows - 1, col)
            bottom += ("---" if self._barrier(cell, (self.rows, col)) else "   ") + "+"
        lines.append(bottom)
        return "\n".join(lines)

    def _rgb_render(self) -> np.ndarray:
        assert self.agent_position is not None
        tile = 24
        image = np.full((self.rows * tile + 1, self.cols * tile + 1, 3), 245, dtype=np.uint8)
        for row in range(self.rows):
            for col in range(self.cols):
                cell = (row, col)
                y0, y1 = row * tile, (row + 1) * tile
                x0, x1 = col * tile, (col + 1) * tile
                if cell in self.blocked_cells:
                    image[y0:y1, x0:x1] = (55, 55, 60)
                elif cell in self.goals:
                    image[y0 + 3 : y1 - 2, x0 + 3 : x1 - 2] = (145, 220, 150)
                if cell in self.hazard_positions:
                    image[y0 + 6 : y1 - 5, x0 + 6 : x1 - 5] = (225, 85, 80)
                if cell == self.agent_position:
                    image[y0 + 7 : y1 - 6, x0 + 7 : x1 - 6] = (55, 110, 225)
                for action, (ys, xs) in {
                    MazeAction.NORTH: (slice(y0, y0 + 2), slice(x0, x1 + 1)),
                    MazeAction.EAST: (slice(y0, y1 + 1), slice(x1 - 1, x1 + 1)),
                    MazeAction.SOUTH: (slice(y1 - 1, y1 + 1), slice(x0, x1 + 1)),
                    MazeAction.WEST: (slice(y0, y1 + 1), slice(x0, x0 + 2)),
                }.items():
                    delta = ACTION_DELTAS[int(action)]
                    neighbor = (row + delta[0], col + delta[1])
                    if self._barrier(cell, neighbor):
                        image[ys, xs] = (20, 20, 25)
        return image

    def close(self) -> None:
        """Gymnasium compatibility; the environment owns no external resources."""

        return None
