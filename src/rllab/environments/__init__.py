"""Research environments."""

from rllab.environments.risky_corridor import HazardMode, RiskyCorridorEnv, RouteLabel
from rllab.environments.stochastic_maze import (
    Coordinate,
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
    WallEdge,
    canonical_edge,
)
from rllab.environments.wrappers import WallStateObservationWrapper

__all__ = [
    "Coordinate",
    "EventWall",
    "ExactModelUnavailable",
    "Goal",
    "Hazard",
    "HazardMode",
    "IndependentWall",
    "MarkovWall",
    "MazeAction",
    "MovingHazard",
    "NonstationarityConfig",
    "ParameterRandomization",
    "RiskyCorridorEnv",
    "RouteLabel",
    "ScheduledWall",
    "StochasticMazeEnv",
    "WallEdge",
    "WallStateObservationWrapper",
    "canonical_edge",
]
