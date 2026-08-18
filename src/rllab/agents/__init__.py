"""Tabular policies, TD-control agents, schedules, and exploration strategies."""

from rllab.agents.base import Agent, TabularAgent, UpdateRecord
from rllab.agents.exploration import (
    UCB,
    Boltzmann,
    BoltzmannExploration,
    ConstantSchedule,
    EpsilonGreedy,
    EpsilonGreedyExploration,
    ExplorationStrategy,
    ExponentialDecaySchedule,
    LinearDecaySchedule,
    Schedule,
    UCBExploration,
)
from rllab.agents.nonlearning import PlannerAgent, RandomAgent
from rllab.agents.td_control import (
    DoubleQLearningAgent,
    ExpectedSARSAAgent,
    ExpectedSarsaAgent,
    QLearningAgent,
    SARSAAgent,
    SarsaAgent,
    TDControlAgent,
)

__all__ = [
    "UCB",
    "Agent",
    "Boltzmann",
    "BoltzmannExploration",
    "ConstantSchedule",
    "DoubleQLearningAgent",
    "EpsilonGreedy",
    "EpsilonGreedyExploration",
    "ExpectedSARSAAgent",
    "ExpectedSarsaAgent",
    "ExplorationStrategy",
    "ExponentialDecaySchedule",
    "LinearDecaySchedule",
    "PlannerAgent",
    "QLearningAgent",
    "RandomAgent",
    "SARSAAgent",
    "SarsaAgent",
    "Schedule",
    "TDControlAgent",
    "TabularAgent",
    "UCBExploration",
    "UpdateRecord",
]
