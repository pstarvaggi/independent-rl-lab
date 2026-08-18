"""Observation wrappers that make selected latent maze state explicit."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium import spaces

from rllab.environments.stochastic_maze import StochasticMazeEnv


class WallStateObservationWrapper(gym.Wrapper[Any, int, Any, int]):
    """Expose position and dynamic-wall bits as one finite observation index.

    The ordering matches ``StochasticMazeEnv.exact_mdp(augment_walls=True)``:
    position is the major index and the binary wall mask is the minor index.
    This wrapper is intentionally explicit because silently feeding wall state
    to a position-only learner would invalidate a partial-observability study.
    """

    def __init__(self, env: StochasticMazeEnv) -> None:
        super().__init__(env)
        if env.observation_mode != "state":
            raise ValueError("WallStateObservationWrapper requires observation_mode='state'")
        if not env.augmented_wall_edges:
            raise ValueError("WallStateObservationWrapper requires at least one dynamic wall")
        self.wall_edges = env.augmented_wall_edges
        self.wall_configurations = 1 << len(self.wall_edges)
        self.n_states = env.n_states * self.wall_configurations
        self.n_actions = env.n_actions
        self.observation_space = spaces.Discrete(self.n_states)

    @property
    def base_env(self) -> StochasticMazeEnv:
        return self.env  # type: ignore[return-value]

    def _wall_mask(self) -> int:
        current = self.base_env.current_walls
        return sum(1 << bit for bit, edge in enumerate(self.wall_edges) if edge in current)

    def _observation(self) -> int:
        return self.base_env.latent_state_index * self.wall_configurations + self._wall_mask()

    @property
    def observation_state_index_identity(self) -> tuple[Any, ...]:
        labels: list[Any] = []
        for position in self.base_env.index_to_state:
            for mask in range(self.wall_configurations):
                bits = tuple(bool(mask & (1 << bit)) for bit in range(len(self.wall_edges)))
                labels.append((position, bits))
        return tuple(labels)

    def reset(self, **kwargs: Any) -> tuple[int, dict[str, Any]]:
        _, info = self.env.reset(**kwargs)
        return self._observation(), info

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        _, reward, terminated, truncated, info = self.env.step(action)
        return self._observation(), float(reward), terminated, truncated, info

    def exact_mdp(self, **kwargs: Any) -> Any:
        requested = kwargs.pop("augment_walls", True)
        if requested is False:
            raise ValueError("Wall-state observations require augment_walls=True")
        return self.base_env.exact_mdp(augment_walls=True, **kwargs)

    def exact_model(self, **kwargs: Any) -> Any:
        return self.exact_mdp(**kwargs)

    def build_exact_mdp(self, **kwargs: Any) -> Any:
        return self.exact_mdp(**kwargs)

    def to_finite_mdp(self, **kwargs: Any) -> Any:
        return self.exact_mdp(**kwargs)


__all__ = ["WallStateObservationWrapper"]
