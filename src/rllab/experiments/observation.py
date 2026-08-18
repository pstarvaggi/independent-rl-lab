"""Explicit boundary between agent observations and privileged diagnostics."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Protocol

import numpy as np


class ObservationEncodingError(ValueError):
    """Raised when a tabular agent cannot consume an environment observation."""


class ObservationEncoder(Protocol):
    """Explicit encoder for a finite observation space."""

    n_observations: int
    indexer_id: str
    state_index_identity: tuple[Hashable, ...]

    def encode(self, observation: Any) -> int: ...


@dataclass(frozen=True, slots=True)
class TabularObservationAdapter:
    """Encode only the returned observation; environment ``info`` is forbidden."""

    n_observations: int
    indexer_id: str
    state_index_identity: tuple[Hashable, ...]
    encoder: Any = None

    @staticmethod
    def _identity(env: Any, count: int) -> tuple[Hashable, ...]:
        raw = getattr(env, "observation_state_index_identity", None)
        raw = raw() if callable(raw) else raw
        if raw is None:
            return tuple(range(count))
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
            raise ObservationEncodingError(
                "observation_state_index_identity must be an ordered sequence"
            )
        identity = tuple(raw)
        if len(identity) != count:
            raise ObservationEncodingError(
                "observation_state_index_identity length does not match the observation space"
            )
        try:
            unique = set(identity)
        except TypeError as error:
            raise ObservationEncodingError(
                "observation_state_index_identity entries must be hashable"
            ) from error
        if len(unique) != count:
            raise ObservationEncodingError(
                "observation_state_index_identity entries must be unique"
            )
        return identity

    @staticmethod
    def _indexer_id(env: Any, mode: str, identity: tuple[Hashable, ...]) -> str:
        declared = getattr(env, "observation_indexer_id", None)
        if declared is not None:
            declared = declared() if callable(declared) else declared
            return str(declared)
        payload = json.dumps(identity, default=repr, separators=(",", ":")).encode()
        digest = hashlib.sha256(payload).hexdigest()[:16]
        return f"{type(env).__module__}.{type(env).__qualname__}:{mode}:{digest}"

    @classmethod
    def from_environment(cls, env: Any) -> TabularObservationAdapter:
        space = getattr(env, "observation_space", None)
        if space is not None and isinstance(getattr(space, "n", None), Integral):
            count = int(space.n)
            mode = getattr(env, "observation_mode", "discrete")
            state_identity = cls._identity(env, count)
            indexer_id = cls._indexer_id(env, str(mode), state_identity)
            return cls(count, indexer_id, state_identity)

        encoder = getattr(env, "observation_to_state", None)
        raw_count = getattr(env, "n_observations", None)
        if callable(encoder) and raw_count is not None:
            count = int(raw_count)
            state_identity = cls._identity(env, count)
            indexer_id = cls._indexer_id(env, "explicit", state_identity)
            return cls(count, indexer_id, state_identity, encoder)

        raise ObservationEncodingError(
            "A tabular agent requires a Discrete observation space or an explicit "
            "observation_to_state(observation) encoder with n_observations. Privileged "
            "state identifiers from info are never used."
        )

    def encode(self, observation: Any) -> int:
        value = self.encoder(observation) if self.encoder is not None else observation
        if isinstance(value, np.ndarray):
            if value.size != 1:
                raise ObservationEncodingError(
                    f"Expected one discrete observation index, received shape {value.shape}"
                )
            value = value.reshape(-1)[0]
        if isinstance(value, np.generic):
            value = value.item()
        if not isinstance(value, (int, np.integer)):
            raise ObservationEncodingError(
                f"Expected an integer observation index, received {type(value).__name__}"
            )
        index = int(value)
        if not 0 <= index < self.n_observations:
            raise ObservationEncodingError(
                f"Observation index {index} lies outside [0, {self.n_observations})"
            )
        return index


def latent_state_from_info(info: Any) -> int | None:
    """Extract privileged truth for diagnostics only."""

    if not isinstance(info, Mapping):
        return None
    value = info.get("latent_state_index", info.get("state_index"))
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "ObservationEncoder",
    "ObservationEncodingError",
    "TabularObservationAdapter",
    "latent_state_from_info",
]
