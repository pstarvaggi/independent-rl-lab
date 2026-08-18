"""Low-overhead episode/step instrumentation used by the experiment runner."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]


def _scalar(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    return None


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, set | frozenset):
        return sorted(value, key=repr)
    return repr(value)


def diagnostic_fields(info: Mapping[str, Any] | None, *, prefix: str = "env_") -> dict[str, Any]:
    """Preserve scalar info fields and JSON-encode structured environment state."""

    if not info:
        return {}
    result: dict[str, Any] = {}
    for key, value in info.items():
        name = key if key.startswith(prefix) else f"{prefix}{key}"
        if value is None:
            # Preserve a genuine missing value instead of the string ``"null"``;
            # mixing that string with numeric diagnostics prevents Parquet from
            # inferring a stable column type.
            result[name] = None
            continue
        scalar = _scalar(value)
        if scalar is not None:
            result[name] = scalar
        else:
            try:
                result[name] = json.dumps(value, default=_json_default, sort_keys=True)
            except (TypeError, ValueError):
                result[name] = repr(value)
    return result


def environment_episode_summary(environment: Any) -> dict[str, Any]:
    """Read an optional environment-owned summary for the current episode.

    Environments can expose ``episode_summary() -> Mapping[str, Any]`` to add
    domain diagnostics without teaching the experiment runner about a particular
    maze, route, or hazard. The runner calls this once after an episode ends and
    before the next reset.
    """

    summary = getattr(environment, "episode_summary", None)
    if not callable(summary):
        return {}
    value = summary()
    if not isinstance(value, Mapping):
        raise TypeError("environment episode_summary() must return a mapping")
    return dict(value)


def update_fields(record: Any) -> dict[str, Any]:
    """Normalize mapping/dataclass/object update diagnostics."""

    if record is None:
        return {}
    if isinstance(record, Mapping):
        values = dict(record)
    elif is_dataclass(record) and not isinstance(record, type):
        values = asdict(record)
    elif hasattr(record, "__dict__"):
        values = vars(record)
    elif isinstance(record, (float, np.floating)):
        values = {"td_error": float(record)}
    else:
        return {}
    aliases = {
        "delta": "td_error",
        "td": "td_error",
        "learning_rate": "alpha",
        "exploration_rate": "epsilon",
    }
    result: dict[str, Any] = {}
    for key, value in values.items():
        scalar = _scalar(value)
        if scalar is not None:
            result[aliases.get(str(key), str(key))] = scalar
    return result


@dataclass(slots=True)
class _EpisodeState:
    episode: int
    episode_return: float = 0.0
    length: int = 0
    td_sum: float = 0.0
    td_abs_sum: float = 0.0
    td_square_sum: float = 0.0
    td_count: int = 0
    alpha_sum: float = 0.0
    alpha_count: int = 0
    epsilon_sum: float = 0.0
    epsilon_count: int = 0


class MetricRecorder:
    """Record tidy rows while maintaining online empirical-model diagnostics."""

    def __init__(
        self,
        *,
        trial_id: str,
        seed: int,
        agent: str,
        environment: str,
        record_steps: bool = True,
        step_retention_mode: str | None = None,
        step_sample_fraction: float = 1.0,
        keep_terminal_steps: bool = True,
        keep_event_steps: bool = True,
        retention_salt: str = "rl-lab-step-retention-v2",
        common: Mapping[str, Any] | None = None,
    ) -> None:
        self.trial_id = trial_id
        self.seed = seed
        self.agent = agent
        self.environment = environment
        self.step_retention_mode = step_retention_mode or ("all" if record_steps else "none")
        if self.step_retention_mode not in {"all", "none", "sample"}:
            raise ValueError("step_retention_mode must be 'all', 'none', or 'sample'")
        if not 0.0 < float(step_sample_fraction) <= 1.0:
            raise ValueError("step_sample_fraction must lie in (0, 1]")
        self.step_sample_fraction = float(step_sample_fraction)
        self.keep_terminal_steps = bool(keep_terminal_steps)
        self.keep_event_steps = bool(keep_event_steps)
        self.retention_salt = str(retention_salt)
        self.record_steps = self.step_retention_mode != "none"
        self.common = dict(common or {})
        self.step_rows: list[dict[str, Any]] = []
        self.episode_rows: list[dict[str, Any]] = []
        self.snapshot_rows: list[dict[str, Any]] = []
        self.q_snapshots: dict[str, np.ndarray] = {}
        self._snapshot_keys: set[str] = set()
        self._snapshot_global_steps: set[int] = set()
        self.state_action_rows: list[dict[str, Any]] = []
        self.state_visits: Counter[int] = Counter()
        self.latent_state_visits: Counter[int] = Counter()
        self.state_action_visits: Counter[tuple[int, int]] = Counter()
        self.transition_counts: Counter[tuple[int, int, int]] = Counter()
        self.reward_sums: defaultdict[tuple[int, int, int], float] = defaultdict(float)
        self.state_action_reward_sums: defaultdict[tuple[int, int], float] = defaultdict(float)
        self.reward_square_sums: defaultdict[tuple[int, int], float] = defaultdict(float)
        self.td_sums: defaultdict[tuple[int, int], float] = defaultdict(float)
        self.td_square_sums: defaultdict[tuple[int, int], float] = defaultdict(float)
        self.td_abs_sums: defaultdict[tuple[int, int], float] = defaultdict(float)
        self.td_counts: Counter[tuple[int, int]] = Counter()
        self.retention_counts: Counter[str] = Counter()
        self.observed_step_count = 0
        self.retained_step_count = 0
        self.cumulative_reward = 0.0
        self.global_step = 0
        self._episode: _EpisodeState | None = None

    @property
    def base(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "seed": self.seed,
            "agent": self.agent,
            "environment": self.environment,
            **self.common,
        }

    @staticmethod
    def _has_structural_event(info: Mapping[str, Any] | None) -> bool:
        if not info:
            return False
        for key in ("structural_events", "structural_event", "wall_events", "event", "events"):
            value = info.get(key)
            if value not in (None, "", (), [], {}, "[]", "null", "none", ["none"]):
                return True
        return False

    def _retention_reason(
        self,
        *,
        global_step: int,
        terminated: bool,
        truncated: bool,
        info: Mapping[str, Any] | None,
    ) -> tuple[str | None, float]:
        if self.keep_terminal_steps and (terminated or truncated):
            return "terminal", 1.0
        if self.keep_event_steps and self._has_structural_event(info):
            return "event", 1.0
        if self.step_retention_mode == "all":
            return "all", 1.0
        if self.step_retention_mode == "none":
            return None, 0.0
        payload = f"{self.trial_id}:{global_step}:{self.retention_salt}".encode()
        draw = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / 2**64
        if draw < self.step_sample_fraction:
            return "sample", self.step_sample_fraction
        return None, self.step_sample_fraction

    def start_episode(
        self,
        episode: int,
        state: int,
        info: Mapping[str, Any] | None = None,
        *,
        latent_state: int | None = None,
    ) -> None:
        if self._episode is not None:
            raise RuntimeError("finish_episode must be called before starting another episode")
        self._episode = _EpisodeState(episode=episode)
        self.state_visits[int(state)] += 1
        if latent_state is not None:
            self.latent_state_visits[int(latent_state)] += 1

    def record_step(
        self,
        *,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        terminated: bool,
        truncated: bool,
        info: Mapping[str, Any] | None = None,
        update: Any = None,
        latent_state: int | None = None,
        next_latent_state: int | None = None,
    ) -> None:
        episode = self._episode
        if episode is None:
            raise RuntimeError("start_episode must be called before record_step")
        state, action, next_state = int(state), int(action), int(next_state)
        reward = float(reward)
        self.global_step += 1
        self.observed_step_count += 1
        episode.length += 1
        episode.episode_return += reward
        self.cumulative_reward += reward
        self.state_visits[next_state] += 1
        if next_latent_state is not None:
            self.latent_state_visits[int(next_latent_state)] += 1
        self.state_action_visits[(state, action)] += 1
        transition = (state, action, next_state)
        self.transition_counts[transition] += 1
        self.reward_sums[transition] += reward
        pair = (state, action)
        self.state_action_reward_sums[pair] += reward
        self.reward_square_sums[pair] += reward * reward

        updates = update_fields(update)
        td_error = updates.get("td_error")
        if td_error is not None and np.isfinite(float(td_error)):
            td = float(td_error)
            episode.td_sum += td
            episode.td_abs_sum += abs(td)
            episode.td_square_sum += td * td
            episode.td_count += 1
            self.td_sums[pair] += td
            self.td_square_sums[pair] += td * td
            self.td_abs_sums[pair] += abs(td)
            self.td_counts[pair] += 1
            updates.update({"absolute_td_error": abs(td), "squared_td_error": td * td})
        for key, sum_name, count_name in (
            ("alpha", "alpha_sum", "alpha_count"),
            ("epsilon", "epsilon_sum", "epsilon_count"),
        ):
            value = updates.get(key)
            if value is not None:
                setattr(episode, sum_name, getattr(episode, sum_name) + float(value))
                setattr(episode, count_name, getattr(episode, count_name) + 1)

        pair_count = self.state_action_visits[(state, action)]
        transition_count = self.transition_counts[transition]
        row = {
            **self.base,
            "episode": episode.episode,
            "step": episode.length - 1,
            "global_step": self.global_step,
            "observed_state": state,
            "state": state,
            "action": action,
            "reward": reward,
            "next_observed_state": next_state,
            "next_state": next_state,
            "latent_state": latent_state,
            "next_latent_state": next_latent_state,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "state_visit_count": self.state_visits[state],
            "next_state_visit_count": self.state_visits[next_state],
            "state_action_visit_count": pair_count,
            "transition_count": transition_count,
            "empirical_observation_transition_probability": transition_count / pair_count,
            "empirical_transition_probability": transition_count / pair_count,
            "empirical_reward_mean": self.reward_sums[transition] / transition_count,
            "cumulative_reward": self.cumulative_reward,
            **updates,
            **diagnostic_fields(info),
        }
        reason, sampling_probability = self._retention_reason(
            global_step=self.global_step,
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=info,
        )
        self.retention_counts[reason or "discarded"] += 1
        if reason is not None:
            row["retention_reason"] = reason
            row["sampling_probability"] = sampling_probability
            self.step_rows.append(row)
            self.retained_step_count += 1

    def finish_episode(
        self,
        *,
        terminated: bool,
        truncated: bool,
        final_info: Mapping[str, Any] | None = None,
        success: bool | None = None,
        failure: bool | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        episode = self._episode
        if episode is None:
            raise RuntimeError("No episode is active")
        final_info = final_info or {}
        if success is None:
            success = bool(final_info.get("success", final_info.get("is_success", False)))
        if failure is None:
            failure = bool(final_info.get("failure", terminated and not success))
        action_counts = np.asarray(list(self.state_action_visits.values()), dtype=float)
        # Per-episode entropy is supplied by the runner when it tracks action counts.
        td_mean = episode.td_sum / episode.td_count if episode.td_count else float("nan")
        if episode.td_count > 1:
            td_variance = (episode.td_square_sum - episode.td_count * td_mean**2) / (
                episode.td_count - 1
            )
        elif episode.td_count == 1:
            td_variance = 0.0
        else:
            td_variance = float("nan")
        row = {
            **self.base,
            "episode": episode.episode,
            "episode_return": episode.episode_return,
            "episode_length": episode.length,
            "success": bool(success),
            "failure": bool(failure),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "cumulative_reward": self.cumulative_reward,
            "mean_td_error": td_mean,
            "mean_absolute_td_error": (
                episode.td_abs_sum / episode.td_count if episode.td_count else float("nan")
            ),
            "td_error_variance": max(0.0, td_variance) if np.isfinite(td_variance) else td_variance,
            "mean_learning_rate": (
                episode.alpha_sum / episode.alpha_count if episode.alpha_count else float("nan")
            ),
            "mean_exploration_rate": (
                episode.epsilon_sum / episode.epsilon_count
                if episode.epsilon_count
                else float("nan")
            ),
            "visited_states": len(self.state_visits),
            "visited_latent_states": len(self.latent_state_visits),
            "visited_state_actions": len(self.state_action_visits),
            "total_action_count": int(np.sum(action_counts)) if action_counts.size else 0,
            "observed_step_count": self.observed_step_count,
            "retained_step_count": self.retained_step_count,
            **diagnostic_fields(final_info),
            **dict(extra or {}),
        }
        self.episode_rows.append(row)
        self._episode = None
        return row

    def record_snapshot(
        self,
        episode: int,
        q_values: np.ndarray | None,
        diagnostics: Mapping[str, Any] | None = None,
        *,
        snapshot_key: str | None = None,
    ) -> None:
        key = snapshot_key or f"episode_{episode:08d}"
        if key in self._snapshot_keys:
            raise ValueError(f"snapshot key {key!r} was already recorded")
        if self.global_step in self._snapshot_global_steps:
            raise ValueError(f"a snapshot was already recorded at global step {self.global_step}")
        self._snapshot_keys.add(key)
        self._snapshot_global_steps.add(self.global_step)
        if q_values is not None:
            self.q_snapshots[key] = np.asarray(q_values, dtype=float).copy()
        self.snapshot_rows.append(
            {
                **self.base,
                "episode": episode,
                "global_step": self.global_step,
                "snapshot_key": key,
                **dict(diagnostics or {}),
            }
        )

        for (state, action), count in sorted(self.state_action_visits.items()):
            reward_sum = self.state_action_reward_sums[(state, action)]
            reward_square_sum = self.reward_square_sums[(state, action)]
            reward_mean = reward_sum / count
            reward_variance = (
                max(0.0, (reward_square_sum - count * reward_mean**2) / (count - 1))
                if count > 1
                else 0.0
            )
            td_count = self.td_counts[(state, action)]
            td_mean = self.td_sums[(state, action)] / td_count if td_count else float("nan")
            td_variance = (
                max(
                    0.0,
                    (self.td_square_sums[(state, action)] - td_count * td_mean**2) / (td_count - 1),
                )
                if td_count > 1
                else (0.0 if td_count == 1 else float("nan"))
            )
            self.state_action_rows.append(
                {
                    **self.base,
                    "episode": episode,
                    "global_step": self.global_step,
                    "observed_state": state,
                    "state": state,
                    "action": action,
                    "visit_count": count,
                    "reward_mean": reward_mean,
                    "reward_variance": reward_variance,
                    "td_count": td_count,
                    "td_error_mean": td_mean,
                    "td_error_variance": td_variance,
                    "mean_absolute_td_error": (
                        self.td_abs_sums[(state, action)] / td_count if td_count else float("nan")
                    ),
                }
            )

    def drain_rows(self, table: str) -> list[dict[str, Any]]:
        """Return and clear one bounded row buffer for streaming persistence."""

        buffers = {
            "training_episodes": self.episode_rows,
            "episodes": self.episode_rows,
            "steps": self.step_rows,
            "snapshots": self.snapshot_rows,
            "state_actions": self.state_action_rows,
        }
        if table not in buffers:
            raise KeyError(table)
        rows = list(buffers[table])
        buffers[table].clear()
        return rows

    def drain_q_snapshots(self) -> dict[str, np.ndarray]:
        snapshots = self.q_snapshots
        self.q_snapshots = {}
        return snapshots

    def state_action_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.state_action_rows)

    def frames(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return (
            pd.DataFrame(self.episode_rows),
            pd.DataFrame(self.step_rows),
            pd.DataFrame(self.snapshot_rows),
        )
