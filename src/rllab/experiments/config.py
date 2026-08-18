"""Versioned experiment configuration and Cartesian sweep expansion.

Protocol v2 distinguishes scientific semantics from execution and artifact capture.
That keeps trial identities stable when only worker count, output location, or
diagnostic retention changes.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import yaml  # type: ignore[import-untyped]

CONFIG_SCHEMA_VERSION = 2
JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


def _jsonable(value: Any) -> JsonValue:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return _jsonable(value.item())
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    raise TypeError(f"Configuration value {value!r} is not JSON serializable")


def canonical_json(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))


def stable_identifier(prefix: str, value: Any, *, length: int = 12) -> str:
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:length]
    clean_prefix = "".join(char if char.isalnum() or char in "-_" else "-" for char in prefix)
    return f"{clean_prefix}-{digest}"


@dataclass(frozen=True, slots=True)
class EnvironmentSpec:
    """An environment implementation and its constructor parameters."""

    name: str = "maze"
    kind: str = "stochastic_maze"
    parameters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> EnvironmentSpec:
        known = {"name", "kind", "parameters"}
        extra = {key: item for key, item in value.items() if key not in known}
        return cls(
            name=str(value.get("name", value.get("kind", "maze"))),
            kind=str(value.get("kind", "stochastic_maze")),
            parameters={**dict(value.get("parameters", {})), **extra},
        )


@dataclass(frozen=True, slots=True)
class AgentSpec:
    """An agent implementation and its constructor parameters."""

    name: str
    kind: str
    parameters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> AgentSpec:
        if "kind" not in value and "name" not in value:
            raise ValueError("Each agent needs at least a 'kind' or 'name'")
        kind = str(value.get("kind", value.get("name")))
        known = {"name", "kind", "parameters"}
        extra = {key: item for key, item in value.items() if key not in known}
        return cls(
            name=str(value.get("name", kind)),
            kind=kind,
            parameters={**dict(value.get("parameters", {})), **extra},
        )


@dataclass(frozen=True, slots=True)
class EvaluationScenario:
    """One named held-out environment/policy view evaluated at every checkpoint.

    ``greedy`` disables exploration. ``behavior`` samples from the cloned
    agent's training-time behavior policy without updates, leaving schedule
    position fixed at the checkpoint. Scenarios that share ``seed_group`` reuse
    the same ordered environment seeds for paired contrasts.
    """

    name: str = "default"
    environment_overrides: dict[str, Any] = field(default_factory=dict)
    policy_mode: Literal["greedy", "behavior"] = "greedy"
    seed_group: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("evaluation scenario name must not be empty")
        if self.policy_mode not in {"greedy", "behavior"}:
            raise ValueError("evaluation scenario policy_mode must be 'greedy' or 'behavior'")
        if self.seed_group is not None and not self.seed_group:
            raise ValueError("evaluation scenario seed_group must not be empty")

    @property
    def resolved_seed_group(self) -> str:
        """Seed-panel identity, shared explicitly across paired scenarios."""

        return self.name if self.seed_group is None else self.seed_group

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> EvaluationScenario:
        known = {
            "name",
            "environment_overrides",
            "parameters",
            "policy_mode",
            "seed_group",
        }
        unknown = set(value) - known
        if unknown:
            raise ValueError(f"Unknown evaluation scenario keys: {sorted(unknown)}")
        overrides = value.get("environment_overrides", value.get("parameters", {}))
        return cls(
            name=str(value.get("name", "default")),
            environment_overrides=dict(overrides),
            policy_mode=str(value.get("policy_mode", "greedy")),  # type: ignore[arg-type]
            seed_group=(None if value.get("seed_group") is None else str(value.get("seed_group"))),
        )


@dataclass(frozen=True, slots=True)
class PolicyEvaluationSpec:
    """Update-free, paired-seed policy evaluation configuration."""

    enabled: bool = False
    interval_episodes: int = 50
    episodes_per_checkpoint: int = 10
    include_initial: bool = False
    include_final: bool = True
    scenarios: tuple[EvaluationScenario, ...] = (EvaluationScenario(),)

    def __post_init__(self) -> None:
        if self.interval_episodes < 1:
            raise ValueError("policy evaluation interval_episodes must be positive")
        if self.episodes_per_checkpoint < 1:
            raise ValueError("policy evaluation episodes_per_checkpoint must be positive")
        if not self.scenarios:
            raise ValueError("policy evaluation requires at least one scenario")
        names = [scenario.name for scenario in self.scenarios]
        if len(names) != len(set(names)):
            raise ValueError("policy evaluation scenario names must be unique")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> PolicyEvaluationSpec:
        if not value:
            return cls()
        raw = dict(value)
        scenarios = raw.pop("scenarios", None)
        if scenarios is not None:
            if isinstance(scenarios, Mapping):
                scenarios = [scenarios]
            raw["scenarios"] = tuple(EvaluationScenario.from_mapping(item) for item in scenarios)
        known = {
            "enabled",
            "interval_episodes",
            "episodes_per_checkpoint",
            "include_initial",
            "include_final",
            "scenarios",
        }
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown policy_evaluation keys: {sorted(unknown)}")
        return cls(**raw)


StepRetentionMode = Literal["all", "none", "sample"]


@dataclass(frozen=True, slots=True)
class StepRetentionSpec:
    """Which interaction rows are persisted; online summaries still see all rows."""

    mode: StepRetentionMode = "all"
    fraction: float = 1.0
    keep_terminal: bool = True
    keep_events: bool = True
    salt: str = "rl-lab-step-retention-v2"

    def __post_init__(self) -> None:
        if self.mode not in {"all", "none", "sample"}:
            raise ValueError("step retention mode must be 'all', 'none', or 'sample'")
        if not 0.0 < self.fraction <= 1.0:
            raise ValueError("step retention fraction must lie in (0, 1]")
        if self.mode != "sample" and self.fraction != 1.0:
            raise ValueError("step retention fraction is only meaningful in sample mode")

    @classmethod
    def from_value(cls, value: bool | Mapping[str, Any] | None) -> StepRetentionSpec:
        if value is None:
            return cls()
        if isinstance(value, bool):
            return (
                cls(mode="all")
                if value
                else cls(mode="none", keep_terminal=False, keep_events=False)
            )
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class ArtifactSpec:
    """Artifact capture and storage settings, excluded from scientific identity."""

    output_dir: Path = Path("results")
    table_format: Literal["auto", "parquet", "csv"] = "auto"
    flush_rows: int = 10_000
    save_q_snapshots: bool = True
    step_retention: StepRetentionSpec = field(default_factory=StepRetentionSpec)

    def __post_init__(self) -> None:
        if self.table_format not in {"auto", "parquet", "csv"}:
            raise ValueError("table_format must be 'auto', 'parquet', or 'csv'")
        if self.flush_rows < 1:
            raise ValueError("flush_rows must be positive")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> ArtifactSpec:
        if not value:
            return cls()
        raw = dict(value)
        raw["output_dir"] = Path(raw.get("output_dir", "results"))
        raw["step_retention"] = StepRetentionSpec.from_value(raw.get("step_retention"))
        known = {"output_dir", "table_format", "flush_rows", "save_q_snapshots", "step_retention"}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown artifact keys: {sorted(unknown)}")
        return cls(**raw)


@dataclass(frozen=True, slots=True)
class ExecutionSpec:
    """Process orchestration settings, excluded from scientific identity."""

    parallel_workers: int = 1
    failure_policy: Literal["fail_fast", "continue"] = "fail_fast"

    def __post_init__(self) -> None:
        if self.parallel_workers < 1:
            raise ValueError("parallel_workers must be positive")
        if self.failure_policy not in {"fail_fast", "continue"}:
            raise ValueError("failure_policy must be 'fail_fast' or 'continue'")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> ExecutionSpec:
        if not value:
            return cls()
        raw = dict(value)
        known = {"parallel_workers", "failure_policy"}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown execution keys: {sorted(unknown)}")
        return cls(**raw)


@dataclass(frozen=True, slots=True)
class TrialSpec:
    """One fully resolved environment-agent-seed run."""

    experiment_name: str
    environment: EnvironmentSpec
    agent: AgentSpec
    seed: int
    episodes: int
    total_interaction_steps: int | None
    max_steps: int | None
    snapshot_interval: int
    snapshot_step_interval: int | None
    step_retention: StepRetentionSpec
    exact_reference: bool
    policy_evaluation: PolicyEvaluationSpec
    tags: dict[str, Any] = field(default_factory=dict)
    sweep_values: dict[str, Any] = field(default_factory=dict)

    @property
    def scenario_id(self) -> str:
        return stable_identifier(
            self.environment.name,
            {"environment": asdict(self.environment), "max_steps": self.max_steps},
        )

    @property
    def condition_id(self) -> str:
        budget_identity = (
            {"total_interaction_steps": self.total_interaction_steps}
            if self.total_interaction_steps is not None
            else {"episodes": self.episodes}
        )
        return stable_identifier(
            self.agent.name,
            {
                "scenario_id": self.scenario_id,
                "agent": asdict(self.agent),
                **budget_identity,
            },
        )

    @property
    def trial_id(self) -> str:
        return stable_identifier(
            f"{self.environment.name}-{self.agent.name}-s{self.seed}",
            {"condition_id": self.condition_id, "seed": self.seed},
        )


@dataclass(frozen=True, slots=True, init=False)
class ExperimentConfig:
    """Protocol-v2 experiment definition with separate execution/capture settings.

    ``total_interaction_steps`` replaces ``episodes`` as the stopping condition
    when present. The final partial episode is runner-truncated at the exact
    interaction boundary. ``snapshot_step_interval`` adds exact interaction-step
    checkpoints to the legacy episode-indexed snapshot schedule.
    """

    name: str = "experiment"
    episodes: int = 500
    total_interaction_steps: int | None = None
    seeds: tuple[int, ...] = (0,)
    environments: tuple[EnvironmentSpec, ...] = (EnvironmentSpec(),)
    agents: tuple[AgentSpec, ...] = (AgentSpec(name="q_learning", kind="q_learning"),)
    sweep: dict[str, tuple[Any, ...]] = field(default_factory=dict)
    max_steps: int | None = None
    snapshot_interval: int = 10
    snapshot_step_interval: int | None = None
    exact_reference: bool = True
    policy_evaluation: PolicyEvaluationSpec = field(default_factory=PolicyEvaluationSpec)
    execution: ExecutionSpec = field(default_factory=ExecutionSpec)
    artifacts: ArtifactSpec = field(default_factory=ArtifactSpec)
    tags: dict[str, Any] = field(default_factory=dict)
    config_schema_version: int = CONFIG_SCHEMA_VERSION

    def __init__(
        self,
        name: str = "experiment",
        episodes: int = 500,
        total_interaction_steps: int | None = None,
        seeds: tuple[int, ...] = (0,),
        environments: tuple[EnvironmentSpec, ...] = (EnvironmentSpec(),),
        agents: tuple[AgentSpec, ...] = (AgentSpec(name="q_learning", kind="q_learning"),),
        sweep: dict[str, tuple[Any, ...]] | None = None,
        max_steps: int | None = None,
        snapshot_interval: int = 10,
        snapshot_step_interval: int | None = None,
        exact_reference: bool | None = None,
        policy_evaluation: PolicyEvaluationSpec | None = None,
        execution: ExecutionSpec | None = None,
        artifacts: ArtifactSpec | None = None,
        tags: dict[str, Any] | None = None,
        config_schema_version: int = CONFIG_SCHEMA_VERSION,
        *,
        output_dir: str | Path | None = None,
        parallel_workers: int | None = None,
        save_q_snapshots: bool | None = None,
        record_steps: bool | None = None,
        exact_evaluation: bool | None = None,
    ) -> None:
        """Build a v2 config, accepting the v1 flat conveniences for one cycle."""

        resolved_execution = execution or ExecutionSpec()
        resolved_artifacts = artifacts or ArtifactSpec()
        if parallel_workers is not None:
            resolved_execution = replace(
                resolved_execution,
                parallel_workers=parallel_workers,
            )
        if output_dir is not None:
            resolved_artifacts = replace(resolved_artifacts, output_dir=Path(output_dir))
        if save_q_snapshots is not None:
            resolved_artifacts = replace(
                resolved_artifacts,
                save_q_snapshots=save_q_snapshots,
            )
        if record_steps is not None:
            resolved_artifacts = replace(
                resolved_artifacts,
                step_retention=StepRetentionSpec.from_value(record_steps),
            )
        if exact_reference is not None and exact_evaluation is not None:
            raise ValueError("Specify exact_reference or exact_evaluation, not both")
        resolved_exact = (
            bool(exact_reference)
            if exact_reference is not None
            else (True if exact_evaluation is None else bool(exact_evaluation))
        )
        values = {
            "name": name,
            "episodes": episodes,
            "total_interaction_steps": total_interaction_steps,
            "seeds": tuple(seeds),
            "environments": tuple(environments),
            "agents": tuple(agents),
            "sweep": dict(sweep or {}),
            "max_steps": max_steps,
            "snapshot_interval": snapshot_interval,
            "snapshot_step_interval": snapshot_step_interval,
            "exact_reference": resolved_exact,
            "policy_evaluation": policy_evaluation or PolicyEvaluationSpec(),
            "execution": resolved_execution,
            "artifacts": resolved_artifacts,
            "tags": dict(tags or {}),
            "config_schema_version": config_schema_version,
        }
        for key, value in values.items():
            object.__setattr__(self, key, value)
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.config_schema_version != CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported config schema {self.config_schema_version}; expected {CONFIG_SCHEMA_VERSION}"
            )
        if self.episodes < 1:
            raise ValueError("episodes must be positive")
        if self.total_interaction_steps is not None and self.total_interaction_steps < 1:
            raise ValueError("total_interaction_steps must be positive when provided")
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if not self.environments or not self.agents:
            raise ValueError("at least one environment and agent are required")
        if self.snapshot_interval < 1:
            raise ValueError("snapshot_interval must be positive")
        if self.snapshot_step_interval is not None and self.snapshot_step_interval < 1:
            raise ValueError("snapshot_step_interval must be positive when provided")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("max_steps must be positive when provided")
        for key, values in self.sweep.items():
            if not values:
                raise ValueError(f"sweep dimension {key!r} is empty")

    @property
    def parallel_workers(self) -> int:
        return self.execution.parallel_workers

    @property
    def output_dir(self) -> Path:
        return self.artifacts.output_dir

    @property
    def save_q_snapshots(self) -> bool:
        return self.artifacts.save_q_snapshots

    @property
    def record_steps(self) -> bool:
        return self.artifacts.step_retention.mode != "none"

    @property
    def exact_evaluation(self) -> bool:
        """Deprecated spelling retained for one compatibility cycle."""

        return self.exact_reference

    def with_output_dir(self, output_dir: str | Path) -> ExperimentConfig:
        return replace(self, artifacts=replace(self.artifacts, output_dir=Path(output_dir)))

    def with_parallel_workers(self, workers: int) -> ExperimentConfig:
        return replace(self, execution=replace(self.execution, parallel_workers=workers))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ExperimentConfig:
        """Parse protocol-v2 documents and normalize legacy flat documents."""

        document = dict(value)
        is_v2_document = any(
            key in document
            for key in ("config_schema_version", "execution", "artifacts", "policy_evaluation")
        )
        raw = dict(document["experiment"]) if "experiment" in document else dict(document)
        schema_version = int(document.get("config_schema_version", CONFIG_SCHEMA_VERSION))
        if schema_version != CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported config schema {schema_version}; expected {CONFIG_SCHEMA_VERSION}"
            )

        env_values = raw.pop("environments", raw.pop("environment", [{}]))
        agent_values = raw.pop("agents", raw.pop("agent", [{"kind": "q_learning"}]))
        if isinstance(env_values, Mapping):
            env_values = [env_values]
        if isinstance(agent_values, Mapping):
            agent_values = [agent_values]

        sweep_value = raw.pop("sweep", {}) or {}
        sweep = {
            str(key): tuple(items if isinstance(items, list) else [items])
            for key, items in sweep_value.items()
        }
        seeds_value = raw.pop("seeds", [0])
        seeds = (
            tuple(range(seeds_value))
            if isinstance(seeds_value, int)
            else tuple(int(seed) for seed in seeds_value)
        )

        execution_raw = dict(document.get("execution", {})) if is_v2_document else {}
        artifact_raw = dict(document.get("artifacts", {})) if is_v2_document else {}
        policy_raw = document.get("policy_evaluation", raw.pop("policy_evaluation", None))

        legacy_workers = raw.pop("parallel_workers", None)
        if legacy_workers is not None:
            if "parallel_workers" in execution_raw:
                raise ValueError("Specify parallel_workers in only one configuration section")
            execution_raw["parallel_workers"] = legacy_workers
        for legacy_key in ("output_dir", "save_q_snapshots"):
            if legacy_key in raw:
                if legacy_key in artifact_raw:
                    raise ValueError(f"Specify {legacy_key} in only one configuration section")
                artifact_raw[legacy_key] = raw.pop(legacy_key)
        if "record_steps" in raw:
            if "step_retention" in artifact_raw:
                raise ValueError("Specify record_steps or artifacts.step_retention, not both")
            artifact_raw["step_retention"] = raw.pop("record_steps")

        exact_reference = raw.pop("exact_reference", None)
        legacy_exact = raw.pop("exact_evaluation", None)
        if exact_reference is not None and legacy_exact is not None:
            raise ValueError("Specify exact_reference or exact_evaluation, not both")
        if exact_reference is None:
            exact_reference = True if legacy_exact is None else bool(legacy_exact)

        known = {
            "name",
            "episodes",
            "total_interaction_steps",
            "max_steps",
            "snapshot_interval",
            "snapshot_step_interval",
            "tags",
        }
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown experiment configuration keys: {sorted(unknown)}")
        if "experiment" in document:
            top_level_known = {
                "config_schema_version",
                "experiment",
                "execution",
                "artifacts",
                "policy_evaluation",
            }
            top_unknown = set(document) - top_level_known
            if top_unknown:
                raise ValueError(f"Unknown top-level configuration keys: {sorted(top_unknown)}")
        elif not is_v2_document and legacy_exact is not None:
            warnings.warn(
                "exact_evaluation is a legacy name; use exact_reference in schema v2",
                DeprecationWarning,
                stacklevel=2,
            )

        return cls(
            environments=tuple(EnvironmentSpec.from_mapping(item) for item in env_values),
            agents=tuple(AgentSpec.from_mapping(item) for item in agent_values),
            seeds=seeds,
            sweep=sweep,
            exact_reference=bool(exact_reference),
            execution=ExecutionSpec.from_mapping(execution_raw),
            artifacts=ArtifactSpec.from_mapping(artifact_raw),
            policy_evaluation=PolicyEvaluationSpec.from_mapping(policy_raw),
            config_schema_version=schema_version,
            **raw,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        source = Path(path).resolve()
        with source.open(encoding="utf-8") as stream:
            value = yaml.safe_load(stream)
        if not isinstance(value, Mapping):
            raise ValueError("The YAML document must contain a mapping")
        config = cls.from_mapping(value)
        if not config.output_dir.is_absolute():
            candidate = source.parent
            while candidate != candidate.parent and not (candidate / "pyproject.toml").exists():
                candidate = candidate.parent
            root = candidate if (candidate / "pyproject.toml").exists() else source.parent
            config = config.with_output_dir(root / config.output_dir)
        return config

    def as_dict(self) -> dict[str, Any]:
        experiment: dict[str, Any] = {
            "name": self.name,
            "episodes": self.episodes,
            "seeds": self.seeds,
            "environments": self.environments,
            "agents": self.agents,
            "sweep": self.sweep,
            "max_steps": self.max_steps,
            "snapshot_interval": self.snapshot_interval,
            "exact_reference": self.exact_reference,
            "tags": self.tags,
        }
        if self.total_interaction_steps is not None:
            experiment["total_interaction_steps"] = self.total_interaction_steps
        if self.snapshot_step_interval is not None:
            experiment["snapshot_step_interval"] = self.snapshot_step_interval
        value = {
            "config_schema_version": self.config_schema_version,
            "experiment": experiment,
            "policy_evaluation": self.policy_evaluation,
            "execution": self.execution,
            "artifacts": self.artifacts,
        }
        return _jsonable(value)  # type: ignore[return-value]

    def trials(self) -> list[TrialSpec]:
        dimensions = sorted(self.sweep)
        combinations: Iterable[tuple[Any, ...]] = (
            itertools.product(*(self.sweep[key] for key in dimensions)) if dimensions else [()]
        )
        trials: list[TrialSpec] = []
        for combination in combinations:
            sweep_values = dict(zip(dimensions, combination, strict=True))
            for environment, agent, seed in itertools.product(
                self.environments, self.agents, self.seeds
            ):
                resolved_environment = environment
                resolved_agent = agent
                for key, item in sweep_values.items():
                    prefix, separator, parameter = key.partition(".")
                    if not separator:
                        prefix, parameter = "environment", prefix
                    if prefix in {"environment", "env"}:
                        resolved_environment = replace(
                            resolved_environment,
                            parameters={**resolved_environment.parameters, parameter: item},
                        )
                    elif prefix == "agent":
                        resolved_agent = replace(
                            resolved_agent,
                            parameters={**resolved_agent.parameters, parameter: item},
                        )
                    else:
                        raise ValueError(
                            f"Sweep key {key!r} must begin with 'environment.' or 'agent.'"
                        )
                trials.append(
                    TrialSpec(
                        experiment_name=self.name,
                        environment=resolved_environment,
                        agent=resolved_agent,
                        seed=seed,
                        episodes=self.episodes,
                        total_interaction_steps=self.total_interaction_steps,
                        max_steps=self.max_steps,
                        snapshot_interval=self.snapshot_interval,
                        snapshot_step_interval=self.snapshot_step_interval,
                        step_retention=self.artifacts.step_retention,
                        exact_reference=self.exact_reference,
                        policy_evaluation=self.policy_evaluation,
                        tags=self.tags,
                        sweep_values=sweep_values,
                    )
                )
        identifiers = [trial.trial_id for trial in trials]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Expanded trials are not unique; check repeated seeds/specifications")
        return trials


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "AgentSpec",
    "ArtifactSpec",
    "EnvironmentSpec",
    "EvaluationScenario",
    "ExecutionSpec",
    "ExperimentConfig",
    "PolicyEvaluationSpec",
    "StepRetentionSpec",
    "TrialSpec",
    "canonical_json",
    "stable_identifier",
]
