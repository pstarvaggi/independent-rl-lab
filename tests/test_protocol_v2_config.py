from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from rllab.experiments import AgentSpec, EnvironmentSpec, ExperimentConfig
from rllab.experiments.config import (
    ArtifactSpec,
    EvaluationScenario,
    ExecutionSpec,
    PolicyEvaluationSpec,
    StepRetentionSpec,
)
from rllab.experiments.preflight import estimate_run

ROOT = Path(__file__).resolve().parents[1]
FULL_CONFIG = ROOT / "configs" / "stochasticity_sweep.yaml"
SHORTCUT_CONFIGS = (
    ROOT / "configs" / "shortcut_or_shelter_recoverable.yaml",
    ROOT / "configs" / "shortcut_or_shelter_lethal.yaml",
)


def _v2_document(*, output_dir: str = "results") -> dict[str, Any]:
    return {
        "config_schema_version": 2,
        "experiment": {
            "name": "protocol-contract",
            "episodes": 20,
            "seeds": [2, 7],
            "snapshot_interval": 5,
            "exact_reference": True,
            "environment": {
                "name": "tiny-maze",
                "kind": "stochastic_maze",
                "parameters": {
                    "shape": [2, 3],
                    "start": [1, 0],
                    "goals": [[0, 2]],
                    "max_episode_steps": 40,
                },
            },
            "agent": {
                "name": "learner",
                "kind": "q_learning",
                "parameters": {"learning_rate": 0.1, "gamma": 0.98},
            },
            "sweep": {"environment.action_reliability": [1.0, 0.8]},
        },
        "policy_evaluation": {
            "enabled": True,
            "interval_episodes": 10,
            "episodes_per_checkpoint": 3,
            "include_initial": True,
            "include_final": True,
        },
        "execution": {"parallel_workers": 2, "failure_policy": "continue"},
        "artifacts": {
            "output_dir": output_dir,
            "table_format": "csv",
            "flush_rows": 250,
            "save_q_snapshots": False,
            "step_retention": {
                "mode": "sample",
                "fraction": 0.25,
                "keep_terminal": True,
                "keep_events": True,
            },
        },
    }


def test_v2_yaml_roundtrip_and_repository_relative_output_resolution(tmp_path: Path) -> None:
    project = tmp_path / "project"
    configs = project / "nested" / "configs"
    configs.mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    source = configs / "experiment.yaml"
    source.write_text(yaml.safe_dump(_v2_document(), sort_keys=False), encoding="utf-8")

    loaded = ExperimentConfig.from_yaml(source)
    assert loaded.output_dir == project / "results"
    assert loaded.config_schema_version == 2
    assert loaded.execution.failure_policy == "continue"
    assert loaded.artifacts.step_retention.fraction == 0.25

    roundtrip_path = configs / "roundtrip.yaml"
    roundtrip_path.write_text(yaml.safe_dump(loaded.as_dict(), sort_keys=False), encoding="utf-8")
    roundtripped = ExperimentConfig.from_yaml(roundtrip_path)
    assert roundtripped == loaded
    assert roundtripped.as_dict() == loaded.as_dict()


def test_trial_identities_ignore_execution_and_artifact_capture_changes(tmp_path: Path) -> None:
    base = ExperimentConfig.from_mapping(_v2_document())
    operationally_changed = replace(
        base,
        execution=ExecutionSpec(parallel_workers=9, failure_policy="fail_fast"),
        artifacts=ArtifactSpec(
            output_dir=tmp_path / "elsewhere",
            table_format="parquet",
            flush_rows=17,
            save_q_snapshots=True,
            step_retention=StepRetentionSpec(mode="none", keep_terminal=False, keep_events=False),
        ),
    )

    original_ids = [
        (trial.scenario_id, trial.condition_id, trial.trial_id) for trial in base.trials()
    ]
    changed_ids = [
        (trial.scenario_id, trial.condition_id, trial.trial_id)
        for trial in operationally_changed.trials()
    ]
    assert changed_ids == original_ids
    assert operationally_changed.parallel_workers == 9
    assert operationally_changed.output_dir == tmp_path / "elsewhere"

    longer_training = replace(base, episodes=base.episodes + 1)
    assert [trial.scenario_id for trial in longer_training.trials()] == [
        trial.scenario_id for trial in base.trials()
    ]
    assert [trial.condition_id for trial in longer_training.trials()] != [
        trial.condition_id for trial in base.trials()
    ]


def test_interaction_budget_roundtrips_and_defines_training_identity() -> None:
    document = _v2_document()
    document["experiment"]["total_interaction_steps"] = 1_234
    document["experiment"]["snapshot_step_interval"] = 125
    document["policy_evaluation"]["scenarios"] = [
        {
            "name": "frozen-greedy",
            "policy_mode": "greedy",
            "seed_group": "deployment-panel",
        },
        {
            "name": "continuing-behavior",
            "policy_mode": "behavior",
            "seed_group": "deployment-panel",
        },
    ]

    config = ExperimentConfig.from_mapping(document)
    assert config.total_interaction_steps == 1_234
    assert config.snapshot_step_interval == 125
    assert config.as_dict()["experiment"]["total_interaction_steps"] == 1_234
    assert config.as_dict()["experiment"]["snapshot_step_interval"] == 125
    assert [scenario.policy_mode for scenario in config.policy_evaluation.scenarios] == [
        "greedy",
        "behavior",
    ]
    assert {scenario.resolved_seed_group for scenario in config.policy_evaluation.scenarios} == {
        "deployment-panel"
    }
    assert all(trial.total_interaction_steps == 1_234 for trial in config.trials())
    assert all(trial.snapshot_step_interval == 125 for trial in config.trials())

    # In interaction-budget mode, the legacy episode setting is not a stopping
    # condition and therefore must not perturb scientific trial identity.
    changed_episode_default = replace(config, episodes=config.episodes + 99)
    assert [trial.condition_id for trial in changed_episode_default.trials()] == [
        trial.condition_id for trial in config.trials()
    ]
    changed_budget = replace(config, total_interaction_steps=1_235)
    assert [trial.condition_id for trial in changed_budget.trials()] != [
        trial.condition_id for trial in config.trials()
    ]
    changed_snapshot_capture = replace(config, snapshot_step_interval=50)
    assert [trial.trial_id for trial in changed_snapshot_capture.trials()] == [
        trial.trial_id for trial in config.trials()
    ]

    with pytest.raises(ValueError, match="snapshot_step_interval"):
        replace(config, snapshot_step_interval=0)


def test_policy_evaluation_scenarios_validate_mode_name_and_seed_group() -> None:
    with pytest.raises(ValueError, match="policy_mode"):
        EvaluationScenario(policy_mode="unsupported")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="name"):
        EvaluationScenario(name="")
    with pytest.raises(ValueError, match="seed_group"):
        EvaluationScenario(seed_group="")
    with pytest.raises(ValueError, match="names must be unique"):
        PolicyEvaluationSpec(
            scenarios=(
                EvaluationScenario(name="same", policy_mode="greedy"),
                EvaluationScenario(name="same", policy_mode="behavior"),
            )
        )


def test_interaction_budget_preflight_is_exact_in_steps() -> None:
    config = ExperimentConfig(
        name="step-budget",
        episodes=1,
        total_interaction_steps=37,
        seeds=(1, 2),
        environments=(EnvironmentSpec(name="unbounded", parameters={}),),
        agents=(AgentSpec(name="q", kind="q_learning"),),
        exact_reference=False,
        policy_evaluation=PolicyEvaluationSpec(enabled=True),
    )
    estimate = estimate_run(config)

    assert estimate.training_budget_unit == "interaction_steps"
    assert estimate.training_budget_per_trial == 37
    assert estimate.training_episode_count is None
    assert estimate.training_interaction_step_count == 74
    assert estimate.maximum_transition_rows == 74
    assert estimate.evaluation_episode_count is None


def test_legacy_flat_mapping_normalizes_to_protocol_v2(tmp_path: Path) -> None:
    legacy = {
        "name": "legacy",
        "episodes": 3,
        "seeds": 2,
        "parallel_workers": 4,
        "output_dir": str(tmp_path / "legacy-results"),
        "record_steps": False,
        "save_q_snapshots": False,
        "exact_evaluation": False,
        "environment": {
            "name": "old-maze",
            "parameters": {"max_episode_steps": 7},
        },
        "agent": {"name": "old-q", "kind": "q_learning"},
    }
    with pytest.warns(DeprecationWarning, match="exact_evaluation"):
        config = ExperimentConfig.from_mapping(legacy)

    assert config.config_schema_version == 2
    assert config.seeds == (0, 1)
    assert config.parallel_workers == 4
    assert config.output_dir == tmp_path / "legacy-results"
    assert not config.record_steps
    assert not config.save_q_snapshots
    assert not config.exact_reference
    assert config.artifacts.step_retention.mode == "none"
    assert config.as_dict()["config_schema_version"] == 2
    assert "execution" in config.as_dict()
    assert "artifacts" in config.as_dict()


def test_direct_constructor_retains_v1_convenience_arguments(tmp_path: Path) -> None:
    config = ExperimentConfig(
        name="direct",
        episodes=4,
        seeds=(3, 5),
        environments=(EnvironmentSpec(name="direct-maze", parameters={"max_episode_steps": 9}),),
        agents=(AgentSpec(name="direct-q", kind="q_learning"),),
        output_dir=tmp_path,
        parallel_workers=3,
        save_q_snapshots=False,
        record_steps=False,
        exact_evaluation=False,
    )
    assert config.output_dir == tmp_path
    assert config.parallel_workers == 3
    assert not config.save_q_snapshots
    assert not config.record_steps
    assert not config.exact_reference
    assert len(config.trials()) == 2
    assert all(trial.step_retention.mode == "none" for trial in config.trials())

    with pytest.raises(ValueError, match="exact_reference or exact_evaluation"):
        ExperimentConfig(exact_reference=True, exact_evaluation=False)


def test_full_stochasticity_config_resource_estimate() -> None:
    config = ExperimentConfig.from_yaml(FULL_CONFIG)
    estimate = estimate_run(config)
    assert estimate.trial_count == 120
    assert estimate.condition_count == 6
    assert estimate.scenario_count == 6
    assert estimate.training_episode_count == 240_000
    assert estimate.evaluation_episode_count == 6_600
    assert estimate.maximum_transition_rows == 72_000_000
    assert estimate.estimated_retained_step_rows == 3_600_000
    assert estimate.step_retention_mode == "sample"
    assert estimate.workers == 4


def test_quick_stochasticity_config_resource_estimate() -> None:
    full = ExperimentConfig.from_yaml(FULL_CONFIG)
    quick = replace(full, seeds=full.seeds[:2], episodes=30).with_parallel_workers(1)
    estimate = estimate_run(quick)
    assert estimate.trial_count == 12
    assert estimate.condition_count == 6
    assert estimate.scenario_count == 6
    assert estimate.training_episode_count == 360
    assert estimate.evaluation_episode_count == 120
    assert estimate.maximum_transition_rows == 108_000
    assert estimate.estimated_retained_step_rows == 5_400
    assert estimate.step_retention_mode == "sample"
    assert estimate.workers == 1


@pytest.mark.parametrize("path", SHORTCUT_CONFIGS)
def test_shortcut_or_shelter_configs_pin_fair_budget_and_coverage(path: Path) -> None:
    config = ExperimentConfig.from_yaml(path)
    estimate = estimate_run(config)

    assert config.total_interaction_steps == 100_000
    assert len(config.seeds) == 20
    assert {agent.kind for agent in config.agents} == {
        "q_learning",
        "sarsa",
        "expected_sarsa",
    }
    assert all(float(agent.parameters["initial_q"]) == 8.0 for agent in config.agents)
    assert all(float(agent.parameters["epsilon"]) == 0.10 for agent in config.agents)
    assert all(float(agent.parameters["learning_rate"]) == 0.05 for agent in config.agents)
    assert {scenario.policy_mode for scenario in config.policy_evaluation.scenarios} == {
        "greedy",
        "behavior",
    }
    assert (
        len({scenario.resolved_seed_group for scenario in config.policy_evaluation.scenarios}) == 1
    )
    assert config.snapshot_step_interval == 2_500
    assert estimate.training_interaction_step_count == estimate.trial_count * 100_000
    assert estimate.estimated_retained_step_rows == 0
