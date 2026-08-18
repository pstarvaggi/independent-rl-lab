"""Structural and quick executable checks for the generated notebooks."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import nbformat
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_checked_in_notebooks_match_generator() -> None:
    subprocess.run(
        [sys.executable, "scripts/build_notebooks.py", "--check"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_all_notebook_code_cells_parse() -> None:
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        assert all(cell.get("id") for cell in notebook.cells)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                assert cell.execution_count is None
                assert cell.outputs == []
                ast.parse(cell.source, filename=f"{path.name}:{cell.id}")


def test_primer_tabular_section_executes(tmp_path: Path) -> None:
    """Run every primer cell before optional PyTorch is imported."""

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    notebook = nbformat.read(ROOT / "notebooks" / "00_rl_primer.ipynb", as_version=4)
    namespace: dict[str, object] = {"__name__": "__notebook_smoke_test__"}
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        if "import torch" in cell.source:
            break
        exec(compile(cell.source, f"00_rl_primer.ipynb:{cell.id}", "exec"), namespace)
    assert "Q_star" in namespace
    assert "results" in namespace


def test_q_learning_protocol_v2_smoke(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    monkeypatch.setenv("RL_LAB_NOTEBOOK_SMOKE", "1")
    monkeypatch.setenv("RL_LAB_NOTEBOOK_RESULTS", str(tmp_path / "results"))
    notebook = nbformat.read(ROOT / "notebooks" / "02_q_learning_experiments.ipynb", as_version=4)
    q_learning_source = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )
    assert 'info.get("state_index"' not in q_learning_source
    namespace: dict[str, object] = {
        "__name__": "__main__",
        "display": lambda *_args, **_kwargs: None,
    }
    for cell in notebook.cells:
        if cell.cell_type == "code":
            exec(
                compile(cell.source, f"02_q_learning_experiments.ipynb:{cell.id}", "exec"),
                namespace,
            )

    config = namespace["config"]
    design = namespace["design"]
    result = namespace["result"]
    training = namespace["training_episodes"]
    evaluations = namespace["held_out_evaluations"]
    retention = namespace["retention_audit"]
    store = namespace["store"]

    assert config.config_schema_version == 2
    assert config.policy_evaluation.enabled
    assert config.artifacts.step_retention.mode == "sample"
    assert design["trial_id"].is_unique
    assert design["condition_id"].nunique() == 2
    assert {"scenario_id", "condition_id", "trial_id"}.issubset(training)
    assert {
        "condition_id",
        "trial_id",
        "evaluation_seed",
        "evaluation_scenario",
        "checkpoint_episode",
    }.issubset(evaluations)
    assert set(evaluations["evaluation_seed"]).isdisjoint(set(design["seed"]))
    assert result.run_directory.is_relative_to(tmp_path)
    assert store.manifest.artifact_schema_version == 2
    assert store.manifest.status == "complete"
    assert retention["retained_steps"].le(retention["observed_steps"]).all()
    namespace["plt"].close("all")


def test_risk_drift_memory_notebook_smoke(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    monkeypatch.setenv("RL_LAB_NOTEBOOK_SMOKE", "1")
    monkeypatch.setenv("RL_LAB_NOTEBOOK_RESULTS", str(tmp_path / "results"))
    notebook = nbformat.read(
        ROOT / "notebooks" / "04_policies_under_risk_drift_and_memory.ipynb",
        as_version=4,
    )
    namespace: dict[str, object] = {
        "__name__": "__main__",
        "display": lambda *_args, **_kwargs: None,
    }
    for cell in notebook.cells:
        if cell.cell_type == "code":
            exec(
                compile(
                    cell.source,
                    f"04_policies_under_risk_drift_and_memory.ipynb:{cell.id}",
                    "exec",
                ),
                namespace,
            )

    risk_result = namespace["risk_result"]
    risk_evaluations = risk_result.evaluations
    assert set(risk_result.training_episodes["agent"]) == {
        "q_learning",
        "sarsa",
        "expected_sarsa",
        "double_q_learning",
    }
    panels = {
        tuple(group.sort_values("evaluation_episode")["evaluation_seed"].tolist())
        for _, group in risk_evaluations.groupby(["agent", "checkpoint_episode"], sort=True)
    }
    assert len(panels) == 1
    assert len(namespace["risk_contrast_summary"]) == 3

    drift_result = namespace["drift_result"]
    assert drift_result.steps["env_action_reliability"].nunique() == 2
    assert not drift_result.snapshots["exact_evaluation_available"].any()

    memory_result = namespace["memory_result"]
    availability = namespace["exact_availability"]
    assert not availability.query("representation == 'position only'")[
        "exact_evaluation_available"
    ].any()
    assert availability.query("representation == 'position + wall'")[
        "exact_evaluation_available"
    ].all()
    final_rows = namespace["memory_final_rows"]
    sample = final_rows.iloc[0]
    assert sample.snapshot_key in memory_result.q_snapshots(sample.trial_id)
    assert len(namespace["memory_contrast_summary"]) == 3
    assert all(
        result.run_directory.is_relative_to(tmp_path)
        for result in (
            risk_result,
            drift_result,
            memory_result,
        )
    )
    namespace["plt"].close("all")


def test_shortcut_or_shelter_notebook_smoke(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    monkeypatch.setenv("RL_LAB_NOTEBOOK_SMOKE", "1")
    monkeypatch.setenv("RL_LAB_NOTEBOOK_RESULTS", str(tmp_path / "results"))
    notebook = nbformat.read(
        ROOT / "notebooks" / "05_shortcut_or_shelter.ipynb",
        as_version=4,
    )
    namespace: dict[str, object] = {
        "__name__": "__main__",
        "display": lambda *_args, **_kwargs: None,
    }
    for cell in notebook.cells:
        if cell.cell_type == "code":
            exec(
                compile(
                    cell.source,
                    f"05_shortcut_or_shelter.ipynb:{cell.id}",
                    "exec",
                ),
                namespace,
            )

    main_design = namespace["main_design"]
    main_training = namespace["main_training"]
    main_evaluations = namespace["main_evaluations"]
    final_choices = namespace["final_choices"]
    exact_thresholds = namespace["exact_thresholds"]
    annealed_choices = namespace["annealed_choices"]
    backup_variance = namespace["backup_variance"]
    checkpoint_choices = namespace["checkpoint_choices"]
    endpoint_calibration = namespace["endpoint_calibration"]
    boundary_contrasts = namespace["boundary_contrasts"]

    assert set(main_design["hazard_mode"]) == {"recoverable", "lethal"}
    assert set(main_design["agent"]) == {
        "q_learning",
        "sarsa",
        "expected_sarsa",
    }
    assert set(main_evaluations["evaluation_policy_mode"]) == {"greedy", "behavior"}
    paired = main_evaluations.pivot_table(
        index=["trial_id", "evaluation_episode"],
        columns="evaluation_policy_mode",
        values="evaluation_seed",
        aggfunc="first",
    ).dropna()
    assert (paired["greedy"] == paired["behavior"]).all()
    assert main_training.groupby("trial_id")["observed_step_count"].max().eq(80).all()
    assert set(final_choices["choice"]).issubset({"corridor", "shelter", "other", "tie"})
    assert final_choices["exact_deployment_regret"].ge(0.0).all()
    assert set(annealed_choices["hazard_mode"]) == {"recoverable", "lethal"}
    assert checkpoint_choices["step_lag"].eq(0).all()
    assert (
        backup_variance["sarsa_variance"] >= backup_variance["expected_sarsa_variance"] - 1e-12
    ).all()
    assert len(endpoint_calibration) == 12
    assert len(boundary_contrasts) == 4

    recoverable_greedy = exact_thresholds.query("hazard_mode == 'recoverable' and epsilon == 0.0")[
        "threshold"
    ].iloc[0]
    lethal_greedy = exact_thresholds.query("hazard_mode == 'lethal' and epsilon == 0.0")[
        "threshold"
    ].iloc[0]
    lethal_soft = exact_thresholds.query("hazard_mode == 'lethal' and epsilon == 0.1")[
        "threshold"
    ].iloc[0]
    assert 0.75 < recoverable_greedy < 0.86
    assert 0.97 < lethal_greedy < 1.0
    assert np.isnan(lethal_soft)
    namespace["plt"].close("all")


def test_lunar_lander_continuous_contract() -> None:
    pytest.importorskip("Box2D")
    gym = pytest.importorskip("gymnasium")

    environment = gym.make("LunarLander-v3", continuous=True, enable_wind=False)
    try:
        observation, _ = environment.reset(seed=23)
        next_observation, reward, terminated, truncated, _ = environment.step(
            np.zeros(2, dtype=np.float32)
        )
        assert observation.shape == (8,)
        assert environment.action_space.shape == (2,)
        assert np.isfinite(observation).all()
        assert np.isfinite(next_observation).all()
        assert np.isfinite(reward)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
    finally:
        environment.close()


def test_lunar_lander_sac_smoke(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pytest.importorskip("Box2D")
    pytest.importorskip("torch")

    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("RL_LAB_NOTEBOOK_SMOKE", "1")
    notebook = nbformat.read(ROOT / "notebooks" / "03_lunar_lander_sac.ipynb", as_version=4)
    namespace: dict[str, object] = {"__name__": "__main__"}
    for cell in notebook.cells:
        if cell.cell_type == "code":
            exec(
                compile(cell.source, f"03_lunar_lander_sac.ipynb:{cell.id}", "exec"),
                namespace,
            )

    result = namespace["result"]
    showcase = namespace["showcase"]
    assert result.updates.shape[0] > 0
    assert result.evaluations["step"].max() == 320
    assert showcase.observations.shape[1] == 8
    assert showcase.actions.shape[1] == 2
    assert len(showcase.frames) > 1
    figure, movie = namespace["make_landing_animation"](showcase, fps=10, max_frames=12)
    animation_html = movie.to_jshtml(fps=10, default_mode="once")
    namespace["plt"].close(figure)
    assert "<script" in animation_html
