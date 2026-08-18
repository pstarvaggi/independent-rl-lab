from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from rllab.experiments.artifacts import (
    RunStore,
    UnsupportedArtifactVersion,
    load_run_manifest,
)
from rllab.experiments.schema import (
    ARTIFACT_SCHEMA_VERSION,
    ArtifactReference,
    RunManifest,
    TrialManifestEntry,
)


def test_schema_rejects_unsafe_paths_hashes_and_extra_fields() -> None:
    with pytest.raises(ValidationError, match="artifact paths"):
        ArtifactReference(
            path="../outside.csv",
            format="csv",
            bytes=0,
            sha256="0" * 64,
        )
    with pytest.raises(ValidationError, match="sha256"):
        ArtifactReference(path="table.csv", format="csv", bytes=0, sha256="BAD")
    with pytest.raises(ValidationError, match="extra"):
        ArtifactReference(
            path="table.csv",
            format="csv",
            bytes=0,
            sha256="0" * 64,
            mystery=True,  # type: ignore[call-arg]
        )


def test_manifest_requires_trial_keys_to_match_entries() -> None:
    entry = TrialManifestEntry(trial_id="trial-1", path="trials/trial-1")
    with pytest.raises(ValidationError, match="do not match"):
        RunManifest(
            run_id="run-1",
            experiment_name="study",
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
            table_format="csv",
            config_sha256="0" * 64,
            provenance_sha256="1" * 64,
            trials={"wrong": entry},
        )


def test_run_store_writes_a_versioned_plan_before_any_attempt(tmp_path: Path) -> None:
    store = RunStore.create(
        tmp_path / "run-1",
        run_id="run-1",
        experiment_name="study",
        trials={"trial-1": {"seed": 7}},
        config={"episodes": 3},
        provenance={"source_sha256": "a" * 64},
        table_format="csv",
    )

    manifest = load_run_manifest(store.run_directory)
    assert manifest.artifact_schema_version == ARTIFACT_SCHEMA_VERSION
    assert manifest.status == "planned"
    assert manifest.trials["trial-1"].status == "pending"
    assert json.loads((store.run_directory / "config.json").read_text()) == {"episodes": 3}
    descriptor = json.loads((store.run_directory / "trials" / "trial-1" / "trial.json").read_text())
    assert descriptor["specification"] == {"seed": 7}


def test_loader_refuses_unknown_artifact_versions(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "artifact_schema_version": 99,
                "document_type": "run_manifest",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(UnsupportedArtifactVersion, match="schema 99"):
        load_run_manifest(path)
