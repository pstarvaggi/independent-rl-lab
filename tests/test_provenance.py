from __future__ import annotations

import subprocess
from pathlib import Path

from rllab.experiments.persistence import provenance
from rllab.experiments.provenance import (
    collect_provenance,
    git_provenance,
    source_tree_hash,
    value_sha256,
)


def _write_source_tree(root: Path) -> Path:
    package = root / "src" / "rllab"
    package.mkdir(parents=True)
    source = package / "example.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    (root / "pyproject.toml").write_text('[project]\nname = "fixture"\n', encoding="utf-8")
    (root / "results").mkdir()
    return source


def _git(root: Path, *arguments: str) -> None:
    subprocess.run(["git", *arguments], cwd=root, check=True, capture_output=True, text=True)


def test_source_hash_tracks_relevant_content_but_not_results(tmp_path: Path) -> None:
    source = _write_source_tree(tmp_path)
    first_hash, first_files = source_tree_hash(tmp_path)
    assert {record.path for record in first_files} == {
        "pyproject.toml",
        "src/rllab/example.py",
    }

    (tmp_path / "results" / "large.csv").write_text("generated", encoding="utf-8")
    ignored_hash, _ = source_tree_hash(tmp_path)
    assert ignored_hash == first_hash

    source.write_text("VALUE = 2\n", encoding="utf-8")
    changed_hash, changed_files = source_tree_hash(tmp_path)
    assert changed_hash != first_hash
    assert all(len(record.sha256) == 64 for record in changed_files)


def test_git_provenance_distinguishes_clean_and_dirty_tree(tmp_path: Path) -> None:
    source = _write_source_tree(tmp_path)
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "fixture@example.invalid")
    _git(tmp_path, "config", "user.name", "Fixture")
    _git(tmp_path, "add", "pyproject.toml", "src")
    _git(tmp_path, "commit", "-qm", "initial")

    clean = git_provenance(tmp_path)
    assert clean.available
    assert clean.commit is not None and len(clean.commit) == 40
    assert clean.dirty is False
    assert clean.diff_sha256 is None

    source.write_text("VALUE = 3\n", encoding="utf-8")
    dirty = git_provenance(tmp_path)
    assert dirty.dirty is True
    assert dirty.diff_sha256 is not None and len(dirty.diff_sha256) == 64
    assert any("src/rllab/example.py" in line for line in dirty.status)


def test_collect_provenance_has_source_runtime_and_config_fingerprints(tmp_path: Path) -> None:
    _write_source_tree(tmp_path)
    config = {"episodes": 10, "seeds": [1, 2]}
    record = collect_provenance(
        repository=tmp_path,
        config=config,
        packages=("pydantic", "definitely-not-an-installed-package"),
    )
    assert record.source_files
    assert record.config_sha256 == value_sha256(config)
    assert record.runtime.machine
    assert record.runtime.package_versions.keys() == {"pydantic"}
    assert len(record.runtime.fingerprint) == 64

    legacy_shape = provenance(repository=tmp_path, config=config)
    assert legacy_shape["config"] == config
    assert legacy_shape["source_sha256"] == record.source_sha256
    assert "git_commit" in legacy_shape
    assert "package_versions" in legacy_shape
