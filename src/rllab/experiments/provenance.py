"""Deterministic source, Git, and runtime provenance for experiment artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rllab.experiments.schema import (
    GitProvenance,
    ProvenanceRecord,
    RuntimeProvenance,
    SourceFileHash,
)

TRACKED_PACKAGES = (
    "independent-rl-lab",
    "numpy",
    "pandas",
    "scipy",
    "gymnasium",
    "matplotlib",
    "pydantic",
    "PyYAML",
)

_IGNORED_SOURCE_PARTS = {
    ".git",
    ".ipynb_checkpoints",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "results",
    "venv",
}


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp with microsecond precision."""

    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def canonical_json_bytes(value: Any) -> bytes:
    """Stable JSON bytes used for configuration and runtime fingerprints."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")


def value_sha256(value: Any) -> str:
    """SHA-256 of a JSON-compatible value in canonical form."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def package_versions(packages: Sequence[str] = TRACKED_PACKAGES) -> dict[str, str]:
    """Installed package versions, omitting unavailable optional packages."""

    versions: dict[str, str] = {}
    for package in packages:
        with suppress(importlib.metadata.PackageNotFoundError):
            versions[package] = importlib.metadata.version(package)
    return versions


def _git(
    repository: Path,
    *arguments: str,
    text: bool = True,
) -> subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes] | None:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=True,
            capture_output=True,
            text=text,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def git_commit(path: str | Path) -> str | None:
    """Current commit, or ``None`` outside Git or before the first commit."""

    result = _git(Path(path), "rev-parse", "HEAD")
    if result is None or not isinstance(result.stdout, str):
        return None
    return result.stdout.strip() or None


def git_provenance(path: str | Path) -> GitProvenance:
    """Capture commit, branch, status, and a digest of tracked modifications."""

    repository = Path(path).resolve()
    root_result = _git(repository, "rev-parse", "--show-toplevel")
    if root_result is None or not isinstance(root_result.stdout, str):
        return GitProvenance(available=False)

    status_result = _git(repository, "status", "--porcelain=v1", "--untracked-files=normal")
    status = (
        tuple(line for line in status_result.stdout.splitlines() if line)
        if status_result is not None and isinstance(status_result.stdout, str)
        else ()
    )
    branch_result = _git(repository, "branch", "--show-current")
    branch = (
        branch_result.stdout.strip()
        if branch_result is not None and isinstance(branch_result.stdout, str)
        else None
    )
    commit = git_commit(repository)

    diff_hasher = hashlib.sha256()
    has_diff = False
    if commit is not None:
        for arguments in (("diff", "--binary", "HEAD"), ("diff", "--cached", "--binary", "HEAD")):
            result = _git(repository, *arguments, text=False)
            if result is not None and isinstance(result.stdout, bytes):
                has_diff = has_diff or bool(result.stdout)
                diff_hasher.update(len(result.stdout).to_bytes(8, "big"))
                diff_hasher.update(result.stdout)
    # The status digest includes untracked path names. Their relevant contents
    # are independently covered by source_sha256 below.
    status_bytes = "\n".join(status).encode("utf-8", errors="surrogateescape")
    if status_bytes:
        has_diff = True
        diff_hasher.update(len(status_bytes).to_bytes(8, "big"))
        diff_hasher.update(status_bytes)

    return GitProvenance(
        available=True,
        commit=commit,
        branch=branch or None,
        dirty=bool(status),
        status=status,
        diff_sha256=diff_hasher.hexdigest() if has_diff else None,
    )


def _is_source_file(repository: Path, path: Path) -> bool:
    try:
        relative = path.relative_to(repository)
    except ValueError:
        return False
    if any(part in _IGNORED_SOURCE_PARTS for part in relative.parts):
        return False
    if relative == Path("pyproject.toml"):
        return True
    if relative.parts[:2] == ("src", "rllab") and path.suffix in {".py", ".pyi"}:
        return True
    return relative.parts[:1] == ("experiments",) and path.suffix == ".py"


def discover_source_files(
    repository: str | Path,
    *,
    paths: Iterable[str | Path] | None = None,
) -> tuple[Path, ...]:
    """Find the execution-relevant files included in a source fingerprint."""

    root = Path(repository).resolve()
    candidates: list[Path]
    if paths is not None:
        candidates = [Path(path) if Path(path).is_absolute() else root / path for path in paths]
    else:
        candidates = []
        pyproject = root / "pyproject.toml"
        if pyproject.is_file():
            candidates.append(pyproject)
        for directory in (root / "src" / "rllab", root / "experiments"):
            if directory.is_dir():
                candidates.extend(path for path in directory.rglob("*") if path.is_file())
    selected = {
        path.resolve() for path in candidates if path.is_file() and _is_source_file(root, path)
    }
    return tuple(sorted(selected, key=lambda path: path.relative_to(root).as_posix()))


def source_tree_hash(
    repository: str | Path,
    *,
    paths: Iterable[str | Path] | None = None,
) -> tuple[str, tuple[SourceFileHash, ...]]:
    """Hash path and raw content, including relevant untracked source files."""

    root = Path(repository).resolve()
    hasher = hashlib.sha256()
    records: list[SourceFileHash] = []
    for path in discover_source_files(root, paths=paths):
        relative = path.relative_to(root).as_posix()
        content = path.read_bytes()
        relative_bytes = relative.encode("utf-8")
        digest = hashlib.sha256(content).hexdigest()
        hasher.update(len(relative_bytes).to_bytes(8, "big"))
        hasher.update(relative_bytes)
        hasher.update(len(content).to_bytes(8, "big"))
        hasher.update(content)
        records.append(SourceFileHash(path=relative, bytes=len(content), sha256=digest))
    return hasher.hexdigest(), tuple(records)


def runtime_provenance(
    packages: Sequence[str] = TRACKED_PACKAGES,
) -> RuntimeProvenance:
    """Interpreter and dependency identity with its own stable fingerprint."""

    values: dict[str, Any] = {
        "python": sys.version,
        "executable": sys.executable,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "package_versions": package_versions(packages),
    }
    return RuntimeProvenance(**values, fingerprint=value_sha256(values))


def collect_provenance(
    *,
    repository: str | Path,
    config: Mapping[str, Any] | None = None,
    packages: Sequence[str] = TRACKED_PACKAGES,
    source_paths: Iterable[str | Path] | None = None,
) -> ProvenanceRecord:
    """Collect a complete Protocol v2 provenance record."""

    source_sha256, source_files = source_tree_hash(repository, paths=source_paths)
    return ProvenanceRecord(
        created_at=utc_timestamp(),
        git=git_provenance(repository),
        source_sha256=source_sha256,
        source_files=source_files,
        runtime=runtime_provenance(packages),
        config_sha256=value_sha256(dict(config)) if config is not None else None,
    )


__all__ = [
    "TRACKED_PACKAGES",
    "canonical_json_bytes",
    "collect_provenance",
    "discover_source_files",
    "git_commit",
    "git_provenance",
    "package_versions",
    "runtime_provenance",
    "source_tree_hash",
    "utc_timestamp",
    "value_sha256",
]
