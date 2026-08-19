"""Atomic, trial-sharded storage for Protocol v2 experiment artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import tempfile
import traceback as traceback_module
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from pydantic import BaseModel, ValidationError

from rllab.experiments.provenance import utc_timestamp, value_sha256
from rllab.experiments.schema import (
    ARTIFACT_SCHEMA_VERSION,
    TABLE_SCHEMA_VERSION,
    ArtifactReference,
    AttemptCommit,
    AttemptState,
    FailureRecord,
    RunManifest,
    TableFormat,
    TrialDescriptor,
    TrialManifestEntry,
    validate_path_component,
)

TableFormatPreference = Literal["auto", "parquet", "csv"]


class ArtifactError(RuntimeError):
    """Base error for an invalid or inconsistent artifact store."""


class UnsupportedArtifactVersion(ArtifactError):
    """Raised instead of guessing how to interpret a future artifact version."""


class ArtifactIntegrityError(ArtifactError):
    """Raised when a committed artifact is absent or fails its checksum."""


class RunLockedError(ArtifactError):
    """Raised when another process already owns a run lock."""


def _json_value(value: Mapping[str, Any] | BaseModel) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return dict(value)


def _fsync_directory(path: Path) -> None:
    """Best-effort directory sync; unsupported filesystems may reject it."""

    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, content: bytes) -> None:
    """Durably replace ``path`` from a unique temporary file beside it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, value: Mapping[str, Any] | BaseModel) -> None:
    """Atomically write a validated model or ordinary JSON mapping."""

    content = (
        json.dumps(
            _json_value(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        + "\n"
    ).encode("utf-8")
    atomic_write_bytes(path, content)


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object and reject other top-level values."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ArtifactError(f"Expected a JSON object in {path}")
    return value


def sha256_file(path: Path, *, block_size: int = 1024 * 1024) -> str:
    """Stream a file into SHA-256 without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def select_table_format(preference: TableFormatPreference = "auto") -> TableFormat:
    """Resolve ``auto`` once so a run never mixes CSV and Parquet parts."""

    if preference == "csv":
        return "csv"
    if preference not in {"auto", "parquet"}:
        raise ValueError("table format must be 'auto', 'parquet', or 'csv'")
    try:
        import pyarrow  # type: ignore[import-untyped]  # noqa: F401
    except ImportError:
        try:
            import fastparquet  # type: ignore[import-not-found]  # noqa: F401
        except ImportError:
            if preference == "parquet":
                raise ImportError(
                    "Parquet was requested but pyarrow/fastparquet is unavailable"
                ) from None
            return "csv"
    return "parquet"


def _write_frame_atomic(frame: pd.DataFrame, path: Path, table_format: TableFormat) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        if table_format == "parquet":
            frame.to_parquet(temporary, index=False)
        else:
            frame.to_csv(temporary, index=False)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        temporary.replace(path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _artifact_reference(
    run_directory: Path,
    path: Path,
    *,
    artifact_format: Literal["parquet", "csv", "npz", "json"],
    table: str | None = None,
    part: int | None = None,
    rows: int = 0,
    columns: Mapping[str, str] | None = None,
) -> ArtifactReference:
    return ArtifactReference(
        path=path.relative_to(run_directory).as_posix(),
        format=artifact_format,
        table=table,
        part=part,
        rows=rows,
        bytes=path.stat().st_size,
        sha256=sha256_file(path),
        table_schema_version=TABLE_SCHEMA_VERSION if table is not None else None,
        columns=dict(columns or {}),
    )


@dataclass(frozen=True, slots=True)
class AttemptReservation:
    """Pickle-friendly assignment passed from the parent to one worker."""

    run_directory: Path
    trial_id: str
    attempt: int
    table_format: TableFormat

    @property
    def directory(self) -> Path:
        return self.run_directory / "trials" / self.trial_id / "attempts" / f"{self.attempt:04d}"


class TrialAttemptWriter:
    """Single-writer sink for one immutable trial attempt.

    Table parts are atomic but invisible to normal readers until :meth:`commit`
    writes ``commit.json``.  A retry uses a different attempt directory.
    """

    def __init__(
        self,
        reservation: AttemptReservation,
        *,
        source_hash: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        validate_path_component(reservation.trial_id, name="trial_id")
        if reservation.attempt < 1:
            raise ValueError("attempt must be positive")
        self.reservation = reservation
        self.run_directory = reservation.run_directory.resolve()
        self.directory = reservation.directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.started_at = utc_timestamp()
        self.source_hash = source_hash
        self.metadata = dict(metadata or {})
        self._artifacts: list[ArtifactReference] = []
        self._row_counts: defaultdict[str, int] = defaultdict(int)
        self._parts: defaultdict[str, int] = defaultdict(int)
        self._closed = False
        attempt_path = self.directory / "attempt.json"
        if attempt_path.exists():
            raise FileExistsError(f"Attempt already has a writer: {attempt_path}")
        self._write_state("running")

    @property
    def trial_id(self) -> str:
        return self.reservation.trial_id

    @property
    def attempt(self) -> int:
        return self.reservation.attempt

    @property
    def artifacts(self) -> tuple[ArtifactReference, ...]:
        return tuple(self._artifacts)

    @property
    def row_counts(self) -> dict[str, int]:
        return dict(self._row_counts)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("trial attempt is already closed")

    def _write_state(self, status: Literal["running", "succeeded", "failed"]) -> None:
        state = AttemptState(
            trial_id=self.trial_id,
            attempt=self.attempt,
            status=status,
            started_at=self.started_at,
            updated_at=utc_timestamp(),
            source_hash=self.source_hash,
            metadata=self.metadata,
        )
        atomic_write_json(self.directory / "attempt.json", state)

    def write_table(self, table: str, frame: pd.DataFrame) -> ArtifactReference | None:
        """Write one bounded table part and return its integrity reference."""

        self._require_open()
        validate_path_component(table, name="table")
        if frame.empty:
            return None
        part = self._parts[table]
        extension = ".parquet" if self.reservation.table_format == "parquet" else ".csv"
        path = self.directory / "tables" / table / f"part-{part:06d}{extension}"
        _write_frame_atomic(frame, path, self.reservation.table_format)
        reference = _artifact_reference(
            self.run_directory,
            path,
            artifact_format=self.reservation.table_format,
            table=table,
            part=part,
            rows=len(frame),
            columns={column: str(dtype) for column, dtype in frame.dtypes.items()},
        )
        self._artifacts.append(reference)
        self._row_counts[table] += len(frame)
        self._parts[table] += 1
        return reference

    def write_q_snapshots(self, snapshots: Mapping[str, np.ndarray]) -> ArtifactReference | None:
        """Write one bounded group of numeric Q snapshots without pickle."""

        self._require_open()
        if not snapshots:
            return None
        table = "q_snapshots"
        part = self._parts[table]
        path = self.directory / table / f"part-{part:06d}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary = Path(temporary_name)
        try:
            arrays: dict[str, Any] = {key: np.asarray(value) for key, value in snapshots.items()}
            with os.fdopen(descriptor, "wb") as stream:
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
            _fsync_directory(path.parent)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        reference = _artifact_reference(
            self.run_directory,
            path,
            artifact_format="npz",
            table="q_snapshots",
            part=part,
            rows=len(snapshots),
        )
        self._artifacts.append(reference)
        self._row_counts[table] += len(snapshots)
        self._parts[table] += 1
        return reference

    def commit(self, *, metadata: Mapping[str, Any] | None = None) -> AttemptCommit:
        """Publish this attempt atomically after all of its parts are durable."""

        self._require_open()
        commit = AttemptCommit(
            trial_id=self.trial_id,
            attempt=self.attempt,
            started_at=self.started_at,
            completed_at=utc_timestamp(),
            source_hash=self.source_hash,
            artifacts=tuple(self._artifacts),
            row_counts=dict(self._row_counts),
            metadata={**self.metadata, **dict(metadata or {})},
        )
        atomic_write_json(self.directory / "commit.json", commit)
        self._closed = True
        self._write_state("succeeded")
        return commit

    def fail(
        self,
        error: BaseException | str,
        *,
        traceback: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> FailureRecord:
        """Close an uncommitted attempt with a structured, atomic failure."""

        self._require_open()
        if isinstance(error, BaseException):
            exception_type = f"{type(error).__module__}.{type(error).__qualname__}"
            message = str(error)
            if traceback is None:
                traceback = "".join(
                    traceback_module.format_exception(type(error), error, error.__traceback__)
                )
        else:
            exception_type = "builtins.RuntimeError"
            message = error
        failure = FailureRecord(
            trial_id=self.trial_id,
            attempt=self.attempt,
            started_at=self.started_at,
            failed_at=utc_timestamp(),
            exception_type=exception_type,
            message=message,
            traceback=traceback,
            artifacts=tuple(self._artifacts),
            row_counts=dict(self._row_counts),
            metadata={**self.metadata, **dict(metadata or {})},
        )
        atomic_write_json(self.directory / "failure.json", failure)
        self._closed = True
        self._write_state("failed")
        return failure


class RunLock:
    """Small exclusive lock for the single parent manifest writer."""

    def __init__(self, run_directory: str | Path) -> None:
        self.path = Path(run_directory) / "run.lock"
        self._owned = False

    def acquire(self) -> Self:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(
            {
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "created_at": utc_timestamp(),
            },
            sort_keys=True,
        ).encode("utf-8")
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError as error:
            raise RunLockedError(f"Run is already locked: {self.path}") from error
        try:
            os.write(descriptor, content)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self._owned = True
        _fsync_directory(self.path.parent)
        return self

    def release(self) -> None:
        if self._owned:
            self.path.unlink(missing_ok=True)
            self._owned = False
            _fsync_directory(self.path.parent)

    def __enter__(self) -> Self:
        return self.acquire()

    def __exit__(self, *_: object) -> None:
        self.release()


def _load_model(path: Path, model: type[BaseModel]) -> BaseModel:
    value = read_json(path)
    version = value.get("artifact_schema_version")
    if version != ARTIFACT_SCHEMA_VERSION:
        raise UnsupportedArtifactVersion(
            f"{path} uses artifact schema {version!r}; supported version is "
            f"{ARTIFACT_SCHEMA_VERSION}"
        )
    try:
        return model.model_validate(value)
    except ValidationError as error:
        raise ArtifactError(f"Invalid artifact document {path}: {error}") from error


def load_run_manifest(path: str | Path) -> RunManifest:
    """Load and validate a Protocol v2 root manifest."""

    source = Path(path)
    if source.is_dir():
        source = source / "manifest.json"
    model = _load_model(source, RunManifest)
    assert isinstance(model, RunManifest)
    return model


def load_attempt_commit(path: str | Path) -> AttemptCommit:
    """Load and validate an immutable attempt commit."""

    source = Path(path)
    if source.is_dir():
        source = source / "commit.json"
    model = _load_model(source, AttemptCommit)
    assert isinstance(model, AttemptCommit)
    return model


def load_failure_record(path: str | Path) -> FailureRecord:
    """Load and validate an attempt failure document."""

    source = Path(path)
    if source.is_dir():
        source = source / "failure.json"
    model = _load_model(source, FailureRecord)
    assert isinstance(model, FailureRecord)
    return model


def verify_artifact(run_directory: Path, reference: ArtifactReference) -> None:
    """Verify containment, size, and SHA-256 for one committed artifact."""

    root = run_directory.resolve()
    path = (root / reference.path).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ArtifactIntegrityError(f"Artifact escapes run directory: {reference.path}") from error
    if not path.is_file():
        raise ArtifactIntegrityError(f"Committed artifact is missing: {reference.path}")
    if path.stat().st_size != reference.bytes:
        raise ArtifactIntegrityError(f"Artifact size changed: {reference.path}")
    if sha256_file(path) != reference.sha256:
        raise ArtifactIntegrityError(f"Artifact checksum changed: {reference.path}")


def _apply_filters(frame: pd.DataFrame, filters: Mapping[str, Any] | None) -> pd.DataFrame:
    if not filters or frame.empty:
        return frame
    selected = np.ones(len(frame), dtype=bool)
    for column, expected in filters.items():
        if column not in frame:
            raise KeyError(f"Filter column {column!r} is absent")
        if isinstance(expected, (list, tuple, set, frozenset)):
            selected &= frame[column].isin(expected).to_numpy()
        else:
            selected &= frame[column].eq(expected).to_numpy()
    return frame.loc[selected]


def _read_reference(
    run_directory: Path,
    reference: ArtifactReference,
    *,
    columns: Sequence[str] | None,
    filters: Mapping[str, Any] | None,
) -> pd.DataFrame:
    path = run_directory / reference.path
    read_columns: list[str] | None = None
    if columns is not None:
        read_columns = list(dict.fromkeys([*columns, *(filters or {}).keys()]))
    if reference.format == "parquet":
        frame = pd.read_parquet(path, columns=read_columns)
    elif reference.format == "csv":
        frame = pd.read_csv(path, usecols=read_columns)
    else:
        raise ArtifactError(f"Artifact {reference.path} is not a tabular part")
    frame = _apply_filters(frame, filters)
    if columns is not None:
        frame = frame.loc[:, list(columns)]
    return frame


class RunStore:
    """Parent-owned manifest plus immutable per-trial attempt artifacts."""

    def __init__(self, run_directory: str | Path, manifest: RunManifest) -> None:
        self.run_directory = Path(run_directory).resolve()
        self._manifest = manifest
        self._committed_attempt_cache: tuple[AttemptCommit, ...] | None = None

    @property
    def manifest(self) -> RunManifest:
        return self._manifest

    @classmethod
    def create(
        cls,
        run_directory: str | Path,
        *,
        run_id: str,
        experiment_name: str,
        trials: Mapping[str, Mapping[str, Any]],
        config: Mapping[str, Any],
        provenance: Mapping[str, Any] | BaseModel,
        table_format: TableFormatPreference = "auto",
        metadata: Mapping[str, Any] | None = None,
    ) -> RunStore:
        """Create and fully plan a run before any trial is scheduled."""

        validate_path_component(run_id, name="run_id")
        root = Path(run_directory).resolve()
        root.mkdir(parents=True, exist_ok=False)
        selected_format = select_table_format(table_format)
        config_value = dict(config)
        provenance_value = _json_value(provenance)
        atomic_write_json(root / "config.json", config_value)
        atomic_write_json(root / "provenance.json", provenance_value)

        entries: dict[str, TrialManifestEntry] = {}
        for trial_id, specification in sorted(trials.items()):
            validate_path_component(trial_id, name="trial_id")
            trial_directory = root / "trials" / trial_id
            trial_directory.mkdir(parents=True)
            descriptor = TrialDescriptor(trial_id=trial_id, specification=dict(specification))
            atomic_write_json(trial_directory / "trial.json", descriptor)
            entries[trial_id] = TrialManifestEntry(
                trial_id=trial_id,
                path=(Path("trials") / trial_id).as_posix(),
            )

        now = utc_timestamp()
        manifest = RunManifest(
            run_id=run_id,
            experiment_name=experiment_name,
            status="planned",
            created_at=now,
            updated_at=now,
            table_format=selected_format,
            config_sha256=value_sha256(config_value),
            provenance_sha256=value_sha256(provenance_value),
            trials=entries,
            metadata=dict(metadata or {}),
        )
        atomic_write_json(root / "manifest.json", manifest)
        return cls(root, manifest)

    @classmethod
    def open(cls, run_directory: str | Path) -> RunStore:
        root = Path(run_directory).resolve()
        return cls(root, load_run_manifest(root))

    def _save(self, *, trials: Mapping[str, TrialManifestEntry] | None = None) -> None:
        entries = dict(trials if trials is not None else self._manifest.trials)
        status = self._derive_status(entries)
        self._manifest = self._manifest.model_copy(
            update={
                "status": status,
                "updated_at": utc_timestamp(),
                "revision": self._manifest.revision + 1,
                "trials": entries,
            }
        )
        atomic_write_json(self.run_directory / "manifest.json", self._manifest)
        self._committed_attempt_cache = None

    @staticmethod
    def _derive_status(trials: Mapping[str, TrialManifestEntry]) -> str:
        if not trials:
            return "complete"
        statuses = [entry.status for entry in trials.values()]
        if all(status == "succeeded" for status in statuses):
            return "complete"
        if all(status == "pending" for status in statuses):
            return "planned"
        if any(status == "running" for status in statuses):
            return "running"
        if any(status == "pending" for status in statuses):
            return "partial"
        succeeded = sum(status == "succeeded" for status in statuses)
        return "partial" if succeeded else "failed"

    def reserve_attempt(self, trial_id: str) -> AttemptReservation:
        """Atomically record a new attempt before handing it to a worker."""

        if trial_id not in self._manifest.trials:
            raise KeyError(trial_id)
        entry = self._manifest.trials[trial_id]
        if entry.status == "succeeded":
            raise ArtifactError(f"Trial {trial_id} already has a committed attempt")
        attempt = entry.attempts + 1
        reservation = AttemptReservation(
            run_directory=self.run_directory,
            trial_id=trial_id,
            attempt=attempt,
            table_format=self._manifest.table_format,
        )
        reservation.directory.mkdir(parents=True, exist_ok=False)
        updated = entry.model_copy(
            update={
                "status": "running",
                "attempts": attempt,
                "active_attempt": attempt,
                "last_error": None,
            }
        )
        entries = {**self._manifest.trials, trial_id: updated}
        self._save(trials=entries)
        return reservation

    def start_attempt(
        self,
        trial_id: str,
        *,
        source_hash: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> TrialAttemptWriter:
        """Reserve and open an attempt for serial execution."""

        return TrialAttemptWriter(
            self.reserve_attempt(trial_id), source_hash=source_hash, metadata=metadata
        )

    def record_commit(self, commit: AttemptCommit, *, verify: bool = False) -> None:
        """Select a worker's committed attempt in the root manifest."""

        if commit.trial_id not in self._manifest.trials:
            raise KeyError(commit.trial_id)
        if verify:
            for reference in commit.artifacts:
                verify_artifact(self.run_directory, reference)
        entry = self._manifest.trials[commit.trial_id]
        if commit.attempt > entry.attempts:
            raise ArtifactError("Commit attempt was not reserved by this manifest")
        updated = entry.model_copy(
            update={
                "status": "succeeded",
                "active_attempt": None,
                "committed_attempt": commit.attempt,
                "row_counts": dict(commit.row_counts),
                "last_error": None,
            }
        )
        self._save(trials={**self._manifest.trials, commit.trial_id: updated})

    def record_failure(self, failure: FailureRecord) -> None:
        """Record a worker failure while keeping successful trials readable."""

        if failure.trial_id not in self._manifest.trials:
            raise KeyError(failure.trial_id)
        entry = self._manifest.trials[failure.trial_id]
        if failure.attempt > entry.attempts:
            raise ArtifactError("Failure attempt was not reserved by this manifest")
        updated = entry.model_copy(
            update={
                "status": "failed",
                "active_attempt": None,
                "last_error": f"{failure.exception_type}: {failure.message}",
            }
        )
        self._save(trials={**self._manifest.trials, failure.trial_id: updated})

    def _attempt_directories(self, entry: TrialManifestEntry) -> list[Path]:
        directory = self.run_directory / entry.path / "attempts"
        if not directory.is_dir():
            return []
        return sorted(
            (
                path
                for path in directory.iterdir()
                if path.is_dir() and path.name.isdigit() and int(path.name) >= 1
            ),
            key=lambda path: int(path.name),
        )

    def reconcile(self, *, verify: bool = False) -> RunManifest:
        """Recover root state from immutable commits after interruption."""

        entries: dict[str, TrialManifestEntry] = {}
        for trial_id, entry in self._manifest.trials.items():
            attempt_directories = self._attempt_directories(entry)
            attempts = max((int(path.name) for path in attempt_directories), default=0)
            commits: list[AttemptCommit] = []
            failures: list[FailureRecord] = []
            for directory in attempt_directories:
                commit_path = directory / "commit.json"
                failure_path = directory / "failure.json"
                if commit_path.is_file():
                    commit = load_attempt_commit(commit_path)
                    if commit.trial_id != trial_id or commit.attempt != int(directory.name):
                        raise ArtifactError(f"Commit identity does not match {directory}")
                    if verify:
                        for reference in commit.artifacts:
                            verify_artifact(self.run_directory, reference)
                    commits.append(commit)
                elif failure_path.is_file():
                    failure = load_failure_record(failure_path)
                    if failure.trial_id != trial_id or failure.attempt != int(directory.name):
                        raise ArtifactError(f"Failure identity does not match {directory}")
                    failures.append(failure)

            if commits:
                selected = max(commits, key=lambda value: value.attempt)
                updated = entry.model_copy(
                    update={
                        "status": "succeeded",
                        "attempts": attempts,
                        "active_attempt": None,
                        "committed_attempt": selected.attempt,
                        "row_counts": dict(selected.row_counts),
                        "last_error": None,
                    }
                )
            elif failures and failures[-1].attempt == attempts:
                selected_failure = failures[-1]
                updated = entry.model_copy(
                    update={
                        "status": "failed",
                        "attempts": attempts,
                        "active_attempt": None,
                        "committed_attempt": None,
                        "row_counts": {},
                        "last_error": (
                            f"{selected_failure.exception_type}: {selected_failure.message}"
                        ),
                    }
                )
            elif attempts:
                updated = entry.model_copy(
                    update={
                        "status": "interrupted",
                        "attempts": attempts,
                        "active_attempt": None,
                        "committed_attempt": None,
                        "row_counts": {},
                        "last_error": "attempt ended without a commit or failure record",
                    }
                )
            else:
                updated = entry.model_copy(
                    update={
                        "status": "pending",
                        "attempts": 0,
                        "active_attempt": None,
                        "committed_attempt": None,
                        "row_counts": {},
                        "last_error": None,
                    }
                )
            entries[trial_id] = updated
        if (
            entries != self._manifest.trials
            or self._derive_status(entries) != self._manifest.status
        ):
            self._save(trials=entries)
        return self._manifest

    def committed_attempts(self) -> tuple[AttemptCommit, ...]:
        """Return selected successful attempts in deterministic trial order."""

        if self._committed_attempt_cache is not None:
            return self._committed_attempt_cache

        commits: list[AttemptCommit] = []
        for trial_id, entry in sorted(self._manifest.trials.items()):
            if entry.status != "succeeded" or entry.committed_attempt is None:
                continue
            directory = (
                self.run_directory / entry.path / "attempts" / f"{entry.committed_attempt:04d}"
            )
            commit = load_attempt_commit(directory)
            if commit.trial_id != trial_id:
                raise ArtifactError(f"Selected commit does not belong to trial {trial_id}")
            commits.append(commit)
        self._committed_attempt_cache = tuple(commits)
        return self._committed_attempt_cache

    def artifact_references(self, table: str) -> tuple[ArtifactReference, ...]:
        """Committed references for a logical table, excluding partial attempts."""

        validate_path_component(table, name="table")
        candidates = (
            (table, "training_episodes")
            if table == "episodes"
            else ((table, "episodes") if table == "training_episodes" else (table,))
        )
        commits = self.committed_attempts()
        for candidate in candidates:
            references = tuple(
                reference
                for commit in commits
                for reference in commit.artifacts
                if reference.table == candidate and reference.format in {"parquet", "csv"}
            )
            if references:
                return references
        return ()

    def iter_table(
        self,
        table: str = "episodes",
        *,
        columns: Sequence[str] | None = None,
        filters: Mapping[str, Any] | None = None,
        batch_size: int | None = None,
        verify: bool = False,
    ) -> Iterator[pd.DataFrame]:
        """Yield bounded committed parts without materializing the full run."""

        if batch_size is not None and batch_size < 1:
            raise ValueError("batch_size must be positive")
        for reference in self.artifact_references(table):
            if verify:
                verify_artifact(self.run_directory, reference)
            frame = _read_reference(
                self.run_directory,
                reference,
                columns=columns,
                filters=filters,
            )
            if frame.empty:
                continue
            size = batch_size or len(frame)
            for start in range(0, len(frame), size):
                yield frame.iloc[start : start + size].reset_index(drop=True)

    def read_table(
        self,
        table: str = "episodes",
        *,
        columns: Sequence[str] | None = None,
        filters: Mapping[str, Any] | None = None,
        verify: bool = False,
    ) -> pd.DataFrame:
        """Materialize one logical table from committed attempts only."""

        parts = list(self.iter_table(table, columns=columns, filters=filters, verify=verify))
        if not parts:
            return pd.DataFrame(columns=list(columns) if columns is not None else None)
        return pd.concat(parts, ignore_index=True)

    def read_q_snapshots(
        self,
        trial_id: str,
        *,
        keys: Sequence[str] | None = None,
        verify: bool = False,
    ) -> dict[str, np.ndarray]:
        """Load selected numeric Q snapshots for one committed trial.

        Snapshot keys are the values recorded in the tidy ``snapshots`` table,
        such as ``episode_00000199`` or ``global_step_000000010000``. The reader
        follows only the attempt selected by the run manifest, so incomplete or
        superseded attempts remain invisible just like their table rows.
        """

        validate_path_component(trial_id, name="trial_id")
        entry = self._manifest.trials.get(trial_id)
        if entry is None:
            raise KeyError(f"Unknown trial_id {trial_id!r}")
        if entry.status != "succeeded" or entry.committed_attempt is None:
            raise ArtifactError(f"Trial {trial_id!r} has no committed Q snapshots")

        commit = next(
            (item for item in self.committed_attempts() if item.trial_id == trial_id),
            None,
        )
        if commit is None:  # pragma: no cover - guarded by manifest validation above
            raise ArtifactError(f"Trial {trial_id!r} has no selected commit")

        requested = None if keys is None else tuple(dict.fromkeys(keys))
        requested_set = None if requested is None else set(requested)
        snapshots: dict[str, np.ndarray] = {}
        references = sorted(
            (
                reference
                for reference in commit.artifacts
                if reference.format == "npz"
                and (reference.table == "q_snapshots" or "/q_snapshots/" in f"/{reference.path}")
            ),
            key=lambda reference: -1 if reference.part is None else reference.part,
        )
        for reference in references:
            if verify:
                verify_artifact(self.run_directory, reference)
            path = self.run_directory / reference.path
            with np.load(path, allow_pickle=False) as archive:
                for key in archive.files:
                    if requested_set is not None and key not in requested_set:
                        continue
                    if key in snapshots:
                        raise ArtifactError(
                            f"Duplicate Q snapshot key {key!r} for trial {trial_id!r}"
                        )
                    snapshots[key] = np.asarray(archive[key]).copy()

        if requested is not None:
            missing = [key for key in requested if key not in snapshots]
            if missing:
                raise KeyError(f"Q snapshot keys are absent for trial {trial_id!r}: {missing!r}")
            return {key: snapshots[key] for key in requested}
        return snapshots

    def verify(self) -> None:
        """Verify every artifact referenced by each selected commit."""

        for commit in self.committed_attempts():
            for reference in commit.artifacts:
                verify_artifact(self.run_directory, reference)


__all__ = [
    "ArtifactError",
    "ArtifactIntegrityError",
    "AttemptReservation",
    "RunLock",
    "RunLockedError",
    "RunStore",
    "TableFormatPreference",
    "TrialAttemptWriter",
    "UnsupportedArtifactVersion",
    "atomic_write_bytes",
    "atomic_write_json",
    "load_attempt_commit",
    "load_failure_record",
    "load_run_manifest",
    "read_json",
    "select_table_format",
    "sha256_file",
    "verify_artifact",
]
