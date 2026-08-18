"""Versioned, validated documents for Protocol v2 experiment artifacts.

The JSON files in a run directory are a public data contract.  Keeping their
models separate from execution code makes version checks and migration rules
available to readers without importing an agent or environment.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CONFIG_SCHEMA_VERSION = 2
ARTIFACT_SCHEMA_VERSION = 2
TABLE_SCHEMA_VERSION = 1

TableFormat = Literal["parquet", "csv"]
ArtifactFormat = Literal["parquet", "csv", "npz", "json"]
RunStatus = Literal["planned", "running", "partial", "complete", "failed"]
TrialStatus = Literal["pending", "running", "succeeded", "failed", "interrupted", "cancelled"]

_COMPONENT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def validate_path_component(value: str, *, name: str = "identifier") -> str:
    """Validate an identifier before it is used as a directory component."""

    if not _COMPONENT_PATTERN.fullmatch(value):
        raise ValueError(
            f"{name} must contain only letters, digits, '.', '_', and '-' and may not be empty"
        )
    return value


def validate_relative_artifact_path(value: str) -> str:
    """Require portable POSIX-style paths contained by a run directory."""

    if not value or value.startswith(("/", "\\")) or "\\" in value:
        raise ValueError("artifact paths must be nonempty relative POSIX paths")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("artifact paths may not contain empty, '.' or '..' components")
    return value


class StrictDocument(BaseModel):
    """Base settings shared by machine-written Protocol v2 documents."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ArtifactReference(StrictDocument):
    """Integrity and logical-schema metadata for one immutable artifact part."""

    path: str
    format: ArtifactFormat
    table: str | None = None
    part: int | None = Field(default=None, ge=0)
    rows: int = Field(default=0, ge=0)
    bytes: int = Field(ge=0)
    sha256: str
    table_schema_version: int | None = Field(default=None, ge=1)
    columns: dict[str, str] = Field(default_factory=dict)

    @field_validator("path")
    @classmethod
    def _path_is_safe(cls, value: str) -> str:
        return validate_relative_artifact_path(value)

    @field_validator("table")
    @classmethod
    def _table_is_safe(cls, value: str | None) -> str | None:
        return None if value is None else validate_path_component(value, name="table")

    @field_validator("sha256")
    @classmethod
    def _sha256_is_canonical(cls, value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("sha256 must contain exactly 64 lowercase hexadecimal characters")
        return value


class AttemptState(StrictDocument):
    """Mutable status document written when an isolated attempt begins."""

    artifact_schema_version: Literal[2] = 2
    document_type: Literal["attempt"] = "attempt"
    trial_id: str
    attempt: int = Field(ge=1)
    status: Literal["running", "succeeded", "failed"] = "running"
    started_at: str
    updated_at: str
    source_hash: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("trial_id")
    @classmethod
    def _trial_id_is_safe(cls, value: str) -> str:
        return validate_path_component(value, name="trial_id")

    @field_validator("source_hash")
    @classmethod
    def _source_hash_is_canonical(cls, value: str | None) -> str | None:
        if value is not None and not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("source_hash must be a SHA-256 hexadecimal digest")
        return value


class AttemptCommit(StrictDocument):
    """Immutable commit point that makes an attempt visible to readers."""

    artifact_schema_version: Literal[2] = 2
    document_type: Literal["trial_commit"] = "trial_commit"
    trial_id: str
    attempt: int = Field(ge=1)
    status: Literal["succeeded"] = "succeeded"
    started_at: str
    completed_at: str
    source_hash: str | None = None
    artifacts: tuple[ArtifactReference, ...] = ()
    row_counts: dict[str, int] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("trial_id")
    @classmethod
    def _trial_id_is_safe(cls, value: str) -> str:
        return validate_path_component(value, name="trial_id")

    @field_validator("source_hash")
    @classmethod
    def _source_hash_is_canonical(cls, value: str | None) -> str | None:
        if value is not None and not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("source_hash must be a SHA-256 hexadecimal digest")
        return value

    @field_validator("row_counts")
    @classmethod
    def _row_counts_are_nonnegative(cls, value: dict[str, int]) -> dict[str, int]:
        if any(count < 0 for count in value.values()):
            raise ValueError("row counts must be nonnegative")
        return value


class FailureRecord(StrictDocument):
    """Structured failure for an attempt that did not commit."""

    artifact_schema_version: Literal[2] = 2
    document_type: Literal["trial_failure"] = "trial_failure"
    trial_id: str
    attempt: int = Field(ge=1)
    status: Literal["failed"] = "failed"
    started_at: str
    failed_at: str
    exception_type: str
    message: str
    traceback: str | None = None
    artifacts: tuple[ArtifactReference, ...] = ()
    row_counts: dict[str, int] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("trial_id")
    @classmethod
    def _trial_id_is_safe(cls, value: str) -> str:
        return validate_path_component(value, name="trial_id")


class TrialDescriptor(StrictDocument):
    """Immutable resolved specification stored before a trial is scheduled."""

    artifact_schema_version: Literal[2] = 2
    document_type: Literal["trial"] = "trial"
    trial_id: str
    specification: dict[str, Any] = Field(default_factory=dict)

    @field_validator("trial_id")
    @classmethod
    def _trial_id_is_safe(cls, value: str) -> str:
        return validate_path_component(value, name="trial_id")


class TrialManifestEntry(StrictDocument):
    """Compact root-manifest index for one trial and its selected attempt."""

    trial_id: str
    path: str
    status: TrialStatus = "pending"
    attempts: int = Field(default=0, ge=0)
    active_attempt: int | None = Field(default=None, ge=1)
    committed_attempt: int | None = Field(default=None, ge=1)
    row_counts: dict[str, int] = Field(default_factory=dict)
    last_error: str | None = None

    @field_validator("trial_id")
    @classmethod
    def _trial_id_is_safe(cls, value: str) -> str:
        return validate_path_component(value, name="trial_id")

    @field_validator("path")
    @classmethod
    def _path_is_safe(cls, value: str) -> str:
        return validate_relative_artifact_path(value)

    @field_validator("row_counts")
    @classmethod
    def _row_counts_are_nonnegative(cls, value: dict[str, int]) -> dict[str, int]:
        if any(count < 0 for count in value.values()):
            raise ValueError("row counts must be nonnegative")
        return value


class RunManifest(StrictDocument):
    """Authoritative, atomically replaced index for a Protocol v2 run."""

    artifact_schema_version: Literal[2] = 2
    document_type: Literal["run_manifest"] = "run_manifest"
    run_id: str
    experiment_name: str
    status: RunStatus = "planned"
    created_at: str
    updated_at: str
    revision: int = Field(default=0, ge=0)
    table_format: TableFormat
    config_sha256: str
    provenance_sha256: str
    trials: dict[str, TrialManifestEntry] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("run_id")
    @classmethod
    def _run_id_is_safe(cls, value: str) -> str:
        return validate_path_component(value, name="run_id")

    @field_validator("config_sha256", "provenance_sha256")
    @classmethod
    def _sha256_is_canonical(cls, value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("fingerprints must be SHA-256 hexadecimal digests")
        return value

    @model_validator(mode="after")
    def _trial_keys_match_entries(self) -> RunManifest:
        mismatches = [key for key, entry in self.trials.items() if key != entry.trial_id]
        if mismatches:
            raise ValueError(f"trial manifest keys do not match their trial_id: {mismatches}")
        return self


class SourceFileHash(StrictDocument):
    """One path/content contribution to a deterministic source-tree hash."""

    path: str
    bytes: int = Field(ge=0)
    sha256: str

    @field_validator("path")
    @classmethod
    def _path_is_safe(cls, value: str) -> str:
        return validate_relative_artifact_path(value)

    @field_validator("sha256")
    @classmethod
    def _sha256_is_canonical(cls, value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("sha256 must be a SHA-256 hexadecimal digest")
        return value


class GitProvenance(StrictDocument):
    """Git identity, including a digest of uncommitted changes when available."""

    available: bool
    commit: str | None = None
    branch: str | None = None
    dirty: bool | None = None
    status: tuple[str, ...] = ()
    diff_sha256: str | None = None


class RuntimeProvenance(StrictDocument):
    """Interpreter, platform, and package versions used by a run."""

    python: str
    executable: str
    implementation: str
    platform: str
    machine: str
    processor: str
    package_versions: dict[str, str] = Field(default_factory=dict)
    fingerprint: str

    @field_validator("fingerprint")
    @classmethod
    def _fingerprint_is_canonical(cls, value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("runtime fingerprint must be a SHA-256 hexadecimal digest")
        return value


class ProvenanceRecord(StrictDocument):
    """Protocol v2 source/runtime provenance kept beside a run manifest."""

    provenance_schema_version: Literal[2] = 2
    document_type: Literal["provenance"] = "provenance"
    created_at: str
    git: GitProvenance
    source_hash_algorithm: Literal["sha256-path-length-content-v1"] = (
        "sha256-path-length-content-v1"
    )
    source_sha256: str
    source_files: tuple[SourceFileHash, ...]
    runtime: RuntimeProvenance
    config_sha256: str | None = None

    @field_validator("source_sha256", "config_sha256")
    @classmethod
    def _hash_is_canonical(cls, value: str | None) -> str | None:
        if value is not None and not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("fingerprints must be SHA-256 hexadecimal digests")
        return value


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "CONFIG_SCHEMA_VERSION",
    "TABLE_SCHEMA_VERSION",
    "ArtifactFormat",
    "ArtifactReference",
    "AttemptCommit",
    "AttemptState",
    "FailureRecord",
    "GitProvenance",
    "ProvenanceRecord",
    "RunManifest",
    "RunStatus",
    "RuntimeProvenance",
    "SourceFileHash",
    "StrictDocument",
    "TableFormat",
    "TrialDescriptor",
    "TrialManifestEntry",
    "TrialStatus",
    "validate_path_component",
    "validate_relative_artifact_path",
]
