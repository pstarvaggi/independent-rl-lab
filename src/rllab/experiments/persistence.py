"""Compatibility façade for legacy flat files and Protocol v2 run stores."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from pydantic import BaseModel

from rllab.experiments.artifacts import (
    RunStore,
    atomic_write_json,
    select_table_format,
)
from rllab.experiments.provenance import (
    TRACKED_PACKAGES,
    collect_provenance,
    git_commit,
    package_versions,
    utc_timestamp,
)


def provenance(*, repository: str | Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """Capture v2 provenance while retaining the original flat metadata keys."""

    record = collect_provenance(repository=repository, config=config)
    return {
        "provenance_schema_version": record.provenance_schema_version,
        "created_at": record.created_at,
        "git_commit": record.git.commit,
        "git_branch": record.git.branch,
        "git_dirty": record.git.dirty,
        "git_status": list(record.git.status),
        "git_diff_sha256": record.git.diff_sha256,
        "source_sha256": record.source_sha256,
        "source_files": [value.model_dump(mode="json") for value in record.source_files],
        "python": record.runtime.python,
        "python_executable": record.runtime.executable,
        "platform": record.runtime.platform,
        "machine": record.runtime.machine,
        "runtime_fingerprint": record.runtime.fingerprint,
        "package_versions": dict(record.runtime.package_versions),
        "config": dict(config),
    }


def write_json(path: Path, value: Mapping[str, Any] | BaseModel) -> None:
    """Atomically replace a JSON mapping beside any existing version."""

    atomic_write_json(path, value)


def _atomic_dataframe_write(frame: pd.DataFrame, path: Path, *, parquet: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        if parquet:
            frame.to_parquet(temporary, index=False)
        else:
            frame.to_csv(temporary, index=False)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_table(frame: pd.DataFrame, stem: Path) -> Path | None:
    """Write one legacy flat table, preferring Parquet with a CSV fallback."""

    if frame.empty:
        return None
    selected = select_table_format("auto")
    if selected == "parquet":
        parquet = stem.with_suffix(".parquet")
        try:
            _atomic_dataframe_write(frame, parquet, parquet=True)
            return parquet
        except (ImportError, ModuleNotFoundError, ValueError):
            # Preserve the v1 promise that unsupported Parquet values still have
            # a machine-readable CSV representation.
            pass
    csv = stem.with_suffix(".csv")
    _atomic_dataframe_write(frame, csv, parquet=False)
    return csv


def _filter_frame(frame: pd.DataFrame, filters: Mapping[str, Any] | None) -> pd.DataFrame:
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


def _direct_table_files(source: Path, table: str) -> tuple[Path, ...]:
    if source.is_file():
        return (source,)
    flat = [source / f"{table}.parquet", source / f"{table}.csv"]
    existing = tuple(path for path in flat if path.is_file())
    if existing:
        return existing[:1]
    parts = tuple(sorted(source.glob("part-*.parquet")))
    if parts:
        return parts
    return tuple(sorted(source.glob("part-*.csv")))


def _read_direct_file(
    source: Path,
    *,
    columns: Sequence[str] | None,
    filters: Mapping[str, Any] | None,
) -> pd.DataFrame:
    read_columns: list[str] | None = None
    if columns is not None:
        read_columns = list(dict.fromkeys([*columns, *(filters or {}).keys()]))
    if source.suffix == ".parquet":
        frame = pd.read_parquet(source, columns=read_columns)
    elif source.suffix == ".csv":
        frame = pd.read_csv(source, usecols=read_columns)
    else:
        raise ValueError(f"Unsupported results table: {source}")
    frame = _filter_frame(frame, filters)
    if columns is not None:
        frame = frame.loc[:, list(columns)]
    return frame


def iter_table(
    path: str | Path,
    table: str = "episodes",
    *,
    columns: Sequence[str] | None = None,
    filters: Mapping[str, Any] | None = None,
    batch_size: int | None = None,
    verify: bool = False,
) -> Iterator[pd.DataFrame]:
    """Yield bounded v1 or v2 table batches.

    A directory containing ``manifest.json`` is Protocol v2. Other directories
    retain the original flat-file behavior or may point directly at a part set.
    """

    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size must be positive")
    source = Path(path)
    if source.is_dir() and (source / "manifest.json").is_file():
        yield from RunStore.open(source).iter_table(
            table,
            columns=columns,
            filters=filters,
            batch_size=batch_size,
            verify=verify,
        )
        return

    files = _direct_table_files(source, table)
    if not files:
        raise FileNotFoundError(f"No {table!r} table found under {source}")
    for table_file in files:
        frame = _read_direct_file(table_file, columns=columns, filters=filters)
        if frame.empty:
            continue
        size = batch_size or len(frame)
        for start in range(0, len(frame), size):
            yield frame.iloc[start : start + size].reset_index(drop=True)


def read_table(
    path: str | Path,
    table: str = "episodes",
    *,
    columns: Sequence[str] | None = None,
    filters: Mapping[str, Any] | None = None,
    verify: bool = False,
) -> pd.DataFrame:
    """Materialize a table from a v1 file/directory or Protocol v2 run."""

    parts = list(
        iter_table(
            path,
            table,
            columns=columns,
            filters=filters,
            verify=verify,
        )
    )
    if not parts:
        return pd.DataFrame(columns=list(columns) if columns is not None else None)
    return pd.concat(parts, ignore_index=True)


def write_q_snapshots(path: Path, snapshots: Mapping[str, np.ndarray]) -> Path | None:
    """Atomically store numeric legacy snapshots without object arrays or pickle."""

    if not snapshots:
        return None
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
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


__all__ = [
    "TRACKED_PACKAGES",
    "git_commit",
    "iter_table",
    "package_versions",
    "provenance",
    "read_table",
    "utc_timestamp",
    "write_json",
    "write_q_snapshots",
    "write_table",
]
