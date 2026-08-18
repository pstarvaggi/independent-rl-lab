from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rllab.experiments.artifacts import (
    ArtifactIntegrityError,
    RunStore,
    TrialAttemptWriter,
)
from rllab.experiments.persistence import iter_table, read_table


@pytest.mark.parametrize("table_format", ["csv", "parquet"])
def test_attempt_writer_streams_parts_and_reader_batches(tmp_path: Path, table_format: str) -> None:
    pytest.importorskip("pyarrow") if table_format == "parquet" else None
    store = RunStore.create(
        tmp_path / f"run-{table_format}",
        run_id=f"run-{table_format}",
        experiment_name="streaming",
        trials={"trial-1": {"seed": 1}},
        config={"episodes": 5},
        provenance={"source_sha256": "a" * 64},
        table_format=table_format,  # type: ignore[arg-type]
    )
    writer = store.start_attempt("trial-1", source_hash="b" * 64)
    writer.write_table(
        "episodes",
        pd.DataFrame(
            {"trial_id": ["trial-1"] * 3, "episode": [0, 1, 2], "return": [0.0, 1.0, 2.0]}
        ),
    )
    writer.write_table(
        "episodes",
        pd.DataFrame({"trial_id": ["trial-1"] * 2, "episode": [3, 4], "return": [3.0, 4.0]}),
    )
    writer.write_q_snapshots(
        {"episode_00000000": np.zeros((2, 2)), "episode_00000004": np.ones((2, 2))}
    )
    commit = writer.commit(metadata={"exact_source": "test"})
    store.record_commit(commit, verify=True)

    assert store.manifest.status == "complete"
    assert store.manifest.trials["trial-1"].row_counts == {
        "episodes": 5,
        "q_snapshots": 2,
    }
    references = store.artifact_references("episodes")
    assert [reference.part for reference in references] == [0, 1]
    assert all(reference.sha256 and reference.bytes > 0 for reference in references)

    snapshots = store.read_q_snapshots("trial-1", keys=("episode_00000004",), verify=True)
    assert list(snapshots) == ["episode_00000004"]
    np.testing.assert_array_equal(snapshots["episode_00000004"], np.ones((2, 2)))

    batches = list(store.iter_table("episodes", batch_size=2))
    assert [len(batch) for batch in batches] == [2, 1, 2]
    result = store.read_table(
        "episodes", columns=("episode", "return"), filters={"episode": [1, 4]}
    )
    pd.testing.assert_frame_equal(
        result.reset_index(drop=True),
        pd.DataFrame({"episode": [1, 4], "return": [1.0, 4.0]}),
        check_dtype=False,
    )
    assert not list(store.run_directory.rglob("*.tmp"))


def test_q_snapshot_reader_rejects_unknown_keys_and_uncommitted_attempts(tmp_path: Path) -> None:
    store = RunStore.create(
        tmp_path / "run-q",
        run_id="run-q",
        experiment_name="snapshots",
        trials={"committed": {}, "pending": {}},
        config={},
        provenance={},
        table_format="csv",
    )
    writer = store.start_attempt("committed")
    writer.write_q_snapshots({"episode_00000000": np.zeros((2, 2))})
    store.record_commit(writer.commit())

    with pytest.raises(KeyError, match="absent"):
        store.read_q_snapshots("committed", keys=("episode_99999999",))
    with pytest.raises(RuntimeError, match="no committed Q snapshots"):
        store.read_q_snapshots("pending")


def test_uncommitted_parts_are_invisible_and_reconcile_recovers_commit(tmp_path: Path) -> None:
    store = RunStore.create(
        tmp_path / "run-1",
        run_id="run-1",
        experiment_name="recovery",
        trials={"trial-1": {}},
        config={},
        provenance={},
        table_format="csv",
    )
    writer = store.start_attempt("trial-1")
    writer.write_table("episodes", pd.DataFrame({"episode": [0, 1]}))
    assert store.read_table("episodes").empty

    writer.commit()
    reopened = RunStore.open(store.run_directory)
    assert reopened.manifest.trials["trial-1"].status == "running"
    reopened.reconcile(verify=True)
    assert reopened.manifest.status == "complete"
    assert reopened.read_table("episodes")["episode"].tolist() == [0, 1]


def test_retry_selects_only_successful_attempt_without_duplicate_rows(tmp_path: Path) -> None:
    store = RunStore.create(
        tmp_path / "run-1",
        run_id="run-1",
        experiment_name="retry",
        trials={"good": {}, "retry": {}},
        config={},
        provenance={},
        table_format="csv",
    )
    good = store.start_attempt("good")
    good.write_table("episodes", pd.DataFrame({"trial_id": ["good"], "episode": [0]}))
    store.record_commit(good.commit())

    failed = store.start_attempt("retry")
    failed.write_table("episodes", pd.DataFrame({"trial_id": ["retry"], "episode": [999]}))
    store.record_failure(failed.fail(ValueError("injected failure")))
    assert store.manifest.status == "partial"
    assert store.read_table("episodes")["trial_id"].tolist() == ["good"]

    reservation = store.reserve_attempt("retry")
    assert reservation.attempt == 2
    retry = TrialAttemptWriter(reservation)
    retry.write_table("episodes", pd.DataFrame({"trial_id": ["retry"], "episode": [0]}))
    store.record_commit(retry.commit())

    result = store.read_table("episodes").sort_values("trial_id").reset_index(drop=True)
    assert result.to_dict("records") == [
        {"trial_id": "good", "episode": 0},
        {"trial_id": "retry", "episode": 0},
    ]
    assert store.manifest.status == "complete"


def test_checksum_verification_detects_committed_file_mutation(tmp_path: Path) -> None:
    store = RunStore.create(
        tmp_path / "run-1",
        run_id="run-1",
        experiment_name="integrity",
        trials={"trial-1": {}},
        config={},
        provenance={},
        table_format="csv",
    )
    writer = store.start_attempt("trial-1")
    reference = writer.write_table("episodes", pd.DataFrame({"episode": [0]}))
    assert reference is not None
    store.record_commit(writer.commit())
    (store.run_directory / reference.path).write_text("corrupt\n", encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match=r"size changed|checksum changed"):
        store.verify()


def test_persistence_facade_reads_v1_and_v2_lazily(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    pd.DataFrame({"episode": [0, 1, 2], "score": [1.0, 2.0, 3.0]}).to_csv(
        legacy / "episodes.csv", index=False
    )
    assert read_table(legacy, columns=("episode",))["episode"].tolist() == [0, 1, 2]
    assert [len(batch) for batch in iter_table(legacy, batch_size=2)] == [2, 1]

    store = RunStore.create(
        tmp_path / "v2",
        run_id="v2-run",
        experiment_name="facade",
        trials={"trial-1": {}},
        config={},
        provenance={},
        table_format="csv",
    )
    writer = store.start_attempt("trial-1")
    writer.write_table("episodes", pd.DataFrame({"episode": [4, 5]}))
    store.record_commit(writer.commit())
    assert read_table(store.run_directory)["episode"].tolist() == [4, 5]
