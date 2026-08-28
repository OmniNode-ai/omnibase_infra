# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The equivalence proof behind a verified checksum adoption (OMN-15857).

``_ledger/verified-checksum-adoptions.tsv`` lets ``bootstrap.sql`` adopt the
manifest checksum for a row whose stored checksum is a hand-written sentinel.
That declaration is only as good as the proof behind it, and bootstrap.sql
cannot check the proof: it compares checksums and never looks at a table.

So the load-bearing assertion lives here.
``scripts/migrations/verify_migration_checksum_adoption.py`` replays the
checked-in migration into a scratch database, derives the object surface that
file is responsible for **by executing it** (snapshot before, snapshot after,
diff -- no SQL parsing), and compares that surface against the live database.

This file proves the verifier is not a rubber stamp:

* it says ``equivalent`` when the live schema really was produced by the file,
* it says ``divergent`` -- and refuses to emit an adoption -- for each way a
  hand-applied schema realistically drifts: a missing column, a widened type, a
  dropped constraint, a missing index, a table that was never created at all,
* it says ``unreachable``, never ``equivalent``, when it cannot decide.

Without the divergent cases a green ``equivalent`` proves nothing, because a
verifier that returns ``equivalent`` unconditionally would pass every other test
in this repo.

Ticket: OMN-15857
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

from tests.integration.migrations.test_application_migration_ledger_omn15413 import (
    Pg16Cluster,
)

pytestmark = [pytest.mark.integration, pytest.mark.postgres, pytest.mark.serial]

REPO_ROOT = Path(__file__).resolve().parents[3]
VERIFIER_PATH = (
    REPO_ROOT / "scripts" / "migrations" / "verify_migration_checksum_adoption.py"
)
VERSION = "node:node_projection_llm_cost:0001_create_llm_call_metrics.sql"
SENTINEL = "hotfix-applied-by-codex"


def _load_verifier() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "verify_migration_checksum_adoption", VERIFIER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


verifier = _load_verifier()


@pytest.fixture
def audited(pg16: Pg16Cluster) -> Iterator[tuple[str, verifier.PsqlClient]]:
    """A database standing in for a live lane, plus a client pointed at it."""
    database = f"omn15857_audited_{next(_COUNTER)}"
    pg16.create_database(database)
    client = verifier.PsqlClient(
        argv=(
            str(pg16.bin_dir / "psql"),
            "-h",
            "127.0.0.1",
            "-p",
            str(pg16.port),
            "-U",
            "postgres",
        ),
        label="audited",
    )
    return database, client


def _counter() -> Iterator[int]:
    value = 0
    while True:
        yield value
        value += 1


_COUNTER = _counter()


def _target_migration() -> Path:
    replay = verifier.resolve_replay_set(VERSION)
    assert replay is not None, "the node migration under test must be checked in"
    target, prefix = replay
    assert prefix == [], "this node declares a single migration; the test assumes it"
    return target


def _apply_faithfully(pg16: Pg16Cluster, database: str) -> None:
    """Put the database in the state a correct apply of the file would leave."""
    pg16.command(
        database, "-f", "-", input_text=_target_migration().read_text(encoding="utf-8")
    )


def _verify(database: str, client: verifier.PsqlClient) -> verifier.RowVerdict:
    bin_dir = verifier._postgres_bin_dir()
    assert bin_dir is not None, "a local Postgres is required to replay migrations"
    import tempfile

    with tempfile.TemporaryDirectory(prefix="omn15857-scratch-") as tmp:
        scratch = verifier.ScratchServer(bin_dir, Path(tmp))
        scratch.start()
        try:
            return verifier.verify_row(
                version=VERSION,
                source_checksum=SENTINEL,
                source_set="node",
                database=database,
                live=client,
                scratch=scratch,
                manifest=verifier.load_manifest(),
                legacy=verifier.load_legacy_declarations(),
            )
        finally:
            scratch.stop()


def test_a_faithfully_applied_schema_is_equivalent(
    pg16: Pg16Cluster, audited: tuple[str, verifier.PsqlClient]
) -> None:
    """The GREEN reference: same file in, same schema out, verdict equivalent."""
    database, client = audited
    _apply_faithfully(pg16, database)

    verdict = _verify(database, client)

    assert verdict.verdict == verifier.VERDICT_EQUIVALENT, verdict.divergences
    assert verdict.divergences == []
    # The surface was derived by execution, so it must actually name the objects
    # the file creates -- an empty surface would make "equivalent" vacuous, and
    # verify_row() refuses that case explicitly.
    assert set(verdict.declared_objects) == {
        "table:llm_call_metrics",
        "enum:usage_source_type",
    }
    assert verdict.manifest_checksum == verdict.artifact_sha256


@pytest.mark.parametrize(
    ("label", "drift_sql", "expected_fragment"),
    [
        (
            "dropped_column",
            "ALTER TABLE llm_call_metrics DROP COLUMN contract_version;",
            "column 'contract_version' missing live",
        ),
        (
            "widened_type",
            "ALTER TABLE llm_call_metrics ALTER COLUMN model_id TYPE TEXT;",
            "column 'model_id'",
        ),
        (
            "dropped_constraint",
            "ALTER TABLE llm_call_metrics DROP CONSTRAINT non_negative_total_tokens;",
            "constraint 'non_negative_total_tokens' missing live",
        ),
        (
            "dropped_index",
            "DROP INDEX idx_llm_call_metrics_model_created;",
            "index 'idx_llm_call_metrics_model_created' missing live",
        ),
        (
            "relaxed_nullability",
            "ALTER TABLE llm_call_metrics ALTER COLUMN model_id DROP NOT NULL;",
            "column 'model_id'",
        ),
    ],
)
def test_a_drifted_schema_is_divergent(
    pg16: Pg16Cluster,
    audited: tuple[str, verifier.PsqlClient],
    label: str,
    drift_sql: str,
    expected_fragment: str,
) -> None:
    """Each realistic hand-apply drift is refused, and named in the report."""
    database, client = audited
    _apply_faithfully(pg16, database)
    pg16.command(database, "-f", "-", input_text=drift_sql)

    verdict = _verify(database, client)

    assert verdict.verdict == verifier.VERDICT_DIVERGENT, label
    assert verdict.divergences, label
    joined = "\n".join(verdict.divergences)
    assert expected_fragment in joined, f"{label}: {joined}"
    assert verdict.verdict not in verifier.ADOPTABLE_VERDICTS


def test_a_table_that_was_never_created_is_divergent(
    audited: tuple[str, verifier.PsqlClient],
) -> None:
    """An empty database is the loudest divergence, not a silent pass."""
    database, client = audited

    verdict = _verify(database, client)

    assert verdict.verdict == verifier.VERDICT_DIVERGENT
    assert any("does not exist" in line for line in verdict.divergences)


def test_a_version_with_no_checked_in_file_is_unreachable_not_equivalent(
    audited: tuple[str, verifier.PsqlClient],
) -> None:
    """Undecidable is its own verdict. It must never collapse into equivalent."""
    database, client = audited
    bin_dir = verifier._postgres_bin_dir()
    assert bin_dir is not None
    import tempfile

    with tempfile.TemporaryDirectory(prefix="omn15857-scratch-") as tmp:
        scratch = verifier.ScratchServer(bin_dir, Path(tmp))
        scratch.start()
        try:
            verdict = verifier.verify_row(
                version="node:node_that_does_not_exist:0001_nothing.sql",
                source_checksum=SENTINEL,
                source_set="node",
                database=database,
                live=client,
                scratch=scratch,
                manifest=verifier.load_manifest(),
                legacy=verifier.load_legacy_declarations(),
            )
        finally:
            scratch.stop()

    assert verdict.verdict == verifier.VERDICT_UNREACHABLE
    assert verdict.verdict not in verifier.ADOPTABLE_VERDICTS


def test_a_legacy_declared_version_is_attested_not_verified(
    audited: tuple[str, verifier.PsqlClient],
) -> None:
    """The two OMN-15717 delegation rows have no file and need no adoption.

    bootstrap.sql already imports them as ``legacy_attestation``, which proves a
    source record rather than file bytes. The verifier must classify them that
    way instead of manufacturing an equivalence claim it cannot support.
    """
    database, client = audited
    legacy = verifier.load_legacy_declarations()
    version = (
        "node:node_projection_delegation:0014_create_live_event_projection_view.sql"
    )
    assert version in legacy, "OMN-15717 declaration expected in the ledger"

    bin_dir = verifier._postgres_bin_dir()
    assert bin_dir is not None
    import tempfile

    with tempfile.TemporaryDirectory(prefix="omn15857-scratch-") as tmp:
        scratch = verifier.ScratchServer(bin_dir, Path(tmp))
        scratch.start()
        try:
            verdict = verifier.verify_row(
                version=version,
                source_checksum=SENTINEL,
                source_set="node",
                database=database,
                live=client,
                scratch=scratch,
                manifest=verifier.load_manifest(),
                legacy=legacy,
            )
        finally:
            scratch.stop()

    assert verdict.verdict == verifier.VERDICT_LEGACY_ATTESTED
    assert verdict.verdict not in verifier.ADOPTABLE_VERDICTS


def test_the_committed_adoption_matches_its_committed_receipt() -> None:
    """The TSV's receipt_sha256 must resolve to a receipt that says equivalent.

    Without this the receipt hash is decoration: a row could cite any 64 hex
    characters. The check binds the declaration to a proof that is in the tree,
    covers this exact version, and reached the only adoptable verdict.
    """
    adoptions = verifier.load_adoptions()
    assert VERSION in adoptions, "the lane-blocking row must be declared"
    declared = adoptions[VERSION]

    receipts_dir = verifier.LEDGER_DIR / "receipts"
    matching = [
        path
        for path in sorted(receipts_dir.glob("*.json"))
        if verifier.file_sha256(path) == declared["receipt_sha256"]
    ]
    assert matching, (
        f"no receipt in {receipts_dir} hashes to "
        f"{declared['receipt_sha256']}; the adoption cites a proof that is not "
        "in the tree"
    )

    receipt = json.loads(matching[0].read_text(encoding="utf-8"))
    entry = next(item for item in receipt["verdicts"] if item["version"] == VERSION)
    assert entry["verdict"] == verifier.VERDICT_EQUIVALENT
    assert entry["source_checksum"] == declared["source_checksum"]
    assert entry["manifest_checksum"] == declared["manifest_checksum"]
    assert entry["declared_objects"], "an empty surface cannot prove equivalence"
