# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The forward migration behind the one row OMN-16919 refused (OMN-16923).

OMN-16919's census ran OMN-16915's replay-and-introspect verifier against the
.201 stability-test lane's ``omnibase_infra`` database and got exactly one
DIVERGENT verdict:
``node:node_projection_savings:074_create_savings_estimates.sql``, 11 structural
differences. It was correctly NOT declared -- an adoption asserts that the
applied SQL produced the schema the checked-in file produces, and here it plainly
had not.

The other six rows in this family were stale-revision rows whose old bytes
provably produced today's schema. This one is a different animal, and the
distinction is the whole ticket:

* Both revisions of the node file (the 2026-07-21 bytes the lane recorded and
  the current 2026-07-29 bytes) declare the IDENTICAL ``CREATE TABLE IF NOT
  EXISTS savings_estimates`` -- TEXT columns, ``NUMERIC(18,6)`` money columns,
  three named CHECKs. Their diff is only OMN-15376's reconciliation block. So
  the revision gap explains none of the 11 differences.
* What explains them is the dual-producer topology. The FLAT producer
  ``docker/migrations/forward/074_create_savings_estimates.sql`` creates this
  table in the SERVICE database with ``VARCHAR(255)``/``VARCHAR(64)`` and
  ``NUMERIC(14,6)``. The node file's guarded ``CREATE TABLE IF NOT EXISTS`` then
  silently no-opped against it.
* OMN-15376's block cannot repair that. It ADDs missing columns, SETs NOT NULL
  and ADDs missing constraints; it cannot widen the TYPE of a column that
  already exists. A clean re-apply of the current node bytes would still leave
  8 of the 11 standing.

So the honest resolution is a forward migration --
``docker/migrations/forward/101_converge_savings_estimates_to_node_declared_shape.sql``
-- and it belongs to the FLAT corpus because, per the OMN-15857 ownership ruling
(``tests/ci/test_llm_call_metrics_ownership_omn15857.py``), the flat set is the
only declaring owner the service database has. A node-scoped 085 would run
against ``omnidash_analytics``, which is already correct.

What this file proves, by execution rather than by restatement:

1. RED -- the drifted service schema, reconstructed by replaying the REAL
   historical corpus (flat 074, flat 076, the node file at the exact bytes the
   lane's checksum names, node 075), reproduces all 11 differences.
2. GREEN -- 101 drives that same database to zero differences, and the verifier
   then reaches ``divergent_verified``.
3. The convergence is LOSSLESS: every row survives with its values intact.
4. Every branch that would NOT be lossless RAISEs and changes nothing. This is
   the half that matters: a migration that widens when it can and truncates when
   it cannot is worse than no migration, because the truncation is silent.

Ticket: OMN-16923. Family: OMN-15857 / OMN-16915 / OMN-16919.
"""

# ruff: noqa: S608 -- the SQL below is assembled from literals defined in this
# file and from checked-in migration paths; there is no untrusted input here,
# and the same suppression is carried by the OMN-16915 sibling for the same
# reason.


# literal defined in this file; there is no untrusted input here.

from __future__ import annotations

import hashlib
import importlib.util
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

from tests.integration.migrations.test_application_migration_ledger_omn15413 import (
    Pg16Cluster,
)

pytestmark = [pytest.mark.integration, pytest.mark.postgres, pytest.mark.serial]

REPO_ROOT = Path(__file__).resolve().parents[3]
FORWARD = REPO_ROOT / "docker" / "migrations" / "forward"
VERIFIER_PATH = (
    REPO_ROOT / "scripts" / "migrations" / "verify_migration_checksum_adoption.py"
)

NODE = "node_projection_savings"
FILENAME = "074_create_savings_estimates.sql"
VERSION = f"node:{NODE}:{FILENAME}"

# The bytes the .201 stability-test lane recorded in
# public.omnimarket_schema_migrations for this version, and the commit whose
# tree carries exactly those bytes (superseded by 78b873110 / OMN-15376 on
# 2026-07-29). Pinned so a test that silently replayed the CURRENT file could
# not pass for the wrong reason.
LANE_CHECKSUM = "d5eedd28f26c32f2e9d2a8554a999209c68216dc6a1ee255a973bd034164ce55"
PRIOR_REVISION = "5b904d881ba51a697e5b3d50b28460abbb2fd5aa"
# What _ledger/application-migrations.tsv declares for the same version.
MANIFEST_CHECKSUM = "b78acc5ba3144f9a7c7d85fd0fd5803b02b60503765fc58f8650a6a2bde27f4e"

MIGRATION_101 = FORWARD / "101_converge_savings_estimates_to_node_declared_shape.sql"
ROLLBACK_101 = (
    REPO_ROOT
    / "docker"
    / "migrations"
    / "rollback"
    / "rollback_101_converge_savings_estimates_to_node_declared_shape.sql"
)
FLAT_074 = FORWARD / "074_create_savings_estimates.sql"
FLAT_076 = FORWARD / "076_add_savings_estimate_provenance.sql"
NODE_075 = FORWARD / "nodes" / NODE / "075_add_savings_estimates_updated_at.sql"

# The 11 differences the OMN-16919 receipt recorded, pinned as (column, live
# type, declared type) and (constraint) rather than as a bare count -- a count
# survives the divergence changing shape entirely.
DRIFTED_COLUMN_TYPES = {
    "session_id": ("character varying(255)", "text"),
    "model_local": ("character varying(255)", "text"),
    "model_cloud_baseline": ("character varying(255)", "text"),
    "repo_name": ("character varying(255)", "text"),
    "machine_id": ("character varying(64)", "text"),
    "local_cost_usd": ("numeric(14,6)", "numeric(18,6)"),
    "cloud_cost_usd": ("numeric(14,6)", "numeric(18,6)"),
    "savings_usd": ("numeric(14,6)", "numeric(18,6)"),
}
DECLARED_CONSTRAINTS = (
    "savings_estimates_local_cost_usd_check",
    "savings_estimates_cloud_cost_usd_check",
    "savings_estimates_amounts_match",
)
EXPECTED_DIVERGENCES = len(DRIFTED_COLUMN_TYPES) + len(DECLARED_CONSTRAINTS)

# Two rows whose values exercise both widenings: a value that needs more than
# NUMERIC(14,6)'s 8 integer digits could not have been stored BEFORE the
# widening, so the fixture stays inside the narrow shape and the test asserts
# byte-equality across it instead.
SEED_ROWS = """
INSERT INTO savings_estimates
  (event_timestamp, session_id, model_local, model_cloud_baseline,
   local_cost_usd, cloud_cost_usd, savings_usd, repo_name, machine_id)
VALUES
  (TIMESTAMPTZ '2026-08-01 00:00:00+00', 'sess-a', 'qwen-local', 'claude-cloud',
   1.250000, 9.750000, 8.500000, 'omnibase_infra', 'mac-01'),
  (TIMESTAMPTZ '2026-08-02 00:00:00+00', 'sess-b', 'qwen-local', 'claude-cloud',
   0.000000, 0.000000, 0.000000, NULL, NULL);
"""


def _load_verifier() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "verify_migration_checksum_adoption_omn16923", VERIFIER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


verifier = _load_verifier()


def _counter() -> Iterator[int]:
    value = 0
    while True:
        yield value
        value += 1


_COUNTER = _counter()


@pytest.fixture
def service_db(pg16: Pg16Cluster) -> tuple[str, object]:
    """An empty database standing in for the SERVICE database, plus a client."""
    database = f"omn16923_service_{next(_COUNTER)}"
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
        label="service",
    )
    return database, client


@pytest.fixture
def pg16_fresh_database(pg16: Pg16Cluster) -> str:
    """An empty database with nothing applied to it at all."""
    database = f"omn16923_fresh_{next(_COUNTER)}"
    pg16.create_database(database)
    return database


def _prior_revision_bytes() -> str:
    """The node file EXACTLY as the lane's recorded checksum names it."""
    artifact = Path("docker/migrations/forward/nodes") / NODE / FILENAME
    revision = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "show", f"{PRIOR_REVISION}:{artifact}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if revision.returncode != 0:
        pytest.skip(
            f"commit {PRIOR_REVISION[:12]} is not present in this clone "
            "(shallow checkout); the historical bytes cannot be replayed"
        )
    old_sql = revision.stdout
    assert hashlib.sha256(old_sql.encode()).hexdigest() == LANE_CHECKSUM, (
        "this fixture is only meaningful if it applies the exact bytes the "
        "lane's ledger row names"
    )
    return old_sql


def _apply(pg16: Pg16Cluster, database: str, sql: str, *, check: bool = True) -> str:
    completed = pg16.command(database, "-f", "-", input_text=sql, check=check)
    return completed.stdout + completed.stderr


def _build_drifted_service_schema(pg16: Pg16Cluster, database: str) -> None:
    """Reconstruct the live service shape by REPLAYING the real corpus.

    Not hand-written DDL: hand-written DDL would only prove that 101 fixes the
    schema someone typed into this file. Replaying the actual producers, in the
    order the runner applies them, is what makes the RED case evidence about the
    lane rather than about the fixture.
    """
    _apply(pg16, database, FLAT_074.read_text(encoding="utf-8"))
    _apply(pg16, database, FLAT_076.read_text(encoding="utf-8"))
    _apply(pg16, database, SEED_ROWS)
    _apply(pg16, database, _prior_revision_bytes())
    _apply(pg16, database, NODE_075.read_text(encoding="utf-8"))


def _verify(database: str, client: object) -> object:
    bin_dir = verifier._postgres_bin_dir()
    assert bin_dir is not None, "a local Postgres is required to replay migrations"
    with tempfile.TemporaryDirectory(prefix="omn16923-scratch-") as tmp:
        scratch = verifier.ScratchServer(bin_dir, Path(tmp))
        scratch.start()
        try:
            return verifier.verify_row(
                version=VERSION,
                source_checksum=LANE_CHECKSUM,
                source_set="omnimarket",
                database=database,
                success_verdict=verifier.VERDICT_DIVERGENT_VERIFIED,
                live=client,
                scratch=scratch,
                manifest=verifier.load_manifest(),
                legacy=verifier.load_legacy_declarations(),
            )
        finally:
            scratch.stop()


def _column_types(pg16: Pg16Cluster, database: str) -> dict[str, str]:
    rows = pg16.sql(
        database,
        "SELECT a.attname || '=' || format_type(a.atttypid, a.atttypmod) "
        "FROM pg_attribute a "
        "WHERE a.attrelid = 'public.savings_estimates'::regclass "
        "AND a.attnum > 0 AND NOT a.attisdropped ORDER BY a.attnum",
    )
    return dict(line.split("=", 1) for line in rows.splitlines() if line)


def _constraint_names(pg16: Pg16Cluster, database: str) -> set[str]:
    rows = pg16.sql(
        database,
        "SELECT conname FROM pg_constraint "
        "WHERE conrelid = 'public.savings_estimates'::regclass ORDER BY conname",
    )
    return {line for line in rows.splitlines() if line}


# ---------------------------------------------------------------------------
# where the migration lives
# ---------------------------------------------------------------------------


def test_the_convergence_lives_in_the_flat_corpus_not_the_node_corpus() -> None:
    """Pins the placement ruling, which is the easiest thing here to get wrong.

    Node migrations are applied against NODE_PGDB, which every compose lane pins
    to omnidash_analytics. The divergent object is in omnibase_infra. A
    node-scoped file would therefore have converged the database that was
    already correct and never touched the one that was not.
    """
    assert MIGRATION_101.is_file(), (
        "the convergence must be a flat migration -- see "
        "tests/ci/test_llm_call_metrics_ownership_omn15857.py for the ruling"
    )
    assert ROLLBACK_101.is_file(), "a flat forward migration ships with its rollback"
    node_dir = FORWARD / "nodes" / NODE
    assert not list(node_dir.glob("085_*savings_estimates*")), (
        "converging the SERVICE database from the node corpus would run the "
        "migration against omnidash_analytics, which is already correct"
    )


def test_the_migration_does_not_edit_an_already_applied_file() -> None:
    """OMN-16705: an in-place edit to an applied migration bricks the lane."""
    changed = subprocess.run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "diff",
            "--name-only",
            "origin/dev...HEAD",
            "--",
            "docker/migrations/forward/074_create_savings_estimates.sql",
            f"docker/migrations/forward/nodes/{NODE}/{FILENAME}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if changed.returncode != 0:
        pytest.skip("origin/dev is not resolvable in this checkout")
    assert changed.stdout.strip() == "", (
        "074 (flat or node) must not be edited to make the shapes agree -- the "
        "convergence is an appended migration, not a rewrite of history"
    )


def test_the_pinned_checksums_still_name_the_manifest_and_the_prior_revision() -> None:
    """Anti-rot: both halves of the divergence are pinned, so a manifest bump
    that invalidates this proof fails here instead of on a lane."""
    manifest = verifier.load_manifest()
    assert manifest[VERSION]["checksum"] == MANIFEST_CHECKSUM
    assert (
        hashlib.sha256((FORWARD / "nodes" / NODE / FILENAME).read_bytes()).hexdigest()
        == MANIFEST_CHECKSUM
    )
    assert LANE_CHECKSUM != MANIFEST_CHECKSUM
    assert hashlib.sha256(_prior_revision_bytes().encode()).hexdigest() == LANE_CHECKSUM


def test_both_revisions_declare_the_same_table_so_the_gap_explains_nothing() -> None:
    """The load-bearing premise of the ticket, checked rather than asserted.

    If the two revisions HAD declared different tables, this would have been an
    OMN-16915-class stale-revision row and a declaration, not a migration, would
    have been the honest fix.
    """
    current = (FORWARD / "nodes" / NODE / FILENAME).read_text(encoding="utf-8")
    prior = _prior_revision_bytes()
    marker = "CREATE TABLE IF NOT EXISTS savings_estimates ("
    for text in (current, prior):
        assert marker in text
    current_block = current.split(marker, 1)[1].split(");", 1)[0]
    prior_block = prior.split(marker, 1)[1].split(");", 1)[0]
    assert current_block == prior_block, (
        "the two revisions declare DIFFERENT tables -- this row is then a "
        "stale-revision case after all and the placement of this whole ticket "
        "needs re-deciding"
    )
    assert "NUMERIC(18, 6)" in current_block
    assert "NUMERIC(14, 6)" in FLAT_074.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# RED / GREEN
# ---------------------------------------------------------------------------


def test_the_drifted_service_schema_reproduces_every_recorded_divergence(
    pg16: Pg16Cluster, service_db: tuple[str, object]
) -> None:
    """RED. Replaying the real corpus reproduces the OMN-16919 receipt exactly."""
    database, client = service_db
    _build_drifted_service_schema(pg16, database)

    types = _column_types(pg16, database)
    for column, (live_type, declared_type) in DRIFTED_COLUMN_TYPES.items():
        assert types[column] == live_type, column
        assert live_type != declared_type

    names = _constraint_names(pg16, database)
    for constraint in DECLARED_CONSTRAINTS:
        assert constraint not in names, constraint
    # The predicates ARE enforced -- under the flat file's own names. That is
    # why adding the declared names below cannot fail on data.
    assert {"non_negative_local", "non_negative_cloud", "savings_consistency"} <= names

    verdict = _verify(database, client)
    assert verdict.verdict == verifier.VERDICT_DIVERGENT, verdict.reason
    assert len(verdict.divergences) == EXPECTED_DIVERGENCES, verdict.divergences


def test_migration_101_converges_the_drifted_schema(
    pg16: Pg16Cluster, service_db: tuple[str, object]
) -> None:
    """GREEN. The same database, one migration later, is indistinguishable from
    what the node file produces."""
    database, client = service_db
    _build_drifted_service_schema(pg16, database)
    _apply(pg16, database, MIGRATION_101.read_text(encoding="utf-8"))

    types = _column_types(pg16, database)
    for column, (_live_type, declared_type) in DRIFTED_COLUMN_TYPES.items():
        assert types[column] == declared_type, column
    assert set(DECLARED_CONSTRAINTS) <= _constraint_names(pg16, database)

    verdict = _verify(database, client)
    assert verdict.verdict == verifier.VERDICT_DIVERGENT_VERIFIED, verdict.divergences
    assert verdict.divergences == []


def test_migration_101_is_idempotent(
    pg16: Pg16Cluster, service_db: tuple[str, object]
) -> None:
    """Re-running must be a no-op: the runner may replay a file after a partial
    lane failure, and a second run that raised would turn a recoverable deploy
    into a stuck one."""
    database, _client = service_db
    _build_drifted_service_schema(pg16, database)
    sql = MIGRATION_101.read_text(encoding="utf-8")
    _apply(pg16, database, sql)
    after_first = (_column_types(pg16, database), _constraint_names(pg16, database))
    # Anti-vacuity: idempotence of a migration that did nothing is not evidence.
    assert after_first[0]["local_cost_usd"] == "numeric(18,6)"
    assert set(DECLARED_CONSTRAINTS) <= after_first[1]
    _apply(pg16, database, sql)
    _apply(pg16, database, sql)
    assert (
        _column_types(pg16, database),
        _constraint_names(pg16, database),
    ) == after_first


def test_migration_101_preserves_every_row_value(
    pg16: Pg16Cluster, service_db: tuple[str, object]
) -> None:
    """The losslessness claim, on data rather than on catalog metadata."""
    database, _client = service_db
    _build_drifted_service_schema(pg16, database)
    projection = (
        "SELECT session_id, model_local, model_cloud_baseline, "
        "local_cost_usd, cloud_cost_usd, savings_usd, "
        "coalesce(repo_name, '<null>'), coalesce(machine_id, '<null>') "
        "FROM savings_estimates ORDER BY event_timestamp"
    )
    before = pg16.sql(database, projection)
    assert before.strip(), "the fixture must carry rows or this proves nothing"

    _apply(pg16, database, MIGRATION_101.read_text(encoding="utf-8"))

    # Anti-vacuity: the widening must actually have happened, or "the rows are
    # unchanged" is a statement about a migration that did nothing.
    assert _column_types(pg16, database)["savings_usd"] == "numeric(18,6)"
    assert pg16.sql(database, projection) == before
    assert pg16.sql(database, "SELECT count(*) FROM savings_estimates") == "2"


def test_the_fresh_service_path_and_the_drifted_path_end_at_one_schema(
    pg16: Pg16Cluster, service_db: tuple[str, object]
) -> None:
    """A fresh service bring-up runs flat 074, flat 076 and 101 -- and NOTHING
    from the node corpus. It must still land on the schema the node file
    declares, or this migration's central claim is false.

    Asserted through the verifier's own surface diff rather than against a
    hand-picked column list. An earlier revision of this test checked only the
    eight widened columns and the three constraints, and passed while
    `updated_at` and `ux_savings_estimates_identity` -- both in the declared
    surface, both left behind by node 074/075 on the live lane and by nothing at
    all on a fresh one -- were absent. The live-lane replica could not have
    surfaced that gap; only the fresh path can.
    """
    database, client = service_db
    _apply(pg16, database, FLAT_074.read_text(encoding="utf-8"))
    _apply(pg16, database, FLAT_076.read_text(encoding="utf-8"))
    _apply(pg16, database, MIGRATION_101.read_text(encoding="utf-8"))

    verdict = _verify(database, client)
    assert verdict.verdict == verifier.VERDICT_DIVERGENT_VERIFIED, verdict.divergences
    assert verdict.divergences == []


def test_101_supplies_the_two_objects_only_the_node_corpus_declares(
    pg16: Pg16Cluster, pg16_fresh_database: str
) -> None:
    """Names the two objects explicitly, so a future edit that drops step 4
    fails here with the reason attached rather than only inside a surface diff."""
    flat_only = (
        FLAT_074.read_text(encoding="utf-8")
        + "\n"
        + FLAT_076.read_text(encoding="utf-8")
    )
    _apply(pg16, pg16_fresh_database, flat_only)
    assert "updated_at" not in _column_types(pg16, pg16_fresh_database), (
        "premise: the flat corpus alone does not declare updated_at"
    )

    _apply(pg16, pg16_fresh_database, MIGRATION_101.read_text(encoding="utf-8"))

    assert _column_types(pg16, pg16_fresh_database)["updated_at"] == (
        "timestamp with time zone"
    )
    indexes = pg16.sql(
        pg16_fresh_database,
        "SELECT indexname FROM pg_indexes WHERE schemaname = 'public' "
        "AND tablename = 'savings_estimates'",
    )
    assert "ux_savings_estimates_identity" in indexes


def test_step_4_converges_a_wrongly_typed_updated_at_instead_of_accepting_it(
    pg16: Pg16Cluster, service_db: tuple[str, object]
) -> None:
    """`IF NOT EXISTS` guards a NAME, never a DEFINITION.

    A pre-existing `updated_at` of the wrong type would survive a bare guarded
    add and leave this file claiming a convergence it did not perform. The
    conversion is therefore unconditional -- and a `timestamp without time zone`
    source is REFUSED rather than cast, because that cast re-stamps every stored
    value against the session time zone: a data change wearing a type change's
    clothes.
    """
    database, _client = service_db
    _build_drifted_service_schema(pg16, database)
    _apply(
        pg16,
        database,
        "ALTER TABLE savings_estimates "
        "ALTER COLUMN updated_at TYPE TIMESTAMP WITHOUT TIME ZONE;",
    )

    completed = pg16.command(
        database,
        "-f",
        "-",
        input_text=MIGRATION_101.read_text(encoding="utf-8"),
        check=False,
    )
    assert completed.returncode != 0
    assert "would REINTERPRET every stored value" in completed.stderr, completed.stderr
    assert _column_types(pg16, database)["updated_at"] == "timestamp without time zone"


def test_step_4_recreates_the_identity_index_rather_than_asserting_its_name(
    pg16: Pg16Cluster, service_db: tuple[str, object]
) -> None:
    """Same class as above, for the index: an object of the right NAME over the
    wrong COLUMNS must not be left standing."""
    database, _client = service_db
    _build_drifted_service_schema(pg16, database)
    _apply(
        pg16,
        database,
        "DROP INDEX ux_savings_estimates_identity; "
        "CREATE UNIQUE INDEX ux_savings_estimates_identity "
        "ON savings_estimates (id);",
    )

    _apply(pg16, database, MIGRATION_101.read_text(encoding="utf-8"))

    definition = pg16.sql(
        database,
        "SELECT indexdef FROM pg_indexes "
        "WHERE indexname = 'ux_savings_estimates_identity'",
    )
    assert "session_id, event_timestamp, model_local, model_cloud_baseline" in (
        definition
    ), definition


def test_step_4_refuses_duplicate_identity_tuples_before_creating_the_index(
    pg16: Pg16Cluster, pg16_fresh_database: str
) -> None:
    """If a lane lost flat 074's unique constraint, 101 emits a ticketed data
    diagnostic instead of relying on PostgreSQL's bare CREATE UNIQUE INDEX
    failure."""
    _apply(pg16, pg16_fresh_database, FLAT_074.read_text(encoding="utf-8"))
    _apply(pg16, pg16_fresh_database, FLAT_076.read_text(encoding="utf-8"))
    _apply(
        pg16,
        pg16_fresh_database,
        """
        ALTER TABLE savings_estimates DROP CONSTRAINT unique_savings_estimate_event;
        INSERT INTO savings_estimates
          (event_timestamp, session_id, model_local, model_cloud_baseline,
           local_cost_usd, cloud_cost_usd, savings_usd, repo_name, machine_id)
        VALUES
          (TIMESTAMPTZ '2026-08-01 00:00:00+00', 'sess-dup', 'qwen-local',
           'claude-cloud', 1.000000, 2.000000, 1.000000, 'omnibase_infra', 'mac-01'),
          (TIMESTAMPTZ '2026-08-01 00:00:00+00', 'sess-dup', 'qwen-local',
           'claude-cloud', 1.000000, 2.000000, 1.000000, 'omnibase_infra', 'mac-02');
        """,
    )

    completed = pg16.command(
        pg16_fresh_database,
        "-f",
        "-",
        input_text=MIGRATION_101.read_text(encoding="utf-8"),
        check=False,
    )

    assert completed.returncode != 0
    assert (
        "OMN-16923: refusing to create ux_savings_estimates_identity"
        in completed.stderr
    )
    indexes = pg16.sql(
        pg16_fresh_database,
        "SELECT indexname FROM pg_indexes WHERE schemaname = 'public' "
        "AND tablename = 'savings_estimates'",
    )
    assert "ux_savings_estimates_identity" not in indexes


def test_the_flat_unique_constraint_already_covers_the_identity_tuple() -> None:
    """Why step 4's CREATE UNIQUE INDEX cannot fail on duplicates.

    Flat 074 declares `unique_savings_estimate_event` over the identical four
    columns in the identical order, so a duplicate tuple was already impossible
    on any database the flat corpus built. Asserted here so a future edit to
    either declaration surfaces as a failing test rather than as a deploy that
    aborts on a unique violation.
    """
    flat = FLAT_074.read_text(encoding="utf-8")
    node = (FORWARD / "nodes" / NODE / FILENAME).read_text(encoding="utf-8")
    tuple_columns = (
        "session_id",
        "event_timestamp",
        "model_local",
        "model_cloud_baseline",
    )

    flat_block = flat.split("CONSTRAINT unique_savings_estimate_event UNIQUE (", 1)[1]
    flat_block = flat_block.split(")", 1)[0]
    assert [c.strip() for c in flat_block.split(",") if c.strip()] == list(
        tuple_columns
    )

    node_block = node.split(
        "CREATE UNIQUE INDEX IF NOT EXISTS ux_savings_estimates_identity", 1
    )[1]
    node_block = node_block.split("(", 1)[1].split(")", 1)[0]
    assert [c.strip() for c in node_block.split(",") if c.strip()] == list(
        tuple_columns
    )


def test_the_rollback_leaves_the_two_step_4_objects_alone(
    pg16: Pg16Cluster, service_db: tuple[str, object]
) -> None:
    """The rollback's stated scope, asserted so it cannot drift from the header.

    It reverses what 101 CONVERTED. It does not remove `updated_at` or
    `ux_savings_estimates_identity`, because on the live lane node 074/075 own
    them and nothing at rollback time distinguishes that case from the fresh one
    -- an unconditional DROP would delete a NOT NULL column, and its data, that
    this migration never created.
    """
    database, _client = service_db
    _build_drifted_service_schema(pg16, database)
    _apply(pg16, database, MIGRATION_101.read_text(encoding="utf-8"))
    _apply(pg16, database, ROLLBACK_101.read_text(encoding="utf-8"))

    assert "updated_at" in _column_types(pg16, database)
    indexes = pg16.sql(
        database,
        "SELECT indexname FROM pg_indexes WHERE schemaname = 'public' "
        "AND tablename = 'savings_estimates'",
    )
    assert "ux_savings_estimates_identity" in indexes
    header = ROLLBACK_101.read_text(encoding="utf-8")
    assert "It does NOT remove the two objects" in header, (
        "the rollback header must keep stating the residual it leaves behind"
    )


def test_101_does_not_install_the_node_write_path_trigger(
    pg16: Pg16Cluster, pg16_fresh_database: str
) -> None:
    """Step 4 converges SHAPE. node 074 also creates
    refresh_savings_estimates_updated_at() and a BEFORE UPDATE trigger; those are
    behaviour, sit outside the surface the verifier measures, and changing the
    service database's write path is not something this ticket adjudicated."""
    _apply(pg16, pg16_fresh_database, FLAT_074.read_text(encoding="utf-8"))
    _apply(pg16, pg16_fresh_database, FLAT_076.read_text(encoding="utf-8"))
    _apply(pg16, pg16_fresh_database, MIGRATION_101.read_text(encoding="utf-8"))

    triggers = pg16.sql(
        pg16_fresh_database,
        "SELECT t.tgname FROM pg_trigger t "
        "JOIN pg_class c ON c.oid = t.tgrelid "
        "WHERE c.relname = 'savings_estimates' AND NOT t.tgisinternal",
    )
    assert triggers.strip() == ""


def test_101_requires_the_table_074_creates_and_074_is_not_skip_manifested(
    pg16: Pg16Cluster, pg16_fresh_database: str
) -> None:
    """101 has no procedural guard for an absent table -- it cannot, because the
    changed-SQL gate rejects every DO block. That is safe for exactly one
    reason, and it is a fact about the runner rather than about this file: the
    flat loop applies in sorted filename order, 074 sorts before 101, and 074 is
    not in the skip manifest. If either of those stopped being true, 101 would
    fail on a fresh service database -- so both are asserted here.
    """
    skip_manifest = (
        REPO_ROOT / "docker" / "migrations" / "skip-manifest.yaml"
    ).read_text(encoding="utf-8")
    assert "074_create_savings_estimates.sql" not in skip_manifest
    assert FLAT_074.name < MIGRATION_101.name

    completed = pg16.command(
        pg16_fresh_database,
        "-f",
        "-",
        input_text=MIGRATION_101.read_text(encoding="utf-8"),
        check=False,
    )
    assert completed.returncode != 0
    assert 'relation "savings_estimates" does not exist' in completed.stderr


# ---------------------------------------------------------------------------
# fail-closed: every branch that would NOT be lossless
# ---------------------------------------------------------------------------


def _drifted_with(pg16: Pg16Cluster, database: str, *statements: str) -> None:
    _build_drifted_service_schema(pg16, database)
    for statement in statements:
        _apply(pg16, database, statement)


# Every value-mutating case below drops 074's own CHECKs first: they are what
# keep the fixture honest in the other tests, and they would reject the drifted
# value before 101 ever saw it.
_UNGUARD = (
    "ALTER TABLE savings_estimates DROP CONSTRAINT savings_consistency; "
    "ALTER TABLE savings_estimates DROP CONSTRAINT non_negative_local; "
    "ALTER TABLE savings_estimates DROP CONSTRAINT non_negative_cloud; "
)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        pytest.param(
            _UNGUARD + "ALTER TABLE savings_estimates ALTER COLUMN local_cost_usd "
            "TYPE NUMERIC(20, 8); "
            "UPDATE savings_estimates SET local_cost_usd = 1.12345678 "
            "WHERE session_id = 'sess-a';",
            "would be ROUNDED",
            id="a-stored-value-with-8-fractional-digits",
        ),
        pytest.param(
            _UNGUARD + "ALTER TABLE savings_estimates ALTER COLUMN cloud_cost_usd "
            "TYPE NUMERIC(24, 6); "
            "UPDATE savings_estimates SET cloud_cost_usd = 1234567890123.5 "
            "WHERE session_id = 'sess-a';",
            "would OVERFLOW",
            id="a-stored-value-with-13-integer-digits",
        ),
        pytest.param(
            _UNGUARD
            + "ALTER TABLE savings_estimates ALTER COLUMN savings_usd TYPE NUMERIC; "
            "UPDATE savings_estimates SET savings_usd = 8.500000001 "
            "WHERE session_id = 'sess-a';",
            "would be ROUNDED",
            id="unconstrained-numeric-holding-an-over-scale-value",
        ),
        pytest.param(
            "ALTER TABLE savings_estimates ALTER COLUMN machine_id "
            "TYPE BYTEA USING machine_id::bytea",
            "outside the character family",
            id="source-type-outside-the-character-family",
        ),
        pytest.param(
            "ALTER TABLE savings_estimates DROP COLUMN repo_name",
            'column "repo_name" does not exist',
            id="declared-column-absent",
        ),
    ],
)
def test_a_conversion_that_would_lose_data_is_refused(
    pg16: Pg16Cluster,
    service_db: tuple[str, object],
    mutation: str,
    expected: str,
) -> None:
    """Each of these is a shape 101 must REFUSE rather than silently mangle.

    Without them the migration is a liability: it would look like a widening in
    review and behave like a narrowing on a lane whose table drifted a second
    time. The assertion is on the message, not just the failure, so a refusal
    for an unrelated reason cannot masquerade as this one.

    `source-type-outside-the-character-family` is the case that would have
    shipped as a silent mangle: `ALTER COLUMN ... TYPE TEXT` does NOT refuse a
    bytea source -- it I/O-converts it to the string `\\x...` and reports
    success. The guard exists because that was checked rather than assumed.

    The whole file is one transaction, so the assertion below is that NOTHING
    changed -- not merely that the failing statement rolled itself back.
    """
    database, _client = service_db
    _drifted_with(pg16, database, mutation)

    before = _column_types(pg16, database)
    completed = pg16.command(
        database,
        "-f",
        "-",
        input_text=MIGRATION_101.read_text(encoding="utf-8"),
        check=False,
    )
    assert completed.returncode != 0, "the migration must not apply this conversion"
    assert expected in completed.stderr, completed.stderr
    assert _column_types(pg16, database) == before


def test_an_untaken_guard_branch_does_not_abort_the_migration(
    pg16: Pg16Cluster, service_db: tuple[str, object]
) -> None:
    """The guard messages are anchored to their column on purpose.

    PostgreSQL resolves a constant cast at PARSE time, so a bare
    `'OMN-16923: ...'::NUMERIC` sitting in a branch no row takes would abort the
    migration on EVERY database -- including every database with nothing wrong
    with it. Concatenating `left(<col>, 0)` makes the expression non-constant
    and defers it to execution. This is the test that would catch a future
    "simplification" that drops the concatenation.
    """
    database, _client = service_db
    _build_drifted_service_schema(pg16, database)
    guarded = MIGRATION_101.read_text(encoding="utf-8")
    assert "left(local_cost_usd::TEXT, 0)" in guarded, (
        "the guard must stay column-anchored or it folds at parse time"
    )
    _apply(pg16, database, guarded)
    assert _column_types(pg16, database)["local_cost_usd"] == "numeric(18,6)"


def test_rows_violating_a_declared_check_are_refused_not_forced(
    pg16: Pg16Cluster, service_db: tuple[str, object]
) -> None:
    """The three CHECKs cannot fail here today because 074's own equivalently
    named constraints already hold. That is a fact about the current corpus, not
    a guarantee -- so the migration counts violations and names them rather than
    letting ADD CONSTRAINT fail with a generic message on a future lane."""
    database, _client = service_db
    _build_drifted_service_schema(pg16, database)
    _apply(
        pg16,
        database,
        "ALTER TABLE savings_estimates DROP CONSTRAINT non_negative_local; "
        "ALTER TABLE savings_estimates DROP CONSTRAINT savings_consistency; "
        "UPDATE savings_estimates SET local_cost_usd = -1 "
        "WHERE session_id = 'sess-a';",
    )

    completed = pg16.command(
        database,
        "-f",
        "-",
        input_text=MIGRATION_101.read_text(encoding="utf-8"),
        check=False,
    )
    assert completed.returncode != 0
    assert "savings_estimates_local_cost_usd_check" in completed.stderr, (
        completed.stderr
    )
    assert "violated by some row" in completed.stderr, completed.stderr
    assert "savings_estimates_local_cost_usd_check" not in _constraint_names(
        pg16, database
    )


# ---------------------------------------------------------------------------
# the rollback is the narrowing, and narrowings are conditional
# ---------------------------------------------------------------------------


def test_the_rollback_refuses_to_truncate_a_value_that_no_longer_fits(
    pg16: Pg16Cluster, service_db: tuple[str, object]
) -> None:
    """After 101, machine_id is unbounded. A rollback to VARCHAR(64) is only
    safe against the rows that happen to be there, so it proves the fit first."""
    database, _client = service_db
    _build_drifted_service_schema(pg16, database)
    _apply(pg16, database, MIGRATION_101.read_text(encoding="utf-8"))
    _apply(
        pg16,
        database,
        "UPDATE savings_estimates SET machine_id = repeat('x', 200) "
        "WHERE session_id = 'sess-a';",
    )

    completed = pg16.command(
        database,
        "-f",
        "-",
        input_text=ROLLBACK_101.read_text(encoding="utf-8"),
        check=False,
    )
    assert completed.returncode != 0
    assert "would be TRUNCATED" in completed.stderr, completed.stderr
    assert _column_types(pg16, database)["machine_id"] == "text"
    assert (
        pg16.sql(
            database,
            "SELECT length(machine_id) FROM savings_estimates "
            "WHERE session_id = 'sess-a'",
        )
        == "200"
    )


def test_the_rollback_returns_the_flat_shape_when_every_row_fits(
    pg16: Pg16Cluster, service_db: tuple[str, object]
) -> None:
    database, _client = service_db
    _build_drifted_service_schema(pg16, database)
    _apply(pg16, database, MIGRATION_101.read_text(encoding="utf-8"))
    # Anti-vacuity: without this the drifted shape and the rolled-back shape are
    # the same thing and a no-op 101 would satisfy the assertions below.
    assert _column_types(pg16, database)["machine_id"] == "text"
    assert set(DECLARED_CONSTRAINTS) <= _constraint_names(pg16, database)
    _apply(pg16, database, ROLLBACK_101.read_text(encoding="utf-8"))

    types = _column_types(pg16, database)
    for column, (live_type, _declared) in DRIFTED_COLUMN_TYPES.items():
        assert types[column] == live_type, column
    names = _constraint_names(pg16, database)
    assert not set(DECLARED_CONSTRAINTS) & names
    # 074's own constraints are not this migration's to drop.
    assert {"non_negative_local", "non_negative_cloud", "savings_consistency"} <= names


# ---------------------------------------------------------------------------
# the committed declaration
# ---------------------------------------------------------------------------


def test_the_committed_divergent_adoption_is_attributed_to_this_ticket() -> None:
    """The row was earned by THIS migration, not by OMN-16915's stale-revision
    argument, and the ledger has to say so or the evidence pointer is a dead
    link -- the reader lands on a ticket that never made this claim."""
    declared = verifier.load_divergent_adoptions()[VERSION]
    assert declared["ticket"] == "OMN-16923"
    assert declared["source_checksum"] == LANE_CHECKSUM
    assert declared["manifest_checksum"] == MANIFEST_CHECKSUM


def test_the_declaration_ticket_override_is_validated_and_version_scoped() -> None:
    """The override must not be able to re-stamp a row it did not earn.

    Both emission loops start from the rows already on disk and overwrite by
    version, so a RUN-scoped override would silently re-attribute every other
    declaration the same run re-proved -- the OMN-16919 census alone covers
    seven. Keyed by version, a run that names one version cannot touch another.
    """
    parsed = verifier.parse_declaration_tickets([f"{VERSION}=OMN-16923"])
    assert parsed == {VERSION: "OMN-16923"}
    assert (
        parsed.get("node:node_projection_dep_health:001_create_dep_health_findings.sql")
        is None
    )

    for bad in ("OMN-16923", f"{VERSION}=see the PR", f"{VERSION}=OMN-", "=OMN-1"):
        with pytest.raises(verifier.VerificationError):
            verifier.parse_declaration_tickets([bad])

    with pytest.raises(verifier.VerificationError):
        verifier.parse_declaration_tickets(
            [f"{VERSION}=OMN-16923", f"{VERSION}=OMN-16915"]
        )

    assert verifier.parse_declaration_tickets(None) == {}


def test_a_later_run_without_an_override_preserves_an_existing_ticket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The override is only half the guarantee.

    A later ORDINARY `--emit-adoptions` run re-proves the same rows with no
    mapping at all. If the ticket column were then reset to the tool's default,
    the provenance the version-scoped override recorded would be erased one run
    later -- the same corruption, arriving by a different route. An existing
    row's ticket is therefore the fallback, and the tool's own ticket is used
    only for a row that is genuinely new.
    """
    existing = {
        VERSION: {
            "version": VERSION,
            "source_checksum": LANE_CHECKSUM,
            "manifest_checksum": MANIFEST_CHECKSUM,
            "ticket": "OMN-16923",
            "receipt_sha256": "0" * 64,
            "verified_at": "2026-08-29",
        }
    }
    # The fallback chain the emission loops use, exercised directly: no override
    # for this version, so the row on disk decides.
    declaration_tickets: dict[str, str] = {}
    resolved = declaration_tickets.get(
        VERSION, existing.get(VERSION, {}).get("ticket", verifier.DIVERGENT_TICKET)
    )
    assert resolved == "OMN-16923"

    fresh = declaration_tickets.get(
        "node:brand_new:0001_x.sql",
        existing.get("node:brand_new:0001_x.sql", {}).get(
            "ticket", verifier.DIVERGENT_TICKET
        ),
    )
    assert fresh == verifier.DIVERGENT_TICKET

    # And the shipped source really does read the row on disk, not a constant.
    source = VERIFIER_PATH.read_text(encoding="utf-8")
    assert "divergent_adoptions.get(verdict.version, {}).get(" in source
    assert 'adoptions.get(verdict.version, {}).get("ticket", TICKET)' in source


def test_the_other_committed_declarations_keep_their_own_ticket() -> None:
    """The six OMN-16915 rows must still say OMN-16915 after this run."""
    declared = verifier.load_divergent_adoptions()
    others = {
        version: row["ticket"]
        for version, row in declared.items()
        if version != VERSION
    }
    assert others, "the sibling declarations must still be present"
    assert set(others.values()) == {"OMN-16915"}, others
