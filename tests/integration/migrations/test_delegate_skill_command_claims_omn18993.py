# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Migration contract checks for delegate_skill_command_claims (OMN-18993).

WHAT THESE ASSERT, AND WHAT THEY DO NOT
    The contract of the VENDORED migration pair that carries the durable
    delegate-skill claim: the key that makes a redelivery distinguishable from
    a reused correlation, the shape-reconciliation block that keeps the create
    idempotent in SHAPE rather than only in existence, and the grant
    assertions whose absence stays invisible until the relation takes traffic.

    They are file-contract checks in the shape of their neighbours in this
    directory, and they need no database, so they never skip and never
    contribute a false green to the merge-gating job. That property is the
    reason they are written this way: a collected-but-never-run test proves
    nothing, which the skip-count ratchet says by name.

    The live-execution half is NOT duplicated here. It already exists and
    already covers this file corpus-wide:
    ``tests/integration/migrations/test_node_migration_shape_drift_omn15376.py``
    derives an unfixed variant by deleting the reconciliation region from the
    real file, then proves RED on a drifted table, GREEN on the same drifted
    table with the region restored, and byte-identical schemas between the
    fresh and drifted paths. This migration carries the region markers that
    suite keys on, so it is picked up without being named.

WHY A CHECKSUM CHECK IS HERE
    These are VENDORED copies whose source of truth is the omnimarket node's
    own migrations directory, and the two drifting apart is a recurring defect
    with six recorded occurrences (OMN-14975 and the five it names). The
    cross-repo parity gate watches the direction where the SOURCE moves; this
    watches the other one, a hand edit landing here, and answers it locally
    against the checksum the forward-migration ledger already records rather
    than by reaching for a cross-repository checkout that is absent in every
    ordinary test split.

Ticket: OMN-18993 (vendoring), OMN-18887 (the idempotency defect),
OMN-19029 (the reconciliation block and the grant this pair now carries)
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[3]
NODE_DIR = (
    REPO_ROOT
    / "docker"
    / "migrations"
    / "forward"
    / "nodes"
    / "node_delegate_skill_orchestrator"
)
CREATE_FILE = NODE_DIR / "0001_delegate_skill_command_claims.sql"
GRANT_FILE = NODE_DIR / "0001_grant_omninode_runtime_delegate_skill_command_claims.sql"
LEDGER = (
    REPO_ROOT
    / "docker"
    / "migrations"
    / "forward"
    / "_ledger"
    / "application-migrations.tsv"
)

TABLE = "omninode_internal.delegate_skill_command_claims"
SCHEMA = "omninode_internal"
INDEX = "delegate_skill_command_claims_correlation_idx"

#: Every column the create declares. Named here rather than parsed out of the
#: file, so a column silently dropped from the migration fails this list
#: instead of quietly shrinking both sides of the comparison together.
DECLARED_COLUMNS: tuple[str, ...] = (
    "delivery_id",
    "correlation_id",
    "claimed_at",
    "terminal_json",
)

RECONCILIATION_BEGIN = "-- ---- BEGIN OMN-15376 shape reconciliation:"
RECONCILIATION_END = "-- ---- END OMN-15376 shape reconciliation:"


def _executable_sql(path: Path) -> str:
    """The file with its ``--`` comment lines removed.

    The absence assertions below have to read this rather than the raw text.
    Both migrations explain IN A COMMENT why they do not use the construct
    being forbidden, so a raw substring search fires on the documentation
    about the rule and reports the file as violating the thing it explains --
    a check that can only ever be satisfied by deleting the explanation.
    """
    return "\n".join(
        line
        for line in path.read_text().splitlines()
        if not line.lstrip().startswith("--")
    )


@pytest.mark.integration
def test_the_create_keys_on_the_delivering_record_not_the_correlation() -> None:
    """The primary key is what separates a redelivery from a reused correlation.

    Correlation is the RETRY identity by construction: it defaults to a fresh
    uuid4 but a caller may supply one, and callers do reuse them. Keyed on
    correlation, a reused one would be answered with a stale terminal and
    never dispatched -- a worse defect than the double-bill this table exists
    to prevent. A redelivery is the same record twice and shares the delivery
    id; a new command reusing a correlation is a different record and does not.
    """
    sql = CREATE_FILE.read_text()
    assert f"CREATE TABLE IF NOT EXISTS {TABLE} (" in sql
    assert "delivery_id    TEXT PRIMARY KEY" in sql


@pytest.mark.integration
def test_the_create_declares_every_column_the_claim_protocol_reads() -> None:
    """Each column answers one question the claim decision asks.

    claimed_at is written on INSERT and never on conflict, so the value
    returned to a caller is the FIRST claimer's and comparing it against what
    that caller passed is the "did I win" answer in one statement.
    terminal_json is how a suppressed redelivery still ANSWERS rather than
    publishing nothing, which would convert a double-bill into the
    missing-envelope defect OMN-15504 exists to prevent.
    """
    sql = CREATE_FILE.read_text()
    for column in DECLARED_COLUMNS:
        assert f"    {column} " in sql, f"{column} is not declared in the create"


@pytest.mark.integration
def test_the_create_reconciles_shape_and_not_merely_existence() -> None:
    """CREATE TABLE IF NOT EXISTS no-ops against a DIFFERENT pre-existing shape.

    Without a guarded ADD COLUMN per declared column, a drifted table of this
    name silently keeps its own shape, and the correlation index that follows
    guards the index NAME rather than the column, so it raises
    ``column "correlation_id" does not exist`` under ON_ERROR_STOP=1 and takes
    the whole forward-migration run with it. That is the OMN-15376 class, with
    two live instances behind it.
    """
    sql = CREATE_FILE.read_text()
    for column in DECLARED_COLUMNS:
        assert f"ADD COLUMN IF NOT EXISTS {column}" in sql, (
            f"{column} is declared but has no guarded ADD COLUMN, so it would "
            "silently not arrive on a drifted database"
        )


@pytest.mark.integration
def test_the_guarded_adds_are_nullable() -> None:
    """A guarded add spelled NOT NULL cannot reconcile a table holding rows.

    This is the OMN-16777 lesson, learned on node_projection_consumer_flow:
    0000 and 0001 both spelled their guarded adds ``... NOT NULL``, Postgres
    refused with "contains null values" under ON_ERROR_STOP=1, and the static
    gate stayed green the whole time because the columns WERE covered. A
    reconciliation that only works on an empty table is not a reconciliation.
    """
    sql = _executable_sql(CREATE_FILE)
    begin = sql.index("ADD COLUMN IF NOT EXISTS")
    region = sql[begin:]
    for column in DECLARED_COLUMNS:
        marker = f"ADD COLUMN IF NOT EXISTS {column}"
        line_start = region.index(marker)
        statement = region[line_start : region.index(";", line_start)]
        assert "NOT NULL" not in statement, (
            f"the guarded add for {column} is spelled NOT NULL, which cannot "
            "reconcile a drifted table that already holds rows"
        )


@pytest.mark.integration
def test_the_reconciliation_precedes_the_column_dependent_index() -> None:
    """Ordering is the whole mechanism, not a style preference.

    The reconciliation has to sit between the create and the first statement
    that names a column, because that statement is what fails. Placed after
    the index, the block would be correct and useless.
    """
    sql = CREATE_FILE.read_text()
    assert RECONCILIATION_BEGIN in sql and RECONCILIATION_END in sql, (
        "the region markers are missing, so the corpus-wide execution proof "
        "in test_node_migration_shape_drift_omn15376.py cannot derive its RED "
        "variant from this file and would silently stop covering it"
    )
    assert (
        sql.index(f"CREATE TABLE IF NOT EXISTS {TABLE}")
        < sql.index(RECONCILIATION_BEGIN)
        < sql.index(RECONCILIATION_END)
        < sql.index(f"CREATE INDEX IF NOT EXISTS {INDEX}")
    )


@pytest.mark.integration
def test_the_grant_covers_and_asserts_every_privilege_it_issues() -> None:
    """Asserting only INSERT is what let a 24-day outage ship.

    pr_merged_events sat behind its topic at consumer lag zero: the consumer
    read fine, every write failed, and the migration's single INSERT assertion
    was true throughout (OMN-17379). Every privilege this file grants is
    asserted back, and the schema USAGE is asserted too -- a table grant with
    no schema usage is a relation the principal cannot reach at all.
    """
    sql = GRANT_FILE.read_text()

    assert f"GRANT USAGE ON SCHEMA {SCHEMA} TO omninode_runtime;" in sql
    assert "GRANT SELECT, INSERT, UPDATE" in sql
    assert f"ON {TABLE}" in sql

    for privilege in ("SELECT", "INSERT", "UPDATE"):
        assert f"AND privilege_type = '{privilege}'" in sql, (
            f"the {privilege} grant is issued but never asserted, which is the "
            "shape that stayed green through the OMN-17379 outage"
        )


@pytest.mark.integration
def test_the_grant_issues_no_delete_and_no_sequence_usage() -> None:
    """The claim is append-and-converge, and its key is TEXT.

    A claim is never deleted -- expiring one on a timer would re-open the
    double-bill for exactly the slow delegations most worth protecting, which
    the create file records as a deliberate non-policy. And the primary key is
    a caller-supplied TEXT delivery id, not a BIGSERIAL, so there is no
    sequence to grant. Granting either would be authority nothing asks for.
    """
    sql = _executable_sql(GRANT_FILE)
    assert "DELETE" not in sql, (
        "a DELETE grant is authority the claim protocol never exercises"
    )
    assert "ON SEQUENCE" not in sql, (
        "this table's key is TEXT, so a sequence grant names a relation that "
        "does not exist and would fail the run"
    )


@pytest.mark.integration
def test_no_create_schema_statement_is_issued() -> None:
    """The node migration loop connects where omninode_internal already exists.

    A schema-creating statement here is what failed with "permission denied
    for database" in OMN-16759, blocking every staging deploy.
    """
    for path in (CREATE_FILE, GRANT_FILE):
        assert "CREATE SCHEMA" not in _executable_sql(path), path.name


@pytest.mark.integration
def test_the_table_declares_no_tenant_column() -> None:
    """An omninode_internal relation receives no stamping and no row-level security.

    A tenant column here would be a posture the schema cannot enforce, which
    the OMN-18774 gate refuses and rightly: a tenant column nothing enforces
    reads like isolation and provides none. The claim keys on the delivering
    record, which is tenant-agnostic.
    """
    sql = _executable_sql(CREATE_FILE)
    assert "tenant_id" not in sql
    assert "ROW LEVEL SECURITY" not in sql


def _ledger_digests() -> dict[str, str]:
    """The sha256 each ledger row records for this node's vendored files.

    Six tab-separated columns; the relative path is first and the digest last.
    """
    digests: dict[str, str] = {}
    for line in LEDGER.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        columns = line.split("\t")
        if len(columns) != 6:
            continue
        relpath, digest = columns[0], columns[5]
        if relpath.startswith("nodes/node_delegate_skill_orchestrator/"):
            digests[Path(relpath).name] = digest
    return digests


@pytest.mark.integration
def test_the_vendored_pair_matches_the_checksum_the_ledger_records() -> None:
    """A hand edit here without a ledger update is the drift nothing else sees.

    The omnimarket-side parity gate watches only the direction where the
    SOURCE moves. This watches the other direction, and does so WITHOUT
    needing the omnimarket tree: the forward-migration ledger already records
    a sha256 per vendored file, and bootstrap resolves a historical migration
    against exactly that checksum. So a file edited here and a ledger left
    alone is both a real defect and a locally answerable question.

    It also pins the pair. This node vendored ONE file before OMN-19029 and
    vendors TWO after it, and the grant arriving without its declaration row
    is precisely the shape that leaves a relation granted and undeclared.
    """
    recorded = _ledger_digests()
    assert set(recorded) == {CREATE_FILE.name, GRANT_FILE.name}, (
        "the ledger does not carry exactly the two rows this node vendors; "
        f"got {sorted(recorded)}"
    )
    for vendored in (CREATE_FILE, GRANT_FILE):
        actual = hashlib.sha256(vendored.read_bytes()).hexdigest()
        assert actual == recorded[vendored.name], (
            f"{vendored.name} does not match the checksum its ledger row "
            "records. Either the file was hand-edited here, or the vendoring "
            "was re-run without re-declaring it"
        )
