# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Integration-facing migration contract checks for dod_verify_runs (OMN-18900).

WHAT THESE ASSERT, AND WHAT THEY DO NOT
    The contract of the VENDORED migration pair: the key that makes the
    attempts metric answerable, the shape-reconciliation block that keeps the
    create idempotent in shape rather than only in existence, and the grant
    assertions whose absence is invisible until a relation takes traffic.

    They are file-contract checks in the shape of their neighbours in this
    directory, and they need no database, so they never skip and never
    contribute a false green to the merge-gating job.

    They are NOT a live-database proof and do not claim to be. That proof was
    taken against real PostgreSQL on the lab dev lane and is quoted in the
    pull request body: the create applied clean, the writer's own upsert was
    prepared and executed, a redelivery converged on its key, a
    re-verification became a second row, and the columns read back as `uuid`
    and `timestamp with time zone`. The omnimarket side additionally carries a
    real-Postgres write-path test over this same file.

WHY BYTE-IDENTITY IS ASSERTED HERE
    This file is a VENDORED copy. Its source of truth is the omnimarket node's
    own migrations directory, and the two drifting apart is a recurring defect
    with six recorded occurrences (OMN-14975 and the five it names). The
    parity gate catches it from the omnimarket side; this catches the half
    where a hand edit lands here.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[3]
NODE_DIR = (
    REPO_ROOT
    / "docker"
    / "migrations"
    / "forward"
    / "nodes"
    / "node_projection_dod_verdict"
)
CREATE_FILE = NODE_DIR / "0000_create_dod_verify_runs.sql"
GRANT_FILE = NODE_DIR / "0001_grant_omninode_runtime_dod_verify_runs.sql"

TABLE = "omninode_internal.dod_verify_runs"
SEQUENCE = "omninode_internal.dod_verify_runs_projection_cursor_seq"

#: Every column the create declares. Named here rather than parsed out of the
#: file, so a column silently dropped from the migration fails this list
#: instead of quietly shrinking both sides of the comparison together.
DECLARED_COLUMNS: tuple[str, ...] = (
    "ticket_id",
    "correlation_id",
    "completed_at",
    "started_at",
    "status",
    "unresolved_cause",
    "total_checks",
    "verified_count",
    "failed_count",
    "skipped_count",
    "superseded_count",
    "non_probative_count",
    "behavior_proving_count",
    "readback_proving_count",
    "unbindable_overlay_count",
    "outcome",
    "outcome_refusal",
    "error_message",
    "projected_at",
    "projection_cursor",
)


@pytest.mark.integration
def test_the_create_declares_the_three_column_run_key() -> None:
    """One row per RUN, which is what makes the attempts metric answerable.

    A table keyed on the ticket alone holds only the last verdict and can
    never be asked how many attempts preceded it. The correlation id is not
    enough either: the verify command-line entry point accepts one from the
    caller, so two runs can share it, and the completion time closes that.
    """
    sql = CREATE_FILE.read_text()
    assert f"CREATE TABLE IF NOT EXISTS {TABLE} (" in sql
    assert "PRIMARY KEY (ticket_id, correlation_id, completed_at)" in sql


@pytest.mark.integration
def test_the_create_declares_every_class_count_the_metric_reads() -> None:
    """The counts are the evidence; the status is a summary of them.

    behavior_proving_count in particular is the conjunct the eval metric's
    done predicate requires, and it was ZERO in the only verify payload the
    2026-09-20 inventory could read, beside 72 non-probative checks out of 88.
    """
    sql = CREATE_FILE.read_text()
    for column in DECLARED_COLUMNS:
        assert f"    {column} " in sql, f"{column} is not declared in the create"


@pytest.mark.integration
def test_the_create_reconciles_shape_and_not_merely_existence() -> None:
    """CREATE TABLE IF NOT EXISTS no-ops against a DIFFERENT pre-existing shape.

    Without a guarded ADD COLUMN per declared column, a drifted table of this
    name silently keeps its own shape and the first column-dependent statement
    after the create fails, taking the whole forward-migration run with it.
    The name has been spelled in a comment in the shared topics module long
    enough that a hand-made table is possible, so this is not hypothetical.
    """
    sql = CREATE_FILE.read_text()
    for column in DECLARED_COLUMNS:
        assert f"ADD COLUMN IF NOT EXISTS {column}" in sql, (
            f"{column} is declared but has no guarded ADD COLUMN, so it would "
            "silently not arrive on a drifted database"
        )


@pytest.mark.integration
def test_the_create_indexes_the_two_questions_the_surface_is_for() -> None:
    """Attempts per ticket in order, and the outcome breakdown.

    The second is what makes a thinning definition of done visible as a rising
    no-behaviour-proving share rather than as an unexplained improvement.
    """
    sql = CREATE_FILE.read_text()
    assert "idx_dod_verify_runs_ticket_time" in sql
    assert "idx_dod_verify_runs_outcome_time" in sql
    assert "idx_dod_verify_runs_cursor" in sql


@pytest.mark.integration
def test_the_grant_covers_and_asserts_every_privilege_including_the_sequence() -> None:
    """Asserting only INSERT is what let a 24-day outage ship.

    pr_merged_events sat behind its topic at consumer lag zero: the consumer
    read fine, every write failed on the BIGSERIAL sequence's own access
    control list, and the migration's single INSERT assertion was true
    throughout (OMN-17379). Every granted privilege is asserted here, and the
    sequence is asserted twice -- once for its identity and once for the grant.
    """
    sql = GRANT_FILE.read_text()

    assert f"GRANT SELECT, INSERT, UPDATE ON {TABLE} TO omninode_runtime;" in sql
    assert f"GRANT USAGE ON SEQUENCE {SEQUENCE} TO omninode_runtime;" in sql

    for privilege in ("INSERT", "SELECT", "UPDATE"):
        assert f"AND privilege_type = '{privilege}'" in sql, (
            f"the {privilege} grant is issued but never asserted, which is the "
            "shape that stayed green through the OMN-17379 outage"
        )

    assert "pg_get_serial_sequence(" in sql
    assert "has_sequence_privilege(" in sql


def _executable_sql(path: Path) -> str:
    """The file with its ``--`` comment lines removed.

    Both absence assertions below have to read this rather than the raw text.
    Each of these migrations explains IN A COMMENT why it does not use the
    construct being forbidden -- the grant file records why the sequence is
    named statically instead of resolved dynamically, and the create file
    records why it issues no schema statement. A raw substring search
    therefore fires on the documentation about the rule and reports the file
    as violating the thing it is explaining, which is a check that can only
    ever be satisfied by deleting the explanation.
    """
    return "\n".join(
        line
        for line in path.read_text().splitlines()
        if not line.lstrip().startswith("--")
    )


@pytest.mark.integration
def test_the_grant_names_the_sequence_statically() -> None:
    """A dynamically composed target is refused, and rightly.

    A static reader cannot prove which relation a runtime-built identifier
    will touch, so the OMN-15361 ownership gate refuses a procedural block
    containing dynamic SQL. The literal name is used instead, and the identity
    assertion is what makes the literal safe.
    """
    sql = _executable_sql(GRANT_FILE)
    assert "EXECUTE format(" not in sql
    assert "DO $$" not in sql


@pytest.mark.integration
def test_no_create_schema_statement_is_issued() -> None:
    """The node migration loop connects where omninode_internal already exists.

    A schema-creating statement here is what failed with "permission denied
    for database" in OMN-16759.
    """
    for path in (CREATE_FILE, GRANT_FILE):
        assert "CREATE SCHEMA" not in _executable_sql(path), path.name


@pytest.mark.integration
def test_the_vendored_pair_is_byte_identical_to_its_omnimarket_source() -> None:
    """Vendored copies drifting from their source is a six-occurrence defect.

    The omnimarket-side parity gate catches the half where the source moves.
    This catches the half where someone hand-edits the copy here, which is the
    direction that gate cannot see. Resolved the same way the vendoring script
    itself resolves the source tree, and SKIPPED only when no omnimarket tree
    is present at all -- which is a missing optional cross-repo checkout, not
    a provisioned service, so it is not a silent-skip false green.
    """
    source_root = os.environ.get("OMNIMARKET_SRC") or os.environ.get("OMNI_HOME", "")
    candidates = []
    if source_root:
        base = Path(source_root)
        candidates.append(base / "src" / "omnimarket" / "nodes")
        candidates.append(base / "omnimarket" / "src" / "omnimarket" / "nodes")
    candidates.append(
        REPO_ROOT
        / ".proof-dependencies"
        / "omnimarket"
        / "src"
        / "omnimarket"
        / "nodes"
    )

    nodes_root = next((c for c in candidates if c.is_dir()), None)
    if nodes_root is None:
        pytest.skip(
            "no omnimarket source tree resolvable for the vendor byte-identity "
            "check; this is an optional cross-repo checkout, not a provisioned "
            "service"
        )

    source_dir = nodes_root / "node_projection_dod_verdict" / "migrations"
    if not source_dir.is_dir():
        pytest.skip(
            "the omnimarket tree resolved here predates node_projection_dod_verdict; "
            "the node package lands in the third pull request of this sequence"
        )

    for vendored in (CREATE_FILE, GRANT_FILE):
        origin = source_dir / vendored.name
        assert origin.is_file(), f"{vendored.name} has no omnimarket source"
        assert (
            hashlib.sha256(vendored.read_bytes()).hexdigest()
            == hashlib.sha256(origin.read_bytes()).hexdigest()
        ), f"{vendored.name} has drifted from its omnimarket source"
