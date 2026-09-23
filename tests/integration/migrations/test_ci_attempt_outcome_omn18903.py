# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Live-apply proof for the ci_attempt_outcome read model (OMN-18903).

These apply the real vendored migration to a throwaway Postgres cluster and
exercise the parts of it that are SQL rather than Python. A unit test that
greps the migration text proves a string is present; it does not prove the
constraint rejects, the key collides, or the guarded reconciliation is
idempotent in SHAPE rather than merely in existence.

Three things here exist nowhere else:

  * **The six-value check constraint.** The vocabulary it mirrors lives in
    omnimarket. A seventh member added there without widening this constraint
    writes a row the database refuses, at runtime, on the lane. This is where
    that is caught instead, and the negative control proves the constraint is
    doing something rather than accepting everything.
  * **The five-column key.** Two attempts of the same run on the same check
    are DIFFERENT rows, and two checks on the same commit are different rows.
    Getting the key wrong collapses attempts the metric counts, which is
    silent: the table still fills, with the wrong denominator.
  * **Idempotency in shape.** The migration is create-if-not-exists plus one
    guarded add-column per declared column, on purpose. Applying it twice must
    converge rather than fail, because the forward runner re-applies the whole
    corpus.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import psycopg2
import pytest

from tests.integration.migrations.conftest import EphemeralPostgres

pytestmark = [pytest.mark.integration, pytest.mark.postgres]

REPO_ROOT = Path(__file__).resolve().parents[3]
FORWARD = (
    REPO_ROOT
    / "docker"
    / "migrations"
    / "forward"
    / "nodes"
    / "node_projection_ci_attempt_outcome"
    / "0000_create_ci_attempt_outcome.sql"
)

#: The six members of the merge-check vocabulary as of OMN-18902. Spelled here
#: rather than imported: this repository does not depend on omnimarket, and the
#: whole point of the constraint is that the two can drift.
CAUSE_CODES = (
    "stale_context",
    "github_api_outage",
    "runner_infra",
    "process_gate_refused",
    "cancelled",
    "product_failed",
)

_T0 = datetime(2026, 9, 20, 12, 0, 0, tzinfo=UTC)
_SHA = "a" * 40
_REPO = "OmniNode-ai/omnimarket"


@pytest.fixture
def applied(
    ephemeral_postgres: EphemeralPostgres,
) -> Iterator[psycopg2.extensions.connection]:
    """Apply the real forward migration, yield an open connection.

    The schema is provisioned first, by the superuser, mirroring the deployed
    lane: the migration ASSERTS omninode_internal and never creates it,
    because creating a schema needs a database-level privilege neither
    migration role holds on the managed instance.
    """
    provisioned = ephemeral_postgres.psql(
        "-v",
        "ON_ERROR_STOP=1",
        "-c",
        "CREATE SCHEMA IF NOT EXISTS omninode_internal;",
    )
    assert provisioned.returncode == 0, provisioned.stderr
    result = ephemeral_postgres.psql("-v", "ON_ERROR_STOP=1", "-f", str(FORWARD))
    assert result.returncode == 0, result.stderr
    conn = ephemeral_postgres.connect()
    try:
        yield conn
    finally:
        conn.close()


def _insert(
    conn: psycopg2.extensions.connection,
    *,
    cause_code: str = "process_gate_refused",
    check_name: str = "verify",
    run_attempt: int = 1,
    head_sha: str = _SHA,
    attempt_ordinal: int = 1,
    ticket_id: str | None = "OMN-18903",
    observed_at: datetime = _T0,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO omninode_internal.ci_attempt_outcome (
                repository, pr_number, head_sha, check_name, run_attempt,
                cause_code, cause_affirmative, attempt_ordinal,
                ticket_id, observed_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                _REPO,
                2730,
                head_sha,
                check_name,
                run_attempt,
                cause_code,
                True,
                attempt_ordinal,
                ticket_id,
                observed_at,
            ),
        )
    conn.commit()


def _count(conn: psycopg2.extensions.connection) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM omninode_internal.ci_attempt_outcome")
        row = cur.fetchone()
    assert row is not None
    return int(row[0])


def test_every_cause_code_is_accepted(
    applied: psycopg2.extensions.connection,
) -> None:
    """All six members round-trip through the real constraint."""
    for index, code in enumerate(CAUSE_CODES, start=1):
        _insert(applied, cause_code=code, check_name=f"check-{index}")
    assert _count(applied) == 6


def test_an_unknown_cause_code_is_refused(
    applied: psycopg2.extensions.connection,
) -> None:
    """The negative control, without which the test above proves nothing.

    A constraint that accepted everything would pass the six-member test and
    tell nobody that the vocabulary and the schema had drifted apart.
    """
    with pytest.raises(psycopg2.errors.CheckViolation):
        _insert(applied, cause_code="not_a_real_cause")
    applied.rollback()


def test_two_attempts_of_one_check_are_two_rows(
    applied: psycopg2.extensions.connection,
) -> None:
    """The run attempt is part of the key, and the metric depends on it.

    Without it the same-commit re-run rescue is inexpressible: a commit that
    failed on attempt one and reached green on attempt two would collapse into
    a single row, and the fact that makes it infrastructure would be gone.
    """
    _insert(applied, run_attempt=1)
    _insert(applied, run_attempt=2, cause_code="runner_infra")
    assert _count(applied) == 2


def test_two_checks_on_one_commit_are_two_rows(
    applied: psycopg2.extensions.connection,
) -> None:
    """A pull request whose lint failed and whose gate refused has two causes."""
    _insert(applied, check_name="lint", cause_code="product_failed")
    _insert(applied, check_name="verify", cause_code="process_gate_refused")
    assert _count(applied) == 2


def test_the_same_attempt_twice_collides_on_the_key(
    applied: psycopg2.extensions.connection,
) -> None:
    """The positive control on the key: it does constrain something."""
    _insert(applied)
    with pytest.raises(psycopg2.errors.UniqueViolation):
        _insert(applied, cause_code="product_failed")
    applied.rollback()


def test_a_null_ticket_is_storable(
    applied: psycopg2.extensions.connection,
) -> None:
    """A row nobody could attribute is recorded, not refused.

    Refusing it would silently shrink the denominator of a metric whose whole
    subject is how many attempts a work unit took.
    """
    _insert(applied, ticket_id=None)
    with applied.cursor() as cur:
        cur.execute("SELECT ticket_id FROM omninode_internal.ci_attempt_outcome")
        row = cur.fetchone()
    assert row is not None
    assert row[0] is None


def test_a_zero_attempt_ordinal_is_refused(
    applied: psycopg2.extensions.connection,
) -> None:
    """Ordinals start at one. A zero is a defaulted value, not a position."""
    with pytest.raises(psycopg2.errors.CheckViolation):
        _insert(applied, attempt_ordinal=0)
    applied.rollback()


def test_the_timestamp_column_is_timezone_aware(
    applied: psycopg2.extensions.connection,
) -> None:
    """Declared with a time zone, so a naive value is not silently accepted.

    The string-where-timestamp class reached a deployed, crash-looping runtime
    once with every layer of mock coverage passing; a real server is the only
    place the column type is enforced.
    """
    _insert(applied, observed_at=_T0)
    with applied.cursor() as cur:
        cur.execute("SELECT observed_at FROM omninode_internal.ci_attempt_outcome")
        row = cur.fetchone()
    assert row is not None
    stored = row[0]
    assert isinstance(stored, datetime)
    assert stored.tzinfo is not None
    assert stored == _T0


def test_reapplying_the_migration_converges(
    applied: psycopg2.extensions.connection,
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    """The forward runner re-applies the whole corpus, so this must be a no-op.

    Asserted over a table that already holds a row, because an idempotency
    check on an empty table cannot tell a converging re-apply from one that
    dropped and recreated the relation.
    """
    _insert(applied)
    assert _count(applied) == 1

    second = ephemeral_postgres.psql("-v", "ON_ERROR_STOP=1", "-f", str(FORWARD))
    assert second.returncode == 0, second.stderr

    assert _count(applied) == 1


def test_the_cursor_column_advances(
    applied: psycopg2.extensions.connection,
) -> None:
    """The exposure pages on this column, so it has to be monotonic."""
    _insert(applied, check_name="first")
    _insert(applied, check_name="second", observed_at=_T0 + timedelta(seconds=1))
    with applied.cursor() as cur:
        cur.execute(
            "SELECT check_name, projection_cursor "
            "FROM omninode_internal.ci_attempt_outcome "
            "ORDER BY projection_cursor"
        )
        rows = cur.fetchall()
    assert [r[0] for r in rows] == ["first", "second"]
    assert rows[0][1] < rows[1][1]
