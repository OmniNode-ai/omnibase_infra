# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-18872 — link 2's bounded poll, against a real PostgreSQL.

Why this exists as an INTEGRATION test and not only as unit coverage
-------------------------------------------------------------------
The unit suite injects a fake ``asyncpg`` and proves the control flow. It
cannot prove the two things that actually decided the live verdict: that a row
written by a *different connection while the readback is already running*
becomes visible to it, and that a real ``permission denied`` from a real
least-privilege role lands in the ERROR arm rather than the absence arm. Both
are properties of PostgreSQL and of the driver, not of the handler's branches,
and the defect this ticket fixes was a timing relationship with a concurrent
writer.

``test_row_written_by_a_concurrent_writer_is_picked_up`` is the live shape
reproduced end to end: the readback starts against an empty table, a separate
connection inserts the row non-terminal a beat later and terminalises it a beat
after that, and the readback must come back TERMINAL. Under the pre-image it
returns ROW_ABSENT, which is run 35482636275.

``test_permission_denied_is_error_not_absence`` is the one that keeps AC5
diagnosable. On the dev lane ``chain_canary_reader`` holds no grant on
``delegation_workflow_state`` at all, so a refusal is a live possibility rather
than a hypothetical; if it were ever folded into the absence arm the canary
would blame the projection for a permissions problem.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from itertools import count

import pytest

from omnibase_infra.nodes.node_chain_canary_effect.handlers.handler_chain_canary import (
    _readback_projection_via_asyncpg,
)
from omnibase_infra.nodes.node_chain_canary_effect.models.enum_projection_readback_status import (
    EnumProjectionReadbackStatus,
)

# Only the TYPE is imported here. The `pg16` fixture itself is re-exported by
# this package's conftest: imported into the module it would shadow every
# `pg16` test parameter below, which is the shadowing the OMN-15857 files
# already record in the migrations conftest.
from tests.integration.migrations.test_application_migration_ledger_omn15413 import (
    Pg16Cluster,
)

pytestmark = [pytest.mark.integration, pytest.mark.postgres, pytest.mark.serial]

_CORRELATION = "756bae29-4d9b-41fb-8d1d-a9666c93bf9f"

# Writes go in on STDIN with a psql variable rather than through `-c` with an
# interpolated f-string. Two reasons, and the second is the load-bearing one:
# the value never becomes part of a query string the linter has to take on
# trust, and psql does NOT expand `:'var'` inside a `-c` argument at all -- it
# expands only while lexing stdin or a file, so the `-c` form fails outright.

# The projection table as the readback sees it. Only the three columns the
# query touches, deliberately: this proves the READBACK, and reproducing the
# lane's full DDL here would make the test fail for reasons that have nothing
# to do with link 2.
_SCHEMA = """
CREATE TABLE delegation_workflow_state (
    correlation_id uuid PRIMARY KEY,
    state text NOT NULL,
    traffic_class text NOT NULL DEFAULT 'unclassified'
);
"""


def _counter() -> Iterator[int]:
    yield from count()


_COUNTER = _counter()


@pytest.fixture
def projection_db(pg16: Pg16Cluster) -> tuple[str, str]:
    """A database with the projection table and a least-privilege reader.

    The reader is NOSUPERUSER NOBYPASSRLS on purpose: the readback asks the
    server for ``current_user``'s own ``pg_roles`` row before it reads anything
    and REFUSES a privileged DSN (OMN-18060), so a superuser DSN here would
    never reach the poll at all and the test would prove nothing.
    """
    database = f"omn18872_projection_{next(_COUNTER)}"
    role = f"omn18872_reader_{next(_COUNTER)}"
    pg16.create_database(database)
    pg16.command(database, "-c", _SCHEMA)
    pg16.command(
        database,
        "-c",
        f"CREATE ROLE {role} LOGIN NOSUPERUSER NOBYPASSRLS",
    )
    pg16.command(database, "-c", f"GRANT USAGE ON SCHEMA public TO {role}")
    pg16.command(
        database,
        "-c",
        f"GRANT SELECT ON delegation_workflow_state TO {role}",
    )
    dsn = f"postgresql://{role}@127.0.0.1:{pg16.port}/{database}"
    return database, dsn


@pytest.mark.asyncio
async def test_row_written_by_a_concurrent_writer_is_picked_up(
    pg16: Pg16Cluster, projection_db: tuple[str, str]
) -> None:
    """The live shape: empty at submission, terminal a few seconds later.

    Run 35482636275 in miniature against a real database. The readback starts
    with nothing to find; a separate connection then creates the row
    non-terminal and terminalises it. Under the single-sample pre-image this
    returns ROW_ABSENT and the canary is red for a chain that completed.
    """
    database, dsn = projection_db

    async def _write_the_row_late() -> None:
        # Non-terminal first, exactly as the projection does it. A poll that
        # stopped at first sight would capture RECEIVED and report STRANDED.
        await asyncio.sleep(1.5)
        pg16.command(
            database,
            "-v",
            f"corr={_CORRELATION}",
            input_text=(
                "INSERT INTO delegation_workflow_state (correlation_id, state) "
                "VALUES (:'corr', 'RECEIVED');"
            ),
        )
        await asyncio.sleep(1.5)
        pg16.command(
            database,
            "-v",
            f"corr={_CORRELATION}",
            input_text=(
                "UPDATE delegation_workflow_state SET state = 'COMPLETED' "
                "WHERE correlation_id = :'corr';"
            ),
        )

    outcome, _ = await asyncio.gather(
        _readback_projection_via_asyncpg(dsn, _CORRELATION, 25.0),
        _write_the_row_late(),
    )

    assert outcome.status is EnumProjectionReadbackStatus.TERMINAL
    assert outcome.state == "COMPLETED"
    assert outcome.traffic_class == "unclassified"
    assert outcome.error == ""


@pytest.mark.asyncio
async def test_absent_row_still_fails_at_the_deadline(
    projection_db: tuple[str, str],
) -> None:
    """A correlation id nothing ever writes is still ROW_ABSENT.

    The poll must not turn every absence into a pass or into a timeout, and it
    must say what it waited so a real absence is distinguishable from the
    one-sample absence this ticket removed.
    """
    _, dsn = projection_db

    outcome = await _readback_projection_via_asyncpg(dsn, _CORRELATION, 3.0)

    assert outcome.status is EnumProjectionReadbackStatus.ROW_ABSENT
    assert "polled for" in outcome.error
    assert "read(s)" in outcome.error


@pytest.mark.asyncio
async def test_row_parked_mid_fsm_is_stranded_with_its_last_state(
    pg16: Pg16Cluster, projection_db: tuple[str, str]
) -> None:
    """OMN-14843's condition survives the fix against a real store."""
    database, dsn = projection_db
    pg16.command(
        database,
        "-v",
        f"corr={_CORRELATION}",
        input_text=(
            "INSERT INTO delegation_workflow_state (correlation_id, state) "
            "VALUES (:'corr', 'INFERENCE_COMPLETED');"
        ),
    )

    outcome = await _readback_projection_via_asyncpg(dsn, _CORRELATION, 3.0)

    assert outcome.status is EnumProjectionReadbackStatus.STRANDED
    assert outcome.state == "INFERENCE_COMPLETED"
    assert "polled for" in outcome.error


@pytest.mark.asyncio
async def test_permission_denied_is_error_not_absence(
    pg16: Pg16Cluster, projection_db: tuple[str, str]
) -> None:
    """A real refusal from a real role lands in ERROR, never in the absence arm.

    This is the live condition behind OMN-18872 AC5: on the dev lane
    ``chain_canary_reader`` has no SELECT on this table. A refusal reported as
    "the projection carries no row" would send the reader to the projection
    for a grant problem, and the poll would spend the whole window re-asking a
    question already answered.
    """
    database, _ = projection_db
    ungranted = f"omn18872_ungranted_{next(_COUNTER)}"
    pg16.command(
        database, "-c", f"CREATE ROLE {ungranted} LOGIN NOSUPERUSER NOBYPASSRLS"
    )
    pg16.command(database, "-c", f"GRANT USAGE ON SCHEMA public TO {ungranted}")
    # Deliberately no GRANT SELECT.
    dsn = f"postgresql://{ungranted}@127.0.0.1:{pg16.port}/{database}"

    outcome = await _readback_projection_via_asyncpg(dsn, _CORRELATION, 10.0)

    assert outcome.status is EnumProjectionReadbackStatus.ERROR
    assert outcome.status is not EnumProjectionReadbackStatus.ROW_ABSENT
    assert "permission denied" in outcome.error.lower()
