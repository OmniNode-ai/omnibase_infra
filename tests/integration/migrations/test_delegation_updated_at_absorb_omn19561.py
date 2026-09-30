# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Live-trigger proof for migration 109 (OMN-19561).

A duplicate or stale event absorbed by the StateStoreAdapter must leave
delegation_workflow_state.updated_at alone, and a real transition must advance
it. The shipped forward migrations run through psql against a throwaway cluster,
the way the forward runner applies them.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from tests.integration.migrations.conftest import EphemeralPostgres

FORWARD_DIR = Path(__file__).resolve().parents[3] / "docker" / "migrations" / "forward"
MIGRATION = (
    FORWARD_DIR / "109_delegation_workflow_state_updated_at_only_on_transition.sql"
)
PREREQUISITES = (
    "090_create_delegation_workflow_state.sql",
    "093_add_delegation_workflow_state_outbox_columns.sql",
    "106_add_delegation_workflow_state_traffic_class.sql",
)


def _apply(pg: EphemeralPostgres, migration: Path) -> None:
    result = pg.psql("-v", "ON_ERROR_STOP=1", "-f", str(migration))
    assert result.returncode == 0, result.stderr


@pytest.mark.integration
@pytest.mark.ephemeral_pg
def test_absorb_preserves_timestamp_and_transition_advances_it(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    for filename in PREREQUISITES:
        _apply(ephemeral_postgres, FORWARD_DIR / filename)
    _apply(ephemeral_postgres, MIGRATION)
    _apply(ephemeral_postgres, MIGRATION)  # Reapplying the migration is idempotent.

    connection = ephemeral_postgres.connect()
    # Each statement commits on its own: NOW() is transaction-scoped.
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO delegation_workflow_state
                    (correlation_id, tenant_id, state, payload)
                VALUES ('absorb', 'tenant', 'ROUTED', '{}'::jsonb)
                RETURNING updated_at
                """
            )
            seeded = cursor.fetchone()
            assert seeded is not None
            seeded_at = seeded[0]

            time.sleep(0.05)
            cursor.execute(
                """
                UPDATE delegation_workflow_state
                SET version = version + 1, state = 'ROUTED',
                    payload = '{}'::jsonb, tenant_id = 'tenant'
                WHERE correlation_id = 'absorb' AND version = 0
                RETURNING updated_at, version
                """
            )
            absorbed = cursor.fetchone()
            assert absorbed is not None
            assert absorbed[1] == 1
            assert absorbed[0] == seeded_at

            cursor.execute(
                """
                UPDATE delegation_workflow_state
                SET version = version + 1, state = 'COMPLETED'
                WHERE correlation_id = 'absorb' AND version = 1
                RETURNING updated_at
                """
            )
            transitioned = cursor.fetchone()
            assert transitioned is not None
            assert transitioned[0] > seeded_at
    finally:
        connection.close()
