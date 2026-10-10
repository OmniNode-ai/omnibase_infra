# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Read back a boot-time completion sweep from a throwaway Postgres cluster."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

import asyncpg
import pytest

from omnibase_infra.runtime.state_io.state_store_adapter import StateStoreAdapter
from tests.integration.migrations.conftest import EphemeralPostgres
from tests.integration.test_state_io_completion_bound_omn18296 import (
    BOUND,
    CID_LIVE,
    TENANT,
    TOPIC_DELEGATION_FAILED,
    _callback,
    _fast_sweeps,
    _RecordingBus,
)

FORWARD_DIR = Path(__file__).resolve().parents[3] / "docker" / "migrations" / "forward"


@pytest.mark.integration
@pytest.mark.ephemeral_pg
def test_restart_sweep_publishes_terminal_and_closes_persisted_row_without_dispatch(
    ephemeral_postgres: EphemeralPostgres,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real adapter selects and CAS-finalizes an abandoned row at wiring."""
    pg = ephemeral_postgres
    for filename in (
        "090_create_delegation_workflow_state.sql",
        "093_add_delegation_workflow_state_outbox_columns.sql",
    ):
        applied = pg.psql("-v", "ON_ERROR_STOP=1", "-f", str(FORWARD_DIR / filename))
        assert applied.returncode == 0, applied.stderr

    with pg.connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO delegation_workflow_state
                (correlation_id, tenant_id, state, in_flight, payload, version,
                 updated_at)
            VALUES (%s, %s, 'ROUTED', TRUE, '{"state":"ROUTED"}', 4,
                    NOW() - INTERVAL '1000 seconds')
            """,
            (CID_LIVE, TENANT),
        )

    _fast_sweeps(monkeypatch)
    bus = _RecordingBus()

    async def drive() -> None:
        pool = await asyncpg.create_pool(
            host=pg.socket_dir,
            port=pg.port,
            user="postgres",
            database="postgres",
            min_size=1,
            max_size=2,
        )

        async def pool_factory() -> asyncpg.Pool:
            return pool

        adapter = StateStoreAdapter(
            "postgresql://postgres@localhost/postgres",
            table="delegation_workflow_state",
            pool_factory=pool_factory,
        )
        try:
            _callback(cast("Any", adapter), bus, completion_bound=BOUND)
            # Only the sweep timer runs: there is no call to the dispatch callback.
            for _ in range(100):
                row = await pool.fetchrow(
                    "SELECT state, in_flight, version, pending_emissions "
                    "FROM delegation_workflow_state WHERE correlation_id = $1",
                    CID_LIVE,
                )
                if row is not None and row["state"] == "FAILED":
                    break
                await asyncio.sleep(0.05)
            assert row is not None
            assert dict(row) == {
                "state": "FAILED",
                "in_flight": False,
                "version": 5,
                "pending_emissions": None,
            }
            assert len(bus.published) == 1
            topic, envelope = bus.published[0]
            assert topic == TOPIC_DELEGATION_FAILED
            assert str(envelope.correlation_id) == CID_LIVE
            assert envelope.payload.terminal_failure_reason == (
                "RuntimeRestartDuringDelegationError: "
                "ONEX_MARKET_DELEGATION_RUNTIME_RESTART"
            )
        finally:
            await pool.close()

    asyncio.run(drive())
