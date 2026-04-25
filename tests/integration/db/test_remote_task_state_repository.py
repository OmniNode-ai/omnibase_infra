# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for RemoteTaskStateRepository."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from omnibase_core.enums.enum_agent_protocol import EnumAgentProtocol
from omnibase_core.enums.enum_agent_task_lifecycle_type import (
    EnumAgentTaskLifecycleType,
)
from omnibase_core.enums.enum_invocation_kind import EnumInvocationKind
from omnibase_core.models.delegation.model_remote_task_state import ModelRemoteTaskState
from omnibase_infra.nodes.node_remote_agent_invoke_effect.persistence import (
    RemoteTaskStateRepository,
)
from tests.helpers.util_postgres import PostgresConfig

pytestmark = [pytest.mark.integration, pytest.mark.postgres, pytest.mark.asyncio]

MIGRATION_SQL = (
    Path(__file__).resolve().parents[3]
    / "docker"
    / "migrations"
    / "forward"
    / "069_create_remote_task_state.sql"
).read_text()


@pytest.fixture
async def postgres_pool():
    config = PostgresConfig.from_env()
    if not config.is_configured:
        pytest.skip(
            "PostgreSQL not configured (set OMNIBASE_INFRA_DB_URL or POSTGRES_HOST/POSTGRES_PASSWORD)"
        )

    import asyncpg

    pool = await asyncpg.create_pool(
        config.build_dsn(), min_size=1, max_size=5, timeout=10.0
    )
    try:
        async with pool.acquire() as conn:
            await conn.execute(MIGRATION_SQL)
            await conn.execute("TRUNCATE TABLE remote_task_state")
        yield pool
    finally:
        await pool.close()


def _row(*, status: EnumAgentTaskLifecycleType) -> ModelRemoteTaskState:
    now = datetime.now(UTC)
    completed_at = (
        now
        if status
        in {
            EnumAgentTaskLifecycleType.COMPLETED,
            EnumAgentTaskLifecycleType.FAILED,
            EnumAgentTaskLifecycleType.TIMED_OUT,
            EnumAgentTaskLifecycleType.CANCELED,
        }
        else None
    )
    return ModelRemoteTaskState(
        task_id=uuid4(),
        invocation_kind=EnumInvocationKind.AGENT,
        protocol=EnumAgentProtocol.A2A,
        target_ref="agent:local-a2a-smoke",
        remote_task_handle=f"remote:{uuid4()}",
        correlation_id=uuid4(),
        status=status,
        last_remote_status=status.value.lower(),
        last_emitted_event_type=status,
        submitted_at=now,
        updated_at=now,
        completed_at=completed_at,
        error="terminal failure"
        if status is EnumAgentTaskLifecycleType.FAILED
        else None,
    )


async def test_upsert_then_load_unfinished(postgres_pool) -> None:
    repo = RemoteTaskStateRepository(postgres_pool)
    row = _row(status=EnumAgentTaskLifecycleType.SUBMITTED)

    await repo.upsert(row)

    loaded = await repo.get(row.task_id)
    unfinished = await repo.load_unfinished()

    assert loaded == row
    assert unfinished == [row]


async def test_terminal_status_excluded_from_unfinished(postgres_pool) -> None:
    repo = RemoteTaskStateRepository(postgres_pool)
    submitted = _row(status=EnumAgentTaskLifecycleType.SUBMITTED)
    completed = _row(status=EnumAgentTaskLifecycleType.COMPLETED)
    failed = _row(status=EnumAgentTaskLifecycleType.FAILED)

    await repo.upsert(submitted)
    await repo.upsert(completed)
    await repo.upsert(failed)

    unfinished = await repo.load_unfinished()

    assert unfinished == [submitted]
