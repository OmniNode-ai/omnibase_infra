# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""PostgreSQL store for session registry entries.

Provides asyncpg-based storage with replay-safe upsert semantics
per Doctrine D3. Schema is created inline via ensure_schema()
(CREATE TABLE IF NOT EXISTS pattern).

Part of the Multi-Session Coordination Layer (OMN-6850, Task 3).

Design decisions:
    - NOT reusing SessionSnapshotStore: that store tracks per-event snapshots
      (prompts, tools). This store is a materialized aggregate -- one row per
      task_id with computed state.
    - Inline DDL via ensure_schema(): omnibase_infra has no SQL migration
      framework. The store creates its own tables idempotently at startup.
    - Replay-safe upsert: arrays deduplicate, scalars never regress (D3).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

import asyncpg

from omnibase_infra.services.session_registry.enum_session_phase import EnumSessionPhase
from omnibase_infra.services.session_registry.enum_session_registry_status import (
    EnumSessionRegistryStatus,
)
from omnibase_infra.services.session_registry.models import (
    ModelSessionRegistryEntry,
)

if TYPE_CHECKING:
    from asyncpg import Connection, Pool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_CREATE_TABLE = """\
CREATE TABLE IF NOT EXISTS session_registry (
    task_id TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'active',
    current_phase TEXT,
    worktree_path TEXT,
    files_touched TEXT[] DEFAULT '{}',
    depends_on TEXT[] DEFAULT '{}',
    session_ids TEXT[] DEFAULT '{}',
    correlation_ids TEXT[] DEFAULT '{}',
    decisions TEXT[] DEFAULT '{}',
    last_activity TIMESTAMPTZ,
    last_event_type TEXT,
    last_event_at TIMESTAMPTZ,
    schema_version TEXT DEFAULT '1.0.0',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
"""

_CREATE_INDEX_STATUS = """\
CREATE INDEX IF NOT EXISTS idx_session_registry_status
    ON session_registry (status);
"""

_CREATE_INDEX_LAST_ACTIVITY = """\
CREATE INDEX IF NOT EXISTS idx_session_registry_last_activity
    ON session_registry (last_activity DESC NULLS LAST);
"""

# ---------------------------------------------------------------------------
# Replay-safe upsert (Doctrine D3)
# ---------------------------------------------------------------------------

_UPSERT = """\
INSERT INTO session_registry (
    task_id, status, current_phase, worktree_path,
    files_touched, depends_on, session_ids, correlation_ids, decisions,
    last_activity, last_event_type, last_event_at, schema_version
) VALUES (
    $1, $2, $3, $4,
    $5, $6, $7, $8, $9,
    $10, $11, $12, $13
)
ON CONFLICT (task_id) DO UPDATE SET
    -- D3: status never regresses from completed to active on late replay
    status = CASE
        WHEN session_registry.status = 'completed' AND EXCLUDED.status = 'active'
        THEN session_registry.status
        ELSE EXCLUDED.status
    END,
    -- D3: current_phase only updates from newer events
    current_phase = CASE
        WHEN EXCLUDED.last_activity > COALESCE(session_registry.last_activity, '1970-01-01'::timestamptz)
        THEN EXCLUDED.current_phase
        ELSE session_registry.current_phase
    END,
    worktree_path = COALESCE(EXCLUDED.worktree_path, session_registry.worktree_path),
    -- D3: deduplicate arrays on upsert
    files_touched = (
        SELECT COALESCE(array_agg(DISTINCT e ORDER BY e), '{}')
        FROM unnest(array_cat(session_registry.files_touched, EXCLUDED.files_touched)) AS e
    ),
    depends_on = (
        SELECT COALESCE(array_agg(DISTINCT e ORDER BY e), '{}')
        FROM unnest(array_cat(session_registry.depends_on, EXCLUDED.depends_on)) AS e
    ),
    session_ids = (
        SELECT COALESCE(array_agg(DISTINCT e ORDER BY e), '{}')
        FROM unnest(array_cat(session_registry.session_ids, EXCLUDED.session_ids)) AS e
    ),
    correlation_ids = (
        SELECT COALESCE(array_agg(DISTINCT e ORDER BY e), '{}')
        FROM unnest(array_cat(session_registry.correlation_ids, EXCLUDED.correlation_ids)) AS e
    ),
    decisions = (
        SELECT COALESCE(array_agg(DISTINCT e ORDER BY e), '{}')
        FROM unnest(array_cat(session_registry.decisions, EXCLUDED.decisions)) AS e
    ),
    -- D3: last_activity never regresses
    last_activity = GREATEST(session_registry.last_activity, EXCLUDED.last_activity),
    last_event_type = EXCLUDED.last_event_type,
    last_event_at = GREATEST(session_registry.last_event_at, EXCLUDED.last_event_at),
    schema_version = EXCLUDED.schema_version,
    updated_at = NOW()
"""

_SELECT_BY_TASK_ID = """\
SELECT task_id, status, current_phase, worktree_path,
       files_touched, depends_on, session_ids, correlation_ids, decisions,
       last_activity, created_at
FROM session_registry
WHERE task_id = $1
"""

_SELECT_ACTIVE = """\
SELECT task_id, status, current_phase, worktree_path,
       files_touched, depends_on, session_ids, correlation_ids, decisions,
       last_activity, created_at
FROM session_registry
WHERE status = 'active'
ORDER BY last_activity DESC NULLS LAST
"""


class SessionRegistryStore:
    """Asyncpg-based store for session registry entries.

    Usage:
        store = SessionRegistryStore(dsn="postgresql://...")
        await store.initialize()
        await store.upsert_entry(entry, event_type="hook.prompt.submitted")
        entry = await store.get_by_task_id("OMN-1234")
        await store.close()
    """

    def __init__(
        self, dsn: str, *, min_pool_size: int = 1, max_pool_size: int = 5
    ) -> None:
        self._dsn = dsn
        self._min_pool_size = min_pool_size
        self._max_pool_size = max_pool_size
        self._pool: Pool | None = None

    async def initialize(self) -> None:
        """Create connection pool and ensure schema exists."""
        self._pool = await asyncpg.create_pool(
            self._dsn,
            min_size=self._min_pool_size,
            max_size=self._max_pool_size,
        )
        await self.ensure_schema()
        logger.info(
            "session_registry_store_initialized",
            extra={"dsn_host": self._dsn.split("@")[-1]},
        )

    async def ensure_schema(self) -> None:
        """Create session_registry table and indexes if they don't exist."""
        if self._pool is None:
            msg = "Store not initialized. Call initialize() first."
            raise RuntimeError(msg)
        async with self._pool.acquire() as conn:
            await conn.execute(_CREATE_TABLE)
            await conn.execute(_CREATE_INDEX_STATUS)
            await conn.execute(_CREATE_INDEX_LAST_ACTIVITY)
        logger.info("session_registry_schema_ensured")

    async def upsert_entry(
        self,
        entry: ModelSessionRegistryEntry,
        *,
        event_type: str | None = None,
        event_at: datetime | None = None,
        schema_version: str = "1.0.0",
    ) -> None:
        """Upsert a session registry entry with replay-safe semantics (D3).

        Args:
            entry: The registry entry to upsert.
            event_type: The Kafka event type that triggered this upsert.
            event_at: Timestamp of the source event.
            schema_version: Schema version string.
        """
        if self._pool is None:
            msg = "Store not initialized. Call initialize() first."
            raise RuntimeError(msg)

        async with self._pool.acquire() as conn:
            await conn.execute(
                _UPSERT,
                entry.task_id,
                entry.status.value,
                entry.current_phase.value if entry.current_phase else None,
                entry.worktree_path,
                entry.files_touched,
                entry.depends_on,
                entry.session_ids,
                entry.correlation_ids,
                entry.decisions,
                entry.last_activity,
                event_type,
                event_at,
                schema_version,
            )

    async def get_by_task_id(self, task_id: str) -> ModelSessionRegistryEntry | None:
        """Fetch a registry entry by task_id.

        Args:
            task_id: The Linear ticket ID (e.g., "OMN-1234").

        Returns:
            The entry if found, None otherwise.
            Note: Callers implementing resume should wrap this in
            Found/NotFound/Unavailable per Doctrine D4.
        """
        if self._pool is None:
            msg = "Store not initialized. Call initialize() first."
            raise RuntimeError(msg)

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(_SELECT_BY_TASK_ID, task_id)

        if row is None:
            return None

        return _row_to_entry(row)

    async def list_active(self) -> list[ModelSessionRegistryEntry]:
        """List all active session registry entries, ordered by last_activity desc."""
        if self._pool is None:
            msg = "Store not initialized. Call initialize() first."
            raise RuntimeError(msg)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(_SELECT_ACTIVE)

        return [_row_to_entry(row) for row in rows]

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None


def _row_to_entry(row: asyncpg.Record) -> ModelSessionRegistryEntry:
    """Convert an asyncpg Record to a ModelSessionRegistryEntry."""
    return ModelSessionRegistryEntry(
        task_id=row["task_id"],
        status=EnumSessionRegistryStatus(row["status"]),
        current_phase=EnumSessionPhase(row["current_phase"])
        if row["current_phase"]
        else None,
        worktree_path=row["worktree_path"],
        files_touched=list(row["files_touched"]) if row["files_touched"] else [],
        depends_on=list(row["depends_on"]) if row["depends_on"] else [],
        session_ids=list(row["session_ids"]) if row["session_ids"] else [],
        correlation_ids=list(row["correlation_ids"]) if row["correlation_ids"] else [],
        decisions=list(row["decisions"]) if row["decisions"] else [],
        last_activity=row["last_activity"],
        created_at=row["created_at"],
    )
