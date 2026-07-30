# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Durable, edge-local SQLite idempotency store.

The gateway runs as a single process on an operator edge and cannot depend on
the cloud database to decide whether a cross-broker delivery already
completed.  This store provides that local durable decision surface while
implementing the same ``ProtocolIdempotencyStore`` contract used by the other
runtime stores.

Each operation uses its own SQLite connection.  That keeps connection/thread
ownership explicit when the blocking standard-library driver is moved through
``asyncio.to_thread``.  WAL plus ``synchronous=FULL`` makes a successful
``mark_processed`` durable before the caller is allowed to commit its source
Kafka offset.
"""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from omnibase_infra.idempotency.protocol_idempotency_store import (
    ProtocolIdempotencyStore,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS idempotency_records (
    domain TEXT NOT NULL,
    message_id TEXT NOT NULL,
    correlation_id TEXT,
    processed_at REAL NOT NULL,
    PRIMARY KEY (domain, message_id)
)
"""


class StoreIdempotencySqlite(ProtocolIdempotencyStore):
    """Coroutine-safe durable idempotency store backed by one SQLite file."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = asyncio.Lock()
        self._started = False

    async def start(self) -> None:
        """Create and verify the store, failing boot when it is unavailable."""
        async with self._lock:
            await asyncio.to_thread(self._initialize_sync)
            self._started = True

    async def close(self) -> None:
        """Close the lifecycle boundary (operations use short-lived connections)."""
        async with self._lock:
            self._started = False

    async def check_and_record(
        self,
        message_id: UUID,
        domain: str | None = None,
        correlation_id: UUID | None = None,
    ) -> bool:
        """Atomically insert a completion marker if one does not exist."""
        async with self._lock:
            self._require_started()
            return await asyncio.to_thread(
                self._check_and_record_sync,
                message_id,
                domain,
                correlation_id,
            )

    async def is_processed(
        self,
        message_id: UUID,
        domain: str | None = None,
    ) -> bool:
        """Return whether the durable completion marker exists."""
        async with self._lock:
            self._require_started()
            return await asyncio.to_thread(
                self._is_processed_sync,
                message_id,
                domain,
            )

    async def mark_processed(
        self,
        message_id: UUID,
        domain: str | None = None,
        correlation_id: UUID | None = None,
        processed_at: datetime | None = None,
    ) -> None:
        """Durably upsert a completion marker before source acknowledgement."""
        timestamp = processed_at or datetime.now(UTC)
        async with self._lock:
            self._require_started()
            await asyncio.to_thread(
                self._mark_processed_sync,
                message_id,
                domain,
                correlation_id,
                timestamp,
            )

    async def cleanup_expired(self, ttl_seconds: int) -> int:
        """Remove completion markers older than the contract-declared TTL."""
        if ttl_seconds < 1:
            raise ValueError("ttl_seconds must be positive")
        cutoff = datetime.now(UTC).timestamp() - ttl_seconds
        async with self._lock:
            self._require_started()
            return await asyncio.to_thread(self._cleanup_expired_sync, cutoff)

    def _initialize_sync(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(_SCHEMA)
            result = connection.execute("PRAGMA quick_check").fetchone()
            if result != ("ok",):
                raise sqlite3.DatabaseError(
                    f"gateway idempotency store quick_check failed: {result!r}"
                )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path, timeout=5.0)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def _check_and_record_sync(
        self,
        message_id: UUID,
        domain: str | None,
        correlation_id: UUID | None,
    ) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO idempotency_records
                    (domain, message_id, correlation_id, processed_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    self._domain(domain),
                    str(message_id),
                    str(correlation_id) if correlation_id is not None else None,
                    datetime.now(UTC).timestamp(),
                ),
            )
            return cursor.rowcount == 1

    def _is_processed_sync(self, message_id: UUID, domain: str | None) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT 1 FROM idempotency_records
                WHERE domain = ? AND message_id = ?
                """,
                (self._domain(domain), str(message_id)),
            ).fetchone()
            return row is not None

    def _mark_processed_sync(
        self,
        message_id: UUID,
        domain: str | None,
        correlation_id: UUID | None,
        processed_at: datetime,
    ) -> None:
        if processed_at.tzinfo is None:
            raise ValueError("processed_at must be timezone-aware")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO idempotency_records
                    (domain, message_id, correlation_id, processed_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(domain, message_id) DO UPDATE SET
                    correlation_id = excluded.correlation_id,
                    processed_at = excluded.processed_at
                """,
                (
                    self._domain(domain),
                    str(message_id),
                    str(correlation_id) if correlation_id is not None else None,
                    processed_at.timestamp(),
                ),
            )

    def _cleanup_expired_sync(self, cutoff: float) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM idempotency_records WHERE processed_at < ?",
                (cutoff,),
            )
            return cursor.rowcount

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError("SQLite idempotency store is not started")

    @staticmethod
    def _domain(domain: str | None) -> str:
        return domain or ""


__all__ = ["StoreIdempotencySqlite"]
