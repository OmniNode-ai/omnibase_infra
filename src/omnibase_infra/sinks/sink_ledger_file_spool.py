# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""File spool ledger sink for durable event storage.

This sink writes events to append-only JSONL files for durability.
Events survive process restart and can be drained/replayed later.

Features:
    - Append-only JSONL files (one JSON object per line)
    - Automatic file rotation based on size
    - Async buffered writes for low latency
    - Background flush for durability
    - Graceful degradation under load

File Format:
    Each file contains newline-delimited JSON (JSONL):
    {"event_type": "db.query.requested", "event_id": "...", ...}
    {"event_type": "db.query.succeeded", "event_id": "...", ...}

File Naming:
    ledger_{timestamp}_{sequence}.jsonl
    Example: ledger_20260205T143000Z_0001.jsonl
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, TYPE_CHECKING

from omnibase_infra.protocols.protocol_ledger_sink import EnumLedgerSinkDropPolicy
from omnibase_infra.sinks.sink_ledger_inmemory import (
    LedgerSinkClosedError,
    LedgerSinkError,
    LedgerSinkFullError,
)

if TYPE_CHECKING:
    from omnibase_infra.models.ledger import ModelLedgerEventBase

logger = logging.getLogger(__name__)


class FileSpoolLedgerSink:
    """File spool ledger sink for durable event storage.

    Writes events to append-only JSONL files with automatic rotation.
    Events are buffered in memory and flushed periodically or on demand.

    Attributes:
        spool_dir: Directory for spool files.
        max_file_size_bytes: Maximum size per file before rotation.
        max_buffer_size: Maximum events to buffer before forcing flush.
        flush_interval_seconds: Background flush interval.
        drop_policy: Policy when buffer is full.

    Example:
        >>> sink = FileSpoolLedgerSink(
        ...     spool_dir="/var/log/omninode/ledger",
        ...     max_file_size_bytes=10 * 1024 * 1024,  # 10MB
        ... )
        >>> await sink.emit(event)
        >>> await sink.flush()
        >>> await sink.close()
    """

    __slots__ = (
        "_buffer",
        "_closed",
        "_current_file_path",
        "_current_file_size",
        "_drop_policy",
        "_file_handle",
        "_file_sequence",
        "_flush_interval",
        "_flush_task",
        "_lock",
        "_max_buffer_size",
        "_max_file_size",
        "_spool_dir",
    )

    def __init__(
        self,
        spool_dir: Path,
        max_file_size_bytes: int = 10 * 1024 * 1024,  # 10MB
        max_buffer_size: int = 1000,
        flush_interval_seconds: float = 1.0,
        drop_policy: EnumLedgerSinkDropPolicy = EnumLedgerSinkDropPolicy.DROP_OLDEST,
    ) -> None:
        """Initialize the file spool sink.

        Args:
            spool_dir: Path to directory for spool files. Created if not exists.
            max_file_size_bytes: Maximum size per file before rotation (default: 10MB).
            max_buffer_size: Maximum events to buffer (default: 1000).
            flush_interval_seconds: Background flush interval (default: 1.0s).
            drop_policy: Policy when buffer is full (default: DROP_OLDEST).
        """
        self._spool_dir = Path(spool_dir)
        self._max_file_size = max_file_size_bytes
        self._max_buffer_size = max_buffer_size
        self._flush_interval = flush_interval_seconds
        self._drop_policy = drop_policy

        self._buffer: deque[ModelLedgerEventBase] = deque(maxlen=max_buffer_size)
        self._lock = asyncio.Lock()
        self._closed = False

        self._file_handle: IO[bytes] | None = None
        self._current_file_path: Path | None = None
        self._current_file_size = 0
        self._file_sequence = 0

        self._flush_task: asyncio.Task[None] | None = None

        # Ensure spool directory exists
        self._spool_dir.mkdir(parents=True, exist_ok=True)

    async def emit(self, event: ModelLedgerEventBase) -> bool:
        """Emit a ledger event to the file spool.

        Args:
            event: Ledger event to emit.

        Returns:
            True if event was accepted.
            False if event was dropped due to policy.

        Raises:
            LedgerSinkClosedError: If sink is closed.
            LedgerSinkFullError: If buffer is full and policy is RAISE.
        """
        if self._closed:
            raise LedgerSinkClosedError("Cannot emit to closed sink")

        async with self._lock:
            if len(self._buffer) >= self._max_buffer_size:
                if self._drop_policy == EnumLedgerSinkDropPolicy.DROP_NEWEST:
                    return False
                elif self._drop_policy == EnumLedgerSinkDropPolicy.RAISE:
                    raise LedgerSinkFullError(
                        f"Sink buffer full ({self._max_buffer_size} events)"
                    )
                elif self._drop_policy == EnumLedgerSinkDropPolicy.BLOCK:
                    # BLOCK policy requires proper condition variable implementation
                    raise NotImplementedError(
                        "BLOCK policy is not yet implemented in FileSpoolLedgerSink. "
                        "Use DROP_OLDEST, DROP_NEWEST, or RAISE."
                    )
                # DROP_OLDEST: deque with maxlen handles this automatically

            self._buffer.append(event)

            # Start background flush task if not running (inside lock to prevent race)
            if self._flush_task is None or self._flush_task.done():
                self._flush_task = asyncio.create_task(self._background_flush_loop())

        return True

    async def flush(self) -> int:
        """Flush all buffered events to disk.

        Returns:
            Number of events flushed.

        Raises:
            LedgerSinkError: If write fails.
        """
        async with self._lock:
            return await self._flush_buffer_locked()

    async def _flush_buffer_locked(self) -> int:
        """Flush buffer while holding lock. Internal method."""
        if not self._buffer:
            return 0

        flushed = 0
        try:
            # Ensure we have an open file
            if self._file_handle is None:
                self._open_new_file()

            while self._buffer:
                event = self._buffer.popleft()
                line = self._serialize_event(event)
                line_bytes = (line + "\n").encode("utf-8")

                # Check if we need to rotate
                if self._current_file_size + len(line_bytes) > self._max_file_size:
                    self._rotate_file()

                assert self._file_handle is not None
                self._file_handle.write(line_bytes)
                self._current_file_size += len(line_bytes)
                flushed += 1

            # Ensure data is written to OS buffer
            if self._file_handle is not None:
                self._file_handle.flush()

        except Exception as e:
            logger.exception("Failed to flush ledger events")
            raise LedgerSinkError(f"Flush failed: {e}") from e

        return flushed

    def _open_new_file(self) -> None:
        """Open a new spool file."""
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self._file_sequence += 1
        filename = f"ledger_{timestamp}_{self._file_sequence:04d}.jsonl"
        self._current_file_path = self._spool_dir / filename

        # Open file for binary append with buffering
        # File handle is manually managed for rotation, not via context manager
        self._file_handle = open(self._current_file_path, "ab", buffering=8192)  # noqa: SIM115
        self._current_file_size = 0

        logger.info(f"Opened new ledger spool file: {self._current_file_path}")

    def _rotate_file(self) -> None:
        """Close current file and open a new one."""
        if self._file_handle is not None:
            self._file_handle.flush()
            os.fsync(self._file_handle.fileno())
            self._file_handle.close()
            self._file_handle = None

        self._open_new_file()

    def _serialize_event(self, event: ModelLedgerEventBase) -> str:
        """Serialize event to JSON string."""
        # Use Pydantic's model_dump with mode='json' for JSON-safe types
        data = event.model_dump(mode="json")
        return json.dumps(data, separators=(",", ":"), sort_keys=True)

    async def _background_flush_loop(self) -> None:
        """Background task that periodically flushes the buffer."""
        while not self._closed:
            await asyncio.sleep(self._flush_interval)
            if self._buffer:
                try:
                    count = await self.flush()
                    if count > 0:
                        logger.debug(f"Background flush: {count} events")
                except Exception:
                    logger.exception("Background flush failed")

    async def close(self) -> None:
        """Close the sink, flushing any pending events.

        Raises:
            LedgerSinkError: If final flush fails.
        """
        self._closed = True

        # Cancel background flush
        if self._flush_task is not None and not self._flush_task.done():
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Final flush
        async with self._lock:
            try:
                await self._flush_buffer_locked()
            finally:
                # Close file with fsync for durability
                if self._file_handle is not None:
                    self._file_handle.flush()
                    os.fsync(self._file_handle.fileno())
                    self._file_handle.close()
                    self._file_handle = None

        logger.info("FileSpoolLedgerSink closed")

    @property
    def drop_policy(self) -> EnumLedgerSinkDropPolicy:
        """Get the configured drop policy."""
        return self._drop_policy

    @property
    def is_closed(self) -> bool:
        """Check if the sink is closed."""
        return self._closed

    @property
    def pending_count(self) -> int:
        """Get the number of events pending in the buffer."""
        return len(self._buffer)

    @property
    def current_file_path(self) -> Path | None:
        """Get the current spool file path (for monitoring)."""
        return self._current_file_path


__all__ = [
    "FileSpoolLedgerSink",
]
