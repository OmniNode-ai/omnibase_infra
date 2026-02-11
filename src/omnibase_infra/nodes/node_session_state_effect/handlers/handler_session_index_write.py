# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Handler for writing the session index (session.json).

Performs atomic writes with ``flock`` for concurrent pipeline safety:
  1. Write to ``session.json.tmp``
  2. ``fsync`` the temp file
  3. ``rename`` over ``session.json`` (atomic on POSIX)

File locking (``flock``) protects ``session.json`` from concurrent writers,
since multiple pipelines may register new runs simultaneously.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from uuid import UUID

from omnibase_infra.nodes.node_session_state_effect.models import (
    ModelSessionIndex,
    ModelSessionStateResult,
)

if sys.platform != "win32":
    import fcntl
else:  # pragma: no cover — flock not available on Windows
    fcntl = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class HandlerSessionIndexWrite:
    """Atomically write the session index to the filesystem with flock.

    This handler uses the write-tmp-fsync-rename pattern to ensure
    that ``session.json`` is never left in a partial state, even on
    power loss or concurrent access.

    Note:
        File locking (``flock``) requires a POSIX platform (Linux, macOS).
        On Windows, the handler raises ``RuntimeError`` at construction time.

        The lock file (``session.json.lock``) persists after use. This is
        intentional — ``flock`` advisory locks do not require file deletion,
        and recreating the file on every write would race with other writers.
    """

    def __init__(self, state_dir: Path) -> None:
        """Initialize with state directory path.

        Args:
            state_dir: Root directory for session state (e.g. ``~/.claude/state``).

        Raises:
            RuntimeError: If ``fcntl`` is not available (Windows).
        """
        if fcntl is None:
            msg = "HandlerSessionIndexWrite requires fcntl (POSIX-only)"
            raise RuntimeError(msg)
        self._state_dir = state_dir

    async def handle(
        self,
        index: ModelSessionIndex,
        correlation_id: UUID,
    ) -> ModelSessionStateResult:
        """Atomically write session.json with flock protection.

        Args:
            index: The session index to persist.
            correlation_id: Correlation ID for distributed tracing.

        Returns:
            Operation result indicating success or failure.
        """
        return await asyncio.to_thread(self._write_sync, index, correlation_id)

    async def handle_read_modify_write(
        self,
        transform: Callable[[ModelSessionIndex], ModelSessionIndex],
        correlation_id: UUID,
    ) -> tuple[ModelSessionIndex | None, ModelSessionStateResult]:
        """Atomically read, transform, and write session.json under flock.

        This method holds the flock for the entire read-modify-write cycle,
        preventing lost-update races when multiple pipelines concurrently
        modify the session index (e.g. adding runs simultaneously).

        Prefer this over separate ``read()`` + ``handle()`` calls whenever
        the write depends on the current index state.

        Args:
            transform: Pure function that receives the current index and
                returns the new index to persist.
            correlation_id: Correlation ID for distributed tracing.

        Returns:
            Tuple of (new index written, operation result). The index is
            ``None`` when the operation fails.
        """
        return await asyncio.to_thread(
            self._read_modify_write_sync, transform, correlation_id
        )

    def _read_modify_write_sync(
        self,
        transform: Callable[[ModelSessionIndex], ModelSessionIndex],
        correlation_id: UUID,
    ) -> tuple[ModelSessionIndex | None, ModelSessionStateResult]:
        """Synchronous read-modify-write under flock."""
        session_path = self._state_dir / "session.json"

        try:
            self._state_dir.mkdir(parents=True, exist_ok=True)

            lock_path = self._state_dir / "session.json.lock"
            lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX)

                # Read current state under the lock (use try/except to
                # avoid TOCTOU if an external process deletes the file)
                try:
                    raw = session_path.read_text(encoding="utf-8")
                    current = ModelSessionIndex.model_validate(json.loads(raw))
                except FileNotFoundError:
                    current = ModelSessionIndex()

                # Apply caller's transform
                new_index = transform(current)
                data = new_index.model_dump(mode="json")

                # Write atomically
                fd, tmp_path = tempfile.mkstemp(
                    dir=str(self._state_dir),
                    prefix=".session_",
                    suffix=".tmp",
                )
                try:
                    file_obj = os.fdopen(fd, "w", encoding="utf-8")
                except BaseException:
                    os.close(fd)
                    raise
                try:
                    with file_obj:
                        json.dump(data, file_obj, indent=2, default=str)
                        file_obj.flush()
                        os.fsync(file_obj.fileno())
                    Path(tmp_path).rename(session_path)
                except BaseException:
                    try:
                        Path(tmp_path).unlink(missing_ok=True)
                    except OSError:
                        pass
                    raise
            finally:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                except OSError:
                    pass
                finally:
                    os.close(lock_fd)

            logger.debug(
                "Atomic read-modify-write session.json with %d runs",
                len(new_index.recent_run_ids),
            )
            return (
                new_index,
                ModelSessionStateResult(
                    success=True,
                    operation="session_index_read_modify_write",
                    correlation_id=correlation_id,
                    files_affected=1,
                ),
            )

        except OSError as e:
            logger.warning("Failed atomic read-modify-write session.json: %s", e)
            return (
                None,
                ModelSessionStateResult(
                    success=False,
                    operation="session_index_read_modify_write",
                    correlation_id=correlation_id,
                    error=f"I/O error in atomic read-modify-write: {e}",
                    error_code="SESSION_INDEX_RMW_ERROR",
                ),
            )
        except Exception as e:
            logger.exception(
                "Unexpected error in atomic read-modify-write session.json: %s", e
            )
            return (
                None,
                ModelSessionStateResult(
                    success=False,
                    operation="session_index_read_modify_write",
                    correlation_id=correlation_id,
                    error=f"Unexpected error in atomic read-modify-write: {e}",
                    error_code="SESSION_INDEX_RMW_UNEXPECTED",
                ),
            )

    def _write_sync(
        self,
        index: ModelSessionIndex,
        correlation_id: UUID,
    ) -> ModelSessionStateResult:
        """Synchronous write logic with flock, executed off the event loop."""
        session_path = self._state_dir / "session.json"

        try:
            # Ensure the state directory exists
            self._state_dir.mkdir(parents=True, exist_ok=True)

            data = index.model_dump(mode="json")

            # Acquire an exclusive flock on a lock file
            lock_path = self._state_dir / "session.json.lock"
            lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX)

                # Write to temp file in the same directory (same filesystem)
                fd, tmp_path = tempfile.mkstemp(
                    dir=str(self._state_dir),
                    prefix=".session_",
                    suffix=".tmp",
                )
                try:
                    file_obj = os.fdopen(fd, "w", encoding="utf-8")
                except BaseException:
                    os.close(fd)
                    raise
                try:
                    with file_obj:
                        json.dump(data, file_obj, indent=2, default=str)
                        file_obj.flush()
                        os.fsync(file_obj.fileno())

                    # Atomic rename (POSIX guarantees)
                    Path(tmp_path).rename(session_path)
                except BaseException:
                    # Clean up temp file on any error
                    try:
                        Path(tmp_path).unlink(missing_ok=True)
                    except OSError:
                        pass
                    raise
            finally:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                except OSError:
                    pass
                finally:
                    os.close(lock_fd)

            logger.debug("Wrote session.json with %d runs", len(index.recent_run_ids))
            return ModelSessionStateResult(
                success=True,
                operation="session_index_write",
                correlation_id=correlation_id,
                files_affected=1,
            )

        except OSError as e:
            logger.warning("Failed to write session.json: %s", e)
            return ModelSessionStateResult(
                success=False,
                operation="session_index_write",
                correlation_id=correlation_id,
                error=f"I/O error writing session.json: {e}",
                error_code="SESSION_INDEX_WRITE_ERROR",
            )
        except Exception as e:
            logger.exception("Unexpected error writing session.json: %s", e)
            return ModelSessionStateResult(
                success=False,
                operation="session_index_write",
                correlation_id=correlation_id,
                error=f"Unexpected error writing session.json: {e}",
                error_code="SESSION_INDEX_WRITE_UNEXPECTED",
            )


__all__: list[str] = ["HandlerSessionIndexWrite"]
