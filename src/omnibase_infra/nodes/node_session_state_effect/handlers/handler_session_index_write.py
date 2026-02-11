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

            data = json.loads(index.model_dump_json())

            # Acquire an exclusive flock on a lock file
            lock_path = self._state_dir / "session.json.lock"
            lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX)

                # Write to temp file in the same directory (same filesystem)
                fd, tmp_path = tempfile.mkstemp(
                    dir=str(self._state_dir),
                    prefix=".session_",
                    suffix=".tmp",
                )
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2, default=str)
                        f.flush()
                        os.fsync(f.fileno())

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
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
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


__all__: list[str] = ["HandlerSessionIndexWrite"]
