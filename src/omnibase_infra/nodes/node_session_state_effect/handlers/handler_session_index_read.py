# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Handler for reading the session index (session.json).

Reads ``~/.claude/state/session.json`` and returns a ``ModelSessionIndex``.
If the file does not exist, returns a default empty index.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from uuid import UUID

from omnibase_infra.nodes.node_session_state_effect.models import (
    ModelSessionIndex,
    ModelSessionStateResult,
)

logger = logging.getLogger(__name__)


class HandlerSessionIndexRead:
    """Read the session index from the filesystem.

    This handler reads ``session.json`` from the configured state directory.
    If the file does not exist, it returns a default empty ``ModelSessionIndex``.
    """

    def __init__(self, state_dir: Path) -> None:
        """Initialize with state directory path.

        Args:
            state_dir: Root directory for session state (e.g. ``~/.claude/state``).
        """
        self._state_dir = state_dir

    async def handle(
        self,
        correlation_id: UUID,
    ) -> tuple[ModelSessionIndex | None, ModelSessionStateResult]:
        """Read session.json and return the parsed index.

        Args:
            correlation_id: Correlation ID for distributed tracing.

        Returns:
            Tuple of (parsed index or None on error, operation result).
            Returns ``None`` for the index when parsing or I/O fails so
            that callers are forced to check ``result.success`` before
            using the index (prevents silent overwrite with empty data).
        """
        return await asyncio.to_thread(self._read_sync, correlation_id)

    def _read_sync(
        self,
        correlation_id: UUID,
    ) -> tuple[ModelSessionIndex | None, ModelSessionStateResult]:
        """Synchronous read logic, executed off the event loop."""
        session_path = self._state_dir / "session.json"

        try:
            raw = session_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            logger.debug(
                "session.json not found at %s, returning default", session_path
            )
            return (
                ModelSessionIndex(),
                ModelSessionStateResult(
                    success=True,
                    operation="session_index_read",
                    correlation_id=correlation_id,
                    files_affected=0,
                ),
            )
        except OSError as e:
            logger.warning("Failed to read session.json: %s", e)
            return (
                None,
                ModelSessionStateResult(
                    success=False,
                    operation="session_index_read",
                    correlation_id=correlation_id,
                    error=f"I/O error reading session.json: {e}",
                    error_code="SESSION_INDEX_IO_ERROR",
                    files_affected=0,
                ),
            )

        try:
            data = json.loads(raw)
            index = ModelSessionIndex.model_validate(data)
            return (
                index,
                ModelSessionStateResult(
                    success=True,
                    operation="session_index_read",
                    correlation_id=correlation_id,
                    files_affected=1,
                ),
            )
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("Failed to parse session.json: %s", e)
            return (
                None,
                ModelSessionStateResult(
                    success=False,
                    operation="session_index_read",
                    correlation_id=correlation_id,
                    error=f"Failed to parse session.json: {e}",
                    error_code="SESSION_INDEX_PARSE_ERROR",
                    files_affected=1,
                ),
            )


__all__: list[str] = ["HandlerSessionIndexRead"]
