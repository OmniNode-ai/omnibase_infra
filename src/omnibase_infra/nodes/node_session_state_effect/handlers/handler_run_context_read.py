# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Handler for reading a run context (runs/{run_id}.json).

Each run context is a single-writer document owned by the pipeline
that created it. No locking is required for reads.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from uuid import UUID

from omnibase_infra.nodes.node_session_state_effect.models import (
    ModelRunContext,
    ModelSessionStateResult,
)

logger = logging.getLogger(__name__)

_SAFE_RUN_ID = re.compile(r"^[a-zA-Z0-9._-]+$")


class HandlerRunContextRead:
    """Read a run context document from the filesystem."""

    def __init__(self, state_dir: Path) -> None:
        """Initialize with state directory path.

        Args:
            state_dir: Root directory for session state (e.g. ``~/.claude/state``).
        """
        self._state_dir = state_dir

    async def handle(
        self,
        run_id: str,
        correlation_id: UUID,
    ) -> tuple[ModelRunContext | None, ModelSessionStateResult]:
        """Read runs/{run_id}.json and return the parsed context.

        Args:
            run_id: The unique run identifier.
            correlation_id: Correlation ID for distributed tracing.

        Returns:
            Tuple of (parsed context or None if not found, operation result).
        """
        if not _SAFE_RUN_ID.match(run_id) or ".." in run_id:
            return (
                None,
                ModelSessionStateResult(
                    success=False,
                    operation="run_context_read",
                    correlation_id=correlation_id,
                    error=f"Invalid run_id: contains path traversal characters: {run_id!r}",
                    error_code="RUN_CONTEXT_INVALID_ID",
                ),
            )

        return await asyncio.to_thread(self._read_sync, run_id, correlation_id)

    def _read_sync(
        self,
        run_id: str,
        correlation_id: UUID,
    ) -> tuple[ModelRunContext | None, ModelSessionStateResult]:
        """Synchronous read logic, executed off the event loop."""
        runs_dir = self._state_dir / "runs"
        run_path = runs_dir / f"{run_id}.json"

        # Defense-in-depth: verify resolved path stays within runs directory
        if run_path.exists() and run_path.resolve().parent != runs_dir.resolve():
            return (
                None,
                ModelSessionStateResult(
                    success=False,
                    operation="run_context_read",
                    correlation_id=correlation_id,
                    error=f"Invalid run_id: resolved path escapes state directory: {run_id!r}",
                    error_code="RUN_CONTEXT_INVALID_ID",
                ),
            )

        if not run_path.exists():
            logger.debug("Run context not found: %s", run_path)
            return (
                None,
                ModelSessionStateResult(
                    success=True,
                    operation="run_context_read",
                    correlation_id=correlation_id,
                    files_affected=0,
                ),
            )

        try:
            raw = run_path.read_text(encoding="utf-8")
            data = json.loads(raw)
            ctx = ModelRunContext.model_validate(data)
            return (
                ctx,
                ModelSessionStateResult(
                    success=True,
                    operation="run_context_read",
                    correlation_id=correlation_id,
                    files_affected=1,
                ),
            )
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("Failed to parse run context %s: %s", run_id, e)
            return (
                None,
                ModelSessionStateResult(
                    success=False,
                    operation="run_context_read",
                    correlation_id=correlation_id,
                    error=f"Failed to parse run context {run_id}: {e}",
                    error_code="RUN_CONTEXT_PARSE_ERROR",
                    files_affected=1,
                ),
            )
        except OSError as e:
            logger.warning("Failed to read run context %s: %s", run_id, e)
            return (
                None,
                ModelSessionStateResult(
                    success=False,
                    operation="run_context_read",
                    correlation_id=correlation_id,
                    error=f"I/O error reading run context {run_id}: {e}",
                    error_code="RUN_CONTEXT_IO_ERROR",
                ),
            )


__all__: list[str] = ["HandlerRunContextRead"]
