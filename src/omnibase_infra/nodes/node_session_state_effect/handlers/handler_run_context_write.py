# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Handler for writing a run context (runs/{run_id}.json).

Run context documents are single-writer (owned by the pipeline that
created the run), so no file locking is required. Uses the same
write-tmp-fsync-rename pattern as session index writes for crash safety.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from uuid import UUID

from omnibase_infra.nodes.node_session_state_effect.models import (
    RUN_ID_PATTERN,
    ModelRunContext,
    ModelSessionStateResult,
)

logger = logging.getLogger(__name__)


class HandlerRunContextWrite:
    """Atomically write a run context document to the filesystem.

    No file locking is used — each run document has a single writer
    (the pipeline that created it).
    """

    def __init__(self, state_dir: Path) -> None:
        """Initialize with state directory path.

        Args:
            state_dir: Root directory for session state (e.g. ``~/.claude/state``).
        """
        self._state_dir = state_dir
        self._runs_dir = (state_dir / "runs").resolve()

    async def handle(
        self,
        context: ModelRunContext,
        correlation_id: UUID,
    ) -> ModelSessionStateResult:
        """Atomically write runs/{run_id}.json.

        Args:
            context: The run context to persist.
            correlation_id: Correlation ID for distributed tracing.

        Returns:
            Operation result indicating success or failure.
        """
        return await asyncio.to_thread(self._write_sync, context, correlation_id)

    def _write_sync(
        self,
        context: ModelRunContext,
        correlation_id: UUID,
    ) -> ModelSessionStateResult:
        """Synchronous write logic, executed off the event loop."""
        # Defense-in-depth: reject unsafe IDs even if model_construct() skipped validators
        if not RUN_ID_PATTERN.match(context.run_id) or ".." in context.run_id:
            return ModelSessionStateResult(
                success=False,
                operation="run_context_write",
                correlation_id=correlation_id,
                error=f"Invalid run_id: contains path traversal characters: {context.run_id!r}",
                error_code="RUN_CONTEXT_INVALID_ID",
            )

        runs_dir = self._runs_dir

        try:
            runs_dir.mkdir(parents=True, exist_ok=True)

            run_path = runs_dir / f"{context.run_id}.json"

            # Defense-in-depth: verify resolved path stays within runs directory
            if run_path.resolve().parent != runs_dir:
                return ModelSessionStateResult(
                    success=False,
                    operation="run_context_write",
                    correlation_id=correlation_id,
                    error=f"Invalid run_id: resolved path escapes state directory: {context.run_id!r}",
                    error_code="RUN_CONTEXT_INVALID_ID",
                )
            data = context.model_dump(mode="json")

            # Write to temp file, fsync, then atomic rename
            fd, tmp_path = tempfile.mkstemp(
                dir=str(runs_dir),
                prefix=f".{context.run_id}_",
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

                Path(tmp_path).rename(run_path)
            except BaseException:
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except OSError:
                    pass
                raise

            logger.debug(
                "Wrote run context %s (status=%s)", context.run_id, context.status.value
            )
            return ModelSessionStateResult(
                success=True,
                operation="run_context_write",
                correlation_id=correlation_id,
                files_affected=1,
            )

        except OSError as e:
            logger.warning("Failed to write run context %s: %s", context.run_id, e)
            return ModelSessionStateResult(
                success=False,
                operation="run_context_write",
                correlation_id=correlation_id,
                error=f"I/O error writing run context {context.run_id}: {e}",
                error_code="RUN_CONTEXT_WRITE_ERROR",
            )
        except Exception as e:
            # Intentional catch-all: this handler must never raise, as an
            # unhandled exception would crash the pipeline.  logger.exception
            # records the full traceback for post-mortem debugging.
            logger.exception(
                "Unexpected error writing run context %s: %s", context.run_id, e
            )
            return ModelSessionStateResult(
                success=False,
                operation="run_context_write",
                correlation_id=correlation_id,
                error=f"Unexpected error writing run context {context.run_id}: {e}",
                error_code="RUN_CONTEXT_WRITE_UNEXPECTED",
            )


__all__: list[str] = ["HandlerRunContextWrite"]
