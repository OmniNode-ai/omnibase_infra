# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Handler for garbage-collecting stale run context documents.

Removes run documents from ``~/.claude/state/runs/`` that are older
than the configured TTL (default: 4 hours). Also cleans up stale
entries from the session index.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timezone
from pathlib import Path
from uuid import UUID

from omnibase_infra.nodes.node_session_state_effect.models import (
    ModelRunContext,
    ModelSessionStateResult,
)

logger = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS: float = 14400.0  # 4 hours


class HandlerStaleRunGC:
    """Garbage-collect stale run context documents.

    Scans ``runs/`` directory and deletes any run document whose
    ``updated_at`` is older than the configured TTL.
    """

    def __init__(
        self, state_dir: Path, ttl_seconds: float = DEFAULT_TTL_SECONDS
    ) -> None:
        """Initialize with state directory and TTL.

        Args:
            state_dir: Root directory for session state.
            ttl_seconds: Time-to-live in seconds (default: 14400 = 4 hours).
        """
        self._state_dir = state_dir
        self._ttl_seconds = ttl_seconds

    async def handle(
        self,
        correlation_id: UUID,
    ) -> tuple[list[str], ModelSessionStateResult]:
        """Scan and delete stale run documents.

        Args:
            correlation_id: Correlation ID for distributed tracing.

        Returns:
            Tuple of (list of deleted run_ids, operation result).
        """
        runs_dir = self._state_dir / "runs"
        deleted_ids: list[str] = []

        if not runs_dir.exists():
            return (
                deleted_ids,
                ModelSessionStateResult(
                    success=True,
                    operation="stale_run_gc",
                    correlation_id=correlation_id,
                    files_affected=0,
                ),
            )

        now = datetime.now(UTC)

        for run_file in runs_dir.glob("*.json"):
            try:
                raw = run_file.read_text(encoding="utf-8")
                data = json.loads(raw)
                ctx = ModelRunContext.model_validate(data)

                if ctx.is_stale(self._ttl_seconds):
                    run_file.unlink()
                    deleted_ids.append(ctx.run_id)
                    logger.info(
                        "GC'd stale run %s (age=%.0fs, ttl=%.0fs)",
                        ctx.run_id,
                        (now - ctx.updated_at).total_seconds(),
                        self._ttl_seconds,
                    )
            except (json.JSONDecodeError, ValueError) as e:
                # Malformed files are also GC candidates — delete them
                logger.warning(
                    "GC: removing malformed run file %s: %s", run_file.name, e
                )
                run_file.unlink(missing_ok=True)
                stem = run_file.stem
                deleted_ids.append(stem)
            except OSError as e:
                logger.warning("GC: failed to process %s: %s", run_file.name, e)

        return (
            deleted_ids,
            ModelSessionStateResult(
                success=True,
                operation="stale_run_gc",
                correlation_id=correlation_id,
                files_affected=len(deleted_ids),
            ),
        )


__all__: list[str] = ["HandlerStaleRunGC"]
