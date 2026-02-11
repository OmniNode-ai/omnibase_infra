# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Handler for garbage-collecting stale run context documents.

Removes run documents from ``~/.claude/state/runs/`` that are older
than the configured TTL (default: 4 hours). Returns the list of deleted
run IDs so that the caller (orchestrator) can update the session index.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from pydantic import ValidationError

from omnibase_infra.nodes.node_session_state_effect.models import (
    ModelRunContext,
    ModelSessionStateResult,
)

logger = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS: float = 14400.0  # 4 hours
DEFAULT_MAX_DELETIONS: int = 500


class HandlerStaleRunGC:
    """Garbage-collect stale run context documents.

    Scans ``runs/`` directory and deletes any run document whose
    ``updated_at`` is older than the configured TTL. The caller is
    responsible for removing deleted run IDs from the session index.
    """

    def __init__(
        self,
        state_dir: Path,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        max_deletions: int = DEFAULT_MAX_DELETIONS,
    ) -> None:
        """Initialize with state directory and TTL.

        Args:
            state_dir: Root directory for session state.
            ttl_seconds: Time-to-live in seconds (default: 14400 = 4 hours).
            max_deletions: Maximum files to delete per GC pass (default: 500).
                Callers should re-invoke if the result count equals this limit.
        """
        self._state_dir = state_dir
        self._ttl_seconds = ttl_seconds
        self._max_deletions = max_deletions

    async def handle(
        self,
        correlation_id: UUID,
    ) -> tuple[list[str], ModelSessionStateResult]:
        """Scan and delete stale run documents.

        Args:
            correlation_id: Correlation ID for distributed tracing.

        Returns:
            Tuple of (list of deleted run_ids, operation result).
            The caller should use the deleted IDs to update the session index.
            Note: deleted IDs may include file stems from malformed documents
            that don't correspond to session index entries; callers should
            silently ignore missing entries during index cleanup.
        """
        return await asyncio.to_thread(self._gc_sync, correlation_id)

    def _gc_sync(
        self,
        correlation_id: UUID,
    ) -> tuple[list[str], ModelSessionStateResult]:
        """Synchronous GC logic, executed off the event loop."""
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

        resolved_runs_dir = runs_dir.resolve()

        # Sort by mtime (oldest first) so max_deletions cap removes the
        # oldest files deterministically, regardless of filesystem order.
        def _safe_mtime(p: Path) -> float:
            try:
                return p.stat().st_mtime
            except OSError:
                return 0.0  # disappeared between glob and sort; sort to front

        run_files = sorted(runs_dir.glob("*.json"), key=_safe_mtime)

        for run_file in run_files:
            if len(deleted_ids) >= self._max_deletions:
                logger.info(
                    "GC: reached max_deletions=%d, stopping", self._max_deletions
                )
                break

            # Skip symlinks and files that resolve outside the runs directory
            if (
                run_file.is_symlink()
                or not run_file.is_file()
                or run_file.resolve().parent != resolved_runs_dir
            ):
                logger.warning(
                    "GC: skipping non-regular or external file %s", run_file.name
                )
                continue

            try:
                raw = run_file.read_text(encoding="utf-8")
                data = json.loads(raw)
                ctx = ModelRunContext.model_validate(data)

                if ctx.is_stale(self._ttl_seconds):
                    # Record IDs before unlink so they're tracked even if
                    # an unexpected error occurs after deletion.
                    stem = run_file.stem
                    deleted_ids.append(ctx.run_id)
                    if stem != ctx.run_id:
                        logger.warning(
                            "GC: file stem %r differs from run_id %r",
                            stem,
                            ctx.run_id,
                        )
                        deleted_ids.append(stem)
                    run_file.unlink(missing_ok=True)
                    logger.info(
                        "GC'd stale run %s (age=%.0fs, ttl=%.0fs)",
                        ctx.run_id,
                        (now - ctx.updated_at).total_seconds(),
                        self._ttl_seconds,
                    )
            except (json.JSONDecodeError, ValueError, ValidationError) as e:
                # Malformed files are also GC candidates — delete them.
                # Note: the stem may not correspond to a valid session index
                # entry; callers should treat deleted_ids as best-effort and
                # silently ignore missing entries when updating the index.
                logger.warning(
                    "GC: removing malformed run file %s: %s", run_file.name, e
                )
                # Track stem before unlink (defensive, consistent with stale path)
                stem = run_file.stem
                deleted_ids.append(stem)
                run_file.unlink(missing_ok=True)
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
