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

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
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

    @property
    def handler_type(self) -> EnumHandlerType:
        """Architectural role: infrastructure handler for filesystem I/O.

        Returns:
            EnumHandlerType.INFRA_HANDLER - This handler is an infrastructure
            handler that garbage-collects stale run context files.
        """
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Behavioral classification: side-effecting filesystem deletion.

        Returns:
            EnumHandlerTypeCategory.EFFECT - This handler performs side-effecting
            I/O operations (filesystem reads and deletes).
        """
        return EnumHandlerTypeCategory.EFFECT

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
            Note: ``len(deleted_ids)`` may exceed ``result.files_affected``
            because a single file can contribute two IDs when its stem differs
            from the embedded ``run_id``.  Malformed documents are deleted but
            their stems are NOT added to ``deleted_ids`` to avoid false-positive
            removal of active runs whose run_id happens to match the file stem.
        """
        return await asyncio.to_thread(self._gc_sync, correlation_id)

    def _gc_sync(
        self,
        correlation_id: UUID,
    ) -> tuple[list[str], ModelSessionStateResult]:
        """Synchronous GC logic, executed off the event loop.

        Scans ``runs/*.json`` sorted by mtime (oldest first), parses
        each document, and deletes any whose ``updated_at`` exceeds
        the configured TTL. Malformed files are also deleted. Stops
        after ``max_deletions`` files to bound execution time.

        Args:
            correlation_id: Correlation ID for distributed tracing.

        Returns:
            Tuple of (list of deleted run IDs, operation result).
        """
        runs_dir = self._state_dir / "runs"
        deleted_ids: list[str] = []
        files_deleted: int = 0

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
            if files_deleted >= self._max_deletions:
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

                age_s = (now - ctx.updated_at).total_seconds()
                if age_s < -5.0:
                    logger.warning(
                        "GC: run %s has updated_at %.0fs in the future (clock skew?)",
                        ctx.run_id,
                        -age_s,
                    )

                if ctx.is_stale(self._ttl_seconds):
                    stem = run_file.stem
                    try:
                        run_file.unlink()
                    except FileNotFoundError:
                        # Concurrently deleted by another process — skip.
                        continue
                    files_deleted += 1
                    deleted_ids.append(ctx.run_id)
                    if stem != ctx.run_id:
                        # Also record the stem so callers can clean up
                        # index entries keyed by either value.  Note:
                        # len(deleted_ids) may exceed files_affected.
                        logger.warning(
                            "GC: file stem %r differs from run_id %r",
                            stem,
                            ctx.run_id,
                        )
                        deleted_ids.append(stem)
                    logger.info(
                        "GC'd stale run %s (age=%.0fs, ttl=%.0fs)",
                        ctx.run_id,
                        (now - ctx.updated_at).total_seconds(),
                        self._ttl_seconds,
                    )
                else:
                    age_seconds = (now - ctx.updated_at).total_seconds()
                    logger.debug(
                        "GC: run %s not stale (age=%.0fs, ttl=%.0fs)",
                        ctx.run_id,
                        age_seconds,
                        self._ttl_seconds,
                    )
            except (json.JSONDecodeError, ValueError, ValidationError) as e:
                # Malformed files are also GC candidates — delete them.
                # Do NOT add stems to deleted_ids: the stem may coincide with
                # a valid run_id of an active run, causing the caller to
                # incorrectly remove that active run from the session index.
                logger.warning(
                    "GC: removing malformed run file %s: %s", run_file.name, e
                )
                try:
                    run_file.unlink()
                except FileNotFoundError:
                    continue  # concurrently deleted
                files_deleted += 1
            except OSError as e:
                logger.warning("GC: failed to process %s: %s", run_file.name, e)

        return (
            deleted_ids,
            ModelSessionStateResult(
                success=True,
                operation="stale_run_gc",
                correlation_id=correlation_id,
                files_affected=files_deleted,
            ),
        )


__all__: list[str] = ["HandlerStaleRunGC"]
