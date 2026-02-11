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
from pathlib import Path
from uuid import UUID

from pydantic import ValidationError

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_session_state_effect.models import (
    RUN_ID_PATTERN,
    ModelRunContext,
    ModelSessionStateResult,
)
from omnibase_infra.utils import sanitize_error_string

logger = logging.getLogger(__name__)


class HandlerRunContextRead:
    """Read a run context document from the filesystem.

    Reads ``runs/{run_id}.json`` from the configured state directory and
    returns a validated ``ModelRunContext``. If the file does not exist,
    returns ``(None, result)`` with ``result.success=True`` and
    ``files_affected=0`` (missing file is not an error).

    Security:
        Run IDs are validated against a filesystem-safe allowlist before
        constructing any file paths, and resolved paths are checked to
        ensure they stay within the ``runs/`` directory (defense-in-depth
        against path traversal).
    """

    def __init__(self, state_dir: Path) -> None:
        """Initialize with state directory path.

        Args:
            state_dir: Root directory for session state (e.g. ``~/.claude/state``).
        """
        self._state_dir = state_dir

    @property
    def handler_type(self) -> EnumHandlerType:
        """Architectural role: infrastructure handler for filesystem I/O.

        Returns:
            EnumHandlerType.INFRA_HANDLER - This handler is an infrastructure
            handler that reads run context files from the filesystem.
        """
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Behavioral classification: side-effecting filesystem read.

        Returns:
            EnumHandlerTypeCategory.EFFECT - This handler performs side-effecting
            I/O operations (filesystem reads).
        """
        return EnumHandlerTypeCategory.EFFECT

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
        if not RUN_ID_PATTERN.match(run_id) or ".." in run_id:
            return (
                None,
                ModelSessionStateResult(
                    success=False,
                    operation="run_context_read",
                    correlation_id=correlation_id,
                    error="Invalid run_id: contains disallowed characters",
                    error_code="RUN_CONTEXT_INVALID_ID",
                ),
            )

        return await asyncio.to_thread(self._read_sync, run_id, correlation_id)

    def _read_sync(
        self,
        run_id: str,
        correlation_id: UUID,
    ) -> tuple[ModelRunContext | None, ModelSessionStateResult]:
        """Synchronous read logic, executed off the event loop.

        Resolves the file path for ``runs/{run_id}.json``, validates
        the resolved path stays within the runs directory, then reads
        and parses the document. Returns ``None`` with a success result
        if the file does not exist (missing is not an error).

        Args:
            run_id: The unique run identifier (pre-validated by caller).
            correlation_id: Correlation ID for distributed tracing.

        Returns:
            Tuple of (parsed context or None, operation result).
        """
        runs_dir = (self._state_dir / "runs").resolve()
        run_path = runs_dir / f"{run_id}.json"

        # Defense-in-depth: structural comparison ensures the resolved path
        # stays within the runs directory, even if run_id contains ".." etc.
        if run_path.resolve().parent != runs_dir:
            return (
                None,
                ModelSessionStateResult(
                    success=False,
                    operation="run_context_read",
                    correlation_id=correlation_id,
                    error="Invalid run_id: resolved path escapes state directory",
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
        except FileNotFoundError:
            # TOCTOU: file was deleted between exists() check and read_text().
            # Treat the same as "not found" -- this is not an error.
            logger.debug(
                "Run context disappeared between exists() and read_text(): %s",
                run_path,
            )
            return (
                None,
                ModelSessionStateResult(
                    success=True,
                    operation="run_context_read",
                    correlation_id=correlation_id,
                    files_affected=0,
                ),
            )
        except (json.JSONDecodeError, ValueError, ValidationError) as e:
            logger.warning("Failed to parse run context %s: %s", run_id, e)
            return (
                None,
                ModelSessionStateResult(
                    success=False,
                    operation="run_context_read",
                    correlation_id=correlation_id,
                    error=sanitize_error_string(
                        f"Failed to parse run context {run_id}: {e}"
                    ),
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
                    error=sanitize_error_string(
                        f"I/O error reading run context {run_id}: {e}"
                    ),
                    error_code="RUN_CONTEXT_IO_ERROR",
                    files_affected=1,
                ),
            )


__all__: list[str] = ["HandlerRunContextRead"]
