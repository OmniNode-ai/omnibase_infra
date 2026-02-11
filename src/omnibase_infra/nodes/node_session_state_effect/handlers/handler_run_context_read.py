# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Handler for reading a run context (runs/{run_id}.json).

Each run context is a single-writer document owned by the pipeline
that created it. No locking is required for reads.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from uuid import UUID

from omnibase_infra.nodes.node_session_state_effect.models import (
    ModelRunContext,
    ModelSessionStateResult,
)

logger = logging.getLogger(__name__)


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
        run_path = self._state_dir / "runs" / f"{run_id}.json"

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
