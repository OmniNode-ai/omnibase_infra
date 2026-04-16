# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler that reads a plan file from the filesystem.

This is an EFFECT handler - it performs I/O (filesystem read).
"""

from __future__ import annotations

import logging
from pathlib import Path
from uuid import UUID

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_scope_file_read_effect.models.model_scope_file_read_result import (
    ModelScopeFileReadResult,
)

logger = logging.getLogger(__name__)


class HandlerScopeFileRead:
    """Reads a plan file from the filesystem and returns its content."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def handle(
        self,
        file_path: str,
        correlation_id: UUID,
        output_path: str = "~/.claude/scope-manifest.json",
    ) -> ModelScopeFileReadResult:
        """Read a file from disk and return its content.

        Args:
            file_path: Absolute path to the plan file.
            correlation_id: Workflow correlation ID.
            output_path: Caller-specified output path to carry forward.

        Returns:
            ModelScopeFileReadResult with file content or error.
        """
        resolved = Path(file_path).expanduser().resolve()
        logger.info(
            "Reading plan file: %s (correlation_id=%s)",
            resolved,
            correlation_id,
        )

        if not resolved.is_file():
            return ModelScopeFileReadResult(
                correlation_id=correlation_id,
                file_path=str(resolved),
                output_path=output_path,
                content="",
                success=False,
                error_message=f"File not found: {resolved}",
            )

        try:
            content = resolved.read_text(encoding="utf-8")
        except OSError as e:
            return ModelScopeFileReadResult(
                correlation_id=correlation_id,
                file_path=str(resolved),
                output_path=output_path,
                content="",
                success=False,
                error_message=f"Read error: {e}",
            )

        return ModelScopeFileReadResult(
            correlation_id=correlation_id,
            file_path=str(resolved),
            output_path=output_path,
            content=content,
            success=True,
        )
