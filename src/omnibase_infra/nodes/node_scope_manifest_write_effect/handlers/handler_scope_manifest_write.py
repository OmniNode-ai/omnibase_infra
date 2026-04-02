# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler that writes a scope manifest JSON file to disk.

This is an EFFECT handler - it performs I/O (filesystem write).
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timezone
from pathlib import Path
from uuid import UUID

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_scope_manifest_write_effect.models.model_scope_manifest_written import (
    ModelScopeManifestWritten,
)

logger = logging.getLogger(__name__)


class HandlerScopeManifestWrite:
    """Writes a scope manifest JSON file to disk."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def handle(
        self,
        output_path: str,
        plan_file_path: str,
        files: tuple[str, ...],
        directories: tuple[str, ...],
        repos: tuple[str, ...],
        systems: tuple[str, ...],
        adjacent_files: tuple[str, ...],
        correlation_id: UUID,
    ) -> ModelScopeManifestWritten:
        """Write scope manifest to a JSON file.

        Args:
            output_path: Path to write the manifest.
            plan_file_path: Source plan file path.
            files: Files in scope.
            directories: Directories in scope.
            repos: Repos in scope.
            systems: Systems in scope.
            adjacent_files: Adjacent files.
            correlation_id: Workflow correlation ID.

        Returns:
            ModelScopeManifestWritten with write confirmation.
        """
        resolved = Path(output_path).expanduser().resolve()
        logger.info(
            "Writing scope manifest to %s (correlation_id=%s)",
            resolved,
            correlation_id,
        )

        manifest = {
            "version": "1.0.0",
            "created_at": datetime.now(tz=UTC).isoformat(),
            "plan_file": plan_file_path,
            "repos": list(repos),
            "files": list(files),
            "directories": list(directories),
            "systems": list(systems),
            "adjacent_files": list(adjacent_files),
        }

        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(
                json.dumps(manifest, indent=2) + "\n",
                encoding="utf-8",
            )
        except OSError as e:
            return ModelScopeManifestWritten(
                correlation_id=correlation_id,
                manifest_path=str(resolved),
                success=False,
                error_message=f"Write error: {e}",
            )

        return ModelScopeManifestWritten(
            correlation_id=correlation_id,
            manifest_path=str(resolved),
            success=True,
        )
