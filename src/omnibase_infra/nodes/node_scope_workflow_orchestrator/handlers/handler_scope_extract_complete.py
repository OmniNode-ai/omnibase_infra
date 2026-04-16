# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler for scope extraction completion.

Receives ModelScopeExtracted and emits a ModelScopeManifestWriteRequest
to trigger the manifest write effect node.

OMN-8735: No-arg constructor required for auto-wiring compliance.
"""

from __future__ import annotations

import logging

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_scope_extract_compute.models.model_scope_extracted import (
    ModelScopeExtracted,
)
from omnibase_infra.nodes.node_scope_manifest_write_effect.models.model_scope_manifest_write_request import (
    ModelScopeManifestWriteRequest,
)

logger = logging.getLogger(__name__)

_DEFAULT_OUTPUT_PATH = "~/.claude/scope-manifest.json"


class HandlerScopeExtractComplete:
    """Processes scope extraction result and prepares manifest write request."""

    def __init__(self) -> None:  # stub-ok: stateless init
        """Initialize the handler (stateless)."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def handle(
        self,
        extracted: ModelScopeExtracted,
    ) -> ModelScopeManifestWriteRequest:
        """Translate scope extraction result to manifest write request."""
        logger.info(
            "Scope extraction complete for plan_file_path=%s "
            "(files=%d, dirs=%d, repos=%d)",
            extracted.plan_file_path,
            len(extracted.files),
            len(extracted.directories),
            len(extracted.repos),
        )
        return ModelScopeManifestWriteRequest(
            correlation_id=extracted.correlation_id,
            output_path=extracted.output_path,
            plan_file_path=extracted.plan_file_path,
            files=extracted.files,
            directories=extracted.directories,
            repos=extracted.repos,
            systems=extracted.systems,
            adjacent_files=extracted.adjacent_files,
        )


__all__ = ["HandlerScopeExtractComplete"]
