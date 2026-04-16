# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler for scope manifest write completion.

Receives ModelScopeManifestWritten and emits a ModelScopeCheckResult
signaling the end of the scope-check workflow.

OMN-8735: No-arg constructor required for auto-wiring compliance.
"""

from __future__ import annotations

import logging

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_scope_manifest_write_effect.models.model_scope_manifest_written import (
    ModelScopeManifestWritten,
)
from omnibase_infra.nodes.node_scope_workflow_orchestrator.models.enum_scope_check_status import (
    EnumScopeCheckStatus,
)
from omnibase_infra.nodes.node_scope_workflow_orchestrator.models.model_scope_check_result import (
    ModelScopeCheckResult,
)

logger = logging.getLogger(__name__)


class HandlerScopeManifestWriteComplete:
    """Processes manifest write result and emits final scope-check result."""

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
        written: ModelScopeManifestWritten,
    ) -> ModelScopeCheckResult:
        """Translate manifest write result to final scope-check result."""
        status = (
            EnumScopeCheckStatus.COMPLETE
            if written.success
            else EnumScopeCheckStatus.FAILED
        )
        logger.info(
            "Scope manifest write complete: manifest_path=%s status=%s",
            written.manifest_path,
            status,
        )
        return ModelScopeCheckResult(
            correlation_id=written.correlation_id,
            manifest_path=written.manifest_path,
            status=status,
        )


__all__ = ["HandlerScopeManifestWriteComplete"]
