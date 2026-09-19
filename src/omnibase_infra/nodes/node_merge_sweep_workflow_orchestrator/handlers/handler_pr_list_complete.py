# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler that receives PR list results and dispatches classification."""

from __future__ import annotations

import logging
from uuid import UUID

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_merge_sweep_classify_compute.models.model_classify_input import (
    ModelClassifyInput,
)
from omnibase_infra.nodes.node_merge_sweep_pr_list_effect.models.model_pr_info import (
    ModelPRInfo,
)

logger = logging.getLogger(__name__)


class HandlerPRListComplete:
    """Receives PR list result and emits classification input."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.COMPUTE

    async def handle(
        self,
        correlation_id: UUID,
        prs: tuple[ModelPRInfo, ...],
        collaborator_logins: tuple[str, ...],
        require_approval: bool = True,
    ) -> ModelClassifyInput:
        """Transform PR list result into classification input.

        Args:
            correlation_id: Workflow correlation ID.
            prs: PRs discovered by the scan.
            collaborator_logins: Accounts whose PRs the sweep must not admit,
                resolved by the caller from the roster the classify node's
                contract names. Positional and required: a caller that never
                resolved a roster must not be able to omit it and get an
                empty one (OMN-18823).
            require_approval: Whether to require review approval.

        Returns:
            ModelClassifyInput for the classify compute node.
        """
        logger.info(
            "PR scan complete: %d PRs to classify against %d excluded "
            "account(s) (correlation_id=%s)",
            len(prs),
            len(collaborator_logins),
            correlation_id,
        )

        return ModelClassifyInput(
            correlation_id=correlation_id,
            prs=prs,
            require_approval=require_approval,
            collaborator_logins=collaborator_logins,
        )
