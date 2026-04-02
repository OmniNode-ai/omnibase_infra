# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler that initiates the merge-sweep workflow by dispatching PR scan."""

from __future__ import annotations

import logging
from typing import Literal
from uuid import UUID

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_merge_sweep_pr_list_effect.models.model_pr_list_request import (
    ModelPRListRequest,
)

logger = logging.getLogger(__name__)


class HandlerMergeSweepInitiate:
    """Receives merge-sweep command and emits PR list request."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.COMPUTE

    async def handle(
        self,
        correlation_id: UUID,
        repos: tuple[str, ...],
        authors: tuple[str, ...] = (),
        labels: tuple[str, ...] = (),
        since: str = "",
        merge_method: Literal["squash", "merge", "rebase"] = "squash",
        require_approval: bool = True,
        dry_run: bool = False,
    ) -> ModelPRListRequest:
        """Transform merge-sweep command into a PR list request.

        Args:
            correlation_id: Workflow correlation ID.
            repos: Repos to scan.
            authors: Author filter.
            labels: Label filter.
            since: Date filter.
            merge_method: Merge strategy (passed through context).
            require_approval: Approval requirement (passed through context).
            dry_run: Dry-run flag (passed through context).

        Returns:
            ModelPRListRequest for the PR list effect node.
        """
        logger.info(
            "Initiating merge-sweep: %d repos (correlation_id=%s)",
            len(repos),
            correlation_id,
        )

        return ModelPRListRequest(
            correlation_id=correlation_id,
            repos=repos,
            authors=authors,
            labels=labels,
            since=since,
        )
