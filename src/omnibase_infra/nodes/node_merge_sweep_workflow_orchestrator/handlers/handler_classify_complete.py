# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler that receives classification results and dispatches auto-merge."""

from __future__ import annotations

import logging
from typing import Literal
from uuid import UUID

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_merge_sweep_auto_merge_effect.models.model_auto_merge_request import (
    ModelAutoMergeRequest,
)
from omnibase_infra.nodes.node_merge_sweep_classify_compute.models.model_pr_classification import (
    ModelPRClassification,
)

logger = logging.getLogger(__name__)


class HandlerClassifyComplete:
    """Receives classification result and emits auto-merge request."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.COMPUTE

    async def handle(
        self,
        correlation_id: UUID,
        track_a: tuple[ModelPRClassification, ...],
        merge_method: Literal["squash", "merge", "rebase"] = "squash",
        dry_run: bool = False,
    ) -> ModelAutoMergeRequest:
        """Transform classification result into auto-merge request.

        Args:
            correlation_id: Workflow correlation ID.
            track_a: Track A (merge-ready) PRs.
            merge_method: Merge strategy.
            dry_run: Whether to simulate.

        Returns:
            ModelAutoMergeRequest for the auto-merge effect node.
        """
        pr_tuples = tuple((c.pr.repo, c.pr.number) for c in track_a)

        logger.info(
            "Classification complete: %d Track A PRs for auto-merge (correlation_id=%s)",
            len(pr_tuples),
            correlation_id,
        )

        return ModelAutoMergeRequest(
            correlation_id=correlation_id,
            prs=pr_tuples,
            merge_method=merge_method,
            dry_run=dry_run,
        )
