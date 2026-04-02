# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler that receives auto-merge results and emits workflow completion."""

from __future__ import annotations

import logging
from uuid import UUID

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_merge_sweep_auto_merge_effect.models.model_auto_merge_outcome import (
    ModelAutoMergeOutcome,
)
from omnibase_infra.nodes.node_merge_sweep_workflow_orchestrator.models.model_merge_sweep_result import (
    ModelMergeSweepResult,
)

logger = logging.getLogger(__name__)


class HandlerAutoMergeComplete:
    """Receives auto-merge result and emits final merge-sweep result."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.COMPUTE

    async def handle(
        self,
        correlation_id: UUID,
        outcomes: tuple[ModelAutoMergeOutcome, ...],
        total_prs_scanned: int = 0,
        track_a_count: int = 0,
        track_b_count: int = 0,
        skipped_count: int = 0,
        repos_scanned: tuple[str, ...] = (),
        repos_failed: tuple[str, ...] = (),
        dry_run: bool = False,
    ) -> ModelMergeSweepResult:
        """Build final merge-sweep result from auto-merge outcomes.

        Args:
            correlation_id: Workflow correlation ID.
            outcomes: Per-PR auto-merge outcomes.
            total_prs_scanned: Total PRs found in scan.
            track_a_count: Track A PRs.
            track_b_count: Track B PRs.
            skipped_count: Skipped PRs.
            repos_scanned: Repos successfully scanned.
            repos_failed: Repos that failed to scan.
            dry_run: Whether this was a dry run.

        Returns:
            ModelMergeSweepResult with final summary.
        """
        enabled = sum(1 for o in outcomes if o.success and not o.dry_run)
        failed = sum(1 for o in outcomes if not o.success)

        if total_prs_scanned == 0:
            status = "nothing_to_merge"
        elif failed > 0:
            status = "partial"
        else:
            status = "complete"

        logger.info(
            "Merge-sweep complete: %d enabled, %d failed (correlation_id=%s)",
            enabled,
            failed,
            correlation_id,
        )

        return ModelMergeSweepResult(
            correlation_id=correlation_id,
            status=status,
            total_prs_scanned=total_prs_scanned,
            track_a_count=track_a_count,
            track_b_count=track_b_count,
            skipped_count=skipped_count,
            auto_merge_enabled=enabled,
            auto_merge_failed=failed,
            repos_scanned=repos_scanned,
            repos_failed=repos_failed,
            dry_run=dry_run,
        )
