# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler that classifies PRs into Track A (merge-ready) vs Track B (needs polish).

This is a COMPUTE handler - pure transformation, no I/O.
"""

from __future__ import annotations

import logging
from uuid import UUID

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_merge_sweep_classify_compute.models.model_classify_result import (
    ModelClassifyResult,
)
from omnibase_infra.nodes.node_merge_sweep_classify_compute.models.model_pr_classification import (
    ModelPRClassification,
)
from omnibase_infra.nodes.node_merge_sweep_pr_list_effect.models.model_pr_info import (
    ModelPRInfo,
)

logger = logging.getLogger(__name__)


def _classify_single(pr: ModelPRInfo, require_approval: bool) -> ModelPRClassification:
    """Classify a single PR into Track A, B, or SKIP."""
    # Skip drafts
    if pr.is_draft:
        return ModelPRClassification(pr=pr, track="SKIP", reason="Draft PR")

    # Skip PRs that already have auto-merge enabled
    if pr.has_auto_merge:
        return ModelPRClassification(
            pr=pr, track="SKIP", reason="Auto-merge already enabled"
        )

    # Track A: merge-ready (CI green, approved, no conflicts)
    ci_green = pr.ci_status == "SUCCESS"
    approved = pr.review_decision == "APPROVED" or not require_approval
    mergeable = pr.mergeable == "MERGEABLE"

    if ci_green and approved and mergeable:
        return ModelPRClassification(
            pr=pr, track="A", reason="CI green, approved, mergeable"
        )

    # Track B: needs polish (has fixable issues)
    reasons: list[str] = []
    if not ci_green:
        reasons.append(f"CI {pr.ci_status}")
    if not approved:
        reasons.append(f"review {pr.review_decision or 'NONE'}")
    if not mergeable:
        reasons.append(f"mergeable={pr.mergeable}")

    return ModelPRClassification(pr=pr, track="B", reason=", ".join(reasons))


class HandlerClassifyPRs:
    """Classifies PRs into merge-ready (Track A) vs needs-polish (Track B)."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.COMPUTE

    async def handle(
        self,
        prs: tuple[ModelPRInfo, ...],
        correlation_id: UUID,
        require_approval: bool = True,
    ) -> ModelClassifyResult:
        """Classify all PRs into Track A / Track B / SKIP.

        Args:
            prs: PRs to classify.
            correlation_id: Workflow correlation ID.
            require_approval: Whether review approval is required for Track A.

        Returns:
            ModelClassifyResult with classified PRs.
        """
        logger.info(
            "Classifying %d PRs (require_approval=%s, correlation_id=%s)",
            len(prs),
            require_approval,
            correlation_id,
        )

        track_a: list[ModelPRClassification] = []
        track_b: list[ModelPRClassification] = []
        skipped: list[ModelPRClassification] = []

        for pr in prs:
            classification = _classify_single(pr, require_approval)
            if classification.track == "A":
                track_a.append(classification)
            elif classification.track == "B":
                track_b.append(classification)
            else:
                skipped.append(classification)

        logger.info(
            "Classification: %d Track A, %d Track B, %d SKIP",
            len(track_a),
            len(track_b),
            len(skipped),
        )

        return ModelClassifyResult(
            correlation_id=correlation_id,
            track_a=tuple(track_a),
            track_b=tuple(track_b),
            skipped=tuple(skipped),
            total_classified=len(prs),
        )
