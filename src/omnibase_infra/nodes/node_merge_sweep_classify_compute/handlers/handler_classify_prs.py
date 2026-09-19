# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler that classifies PRs into Track A (merge-ready) vs Track B (needs polish).

This is a COMPUTE handler - pure transformation, no I/O. Canonical definition B:
``handle(request: ModelClassifyInput) -> ModelClassifyResult`` — a single
typed-request entrypoint the shared runtime adapter drives, with no runtime
envelope type in the core.
"""

from __future__ import annotations

import logging

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_merge_sweep_classify_compute.models.enum_classify_skip_reason import (
    EnumClassifySkipReason,
)
from omnibase_infra.nodes.node_merge_sweep_classify_compute.models.model_classify_input import (
    ModelClassifyInput,
)
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


def _matched_collaborator(pr: ModelPRInfo, collaborator_logins: tuple[str, ...]) -> str:
    """Return the first roster account this PR has been handed to, else "".

    Assignment and requested review are both hand-offs: either one means a
    person is expected to act on the PR, and the sweep is not that person.
    Comparison is case-insensitive because GitHub logins are.
    """
    roster = {login.casefold() for login in collaborator_logins}
    if not roster:
        return ""
    for login in (*pr.assignees, *pr.requested_reviewers):
        if login.casefold() in roster:
            return login
    return ""


def _classify_single(
    pr: ModelPRInfo, require_approval: bool, collaborator_logins: tuple[str, ...]
) -> ModelPRClassification:
    """Classify a single PR into Track A, B, or SKIP."""
    # Skip drafts
    if pr.is_draft:
        return ModelPRClassification(
            pr=pr,
            track="SKIP",
            reason="Draft PR",
            skip_reason=EnumClassifySkipReason.DRAFT,
        )

    # Skip PRs that already have auto-merge enabled
    if pr.has_auto_merge:
        return ModelPRClassification(
            pr=pr,
            track="SKIP",
            reason="Auto-merge already enabled",
            skip_reason=EnumClassifySkipReason.AUTO_MERGE_ENABLED,
        )

    # Skip PRs handed to a collaborator (OMN-18823, occurrence OMN-18794).
    # This runs BEFORE any track determination on purpose: Track B is a polish
    # track that mutates the branch, so a collaborator's PR falling through to
    # it would be as wrong as merging it.
    matched = _matched_collaborator(pr, collaborator_logins)
    if matched:
        return ModelPRClassification(
            pr=pr,
            track="SKIP",
            reason=f"Handed to collaborator {matched} (assignee or requested reviewer)",
            skip_reason=EnumClassifySkipReason.COLLABORATOR_EXCLUDED,
            excluded_account=matched,
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

    async def handle(self, request: ModelClassifyInput) -> ModelClassifyResult:
        """Classify all PRs into Track A / Track B / SKIP.

        Args:
            request: Classification input carrying the PRs, correlation id, and
                whether review approval is required for Track A.

        Returns:
            ModelClassifyResult with classified PRs.
        """
        prs = request.prs
        correlation_id = request.correlation_id
        require_approval = request.require_approval
        collaborator_logins = request.collaborator_logins

        logger.info(
            "Classifying %d PRs (require_approval=%s, excluded_accounts=%d, "
            "correlation_id=%s)",
            len(prs),
            require_approval,
            len(collaborator_logins),
            correlation_id,
        )

        track_a: list[ModelPRClassification] = []
        track_b: list[ModelPRClassification] = []
        skipped: list[ModelPRClassification] = []

        for pr in prs:
            classification = _classify_single(pr, require_approval, collaborator_logins)
            if classification.track == "A":
                track_a.append(classification)
            elif classification.track == "B":
                track_b.append(classification)
            else:
                skipped.append(classification)

        excluded = [
            c.excluded_account
            for c in skipped
            if c.skip_reason is EnumClassifySkipReason.COLLABORATOR_EXCLUDED
        ]
        logger.info(
            "Classification: %d Track A, %d Track B, %d SKIP (%d withheld as "
            "handed to a collaborator: %s)",
            len(track_a),
            len(track_b),
            len(skipped),
            len(excluded),
            ", ".join(sorted(set(excluded))) or "none",
        )

        return ModelClassifyResult(
            correlation_id=correlation_id,
            track_a=tuple(track_a),
            track_b=tuple(track_b),
            skipped=tuple(skipped),
            total_classified=len(prs),
        )
