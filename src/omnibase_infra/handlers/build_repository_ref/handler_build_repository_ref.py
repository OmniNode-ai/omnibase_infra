# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Whether a queued build may start, from what the build repository holds (OMN-26724).

On 2026-11-26 overdue dev builds published at 18:16Z by the old agent process carried a sibling
repository's sha as ``git_ref``. Nothing checked the clone held the commit, so each window ended
in "Could not parse object" and one stale record was spent per window while 19 commands waited.

Pure: ``handle(request) -> decision``, no clock, no I/O. The caller reads the clone (a cat-file,
and one fetch on a miss) and passes what it read; this decides.
"""

from __future__ import annotations

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.enums.enum_build_repository_ref_verdict import (
    EnumBuildRepositoryRefVerdict,
)
from omnibase_infra.models.model_build_repository_ref_decision import (
    ModelBuildRepositoryRefDecision,
)
from omnibase_infra.models.model_build_repository_ref_request import (
    ModelBuildRepositoryRefRequest,
)


class HandlerBuildRepositoryRef:
    """Decide whether the build repository holds a request's commit (contract-bound, pure)."""

    @property
    def handler_type(self) -> EnumHandlerType:
        """Return the pure compute role."""
        return EnumHandlerType.COMPUTE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Return the deterministic compute category."""
        return EnumHandlerTypeCategory.COMPUTE

    def handle(
        self, request: ModelBuildRepositoryRefRequest
    ) -> ModelBuildRepositoryRefDecision:
        verdict = EnumBuildRepositoryRefVerdict
        if request.held:
            return ModelBuildRepositoryRefDecision(verdict=verdict.HELD)
        if request.fetch_exit_code != 0:
            return ModelBuildRepositoryRefDecision(
                verdict=verdict.UNVERIFIABLE,
                reason=f"the fetch for {request.git_ref} did not answer, so its absence is unproven",
            )
        if request.held_after_fetch:
            return ModelBuildRepositoryRefDecision(verdict=verdict.HELD_AFTER_FETCH)
        return ModelBuildRepositoryRefDecision(
            verdict=verdict.MISSING,
            reason=(
                f"build repository does not hold {request.git_ref}, before or after one "
                "fetch; refused so the next queued command runs"
            ),
        )
