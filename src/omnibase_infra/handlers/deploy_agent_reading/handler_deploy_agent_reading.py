# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Judge the dev deploy agent ticket's three live readings (OMN-26724).

AC1: the dev agent loaded code containing the merge and was restarted after it (uptime is positive
and shorter than the merge's age). AC2: the dev agent's last result is a success for the dev lane
with every phase recorded. AC3: the stability and prod agents ran longer than the dev agent, so
neither was restarted at or after the dev restart.

Pure: ``handle(request) -> verdict``, no clock, no I/O. The reader gathers the agents' health
answers and passes them; this decides, and an unanswered health is never a pass.
"""

from __future__ import annotations

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.model_deploy_agent_reading_request import (
    ModelDeployAgentReadingRequest,
)
from omnibase_infra.models.model_deploy_agent_reading_verdict import (
    ModelDeployAgentReadingVerdict,
)


class HandlerDeployAgentReading:
    """Judge the three readings of the dev deploy agent ticket (contract-bound, pure)."""

    @property
    def handler_type(self) -> EnumHandlerType:
        """Return the pure compute role."""
        return EnumHandlerType.COMPUTE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Return the deterministic compute category."""
        return EnumHandlerTypeCategory.COMPUTE

    def handle(
        self, request: ModelDeployAgentReadingRequest
    ) -> ModelDeployAgentReadingVerdict:
        reasons: list[str] = []
        dev = request.dev_uptime_seconds

        ac1 = (
            bool(request.dev_loaded_code_sha)
            and dev is not None
            and 0 < dev < request.merge_age_seconds
            and request.dev_loaded_sha_has_merge
        )
        if not ac1:
            reasons.append(
                "AC1: the dev agent has no loaded sha, no positive uptime, a loaded sha "
                "without the merge, or an uptime older than the merge"
            )

        ac2 = (
            request.dev_last_status == "success"
            and request.dev_last_runtime_lane == "dev"
            and request.dev_last_phase_result_count > 0
        )
        if not ac2:
            reasons.append(
                "AC2: the dev agent's last result is not a success for the dev lane "
                "with phase results"
            )

        others = (request.stability_uptime_seconds, request.prod_uptime_seconds)
        ac3 = dev is not None and all(up is not None and up > dev for up in others)
        if not ac3:
            reasons.append(
                "AC3: the stability or prod agent did not answer or ran no longer "
                "than the dev agent"
            )

        return ModelDeployAgentReadingVerdict(
            ac1_met=ac1, ac2_met=ac2, ac3_met=ac3, reasons=tuple(reasons)
        )
