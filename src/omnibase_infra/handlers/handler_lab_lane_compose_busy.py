# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Wait-while-busy deploy test for a lab lane that answers (OMN-21866).

Bound from ``node_lab_lane_compose_busy_compute``'s contract. The
resolve-lab-lane action asks it on its wait-while-busy path only (OMN-20410);
the no-answer wait keeps the action's shared busy test and so still waits out
every settle (OMN-20509).
"""

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.health.model_lab_lane_compose_busy_decision import (
    ModelLabLaneComposeBusyDecision,
)
from omnibase_infra.models.health.model_lab_lane_compose_busy_request import (
    ModelLabLaneComposeBusyRequest,
)


class HandlerLabLaneComposeBusy:
    """Decide from one agent reading, without I/O or a clock."""

    @property
    def handler_type(self) -> EnumHandlerType:
        """Return the pure compute role."""
        return EnumHandlerType.COMPUTE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Return the deterministic compute category."""
        return EnumHandlerTypeCategory.COMPUTE

    def handle(
        self, request: ModelLabLaneComposeBusyRequest
    ) -> ModelLabLaneComposeBusyDecision:
        """A settle of a declared non-deploy stage with no active job is not busy."""
        if (
            request.state == "settling"
            and not request.active_job
            and request.settling_stage in request.policy.non_deploy_settling_stages
        ):
            return ModelLabLaneComposeBusyDecision()
        return ModelLabLaneComposeBusyDecision(busy_reason=request.shared_busy_reason)


__all__ = ["HandlerLabLaneComposeBusy"]
