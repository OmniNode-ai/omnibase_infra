# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Pure bounded coordinator recovery decision (OMN-21337)."""

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.enums.enum_consumer_coordinator_rejoin_reason import (
    EnumConsumerCoordinatorRejoinReason,
)
from omnibase_infra.models.health.model_consumer_coordinator_rejoin_decision import (
    ModelConsumerCoordinatorRejoinDecision,
)
from omnibase_infra.models.health.model_consumer_coordinator_rejoin_request import (
    ModelConsumerCoordinatorRejoinRequest,
)


class HandlerConsumerCoordinatorRejoin:
    """Decide from measurements only, without I/O or a clock."""

    @property
    def handler_type(self) -> EnumHandlerType:
        """Return the pure compute role."""
        return EnumHandlerType.COMPUTE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Return the deterministic compute category."""
        return EnumHandlerTypeCategory.COMPUTE

    def handle(
        self, request: ModelConsumerCoordinatorRejoinRequest
    ) -> ModelConsumerCoordinatorRejoinDecision:
        """Apply availability bounds and alarm independently of the cooldown."""
        reasons = EnumConsumerCoordinatorRejoinReason
        if request.coordinator_known:
            reason = reasons.COORDINATOR_KNOWN
        elif not request.broker_reachable:
            reason = reasons.BROKER_UNREACHABLE
        elif (
            request.coordinator_unavailable_seconds < request.unavailable_bound_seconds
        ):
            reason = reasons.WITHIN_BOUND
        elif (
            request.seconds_since_last_rejoin is not None
            and request.seconds_since_last_rejoin < request.rejoin_cooldown_seconds
        ):
            reason = reasons.WITHIN_COOLDOWN
        else:
            reason = reasons.COORDINATOR_UNAVAILABLE_PAST_BOUND
        alarm = (
            request.failed_rejoins >= 1
            and not request.alarm_raised
            and reason is not reasons.COORDINATOR_KNOWN
        )
        message = ""
        if alarm and reason is reasons.BROKER_UNREACHABLE:
            # The leaders did not answer: name reachability, not the coordinator.
            message = (
                f"consumer group {request.consumer_group} cannot rejoin: broker {request.broker} "
                f"is unreachable after {request.failed_rejoins} rejoin attempt(s) "
                f"(topic {request.topic})"
            )
        elif alarm:
            message = (
                f"consumer group {request.consumer_group} cannot rejoin: broker {request.broker} "
                f"has answered for {request.coordinator_unavailable_seconds:g}s but the group "
                f"coordinator is unavailable after {request.failed_rejoins} rejoin attempt(s) "
                f"(topic {request.topic})"
            )
        return ModelConsumerCoordinatorRejoinDecision(
            reason=reason,
            rejoin=reason is reasons.COORDINATOR_UNAVAILABLE_PAST_BOUND,
            alarm=alarm,
            alarm_message=message,
        )


__all__ = ["HandlerConsumerCoordinatorRejoin"]
