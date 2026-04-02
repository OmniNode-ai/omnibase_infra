# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler for scoring completion — emits final routing result."""

from __future__ import annotations

import logging

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_model_router_compute.models.model_routing_decision import (
    ModelRoutingDecision,
)
from omnibase_infra.nodes.node_routing_orchestrator.models.model_routing_result import (
    ModelRoutingResult,
)

logger = logging.getLogger(__name__)


class HandlerScoringComplete:
    """Receives scoring decision, emits final routing result."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.COMPUTE

    def finalize_routing(self, decision: ModelRoutingDecision) -> ModelRoutingResult:
        """Transform routing decision into final result for caller."""
        logger.info(
            "Routing complete: %s (correlation_id=%s)",
            decision.selected_model_key,
            decision.correlation_id,
        )

        return ModelRoutingResult(
            correlation_id=decision.correlation_id,
            selected_model_key=decision.selected_model_key,
            selected_endpoint_env=decision.selected_endpoint_env,
            fallback_model_key=decision.fallback_model_key,
            rationale=decision.rationale,
            estimated_cost=decision.estimated_cost,
            estimated_latency_ms=decision.estimated_latency_ms,
            success=decision.success,
            error_message=decision.error_message,
        )
