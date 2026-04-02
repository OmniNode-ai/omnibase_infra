# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler for health probe completion — dispatches scoring."""

from __future__ import annotations

import logging

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_model_health_effect.models.model_health_snapshot import (
    ModelHealthSnapshot,
)
from omnibase_infra.nodes.node_model_router_compute.models.enum_task_type import (
    EnumTaskType,
)
from omnibase_infra.nodes.node_model_router_compute.models.model_routing_constraints import (
    ModelRoutingConstraints,
)
from omnibase_infra.nodes.node_model_router_compute.models.model_scoring_input import (
    ModelScoringInput,
)

logger = logging.getLogger(__name__)


class HandlerHealthComplete:
    """Receives health snapshot, emits scoring input for compute node."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.COMPUTE

    def build_scoring_input(
        self,
        health: ModelHealthSnapshot,
        *,
        task_type: EnumTaskType,
        task_description: str,
        constraints: ModelRoutingConstraints,
        context_length_estimate: int,
        chain_hit: bool,
        chain_hit_model_key: str | None,
    ) -> ModelScoringInput:
        """Build scoring input from health snapshot + workflow context.

        The orchestrator enriches with registry and live_metrics before
        dispatching to node_model_router_compute.
        """
        logger.info(
            "Health probe complete, building scoring input (correlation_id=%s)",
            health.correlation_id,
        )

        return ModelScoringInput(
            correlation_id=health.correlation_id,
            task_type=task_type,
            task_description=task_description,
            constraints=constraints,
            context_length_estimate=context_length_estimate,
            chain_hit=chain_hit,
            chain_hit_model_key=chain_hit_model_key,
            registry=(),  # Populated by orchestrator from registry
            health=health.endpoints,
            live_metrics=(),  # Populated by orchestrator from reducer state
        )
