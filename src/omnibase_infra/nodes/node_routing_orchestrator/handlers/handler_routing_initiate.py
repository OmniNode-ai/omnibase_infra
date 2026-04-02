# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler for initiating the routing workflow — dispatches health probe."""

from __future__ import annotations

import logging

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_model_health_effect.models.model_health_request import (
    ModelHealthRequest,
)
from omnibase_infra.nodes.node_routing_orchestrator.models.model_routing_command import (
    ModelRoutingCommand,
)

logger = logging.getLogger(__name__)


class HandlerRoutingInitiate:
    """Receives routing command, emits health probe request."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.COMPUTE

    def initiate_routing(self, command: ModelRoutingCommand) -> ModelHealthRequest:
        """Transform routing command into a health probe request.

        The orchestrator will dispatch this to node_model_health_effect.
        Targets are populated by the orchestrator from the model registry.
        """
        logger.info(
            "Routing workflow initiated for task_type=%s (correlation_id=%s)",
            command.task_type.value,
            command.correlation_id,
        )

        # Return a health request with empty targets — the orchestrator
        # populates targets from the registry before dispatching.
        return ModelHealthRequest(
            correlation_id=command.correlation_id,
            targets=(),
        )
