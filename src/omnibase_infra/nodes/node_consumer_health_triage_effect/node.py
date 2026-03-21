# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Node Consumer Health Triage Effect — graduated response for consumer health events.

Consumes onex.evt.omnibase-infra.consumer-health.v1 events and applies a
graduated triage response:
    1st occurrence  -> Slack WARNING
    2nd occurrence  -> Slack REPEATED
    3rd in 30 min   -> Restart command (gated by ENABLE_CONSUMER_AUTO_RESTART)
    Restart failure -> Linear ticket

Gated by ENABLE_CONSUMER_HEALTH_TRIAGE (default off).

Architecture:
    onex.evt.omnibase-infra.consumer-health.v1 (Kafka)
        -> NodeConsumerHealthTriageEffect (this declarative shell)
        -> HandlerConsumerHealthTriage
        -> PostgreSQL consumer_health_triage + consumer_restart_state tables
        -> Slack webhook + Linear API (escalation)
        -> onex.cmd.omnibase-infra.consumer-restart.v1 (restart commands)

Related Tickets:
    - OMN-5520: Create NodeConsumerHealthTriageEffect
    - OMN-5529: Runtime Health Event Pipeline (epic)
"""

from __future__ import annotations

from omnibase_core.models.container import ModelONEXContainer
from omnibase_core.nodes.node_effect import NodeEffect


class NodeConsumerHealthTriageEffect(NodeEffect):
    """Declarative effect node for consumer health triage with graduated response.

    This effect node is a lightweight shell that defines the I/O contract
    for consumer health triage operations. All routing and execution
    logic is driven by contract.yaml — this class contains NO custom
    routing code.

    Supported Operations (defined in contract.yaml handler_routing):
        - triage_health_event: Apply graduated triage to a consumer health event

    Dependency Injection:
        Callers must instantiate HandlerConsumerHealthTriage with
        db_pool, slack_handler, and optional linear_handler, then pass
        via the container. This node does not retain handler instances
        as attributes.

    Example:
        ```python
        from omnibase_core.models.container import ModelONEXContainer
        from omnibase_infra.nodes.node_consumer_health_triage_effect import (
            NodeConsumerHealthTriageEffect,
        )
        from omnibase_infra.nodes.node_consumer_health_triage_effect.handlers import (
            HandlerConsumerHealthTriage,
        )

        container = ModelONEXContainer()
        effect = NodeConsumerHealthTriageEffect(container)

        handler = HandlerConsumerHealthTriage(
            db_pool=pool,
            event_bus=event_bus,
        )
        result = await handler.handle(health_event)
        ```
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        super().__init__(container)

    # Pure declarative shell — all behaviour defined in contract.yaml


__all__ = ["NodeConsumerHealthTriageEffect"]
