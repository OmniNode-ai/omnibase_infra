# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Node Runtime Error Triage Effect — first-match-wins triage for runtime errors.

Consumes onex.evt.omnibase-infra.runtime-error.v1 events and applies a
first-match-wins triage rule engine with cross-layer correlation to
Layer 1 consumer health incidents.

Default rules cover aiokafka, asyncpg, and aiohttp error patterns.

Architecture:
    onex.evt.omnibase-infra.runtime-error.v1 (Kafka)
        -> NodeRuntimeErrorTriageEffect (this declarative shell)
        -> HandlerRuntimeErrorTriage
        -> PostgreSQL runtime_error_triage table
        -> Cross-layer correlation with consumer_health_triage
        -> Slack webhook + Linear API (escalation)

Related Tickets:
    - OMN-5522: Create NodeRuntimeErrorTriageEffect
    - OMN-5529: Runtime Health Event Pipeline (epic)
"""

from __future__ import annotations

from omnibase_core.models.container import ModelONEXContainer
from omnibase_core.nodes.node_effect import NodeEffect


class NodeRuntimeErrorTriageEffect(NodeEffect):
    """Declarative effect node for runtime error triage.

    This effect node is a lightweight shell that defines the I/O contract
    for runtime error triage operations. All routing and execution logic
    is driven by contract.yaml — this class contains NO custom routing code.

    Supported Operations (defined in contract.yaml handler_routing):
        - triage_runtime_error: Apply first-match-wins triage with cross-layer correlation

    Example:
        ```python
        from omnibase_core.models.container import ModelONEXContainer
        from omnibase_infra.nodes.node_runtime_error_triage_effect import (
            NodeRuntimeErrorTriageEffect,
        )
        from omnibase_infra.nodes.node_runtime_error_triage_effect.handlers import (
            HandlerRuntimeErrorTriage,
        )

        container = ModelONEXContainer()
        effect = NodeRuntimeErrorTriageEffect(container)

        handler = HandlerRuntimeErrorTriage(db_pool=pool)
        result = await handler.handle(runtime_error_event)
        ```
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        super().__init__(container)

    # Pure declarative shell — all behaviour defined in contract.yaml


__all__ = ["NodeRuntimeErrorTriageEffect"]
