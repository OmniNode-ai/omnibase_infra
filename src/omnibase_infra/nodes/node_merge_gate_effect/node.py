# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""NodeMergeGateEffect -- declarative effect node for merge gate decision persistence.

This node follows the ONEX declarative pattern:
    - DECLARATIVE effect driven by contract.yaml
    - Zero custom routing logic -- all behavior from handler_routing
    - Lightweight shell that delegates to handlers via container resolution
    - Used for ONEX-compliant runtime execution via RuntimeHostProcess
    - Pattern: "Contract-driven, handlers wired externally"

Extends NodeEffect from omnibase_core for infrastructure I/O operations.
All handler routing is 100% driven by contract.yaml, not Python code.

Architecture:
    onex.evt.platform.merge-gate-decision.v1 (Kafka)
        -> NodeMergeGateEffect (this declarative shell)
        -> HandlerUpsertMergeGate
        -> PostgreSQL merge_gate_decisions table (upsert)
        -> Linear GraphQL API (quarantine tickets)

Handler Routing Pattern:
    1. Receive merge gate decision event (ModelMergeGateResult)
    2. Route to HandlerUpsertMergeGate based on "upsert_merge_gate" operation
    3. Upsert into merge_gate_decisions ON CONFLICT (pr_ref, head_sha) DO UPDATE
    4. If QUARANTINE: open Linear ticket with violation details
    5. Return structured ModelBackendResult

Kafka Subscription Wiring:
    The kernel wires Kafka subscriptions via EventBusSubcontractWiring by
    reading ``event_bus.subscribe_topics`` from contract.yaml. Any
    ProtocolDomainPlugin that calls ``load_event_bus_subcontract`` on this
    contract will subscribe to ``onex.evt.platform.merge-gate-decision.v1``
    and route messages through the MessageDispatchEngine to this node's
    handler.

Design Decisions:
    - 100% Contract-Driven: All routing logic in YAML, not Python
    - Zero Custom Routing: Base class handles handler dispatch via contract
    - Idempotent: UNIQUE(pr_ref, head_sha) with ON CONFLICT DO UPDATE
    - Side-effect only: emits no outbound events

Related Tickets:
    - OMN-3140: NodeMergeGateEffect + migration
"""

from __future__ import annotations

from omnibase_core.models.container import ModelONEXContainer
from omnibase_core.nodes.node_effect import NodeEffect


class NodeMergeGateEffect(NodeEffect):
    """Declarative effect node for merge gate decision persistence.

    This effect node is a lightweight shell that defines the I/O contract
    for merge gate decision persistence. All routing and execution logic
    is driven by contract.yaml -- this class contains NO custom routing code.

    Supported Operations (defined in contract.yaml handler_routing):
        - upsert_merge_gate: Upsert decision + optional QUARANTINE Linear ticket

    Dependency Injection:
        Callers must instantiate HandlerUpsertMergeGate with a db_pool and
        pass it via the container. This node does not retain a handler
        instance as an attribute.

    Example:
        ```python
        from omnibase_core.models.container import ModelONEXContainer
        from omnibase_infra.nodes.node_merge_gate_effect import (
            NodeMergeGateEffect,
        )

        container = ModelONEXContainer()
        effect = NodeMergeGateEffect(container)
        ```
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        super().__init__(container)

    # Pure declarative shell -- all behaviour defined in contract.yaml


__all__ = ["NodeMergeGateEffect"]
