# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Node Db Error Linear Effect - Declarative effect node for Linear ticket creation.

This node follows the ONEX declarative pattern:
    - DECLARATIVE effect driven by contract.yaml
    - Zero custom routing logic — all behaviour from handler_routing
    - Lightweight shell that delegates to handlers via container resolution
    - Used for ONEX-compliant runtime execution via RuntimeHostProcess
    - Pattern: "Contract-driven, handlers wired externally"

Extends NodeEffect from omnibase_core for infrastructure I/O operations.
All handler routing is 100% driven by contract.yaml, not Python code.

Architecture:
    onex.evt.omnibase-infra.db-error.v1 (Kafka)
        -> NodeDbErrorLinearEffect  (this declarative shell)
        -> HandlerLinearDbErrorReporter
        -> PostgreSQL db_error_tickets table  (dedup + frequency)
        -> Linear GraphQL API -> new issue

Handler Routing Pattern:
    1. Receive db error event (ModelDbErrorEvent, from contract input_model)
    2. Route to HandlerLinearDbErrorReporter based on "report_error" operation
    3. Execute dedup check + Linear API call + DB insert via handler
    4. Return structured response (ModelDbErrorTicketResult)

Kafka Subscription Wiring:
    The kernel wires Kafka subscriptions via EventBusSubcontractWiring by
    reading ``event_bus.subscribe_topics`` from contract.yaml.  Any
    ProtocolDomainPlugin that calls ``load_event_bus_subcontract`` on this
    contract will subscribe to ``onex.evt.omnibase-infra.db-error.v1``
    and route messages through the MessageDispatchEngine to this node's
    handler.

    The topic constant is::

        TOPIC_DB_ERROR_V1 = "onex.evt.omnibase-infra.db-error.v1"

    (defined in ``omnibase_infra.topics.platform_topic_suffixes``)

Design Decisions:
    - 100% Contract-Driven: All routing logic in YAML, not Python
    - Zero Custom Routing: Base class handles handler dispatch via contract
    - Declarative Handlers: handler_routing section defines dispatch rules
    - Constructor DI: Handler dependencies resolved externally

Node Responsibilities:
    - Define I/O model contract (ModelDbErrorEvent -> ModelDbErrorTicketResult)
    - Delegate all execution to handlers via base class
    - NO custom logic — pure declarative shell

Related Tickets:
    - OMN-3408: Kafka Consumer -> Linear Ticket Reporter (ONEX Node)
    - OMN-3407: PostgreSQL Error Emitter (hard prerequisite)
"""

from __future__ import annotations

from omnibase_core.models.container import ModelONEXContainer
from omnibase_core.nodes.node_effect import NodeEffect


class NodeDbErrorLinearEffect(NodeEffect):
    """Declarative effect node for PostgreSQL error -> Linear ticket reporting.

    This effect node is a lightweight shell that defines the I/O contract
    for database error reporting operations.  All routing and execution
    logic is driven by contract.yaml — this class contains NO custom
    routing code.

    Supported Operations (defined in contract.yaml handler_routing):
        - report_error: Dedup check -> Linear ticket create -> DB insert

    Dependency Injection:
        Callers must instantiate HandlerLinearDbErrorReporter with
        linear_api_key, linear_team_id, and db_pool and pass it via the
        container.  This node does not retain a handler instance as an
        attribute — no handler dependencies are stored as instance variables.

    Example:
        ```python
        from omnibase_core.models.container import ModelONEXContainer
        from omnibase_infra.nodes.node_db_error_linear_effect import (
            NodeDbErrorLinearEffect,
        )
        from omnibase_infra.handlers.handler_linear_db_error_reporter import (
            HandlerLinearDbErrorReporter,
        )
        from omnibase_infra.handlers.models.model_db_error_event import (
            ModelDbErrorEvent,
        )

        container = ModelONEXContainer()
        effect = NodeDbErrorLinearEffect(container)

        handler = HandlerLinearDbErrorReporter(
            linear_api_key=os.environ["LINEAR_API_KEY"],
            linear_team_id=os.environ["LINEAR_TEAM_ID"],
            db_pool=pool,
        )

        event = ModelDbErrorEvent(
            error_code="42883",
            error_message="operator does not exist: character varying = uuid",
            table_name="learned_patterns",
            fingerprint="a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4",
            first_seen_at=datetime.utcnow(),
            service="omnibase-infra-postgres",
        )
        result = await handler.handle(event)
        ```
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        super().__init__(container)

    # Pure declarative shell — all behaviour defined in contract.yaml


__all__ = ["NodeDbErrorLinearEffect"]
