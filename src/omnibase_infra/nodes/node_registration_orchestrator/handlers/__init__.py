# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handlers for NodeRegistrationOrchestrator.

All handlers implement ProtocolMessageHandler with standard signature:
    async def handle(envelope: ModelEventEnvelope[T]) -> ModelHandlerOutput

Properties required by ProtocolMessageHandler:
    - handler_id: str
    - category: EnumMessageCategory
    - message_types: set[str]
    - node_kind: EnumNodeKind

Handlers:
    - HandlerNodeIntrospected: Processes node introspection events
    - HandlerRuntimeTick: Processes runtime tick events for timeout detection
    - HandlerNodeRegistrationAcked: Processes registration acknowledgment commands
    - HandlerNodeHeartbeat: Processes heartbeat events for liveness tracking

Handler Architecture:
    - Handlers are stateless classes (no mutable state between calls)
    - Handlers receive all context via the envelope (event, correlation_id, timestamp)
    - Handlers return ModelHandlerOutput containing events and metadata
    - Time-based decisions use envelope.timestamp or datetime.now(UTC)

Related Tickets:
    - OMN-888 (C1): Registration Orchestrator
    - OMN-932 (C2): Durable Timeout Handling
    - OMN-1006: Node Heartbeat for Liveness Tracking
    - OMN-892: 2-Way Registration E2E Integration Test
    - OMN-1102: Refactor to pure declarative with standard handler signatures
"""

from omnibase_infra.nodes.node_registration_orchestrator.handlers.handler_node_heartbeat import (
    DEFAULT_LIVENESS_WINDOW_SECONDS,
    HandlerNodeHeartbeat,
    ModelHeartbeatHandlerResult,
)
from omnibase_infra.nodes.node_registration_orchestrator.handlers.handler_node_introspected import (
    HandlerNodeIntrospected,
)
from omnibase_infra.nodes.node_registration_orchestrator.handlers.handler_node_registration_acked import (
    DEFAULT_LIVENESS_INTERVAL_SECONDS,
    ENV_LIVENESS_INTERVAL_SECONDS,
    HandlerNodeRegistrationAcked,
    get_liveness_interval_seconds,
)
from omnibase_infra.nodes.node_registration_orchestrator.handlers.handler_runtime_tick import (
    HandlerRuntimeTick,
)

__all__: list[str] = [
    "DEFAULT_LIVENESS_INTERVAL_SECONDS",
    "DEFAULT_LIVENESS_WINDOW_SECONDS",
    "ENV_LIVENESS_INTERVAL_SECONDS",
    "HandlerNodeHeartbeat",
    "HandlerNodeIntrospected",
    "HandlerNodeRegistrationAcked",
    "HandlerRuntimeTick",
    "ModelHeartbeatHandlerResult",
    "get_liveness_interval_seconds",
]
