# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registration dispatcher adapters for MessageDispatchEngine integration.

This module provides ProtocolMessageDispatcher adapters that wrap the
registration orchestrator handlers for integration with MessageDispatchEngine's
category-based routing.

Available Dispatchers:
    - DispatcherNodeIntrospected: Handles ModelNodeIntrospectionEvent (EVENT)
    - DispatcherRuntimeTick: Handles ModelRuntimeTick (EVENT)
    - DispatcherNodeRegistrationAcked: Handles ModelNodeRegistrationAcked (COMMAND)

Usage:
    The dispatchers are typically wired via wire_registration_dispatchers()
    in the domain wiring module (wiring.py), but can also be created directly:

    >>> from omnibase_infra.nodes.node_registration_orchestrator.dispatchers import (
    ...     DispatcherNodeIntrospected,
    ... )
    >>> dispatcher = DispatcherNodeIntrospected(handler_instance)
    >>> engine.register_dispatcher(dispatcher)

Design:
    Each dispatcher:
    - Implements ProtocolMessageDispatcher protocol
    - Wraps a registration handler (HandlerNode*, HandlerRuntimeTick)
    - Handles deserialization from ModelEventEnvelope to typed payload
    - Injects time via dispatch context (ORCHESTRATOR node kind)
    - Returns ModelDispatchResult with structured success/error info

Related:
    - OMN-888: Registration Orchestrator
    - OMN-892: 2-way Registration E2E Integration Test
    - OMN-934: Message Dispatch Engine
    - OMN-1346: Registration Code Extraction
"""

from omnibase_infra.nodes.node_registration_orchestrator.dispatchers.dispatcher_node_introspected import (
    DispatcherNodeIntrospected,
)
from omnibase_infra.nodes.node_registration_orchestrator.dispatchers.dispatcher_node_registration_acked import (
    DispatcherNodeRegistrationAcked,
)
from omnibase_infra.nodes.node_registration_orchestrator.dispatchers.dispatcher_runtime_tick import (
    DispatcherRuntimeTick,
)

__all__: list[str] = [
    "DispatcherNodeIntrospected",
    "DispatcherNodeRegistrationAcked",
    "DispatcherRuntimeTick",
]
