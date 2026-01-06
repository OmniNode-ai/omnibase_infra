# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handlers for NodeRegistryEffect operations.

This package contains the extracted handlers for the NodeRegistryEffect node,
following the declarative node pattern where backend-specific operations are
encapsulated in dedicated handler classes.

Available Handlers:
    HandlerConsulRegister: Consul service registration handler.
    HandlerConsulDeregister: Consul service deregistration handler.
    HandlerPostgresUpsert: PostgreSQL registration record upsert handler.
    HandlerPostgresDeactivate: PostgreSQL registration deactivation handler.
    HandlerPartialRetry: Targeted retry handler for partial failures.

Architecture:
    These handlers are used by NodeRegistryEffect to execute backend-specific
    operations while maintaining clean separation of concerns. Each handler
    is responsible for:
    - Operation timing and observability
    - Error sanitization for security
    - Structured result construction

Related:
    - NodeRegistryEffect: Parent effect node coordinating handlers
    - OMN-1103: Refactoring ticket for handler extraction
"""

from __future__ import annotations

from omnibase_infra.nodes.node_registry_effect.handlers.handler_consul_deregister import (
    HandlerConsulDeregister,
)
from omnibase_infra.nodes.node_registry_effect.handlers.handler_consul_register import (
    HandlerConsulRegister,
)
from omnibase_infra.nodes.node_registry_effect.handlers.handler_partial_retry import (
    HandlerPartialRetry,
)
from omnibase_infra.nodes.node_registry_effect.handlers.handler_postgres_deactivate import (
    HandlerPostgresDeactivate,
)
from omnibase_infra.nodes.node_registry_effect.handlers.handler_postgres_upsert import (
    HandlerPostgresUpsert,
)

__all__: list[str] = [
    "HandlerConsulDeregister",
    "HandlerConsulRegister",
    "HandlerPartialRetry",
    "HandlerPostgresDeactivate",
    "HandlerPostgresUpsert",
]
