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

Shared Patterns:
    All handlers share a common error handling pattern:
    - TimeoutError/InfraTimeoutError: Returns *_TIMEOUT_ERROR code
    - InfraAuthenticationError: Returns *_AUTH_ERROR code (non-retriable)
    - InfraConnectionError: Returns *_CONNECTION_ERROR code (retriable)
    - Exception: Returns *_UNKNOWN_ERROR code

    Each handler sanitizes errors via sanitize_backend_error() or
    sanitize_error_message() to prevent credential exposure.

    NOTE: The exception handling blocks are intentionally duplicated across
    handlers rather than extracted to a shared helper. This keeps each handler
    self-contained and allows for handler-specific error codes (e.g.,
    CONSUL_REGISTRATION_ERROR vs POSTGRES_UPSERT_ERROR). The ~50 lines of
    duplication per handler was evaluated and deemed acceptable for clarity
    over creating a parameterized helper that would reduce readability.

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
