# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handlers for NodeContractPersistenceEffect operations.

This package contains the handlers for the NodeContractPersistenceEffect node,
following the declarative node pattern where PostgreSQL operations are
encapsulated in dedicated handler classes.

Available Handlers:
    HandlerPostgresContractUpsert: Upsert contract record handler.
    HandlerPostgresTopicUpdate: Update topic routing handler.
    HandlerPostgresMarkStale: Batch mark stale contracts handler.
    HandlerPostgresHeartbeat: Update heartbeat timestamp handler.
    HandlerPostgresDeactivate: Deactivate contract handler.
    HandlerPostgresCleanupTopics: Cleanup topic references handler.

Architecture:
    These handlers are used by NodeContractPersistenceEffect to execute
    PostgreSQL operations based on intents from ContractRegistryReducer.
    Each handler is responsible for:
    - Operation timing and observability
    - Error sanitization for security
    - Structured result construction
    - Retry logic per retry_policy configuration

Shared Patterns:
    All handlers share a common error handling pattern:
    - TimeoutError/InfraTimeoutError: Returns *_TIMEOUT_ERROR code
    - InfraAuthenticationError: Returns *_AUTH_ERROR code (non-retriable)
    - InfraConnectionError: Returns *_CONNECTION_ERROR code (retriable)
    - RepositoryExecutionError: Returns operation-specific error code
    - Exception: Returns *_UNKNOWN_ERROR code

    Each handler sanitizes errors via sanitize_error_message() to prevent
    credential exposure in logs and error responses.

Related:
    - NodeContractPersistenceEffect: Parent effect node coordinating handlers
    - ContractRegistryReducer: Source of intents
    - OMN-1845: Implementation ticket
    - OMN-1653: ContractRegistryReducer ticket
"""

from __future__ import annotations

from omnibase_infra.nodes.node_contract_persistence_effect.handlers.handler_postgres_cleanup_topics import (
    HandlerPostgresCleanupTopics,
)
from omnibase_infra.nodes.node_contract_persistence_effect.handlers.handler_postgres_contract_upsert import (
    HandlerPostgresContractUpsert,
)
from omnibase_infra.nodes.node_contract_persistence_effect.handlers.handler_postgres_deactivate import (
    HandlerPostgresDeactivate,
)
from omnibase_infra.nodes.node_contract_persistence_effect.handlers.handler_postgres_heartbeat import (
    HandlerPostgresHeartbeat,
)
from omnibase_infra.nodes.node_contract_persistence_effect.handlers.handler_postgres_mark_stale import (
    HandlerPostgresMarkStale,
)
from omnibase_infra.nodes.node_contract_persistence_effect.handlers.handler_postgres_topic_update import (
    HandlerPostgresTopicUpdate,
    normalize_topic_for_storage,
)

__all__: list[str] = [
    "HandlerPostgresCleanupTopics",
    "HandlerPostgresContractUpsert",
    "HandlerPostgresDeactivate",
    "HandlerPostgresHeartbeat",
    "HandlerPostgresMarkStale",
    "HandlerPostgresTopicUpdate",
    "normalize_topic_for_storage",
]
