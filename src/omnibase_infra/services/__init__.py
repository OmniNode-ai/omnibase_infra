# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""ONEX Infrastructure Services Module.

Provides high-level services that compose infrastructure components for
use by orchestrators and runtime hosts. Services provide clean interfaces
for common operations and encapsulate complexity.

Exports:
    DEFAULT_SELECTION_KEY: Default key for round-robin state tracking
    EnumSelectionStrategy: Selection strategies for capability-based discovery
    ModelTimeoutEmissionConfig: Configuration for timeout emitter
    ModelTimeoutEmissionResult: Result model for timeout emission processing
    ModelTimeoutQueryResult: Result model for timeout queries
    ServiceCapabilityQuery: Query nodes by capability, not by name
    ServiceNodeSelector: Select nodes from candidates using various strategies
    ServiceSnapshot: Generic snapshot service for point-in-time state capture
    ServiceTimeoutEmitter: Emitter for timeout events with markers
    ServiceTimeoutScanner: Scanner for querying overdue registration entities
    StoreSnapshotInMemory: In-memory snapshot store for testing
    StoreSnapshotPostgres: PostgreSQL snapshot store for production
    TimeoutEmitter: Alias for ServiceTimeoutEmitter
    TimeoutScanner: Alias for ServiceTimeoutScanner
"""

from omnibase_infra.enums import EnumSelectionStrategy
from omnibase_infra.services.corpus_capture import CorpusCapture
from omnibase_infra.services.service_capability_query import ServiceCapabilityQuery
from omnibase_infra.services.service_node_selector import (
    DEFAULT_SELECTION_KEY,
    ServiceNodeSelector,
)
from omnibase_infra.services.service_timeout_emitter import (
    ModelTimeoutEmissionConfig,
    ModelTimeoutEmissionResult,
    ServiceTimeoutEmitter,
)
from omnibase_infra.services.service_timeout_scanner import (
    ModelTimeoutQueryResult,
    ServiceTimeoutScanner,
)

# Session services (moved from omniclaude in OMN-1526)
from omnibase_infra.services.session import (
    ConfigSessionConsumer,
    ConfigSessionStorage,
    ConsumerMetrics,
    EnumCircuitState,
    ProtocolSessionAggregator,
    SessionEventConsumer,
    SessionSnapshotStore,
    SessionStoreNotInitializedError,
)
from omnibase_infra.services.snapshot import (
    ServiceSnapshot,
    StoreSnapshotInMemory,
    StoreSnapshotPostgres,
)

# Aliases for convenience
TimeoutEmitter = ServiceTimeoutEmitter
TimeoutScanner = ServiceTimeoutScanner

__all__ = [
    "DEFAULT_SELECTION_KEY",
    "EnumSelectionStrategy",
    "ModelTimeoutEmissionConfig",
    "ModelTimeoutEmissionResult",
    "ModelTimeoutQueryResult",
    "ServiceCapabilityQuery",
    "CorpusCapture",
    "ServiceNodeSelector",
    "ServiceSnapshot",
    "ServiceTimeoutEmitter",
    "ServiceTimeoutScanner",
    "StoreSnapshotInMemory",
    "StoreSnapshotPostgres",
    "TimeoutEmitter",
    "TimeoutScanner",
    # Session services (OMN-1526)
    "ConfigSessionConsumer",
    "ConfigSessionStorage",
    "ConsumerMetrics",
    "EnumCircuitState",
    "ProtocolSessionAggregator",
    "SessionEventConsumer",
    "SessionSnapshotStore",
    "SessionStoreNotInitializedError",
]
