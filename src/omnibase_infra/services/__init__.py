# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
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
    ServiceContractPublisher: Publish contracts to Kafka for dynamic discovery
    ServiceNodeSelector: Select nodes from candidates using various strategies
    ServiceSnapshot: Generic snapshot service for point-in-time state capture
    ServiceTimeoutEmitter: Emitter for timeout events with markers
    ServiceTimeoutScanner: Scanner for querying overdue registration entities
    ServiceTopicCatalog: Topic catalog with KV precedence and caching (OMN-2311)
    StoreSnapshotInMemory: In-memory snapshot store for testing
    StoreSnapshotPostgres: PostgreSQL snapshot store for production
    TimeoutEmitter: Alias for ServiceTimeoutEmitter
    TimeoutScanner: Alias for ServiceTimeoutScanner
"""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibase_infra.enums import EnumSelectionStrategy

    # Contract publisher service (OMN-1752)
    from omnibase_infra.services.contract_publisher import (
        ContractPublisherError,
        ContractPublishingInfraError,
        ContractSourceNotConfiguredError,
        ModelContractError,
        ModelContractPublisherConfig,
        ModelDiscoveredContract,
        ModelInfraError,
        ModelPublishResult,
        ModelPublishStats,
        NoContractsFoundError,
        ProtocolContractPublisherSource,
        ServiceContractPublisher,
        SourceContractComposite,
        SourceContractFilesystem,
        SourceContractPackage,
    )
    from omnibase_infra.services.corpus_capture import CorpusCapture
    from omnibase_infra.services.service_capability_query import ServiceCapabilityQuery
    from omnibase_infra.services.service_circuit_breaker_event_publisher import (
        CircuitBreakerEventPublisher,
    )
    from omnibase_infra.services.service_llm_endpoint_health import (
        ServiceLlmEndpointHealth,
    )
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
    from omnibase_infra.services.service_topic_catalog import ServiceTopicCatalog

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

# OMN-19444: Lazy imports keep `onex <cmd> --help` fast.
_LAZY_EXPORTS: dict[str, str] = {
    "CircuitBreakerEventPublisher": "omnibase_infra.services.service_circuit_breaker_event_publisher",
    "ConfigSessionConsumer": "omnibase_infra.services.session",
    "ConfigSessionStorage": "omnibase_infra.services.session",
    "ConsumerMetrics": "omnibase_infra.services.session",
    "ContractPublisherError": "omnibase_infra.services.contract_publisher",
    "ContractPublishingInfraError": "omnibase_infra.services.contract_publisher",
    "ContractSourceNotConfiguredError": "omnibase_infra.services.contract_publisher",
    "CorpusCapture": "omnibase_infra.services.corpus_capture",
    "DEFAULT_SELECTION_KEY": "omnibase_infra.services.service_node_selector",
    "EnumCircuitState": "omnibase_infra.services.session",
    "EnumSelectionStrategy": "omnibase_infra.enums",
    "ModelContractError": "omnibase_infra.services.contract_publisher",
    "ModelContractPublisherConfig": "omnibase_infra.services.contract_publisher",
    "ModelDiscoveredContract": "omnibase_infra.services.contract_publisher",
    "ModelInfraError": "omnibase_infra.services.contract_publisher",
    "ModelPublishResult": "omnibase_infra.services.contract_publisher",
    "ModelPublishStats": "omnibase_infra.services.contract_publisher",
    "ModelTimeoutEmissionConfig": "omnibase_infra.services.service_timeout_emitter",
    "ModelTimeoutEmissionResult": "omnibase_infra.services.service_timeout_emitter",
    "ModelTimeoutQueryResult": "omnibase_infra.services.service_timeout_scanner",
    "NoContractsFoundError": "omnibase_infra.services.contract_publisher",
    "ProtocolContractPublisherSource": "omnibase_infra.services.contract_publisher",
    "ProtocolSessionAggregator": "omnibase_infra.services.session",
    "ServiceCapabilityQuery": "omnibase_infra.services.service_capability_query",
    "ServiceContractPublisher": "omnibase_infra.services.contract_publisher",
    "ServiceLlmEndpointHealth": "omnibase_infra.services.service_llm_endpoint_health",
    "ServiceNodeSelector": "omnibase_infra.services.service_node_selector",
    "ServiceSnapshot": "omnibase_infra.services.snapshot",
    "ServiceTimeoutEmitter": "omnibase_infra.services.service_timeout_emitter",
    "ServiceTimeoutScanner": "omnibase_infra.services.service_timeout_scanner",
    "ServiceTopicCatalog": "omnibase_infra.services.service_topic_catalog",
    "SessionEventConsumer": "omnibase_infra.services.session",
    "SessionSnapshotStore": "omnibase_infra.services.session",
    "SessionStoreNotInitializedError": "omnibase_infra.services.session",
    "SourceContractComposite": "omnibase_infra.services.contract_publisher",
    "SourceContractFilesystem": "omnibase_infra.services.contract_publisher",
    "SourceContractPackage": "omnibase_infra.services.contract_publisher",
    "StoreSnapshotInMemory": "omnibase_infra.services.snapshot",
    "StoreSnapshotPostgres": "omnibase_infra.services.snapshot",
}

_LAZY_ALIASES: dict[str, tuple[str, str]] = {
    "TimeoutEmitter": (
        "omnibase_infra.services.service_timeout_emitter",
        "ServiceTimeoutEmitter",
    ),
    "TimeoutScanner": (
        "omnibase_infra.services.service_timeout_scanner",
        "ServiceTimeoutScanner",
    ),
}

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
    # LLM endpoint health checker (OMN-2255)
    "ServiceLlmEndpointHealth",
    # Topic catalog service (OMN-2311)
    "ServiceTopicCatalog",
    # Circuit breaker event publisher (OMN-5293)
    "CircuitBreakerEventPublisher",
    # Contract publisher service (OMN-1752)
    "ContractPublisherError",
    "ContractPublishingInfraError",
    "ContractSourceNotConfiguredError",
    "ModelContractError",
    "ModelContractPublisherConfig",
    "ModelDiscoveredContract",
    "ModelInfraError",
    "ModelPublishResult",
    "ModelPublishStats",
    "NoContractsFoundError",
    "ProtocolContractPublisherSource",
    "ServiceContractPublisher",
    "SourceContractComposite",
    "SourceContractFilesystem",
    "SourceContractPackage",
]


def __getattr__(name: str) -> object:
    if name in _LAZY_EXPORTS:
        module = importlib.import_module(_LAZY_EXPORTS[name])
        value: object = getattr(module, name)
        globals()[name] = value
        return value
    if name in _LAZY_ALIASES:
        module_name, attribute_name = _LAZY_ALIASES[name]
        module = importlib.import_module(module_name)
        value = getattr(module, attribute_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
