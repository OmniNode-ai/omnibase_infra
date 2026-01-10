# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""ONEX Infrastructure Mixins.

Reusable mixin classes providing:
- Coroutine-safe async operations (using asyncio.Lock)
- Infrastructure error integration
- Correlation ID propagation
- Configurable behavior

Exports (in __all__):
    Mixins:
        - MixinAsyncCircuitBreaker: Coroutine-safe circuit breaker implementation
        - MixinDictLikeAccessors: Dictionary-style access helpers
        - MixinEnvelopeExtraction: Event envelope extraction utilities
        - MixinNodeIntrospection: Node capability introspection
        - MixinRetryExecution: Retry logic with exponential backoff

    Protocols:
        - ProtocolCircuitBreakerAware: Interface for circuit breaker capability
          (tightly coupled to MixinAsyncCircuitBreaker, co-located here)

    Enums:
        - EnumCircuitState: Circuit breaker states (CLOSED, OPEN, HALF_OPEN)
        - EnumRetryErrorCategory: Error categorization for retry logic

    Models:
        - ModelCircuitBreakerConfig: Circuit breaker configuration
        - ModelRetryErrorClassification: Retry error classification result

    TypedDicts:
        - TypedDictPerformanceMetricsCache: Performance metrics cache structure

NOT Exported (import from canonical locations):
    - ProtocolEventBusLike: General-purpose event bus protocol.
      Import from: omnibase_infra.protocols

    - ModelIntrospectionConfig, ModelIntrospectionTaskConfig: Introspection settings.
      Import from: omnibase_infra.models.discovery

    - ModelIntrospectionPerformanceMetrics, ModelDiscoveredCapabilities: Discovery models.
      Import from: omnibase_infra.models.discovery
"""

from omnibase_infra.enums import EnumCircuitState, EnumRetryErrorCategory
from omnibase_infra.mixins.mixin_async_circuit_breaker import MixinAsyncCircuitBreaker
from omnibase_infra.mixins.mixin_dict_like_accessors import MixinDictLikeAccessors
from omnibase_infra.mixins.mixin_envelope_extraction import MixinEnvelopeExtraction
from omnibase_infra.mixins.mixin_node_introspection import MixinNodeIntrospection
from omnibase_infra.mixins.mixin_retry_execution import MixinRetryExecution
from omnibase_infra.mixins.protocol_circuit_breaker_aware import (
    ProtocolCircuitBreakerAware,
)
from omnibase_infra.models import ModelRetryErrorClassification
from omnibase_infra.models.resilience import ModelCircuitBreakerConfig
from omnibase_infra.types.typed_dict import TypedDictPerformanceMetricsCache

__all__: list[str] = [
    "EnumCircuitState",
    "EnumRetryErrorCategory",
    "MixinAsyncCircuitBreaker",
    "MixinDictLikeAccessors",
    "MixinEnvelopeExtraction",
    "MixinNodeIntrospection",
    "MixinRetryExecution",
    "ModelCircuitBreakerConfig",
    "ModelRetryErrorClassification",
    "ProtocolCircuitBreakerAware",
    "TypedDictPerformanceMetricsCache",
]
