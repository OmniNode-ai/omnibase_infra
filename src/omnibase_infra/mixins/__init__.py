# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""ONEX Infrastructure Mixins.

Reusable mixin classes providing:
- Coroutine-safe async operations (using asyncio.Lock)
- Infrastructure error integration
- Correlation ID propagation
- Configurable behavior

Protocols:
    - ProtocolCircuitBreakerAware: Interface for circuit breaker capability
      (tightly coupled to MixinAsyncCircuitBreaker)

Note:
    TypedDicts and model types used by mixins (e.g., CapabilitiesTypedDict,
    ModelIntrospectionPerformanceMetrics) should be imported from their
    canonical locations in omnibase_infra.models.discovery, not from this
    module.

    Configuration models for mixins:
    - ModelCircuitBreakerConfig: from omnibase_infra.models.resilience
    - ModelIntrospectionConfig: from omnibase_infra.models.discovery
    - ModelIntrospectionTaskConfig: from omnibase_infra.models.discovery

    ProtocolEventBusLike is defined in omnibase_infra.protocols (general-purpose
    event bus abstraction). Import it from there, not from this package.
"""

from omnibase_infra.enums import EnumCircuitState, EnumRetryErrorCategory
from omnibase_infra.mixins.mixin_async_circuit_breaker import (
    MixinAsyncCircuitBreaker,
    ModelCircuitBreakerConfig,
)
from omnibase_infra.mixins.mixin_envelope_extraction import MixinEnvelopeExtraction
from omnibase_infra.mixins.mixin_node_introspection import (
    MixinNodeIntrospection,
    PerformanceMetricsCacheDict,
)
from omnibase_infra.mixins.mixin_retry_execution import MixinRetryExecution
from omnibase_infra.mixins.protocol_circuit_breaker_aware import (
    ProtocolCircuitBreakerAware,
)
from omnibase_infra.models import ModelRetryErrorClassification

__all__: list[str] = [
    "EnumCircuitState",
    "EnumRetryErrorCategory",
    "MixinAsyncCircuitBreaker",
    "MixinEnvelopeExtraction",
    "MixinNodeIntrospection",
    "MixinRetryExecution",
    "ModelCircuitBreakerConfig",
    "ModelRetryErrorClassification",
    "PerformanceMetricsCacheDict",
    "ProtocolCircuitBreakerAware",
]
