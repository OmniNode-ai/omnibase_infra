# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""ONEX Infrastructure Mixins.

Reusable mixin classes providing:
- Coroutine-safe async operations (using asyncio.Lock)
- Infrastructure error integration
- Correlation ID propagation
- Configurable behavior

Note:
    TypedDicts and model types used by mixins (e.g., CapabilitiesTypedDict,
    ModelIntrospectionPerformanceMetrics) should be imported from their
    canonical locations in omnibase_infra.models.discovery, not from this
    module.

    Configuration models for mixins:
    - ModelCircuitBreakerConfig: from omnibase_infra.models.resilience
    - ModelIntrospectionConfig: from omnibase_infra.models.discovery
    - ModelIntrospectionTaskConfig: from omnibase_infra.models.discovery
"""

from omnibase_infra.mixins.mixin_async_circuit_breaker import (
    CircuitState,
    MixinAsyncCircuitBreaker,
    ModelCircuitBreakerConfig,
)
from omnibase_infra.mixins.mixin_envelope_extraction import MixinEnvelopeExtraction
from omnibase_infra.mixins.mixin_node_introspection import (
    MixinNodeIntrospection,
    PerformanceMetricsCacheDict,
)
from omnibase_infra.mixins.mixin_retry_execution import (
    EnumRetryErrorCategory,
    MixinRetryExecution,
    RetryErrorClassification,
)
from omnibase_infra.mixins.protocol_circuit_breaker_aware import (
    ProtocolCircuitBreakerAware,
)
from omnibase_infra.mixins.protocol_event_bus_like import ProtocolEventBusLike

__all__: list[str] = [
    "CircuitState",
    "EnumRetryErrorCategory",
    "MixinAsyncCircuitBreaker",
    "MixinEnvelopeExtraction",
    "MixinNodeIntrospection",
    "MixinRetryExecution",
    "ModelCircuitBreakerConfig",
    "PerformanceMetricsCacheDict",
    "ProtocolCircuitBreakerAware",
    "ProtocolEventBusLike",
    "RetryErrorClassification",
]
