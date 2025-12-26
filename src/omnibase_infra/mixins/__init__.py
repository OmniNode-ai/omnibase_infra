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
"""

from omnibase_infra.mixins.mixin_async_circuit_breaker import (
    CircuitState,
    MixinAsyncCircuitBreaker,
)
from omnibase_infra.mixins.mixin_envelope_extraction import MixinEnvelopeExtraction
from omnibase_infra.mixins.mixin_node_introspection import (
    MixinNodeIntrospection,
    PerformanceMetricsCacheDict,
)
from omnibase_infra.mixins.protocol_event_bus_like import ProtocolEventBusLike

__all__ = [
    "CircuitState",
    "MixinAsyncCircuitBreaker",
    "MixinEnvelopeExtraction",
    "MixinNodeIntrospection",
    "PerformanceMetricsCacheDict",
    "ProtocolEventBusLike",
]
