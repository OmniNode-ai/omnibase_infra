# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Resilience models for infrastructure fault tolerance.

``ModelOperationContext`` and ``ModelRetryState`` live here, beside the circuit
breaker models, because they are the retry seam's own state and no node contract
binds them as handlers (OMN-17427; omnibase_infra#5683). Their consumers are
``omnibase_infra.mixins.mixin_retry_execution`` and
``omnibase_infra.mixins.mixin_llm_http_transport``.
"""

from omnibase_infra.models.resilience.model_circuit_breaker_config import (
    ModelCircuitBreakerConfig,
)
from omnibase_infra.models.resilience.model_circuit_breaker_state_event import (
    ModelCircuitBreakerStateEvent,
)
from omnibase_infra.models.resilience.model_operation_context import (
    ModelOperationContext,
)
from omnibase_infra.models.resilience.model_retry_state import ModelRetryState

__all__ = [
    "ModelCircuitBreakerConfig",
    "ModelCircuitBreakerStateEvent",
    "ModelOperationContext",
    "ModelRetryState",
]
