# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Runtime protocols for ONEX Infrastructure.

This package contains protocol interfaces for runtime components. Protocols
define structural subtyping (duck typing) interfaces per PEP 544.

Available Protocols:
    ProtocolRuntimeScheduler: Interface for runtime tick scheduler.
        The scheduler is the single source of truth for 'now' across orchestrators.
        It emits RuntimeTick events at configured intervals.

    ProtocolTransitionNotificationPublisher: Interface for transition notification
        publishing. Used by TransitionNotificationOutbox for delivering state
        transition notifications to event buses.

Related:
    - OMN-953: RuntimeTick scheduler implementation
    - OMN-1139: TransitionNotificationOutbox implementation
    - See also: runtime.dispatcher_registry.ProtocolMessageDispatcher
    - See also: runtime.protocol_policy.ProtocolPolicy
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Re-export from omnibase_core for convenience
from omnibase_core.protocols.notifications import (
    ProtocolTransitionNotificationPublisher,
)
from omnibase_infra.runtime.protocols.protocol_intent_executor import (
    PayloadT_contra,
    ProtocolIntentExecutor,
)
from omnibase_infra.runtime.protocols.protocol_intent_payload import (
    ProtocolIntentPayload,
)
from omnibase_infra.runtime.protocols.protocol_runtime_scheduler import (
    ProtocolRuntimeScheduler,
)

if TYPE_CHECKING:
    # IntentPayloadType is only available for type checking - it references
    # models from nodes.* which aren't loaded during package initialization.
    from omnibase_infra.runtime.protocols.protocol_intent_executor import (
        IntentPayloadType,
    )

# NOTE: IntentPayloadType is only available under TYPE_CHECKING (see above)
__all__: list[str] = [
    "PayloadT_contra",
    "ProtocolIntentExecutor",
    "ProtocolIntentPayload",
    "ProtocolRuntimeScheduler",
    "ProtocolTransitionNotificationPublisher",
]
