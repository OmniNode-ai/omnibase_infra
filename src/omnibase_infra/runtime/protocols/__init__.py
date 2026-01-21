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

from omnibase_infra.runtime.protocols.protocol_runtime_scheduler import (
    ProtocolRuntimeScheduler,
)
from omnibase_infra.runtime.protocols.protocol_transition_notification_publisher import (
    ProtocolTransitionNotificationPublisher,
)

__all__: list[str] = [
    "ProtocolRuntimeScheduler",
    "ProtocolTransitionNotificationPublisher",
]
