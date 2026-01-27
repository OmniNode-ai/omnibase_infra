# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registration models for ONEX 2-way registration pattern.

This module provides models for the ONEX 2-way registration workflow,
including introspection events, heartbeats, and orchestrator decision events.

Submodules:
    events: Registration decision events emitted by the C1 Orchestrator
"""

from omnibase_infra.models.registration.events import (
    ModelNodeBecameActive,
    ModelNodeLivenessExpired,
    ModelNodeRegistrationAccepted,
    ModelNodeRegistrationAckReceived,
    ModelNodeRegistrationAckTimedOut,
    ModelNodeRegistrationInitiated,
    ModelNodeRegistrationRejected,
)
from omnibase_infra.models.registration.model_event_bus_topic_entry import (
    ModelEventBusTopicEntry,
)
from omnibase_infra.models.registration.model_introspection_metrics import (
    ModelIntrospectionMetrics,
)
from omnibase_infra.models.registration.model_node_capabilities import (
    ModelNodeCapabilities,
)
from omnibase_infra.models.registration.model_node_event_bus_config import (
    ModelNodeEventBusConfig,
)
from omnibase_infra.models.registration.model_node_heartbeat_event import (
    ModelNodeHeartbeatEvent,
)
from omnibase_infra.models.registration.model_node_introspection_event import (
    ModelNodeIntrospectionEvent,
)
from omnibase_infra.models.registration.model_node_metadata import ModelNodeMetadata
from omnibase_infra.models.registration.model_node_registration import (
    ModelNodeRegistration,
)
from omnibase_infra.models.registration.model_node_registration_record import (
    ModelNodeRegistrationRecord,
)

__all__ = [
    # Event bus configuration
    "ModelEventBusTopicEntry",
    "ModelNodeEventBusConfig",
    # Metrics
    "ModelIntrospectionMetrics",
    # Decision events (C1 Orchestrator output)
    "ModelNodeBecameActive",
    # Core registration models
    "ModelNodeCapabilities",
    "ModelNodeHeartbeatEvent",
    "ModelNodeIntrospectionEvent",
    "ModelNodeLivenessExpired",
    "ModelNodeMetadata",
    "ModelNodeRegistration",
    "ModelNodeRegistrationAccepted",
    "ModelNodeRegistrationAckReceived",
    "ModelNodeRegistrationAckTimedOut",
    "ModelNodeRegistrationInitiated",
    "ModelNodeRegistrationRecord",
    "ModelNodeRegistrationRejected",
]
