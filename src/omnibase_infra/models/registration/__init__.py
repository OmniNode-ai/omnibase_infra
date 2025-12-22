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
from omnibase_infra.models.registration.model_node_capabilities import (
    ModelNodeCapabilities,
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
    # Decision events (C1 Orchestrator output)
    "ModelNodeBecameActive",
    "ModelNodeLivenessExpired",
    "ModelNodeRegistrationAccepted",
    "ModelNodeRegistrationAckReceived",
    "ModelNodeRegistrationAckTimedOut",
    "ModelNodeRegistrationInitiated",
    "ModelNodeRegistrationRejected",
    # Core registration models
    "ModelNodeCapabilities",
    "ModelNodeHeartbeatEvent",
    "ModelNodeIntrospectionEvent",
    "ModelNodeMetadata",
    "ModelNodeRegistration",
    "ModelNodeRegistrationRecord",
]
