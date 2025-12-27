# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registration decision events for C1 Registration Orchestrator.

This module provides event models emitted by the Registration Orchestrator
during the ONEX 2-way registration workflow. These events represent
orchestrator decisions and are consumed by the Registration Reducer.

Event Flow:
    NodeIntrospected (input) -> Orchestrator -> Decision Events (output)

Decision Events:
    - ModelNodeRegistrationInitiated: Registration attempt started
    - ModelNodeRegistrationAccepted: Registration accepted by orchestrator
    - ModelNodeRegistrationRejected: Registration rejected by orchestrator
    - ModelNodeRegistrationAckTimedOut: Ack deadline passed without acknowledgment
    - ModelNodeRegistrationAckReceived: Node acknowledged registration
    - ModelNodeBecameActive: Node transitioned to active state
    - ModelNodeLivenessExpired: Liveness deadline passed without heartbeat

See Also:
    - docs/design/ONEX_RUNTIME_REGISTRATION_TICKET_PLAN.md (C1 section)
    - DESIGN_TWO_WAY_REGISTRATION_ARCHITECTURE.md
"""

from omnibase_infra.models.registration.events.model_node_became_active import (
    ModelNodeBecameActive,
)
from omnibase_infra.models.registration.events.model_node_liveness_expired import (
    ModelNodeLivenessExpired,
)
from omnibase_infra.models.registration.events.model_node_registration_accepted import (
    ModelNodeRegistrationAccepted,
)
from omnibase_infra.models.registration.events.model_node_registration_ack_received import (
    ModelNodeRegistrationAckReceived,
)
from omnibase_infra.models.registration.events.model_node_registration_ack_timed_out import (
    ModelNodeRegistrationAckTimedOut,
)
from omnibase_infra.models.registration.events.model_node_registration_initiated import (
    ModelNodeRegistrationInitiated,
)
from omnibase_infra.models.registration.events.model_node_registration_rejected import (
    ModelNodeRegistrationRejected,
)

__all__ = [
    "ModelNodeBecameActive",
    "ModelNodeLivenessExpired",
    "ModelNodeRegistrationAccepted",
    "ModelNodeRegistrationAckReceived",
    "ModelNodeRegistrationAckTimedOut",
    "ModelNodeRegistrationInitiated",
    "ModelNodeRegistrationRejected",
]
