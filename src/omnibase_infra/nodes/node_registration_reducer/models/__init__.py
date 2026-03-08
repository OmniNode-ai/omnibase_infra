# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Models for NodeRegistrationReducer.

This module exports models used by the NodeRegistrationReducer (FSM-driven pattern).
All models are now local to this package (OMN-3989: migrated from nodes.reducers.models).

Available Models:
    - ModelValidationResult: Validation result with error details
    - ModelRegistrationState: Immutable state for reducer FSM
    - ModelRegistrationConfirmation: Confirmation event from Effect layer
    - ModelRegistrationAckUpdate: Acknowledgement update from Effect layer
    - ModelRegistrationHeartbeatUpdate: Heartbeat update
    - ModelPayloadPostgresUpsertRegistration: Payload for PostgreSQL upsert intents
    - ModelPayloadPostgresUpdateRegistration: Payload for PostgreSQL update intents
    - ModelPayloadLedgerAppend: Payload for ledger append intents
"""

from __future__ import annotations

# Registration state models (migrated from nodes.reducers.models)
from omnibase_infra.nodes.node_registration_reducer.models.model_payload_ledger_append import (
    ModelPayloadLedgerAppend,
)
from omnibase_infra.nodes.node_registration_reducer.models.model_payload_postgres_update_registration import (
    ModelPayloadPostgresUpdateRegistration,
)
from omnibase_infra.nodes.node_registration_reducer.models.model_payload_postgres_upsert_registration import (
    ModelPayloadPostgresUpsertRegistration,
)
from omnibase_infra.nodes.node_registration_reducer.models.model_registration_ack_update import (
    ModelRegistrationAckUpdate,
)
from omnibase_infra.nodes.node_registration_reducer.models.model_registration_confirmation import (
    ModelRegistrationConfirmation,
)
from omnibase_infra.nodes.node_registration_reducer.models.model_registration_heartbeat_update import (
    ModelRegistrationHeartbeatUpdate,
)
from omnibase_infra.nodes.node_registration_reducer.models.model_registration_state import (
    ModelRegistrationState,
)

# Node-specific model
from omnibase_infra.nodes.node_registration_reducer.models.model_validation_result import (
    ModelValidationResult,
    ValidationErrorCode,
    ValidationResult,
)

__all__ = [
    "ModelPayloadLedgerAppend",
    "ModelPayloadPostgresUpdateRegistration",
    "ModelPayloadPostgresUpsertRegistration",
    "ModelRegistrationAckUpdate",
    "ModelRegistrationConfirmation",
    "ModelRegistrationHeartbeatUpdate",
    "ModelRegistrationState",
    "ModelValidationResult",
    "ValidationErrorCode",
    "ValidationResult",
]
