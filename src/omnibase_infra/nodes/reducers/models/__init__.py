# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Models for Reducers.

This module exports models used by infrastructure reducers (pure function pattern).

Available Models:
    - ModelRegistrationState: Immutable state for pure reducer pattern
    - ModelRegistrationConfirmation: Confirmation event from Effect layer (Phase 2)
    - ModelPayloadLedgerAppend: Payload for audit ledger append intents
    - ModelPayloadPostgresUpsertRegistration: Payload for PostgreSQL upsert intents
    - ModelPayloadPostgresUpdateRegistration: Payload for PostgreSQL UPDATE intents
"""

from omnibase_infra.nodes.reducers.models.model_payload_ledger_append import (
    ModelPayloadLedgerAppend,
)
from omnibase_infra.nodes.reducers.models.model_payload_postgres_update_registration import (
    ModelPayloadPostgresUpdateRegistration,
)
from omnibase_infra.nodes.reducers.models.model_payload_postgres_upsert_registration import (
    ModelPayloadPostgresUpsertRegistration,
)
from omnibase_infra.nodes.reducers.models.model_registration_ack_update import (
    ModelRegistrationAckUpdate,
)
from omnibase_infra.nodes.reducers.models.model_registration_confirmation import (
    ModelRegistrationConfirmation,
)
from omnibase_infra.nodes.reducers.models.model_registration_heartbeat_update import (
    ModelRegistrationHeartbeatUpdate,
)
from omnibase_infra.nodes.reducers.models.model_registration_state import (
    ModelRegistrationState,
)

__all__ = [
    "ModelPayloadLedgerAppend",
    "ModelPayloadPostgresUpdateRegistration",
    "ModelPayloadPostgresUpsertRegistration",
    "ModelRegistrationAckUpdate",
    "ModelRegistrationConfirmation",
    "ModelRegistrationHeartbeatUpdate",
    "ModelRegistrationState",
]
