# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Models for Reducers.

This module exports models used by infrastructure reducers (pure function pattern).

Available Models:
    - ModelRegistrationState: Immutable state for pure reducer pattern
    - ModelRegistrationConfirmation: Confirmation event from Effect layer (Phase 2)
    - ModelPayloadConsulRegister: Payload for Consul registration intents
    - ModelPayloadLedgerAppend: Payload for audit ledger append intents
    - ModelPayloadPostgresUpsertRegistration: Payload for PostgreSQL upsert intents
"""

from omnibase_infra.nodes.reducers.models.model_payload_consul_register import (
    ModelPayloadConsulRegister,
)
from omnibase_infra.nodes.reducers.models.model_payload_ledger_append import (
    ModelPayloadLedgerAppend,
)
from omnibase_infra.nodes.reducers.models.model_payload_postgres_upsert_registration import (
    ModelPayloadPostgresUpsertRegistration,
)
from omnibase_infra.nodes.reducers.models.model_registration_confirmation import (
    ModelRegistrationConfirmation,
)
from omnibase_infra.nodes.reducers.models.model_registration_state import (
    ModelRegistrationState,
)

__all__ = [
    "ModelPayloadConsulRegister",
    "ModelPayloadLedgerAppend",
    "ModelPayloadPostgresUpsertRegistration",
    "ModelRegistrationConfirmation",
    "ModelRegistrationState",
]
