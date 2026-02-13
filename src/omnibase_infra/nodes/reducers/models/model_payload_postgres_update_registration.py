# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""PostgreSQL UPDATE registration payload model for registration reducer.

Unlike ModelPayloadPostgresUpsertRegistration (INSERT ... ON CONFLICT),
this payload performs a conditional UPDATE WHERE with monotonic guard
for idempotent heartbeat and state-transition processing.

Intent Type:
    ``postgres.update_registration`` -- routed by IntentExecutor to the
    effect handler that executes a plain UPDATE query.

Related:
    - ModelPayloadPostgresUpsertRegistration: Full upsert (INSERT ... ON CONFLICT)
    - RegistrationReducerService: Emits this intent from decide_ack / decide_heartbeat
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.reducers.models.model_registration_ack_update import (
    ModelRegistrationAckUpdate,
)
from omnibase_infra.nodes.reducers.models.model_registration_heartbeat_update import (
    ModelRegistrationHeartbeatUpdate,
)


class ModelPayloadPostgresUpdateRegistration(BaseModel):
    """Payload for plain UPDATE registration projection intent.

    Unlike ModelPayloadPostgresUpsertRegistration (INSERT...ON CONFLICT),
    this payload performs a conditional UPDATE WHERE with monotonic guard
    for idempotent heartbeat processing.

    Attributes:
        intent_type: Discriminator literal for intent routing.
        correlation_id: Correlation ID for distributed tracing.
        entity_id: Entity UUID for WHERE clause.
        domain: Domain for WHERE clause (composite PK).
        updates: Typed column set for SET clause (structural union).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    intent_type: Literal["postgres.update_registration"] = Field(
        default="postgres.update_registration",
        description="Discriminator literal for intent routing.",
    )
    correlation_id: UUID = Field(
        ...,
        description="Correlation ID for distributed tracing.",
    )
    entity_id: UUID = Field(
        ...,
        description="Entity UUID for WHERE clause.",
    )
    domain: str = Field(
        default="registration",
        description="Domain for WHERE clause (composite PK).",
    )
    updates: ModelRegistrationAckUpdate | ModelRegistrationHeartbeatUpdate = Field(
        ...,
        description="Typed column set for SET clause.",
    )


__all__: list[str] = [
    "ModelPayloadPostgresUpdateRegistration",
]
