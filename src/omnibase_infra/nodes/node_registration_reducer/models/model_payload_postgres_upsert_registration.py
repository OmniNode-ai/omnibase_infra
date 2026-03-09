# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""PostgreSQL upsert registration payload model for registration reducer.

This payload implements ProtocolIntentPayload for use with ModelIntent.
It contains the same data as ModelPostgresUpsertRegistrationIntent but with an
`intent_type` field instead of `kind` to satisfy the protocol.

Related:
    - ModelPostgresUpsertRegistrationIntent: Core intent model (uses `kind` discriminator)
    - ProtocolIntentPayload: Protocol requiring `intent_type` property
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny

# NOTE: ModelIntentPayloadBase was removed in omnibase_core 0.6.2
# Using pydantic.BaseModel directly as the base class


class ModelPayloadPostgresUpsertRegistration(BaseModel):
    """Payload for PostgreSQL upsert registration intents.

    This payload follows the ONEX intent payload pattern for use with ModelIntent.

    Uses SerializeAsAny wrapper: the `record` field accepts any BaseModel subclass,
    and serialization preserves all subclass fields via Pydantic's SerializeAsAny
    type wrapper.

    Attributes:
        intent_type: Discriminator literal for intent routing. Always "postgres.upsert_registration".
        correlation_id: Correlation ID for distributed tracing.
        record: The registration record to upsert (uses SerializeAsAny for subclass preservation).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    intent_type: Literal["postgres.upsert_registration"] = Field(
        default="postgres.upsert_registration",
        description="Discriminator literal for intent routing.",
    )

    correlation_id: UUID = Field(
        ...,
        description="Correlation ID for distributed tracing.",
    )

    record: SerializeAsAny[BaseModel] = Field(
        ...,
        description="Registration record to upsert. Accepts any BaseModel subclass.",
    )


__all__ = [
    "ModelPayloadPostgresUpsertRegistration",
]
