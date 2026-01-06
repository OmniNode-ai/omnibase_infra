# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""PostgreSQL upsert registration payload model for registration reducer.

This payload implements ProtocolIntentPayload for use with ModelIntent.
It contains the same data as ModelPostgresUpsertRegistrationIntent but with an
`intent_type` field instead of `kind` to satisfy the protocol.

Related:
    - ModelPostgresUpsertRegistrationIntent: Core intent model (uses `kind` discriminator)
    - ProtocolIntentPayload: Protocol requiring `intent_type` property
    - OMN-1260: Fix JsonValue/JsonType and validation import migration
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from omnibase_core.models.reducer.payloads import ModelIntentPayloadBase
from pydantic import BaseModel, Field


class ModelPayloadPostgresUpsertRegistration(ModelIntentPayloadBase):
    """Payload for PostgreSQL upsert registration intents.

    This payload extends ModelIntentPayloadBase to satisfy ProtocolIntentPayload,
    enabling use with ModelIntent for reducer output.

    Uses serialize_as_any=True semantics: the `record` field is typed as BaseModel
    to accept any registration record type, but serialization preserves subclass
    fields via Pydantic's serialize_as_any configuration.

    Attributes:
        intent_type: Discriminator literal for intent routing. Always "postgres.upsert_registration".
        correlation_id: Correlation ID for distributed tracing.
        record: The registration record to upsert (typed as BaseModel for flexibility).
    """

    intent_type: Literal["postgres.upsert_registration"] = Field(
        default="postgres.upsert_registration",
        description="Discriminator literal for intent routing.",
    )

    correlation_id: UUID = Field(
        ...,
        description="Correlation ID for distributed tracing.",
    )

    record: BaseModel = Field(
        ...,
        description="Registration record to upsert. Accepts any BaseModel subclass.",
    )


__all__ = [
    "ModelPayloadPostgresUpsertRegistration",
]
