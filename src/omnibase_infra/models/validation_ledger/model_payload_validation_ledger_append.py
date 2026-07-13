# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Validation ledger append payload model (OMN-14524).

This payload implements the ONEX intent payload pattern for use with
ModelIntent, mirroring ModelPayloadLedgerAppend (event_ledger's equivalent).
It captures a projected validation event for append-only
validation_event_ledger persistence.

Field Mapping:
    Every field here maps 1:1 to a keyword argument of
    ``PostgresValidationLedgerRepository.append()`` /
    ``ProtocolValidationLedgerRepository.append()`` and to the dict keys
    already returned by ``HandlerValidationLedgerProjection.project()``.

Bytes Encoding:
    Kafka event values are bytes. Since bytes cannot safely cross intent
    boundaries (serialization issues), ``envelope_bytes`` is base64-encoded
    at this boundary -- the same pattern ModelPayloadLedgerAppend uses for
    ``event_value``. The Effect layer (HandlerValidationLedgerAppend) decodes
    it before storage.

Unlike the generic event_ledger payload, validation events have a
well-defined schema, so ``run_id``, ``repo_id``, ``event_type``,
``event_version``, and ``occurred_at`` are all REQUIRED (not nullable) --
matching the NOT NULL columns on validation_event_ledger
(docker/migrations/forward/045_create_validation_event_ledger.sql).
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

# NOTE: ModelIntentPayloadBase was removed in omnibase_core 0.6.2
# Using pydantic.BaseModel directly as the base class, mirroring
# ModelPayloadLedgerAppend.


class ModelPayloadValidationLedgerAppend(BaseModel):
    """Payload for validation ledger append intents.

    Attributes:
        intent_type: Discriminator literal for intent routing. Always
            "validation_ledger.append".
        run_id: Validation run correlation ID.
        repo_id: Repository identifier being validated.
        event_type: Fully qualified event type name.
        event_version: Schema version of the event type.
        occurred_at: When the validation event occurred (from payload).
        kafka_topic: Kafka topic the event was consumed from
            (idempotency key component).
        kafka_partition: Kafka partition number (idempotency key component).
        kafka_offset: Kafka offset within partition (idempotency key
            component).
        envelope_bytes: Base64-encoded raw envelope bytes (required -- the
            raw event, decoded to BYTEA by the Effect layer).
        envelope_hash: SHA-256 hex digest of the raw envelope bytes.
        correlation_id: Correlation ID for distributed tracing (optional).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    intent_type: Literal["validation_ledger.append"] = Field(
        default="validation_ledger.append",
        description="Discriminator literal for intent routing.",
    )

    run_id: UUID = Field(..., description="Validation run correlation ID.")
    repo_id: str = Field(
        ..., min_length=1, description="Repository identifier being validated."
    )
    event_type: str = Field(
        ..., min_length=1, description="Fully qualified event type name."
    )
    event_version: str = Field(
        ..., min_length=1, description="Schema version of the event type."
    )
    occurred_at: datetime = Field(
        ..., description="When the validation event occurred (from payload)."
    )

    # Kafka position -- required for idempotency
    kafka_topic: str = Field(
        ...,
        min_length=1,
        description="Kafka topic the event was consumed from.",
    )
    kafka_partition: int = Field(
        ...,
        ge=0,
        description="Kafka partition number (idempotency key component).",
    )
    kafka_offset: int = Field(
        ...,
        ge=0,
        description="Kafka offset within partition (idempotency key component).",
    )

    # Raw event data as base64 string (bytes never cross intents)
    envelope_bytes: str = Field(
        ...,
        min_length=1,
        description="Base64-encoded raw envelope bytes (required).",
    )
    envelope_hash: str = Field(
        ...,
        min_length=1,
        description="SHA-256 hex digest of the raw envelope bytes.",
    )

    correlation_id: UUID | None = Field(
        default=None,
        description="Correlation ID for distributed tracing (optional).",
    )


__all__ = [
    "ModelPayloadValidationLedgerAppend",
]
