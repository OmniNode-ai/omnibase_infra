# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Ledger append payload model for audit ledger reducer.

This payload implements ProtocolIntentPayload for use with ModelIntent.
It captures Kafka events for append-only audit ledger storage with
metadata extraction best-effort (nullable metadata ensures events are
never dropped due to parsing failures).

Design Rationale - Nullable Metadata:
    The audit ledger serves as the system's source of truth. Events must NEVER
    be dropped due to metadata extraction failures. By making all metadata fields
    nullable, we guarantee:
    1. Malformed events are captured with raw data intact
    2. Parsing errors don't cause event loss
    3. Metadata can be re-extracted later from raw event_value
    4. The ledger remains complete even during schema evolution

Bytes Encoding:
    Kafka event keys and values are bytes. Since bytes cannot safely cross
    intent boundaries (serialization issues), they are base64-encoded at the
    boundary. The Effect layer decodes before storage.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.types import JsonType

# NOTE: ModelIntentPayloadBase was removed in omnibase_core 0.6.2
# Using pydantic.BaseModel directly as the base class


class ModelPayloadLedgerAppend(BaseModel):
    """Payload for audit ledger append intents.

    This payload follows the ONEX intent payload pattern for use with ModelIntent.
    All metadata fields are intentionally OPTIONAL to ensure the audit ledger
    never drops events due to metadata extraction failures.

    Attributes:
        intent_type: Discriminator literal for intent routing. Always "ledger.append".
        topic: Kafka topic the event originated from.
        partition: Kafka partition number (idempotency key component).
        kafka_offset: Kafka offset within partition (idempotency key component).
        event_key: Base64-encoded event key bytes (None for keyless events).
        event_value: Base64-encoded event value bytes (required - the raw event).
        onex_headers: Extracted ONEX headers from Kafka message headers.
        correlation_id: Correlation ID for distributed tracing (extracted, nullable).
        envelope_id: Unique envelope identifier (extracted, nullable).
        event_type: Event type discriminator (extracted, nullable).
        source: Event source identifier (extracted, nullable).
        event_timestamp: Event timestamp (extracted, nullable).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    intent_type: Literal["ledger.append"] = Field(
        default="ledger.append",
        description="Discriminator literal for intent routing.",
    )

    # Kafka position - required for idempotency
    topic: str = Field(
        ...,
        min_length=1,
        description="Kafka topic the event originated from.",
    )

    partition: int = Field(
        ...,
        ge=0,
        description="Kafka partition number (idempotency key component).",
    )

    kafka_offset: int = Field(
        ...,
        ge=0,
        description="Kafka offset within partition (idempotency key component).",
    )

    # Raw event data as base64 strings (bytes never cross intents)
    event_key: str | None = Field(
        default=None,
        description="Base64-encoded event key bytes (None for keyless events).",
    )

    event_value: str = Field(
        ...,
        min_length=1,
        description="Base64-encoded event value bytes (required - the raw event).",
    )

    onex_headers: dict[str, JsonType] = Field(
        default_factory=dict,
        description="Extracted ONEX headers from Kafka message headers.",
    )

    # Extracted metadata - ALL OPTIONAL (audit ledger must never drop events)
    correlation_id: UUID | None = Field(
        default=None,
        description="Correlation ID for distributed tracing (extracted, nullable).",
    )

    envelope_id: UUID | None = Field(
        default=None,
        description="Unique envelope identifier (extracted, nullable).",
    )

    event_type: str | None = Field(
        default=None,
        description="Event type discriminator (extracted, nullable).",
    )

    source: str | None = Field(
        default=None,
        description="Event source identifier (extracted, nullable).",
    )

    event_timestamp: datetime | None = Field(
        default=None,
        description="Event timestamp (extracted, nullable).",
    )


__all__ = [
    "ModelPayloadLedgerAppend",
]
