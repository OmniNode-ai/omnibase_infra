"""Ledger entry model for event ledger storage.

This module defines the data structure for a single event ledger entry,
representing one row in the event_ledger table.
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.types import JsonType


class ModelLedgerEntry(BaseModel):
    """Represents a single event ledger entry (one row in event_ledger table).

    This model captures the complete state of an event as it was received
    from Kafka, including the raw payload, Kafka position metadata, and
    extracted envelope fields for queryability.

    All extracted metadata fields are nullable because:
    1. Events may not conform to the ONEX envelope schema
    2. Malformed events should still be ledgered for debugging
    3. Legacy events may lack certain fields
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    ledger_entry_id: UUID = Field(
        ...,
        description="Unique identifier for this ledger entry",
    )

    # Kafka position - uniquely identifies the event in the stream
    topic: str = Field(
        ...,
        min_length=1,
        description="Kafka topic from which the event was consumed",
    )
    partition: int = Field(
        ...,
        ge=0,
        description="Kafka partition number",
    )
    kafka_offset: int = Field(
        ...,
        ge=0,
        description="Kafka offset within the partition",
    )

    # Raw event data (base64 encoded for transport)
    event_key: str | None = Field(
        default=None,
        description="Base64-encoded Kafka message key, if present",
    )
    event_value: str = Field(
        ...,
        description="Base64-encoded Kafka message value (the event payload)",
    )
    onex_headers: dict[str, JsonType] = Field(
        default_factory=dict,
        description="ONEX-specific headers extracted from Kafka headers",
    )

    # Extracted metadata (ALL NULLABLE for schema flexibility)
    envelope_id: UUID | None = Field(
        default=None,
        description="Extracted envelope ID from the event, if present",
    )
    correlation_id: UUID | None = Field(
        default=None,
        description="Extracted correlation ID for request tracing",
    )
    event_type: str | None = Field(
        default=None,
        description="Extracted event type identifier",
    )
    source: str | None = Field(
        default=None,
        description="Extracted source system or service identifier",
    )

    # Timestamps
    event_timestamp: datetime | None = Field(
        default=None,
        description="Original event timestamp from the envelope, if present",
    )
    ledger_written_at: datetime = Field(
        ...,
        description="Timestamp when this entry was written to the ledger",
    )
