"""Ledger append result model for write operation outcomes.

This module defines the result structure returned after attempting
to append an event to the ledger.
"""

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelLedgerAppendResult(BaseModel):
    """Result of a ledger append operation.

    This model captures the outcome of attempting to write an event
    to the ledger, including handling of duplicate detection via
    the (topic, partition, kafka_offset) unique constraint.

    The duplicate flag indicates when ON CONFLICT DO NOTHING was
    triggered, meaning the event was already in the ledger. This
    is not an error condition - it enables idempotent replay.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    success: bool = Field(
        ...,
        description="Whether the append operation completed without error",
    )
    ledger_entry_id: UUID | None = Field(
        default=None,
        description="ID of the created entry, None if duplicate",
    )
    duplicate: bool = Field(
        default=False,
        description="True if ON CONFLICT DO NOTHING matched existing entry",
    )

    # Kafka position that was attempted
    topic: str = Field(
        ...,
        min_length=1,
        description="Kafka topic of the event",
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
