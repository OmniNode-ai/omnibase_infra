# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Validation ledger append result model for write operation outcomes.

This module defines the result structure returned after attempting
to append a validation event to the ledger.

Ticket: OMN-1908
"""

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelValidationLedgerAppendResult(BaseModel):
    """Result of a validation ledger append operation.

    Captures the outcome of attempting to write a validation event
    to the ledger, including duplicate detection via the
    (kafka_topic, kafka_partition, kafka_offset) unique constraint.

    The duplicate flag indicates when ON CONFLICT DO NOTHING was
    triggered, meaning the event was already in the ledger. This
    is not an error - it enables idempotent replay.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    success: bool = Field(
        ...,
        description="Whether the append operation completed without error",
    )
    entry_id: UUID | None = Field(
        default=None,
        description="ID of the created entry, None if duplicate",
    )
    duplicate: bool = Field(
        default=False,
        description="True if ON CONFLICT DO NOTHING matched existing entry",
    )
    kafka_topic: str = Field(
        ...,
        min_length=1,
        description="Kafka topic of the event",
    )
    kafka_partition: int = Field(
        ...,
        ge=0,
        description="Kafka partition number",
    )
    kafka_offset: int = Field(
        ...,
        ge=0,
        description="Kafka offset within the partition",
    )


__all__ = ["ModelValidationLedgerAppendResult"]
