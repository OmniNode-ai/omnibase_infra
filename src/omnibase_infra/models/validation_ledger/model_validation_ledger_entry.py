# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Validation ledger entry model for cross-repo validation event storage.

This module defines the data structure for a single validation event ledger entry,
representing one row in the validation_event_ledger table.

Design Rationale:
    envelope_bytes is stored as BYTEA in PostgreSQL but transported as a
    base64-encoded string in the model. This follows the established pattern
    from ModelLedgerEntry where bytes never cross intent/transport boundaries.
    Consumers decode the base64 string to recover the exact original bytes
    for deterministic replay.

Ticket: OMN-1908
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelValidationLedgerEntry(BaseModel):
    """Represents a single validation event ledger entry.

    Maps to one row in the validation_event_ledger table. Contains
    domain-specific fields (run_id, repo_id, event_type) for efficient
    querying, plus the raw envelope bytes for deterministic replay.

    Attributes:
        id: Unique identifier for this ledger entry.
        run_id: Validation run identifier linking started/violations/completed.
        repo_id: Human-readable repository identifier.
        event_type: Event type discriminator.
        event_version: Event schema version.
        occurred_at: When the validation event occurred.
        kafka_topic: Kafka topic from which the event was consumed.
        kafka_partition: Kafka partition number.
        kafka_offset: Kafka offset within partition.
        envelope_bytes: Base64-encoded raw Kafka envelope for replay.
        envelope_hash: SHA-256 hex digest of envelope_bytes.
        created_at: When this entry was written to the ledger.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    id: UUID = Field(
        ...,
        description="Unique identifier for this ledger entry",
    )
    run_id: UUID = Field(
        ...,
        description="Validation run identifier linking started/violations/completed events",
    )
    repo_id: str = Field(
        ...,
        min_length=1,
        description="Human-readable repository identifier",
    )
    event_type: str = Field(
        ...,
        min_length=1,
        description="Event type discriminator",
    )
    event_version: str = Field(
        ...,
        min_length=1,
        description="Event schema version",
    )
    occurred_at: datetime = Field(
        ...,
        description="When the validation event occurred",
    )
    kafka_topic: str = Field(
        ...,
        min_length=1,
        description="Kafka topic from which the event was consumed",
    )
    kafka_partition: int = Field(
        ...,
        ge=0,
        description="Kafka partition number",
    )
    kafka_offset: int = Field(
        ...,
        ge=0,
        description="Kafka offset within partition",
    )
    envelope_bytes: str = Field(
        ...,
        min_length=1,
        description="Base64-encoded raw Kafka envelope for deterministic replay",
    )
    envelope_hash: str = Field(
        ...,
        min_length=1,
        description="SHA-256 hex digest of envelope_bytes for integrity verification",
    )
    created_at: datetime = Field(
        ...,
        description="When this entry was written to the ledger",
    )


__all__ = ["ModelValidationLedgerEntry"]
