# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Base model for ledger events.

This module provides the common envelope fields shared by all ledger events.
All ledger event types inherit from ModelLedgerEventBase.

Design Decisions:
    - Frozen models: Immutability for thread safety and hashability
    - Required envelope fields: Ensures traceability and deduplication
    - No raw SQL/params: Security constraint enforced by excluding such fields
    - Deterministic idempotency_key: Enables reconciliation on retries
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelLedgerEventBase(BaseModel):
    """Base class for all ledger events.

    All ledger events share common envelope fields for traceability,
    correlation, and idempotency. Event-specific fields are added in subclasses.

    Attributes:
        event_type: Discriminator for event type (e.g., "db.query.requested").
        event_id: Unique identifier for this specific event emission.
        correlation_id: Distributed tracing ID linking related operations.
        causation_id: ID of the event/action that caused this event (optional).
        idempotency_key: Deterministic key for deduplication on retries.
            Format: "{correlation_id}:{operation_name}:{event_type}"
        contract_id: Identifier of the contract (e.g., repository name).
        contract_fingerprint: SHA256 hash of canonical contract JSON.
        operation_name: Name of the operation being traced.
        emitted_at: Timestamp when the event was emitted.

    Security:
        This model explicitly excludes raw SQL and unredacted parameters.
        Subclasses must not add fields that expose sensitive data.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    event_type: str = Field(
        ...,
        min_length=1,
        description="Event type discriminator (e.g., 'db.query.requested').",
    )

    event_id: UUID = Field(
        ...,
        description="Unique identifier for this specific event emission.",
    )

    correlation_id: UUID = Field(
        ...,
        description="Distributed tracing ID linking related operations.",
    )

    causation_id: UUID | None = Field(
        default=None,
        description="ID of the event/action that caused this event.",
    )

    idempotency_key: str = Field(
        ...,
        min_length=1,
        description=(
            "Deterministic key for deduplication. "
            "Format: '{correlation_id}:{operation_name}:{event_type}'"
        ),
    )

    contract_id: str = Field(
        ...,
        min_length=1,
        description="Identifier of the contract (e.g., repository name).",
    )

    contract_fingerprint: str = Field(
        ...,
        min_length=1,
        description="SHA256 hash of canonical contract JSON.",
    )

    operation_name: str = Field(
        ...,
        min_length=1,
        description="Name of the operation being traced.",
    )

    emitted_at: datetime = Field(
        ...,
        description="Timestamp when the event was emitted.",
    )

    @classmethod
    def build_idempotency_key(
        cls,
        correlation_id: UUID,
        operation_name: str,
        event_type: str,
    ) -> str:
        """Build a deterministic idempotency key.

        Args:
            correlation_id: The correlation ID for this operation.
            operation_name: The operation name being executed.
            event_type: The event type (e.g., "db.query.requested").

        Returns:
            Idempotency key in format "{correlation_id}:{operation_name}:{event_type}".
        """
        return f"{correlation_id}:{operation_name}:{event_type}"


__all__ = ["ModelLedgerEventBase"]
