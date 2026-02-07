# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Validation ledger entry model.

This module defines the data structure for a single validation event ledger
entry, representing one row in the validation_event_ledger table.
"""

from __future__ import annotations

import base64
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelValidationLedgerEntry(BaseModel):
    """Single row in the validation_event_ledger table.

    Unlike the generic event_ledger, all metadata fields are required (NOT NULL)
    because validation events have a well-defined schema. envelope_bytes is
    stored as base64-encoded string for transport safety (BYTEA in database).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    id: UUID = Field(..., description="Primary key")
    run_id: UUID = Field(..., description="Validation run correlation ID")
    repo_id: str = Field(..., min_length=1, description="Repository being validated")
    event_type: str = Field(
        ...,
        min_length=1,
        description="Event type (e.g., onex.evt.validation.cross-repo-run-started.v1)",
    )
    event_version: str = Field(..., min_length=1, description="Event schema version")
    occurred_at: datetime = Field(
        ..., description="When the validation event occurred (UTC)"
    )
    kafka_topic: str = Field(..., min_length=1, description="Kafka topic")
    kafka_partition: int = Field(..., ge=0, description="Kafka partition")
    kafka_offset: int = Field(..., ge=0, description="Kafka offset")
    envelope_bytes: str = Field(
        ...,
        description="Base64-encoded raw envelope bytes for deterministic replay",
    )
    envelope_hash: str = Field(
        ...,
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
        description="SHA-256 hex digest of the raw envelope bytes (before base64 encoding)",
    )
    created_at: datetime = Field(..., description="When this ledger entry was created")

    @field_validator("envelope_bytes")
    @classmethod
    def validate_base64(cls, v: str) -> str:
        """Validate that envelope_bytes is valid base64-encoded data.

        PostgreSQL ``encode(bytea, 'base64')`` inserts newlines every 76
        characters per RFC 2045. We strip whitespace before strict validation
        so that PG-produced base64 passes the check.

        Args:
            v: Base64-encoded string, potentially containing newlines
                and spaces from PostgreSQL encoding.

        Returns:
            The original string unchanged. Whitespace is stripped only
            for validation; the stored value preserves PostgreSQL formatting.

        Raises:
            ValueError: If the string is not valid base64.
        """
        try:
            stripped = v.translate({10: None, 13: None, 32: None})
            base64.b64decode(stripped, validate=True)
        except Exception as exc:
            raise ValueError(
                "envelope_bytes must be valid base64-encoded data"
            ) from exc
        return v
