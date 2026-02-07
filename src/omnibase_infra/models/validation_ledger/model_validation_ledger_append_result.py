# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Validation ledger append result model.

This module defines the result returned after appending a validation event
to the ledger, including success status, the created entry ID, and
duplicate detection.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelValidationLedgerAppendResult(BaseModel):
    """Result of appending a validation event to the ledger."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    success: bool = Field(..., description="Whether append completed without error")
    ledger_entry_id: UUID | None = Field(
        default=None, description="ID of created entry, None if duplicate"
    )
    duplicate: bool = Field(default=False, description="True if ON CONFLICT triggered")
    kafka_topic: str = Field(..., description="Topic of the appended event")
    kafka_partition: int = Field(
        ..., ge=0, description="Partition of the appended event"
    )
    kafka_offset: int = Field(..., ge=0, description="Offset of the appended event")
