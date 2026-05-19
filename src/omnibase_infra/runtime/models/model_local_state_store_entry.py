# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed entry model for the local runtime state store (OMN-8701)."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelLocalStateStoreEntry(BaseModel):
    """A single typed entry in the local state store.

    Attributes:
        key: Unique string key for this entry.
        value: Stored payload (JSON-serialisable dict).
        stored_at: Timestamp when the entry was written.
        correlation_id: Optional tracing correlation ID for the write.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str = Field(..., description="Unique key for this state entry")
    value: dict[str, object] = Field(default_factory=dict, description="State payload")
    stored_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when the entry was written",
    )
    correlation_id: UUID | None = Field(
        default=None,
        description="Optional correlation ID of the invocation that wrote this entry",
    )


__all__ = ["ModelLocalStateStoreEntry"]
