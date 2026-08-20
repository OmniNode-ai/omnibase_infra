# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Server-verified identity of one live PostgreSQL backend connection."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelConnectionIdentity(BaseModel):
    """Bind evidence to the exact backend that produced it.

    Every field here must be populated from a live server-side readback
    (``current_database()``, ``pg_backend_pid()``, ``clock_timestamp()``) by
    the collector that queried the connection -- never typed by a caller.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    database: str = Field(..., min_length=1, max_length=200)
    backend_pid: int = Field(..., ge=1)
    collected_at: datetime

    @field_validator("collected_at")
    @classmethod
    def _timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("connection identity timestamp must be timezone-aware")
        return value


__all__ = ["ModelConnectionIdentity"]
