# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Durable full-day activity evidence for an application database."""

from __future__ import annotations

from datetime import datetime, timedelta

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator


class ModelApplicationDatabaseActivityEvidence(BaseModel):
    """Hash-addressed query/result evidence covering at least 24 hours."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    window_started_at: AwareDatetime
    window_ended_at: AwareDatetime
    query_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    result_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    query_source_key: str = Field(..., pattern=r"^[a-z][a-z0-9_]*$")
    result_source_key: str = Field(..., pattern=r"^[a-z][a-z0-9_]*$")
    observation_count: int = Field(..., ge=0)

    @model_validator(mode="after")
    def validate_full_day_window(self) -> ModelApplicationDatabaseActivityEvidence:
        """Require an ordered interval spanning a complete day."""
        started_at = datetime.fromisoformat(self.window_started_at.isoformat())
        ended_at = datetime.fromisoformat(self.window_ended_at.isoformat())
        if ended_at - started_at < timedelta(hours=24):
            raise ValueError("activity evidence window must span at least 24 hours")
        return self


__all__ = ["ModelApplicationDatabaseActivityEvidence"]
