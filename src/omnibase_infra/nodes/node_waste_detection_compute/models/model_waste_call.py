# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Normalized LLM call telemetry model for waste analyzers."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelWasteCall(BaseModel):
    """Normalized LLM call telemetry used by waste analyzers."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    session_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    cost_usd: float = Field(ge=0.0)
    request_type: str = Field(min_length=1)
    emitted_at: datetime
    correlation_id: str | None = None
    endpoint_url: str | None = None
    repo_name: str | None = None
    machine_id: str | None = None
    tool_name: str | None = None
    tool_input_hash: str | None = None
    status: str | None = None
    error_type: str | None = None
    cache_read_tokens: int | None = Field(default=None, ge=0)
    cache_creation_tokens: int | None = Field(default=None, ge=0)

    @field_validator("emitted_at")
    @classmethod
    def validate_emitted_at_tz_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("emitted_at must be timezone-aware")
        return value

    @property
    def request_signature(self) -> str:
        """Stable signature used by retry and loop analyzers."""
        parts = (
            self.tool_name or self.request_type,
            self.tool_input_hash or self.correlation_id or self.model_id,
        )
        return ":".join(parts)


__all__ = ["ModelWasteCall"]
