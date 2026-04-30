# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Cost token-usage projection snapshot row model."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ModelCostTokenUsageSnapshotRow(BaseModel):
    """One model/time bucket in a token-usage snapshot."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
        populate_by_name=True,
        serialize_by_alias=True,
    )

    bucket_timestamp: datetime
    provider_model_key: str = Field(
        min_length=1,
        validation_alias="model_id",
        serialization_alias="model_id",
    )
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    call_count: int = Field(ge=0)

    @property
    def model_id(self) -> str:
        """Backward-compatible accessor for the wire field name."""
        return self.provider_model_key

    @field_validator("bucket_timestamp")
    @classmethod
    def validate_bucket_tz_aware(cls, value: datetime) -> datetime:
        """Require timezone-aware bucket timestamps."""
        if value.tzinfo is None:
            raise ValueError("bucket_timestamp must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_total_tokens(self) -> ModelCostTokenUsageSnapshotRow:
        """Token total must match prompt plus completion tokens."""
        if self.total_tokens != self.prompt_tokens + self.completion_tokens:
            raise ValueError(
                "total_tokens must equal prompt_tokens + completion_tokens"
            )
        return self


__all__ = ["ModelCostTokenUsageSnapshotRow"]
