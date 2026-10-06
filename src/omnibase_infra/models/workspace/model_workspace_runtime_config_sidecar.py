# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Source attribution for a materialized workspace runtime config."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelWorkspaceRuntimeConfigSidecar(BaseModel):
    """Record the git source and last successful check in UTC."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    source_repository: str = Field(
        description="Absolute path of the owning config repository."
    )
    source_ref: str = Field(description="Git reference carrying the runtime config.")
    source_path: str = Field(description="Runtime config path inside the repository.")
    sha: str = Field(
        pattern=r"^[0-9a-fA-F]{40}$", description="Full source commit SHA."
    )
    materialized_at: datetime = Field(description="Time of the successful check, UTC.")

    @field_validator("materialized_at")
    @classmethod
    def validate_materialized_at(cls, value: datetime) -> datetime:
        """Require a timezone-aware check time and normalize it to UTC."""
        if value.utcoffset() is None:
            raise ValueError("materialized_at must be timezone-aware")
        return value.astimezone(UTC)
