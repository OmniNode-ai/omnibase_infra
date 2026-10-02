# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""An attributable workspace runtime config available to the resolver."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelMaterializedWorkspaceRuntimeConfig(BaseModel):
    """Describe the materialized config, its provenance and its freshness."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    contracts_dir: Path = Field(description="Contracts directory containing runtime/.")
    config_path: Path = Field(description="Materialized runtime_config.yaml path.")
    sha: str = Field(
        pattern=r"^[0-9a-fA-F]{40}$", description="Full source commit SHA."
    )
    materialized_at: datetime = Field(description="Time of the successful check, UTC.")
    stale: bool = Field(description="Whether the copy exceeds the freshness window.")

    @field_validator("materialized_at")
    @classmethod
    def validate_materialized_at(cls, value: datetime) -> datetime:
        """Require a timezone-aware check time and normalize it to UTC."""
        if value.utcoffset() is None:
            raise ValueError("materialized_at must be timezone-aware")
        return value.astimezone(UTC)
