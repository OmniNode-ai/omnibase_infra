# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The typed census record of one lane's runtime-image containers."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.lane_entry_point_census.model_container_entry_point import (
    ModelContainerEntryPoint,
)


class ModelLaneEntryPointCensus(BaseModel):
    """Every runtime-image container of one lane, each with its command."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    lane: str = Field(
        ...,
        min_length=1,
        description="Lane name; the compose project when the manifest names none.",
    )
    compose_project: str = Field(default="")
    container_count: int = Field(..., ge=0)
    allowed_count: int = Field(..., ge=0)
    remove_count: int = Field(..., ge=0)
    remove: tuple[str, ...] = Field(
        default=(),
        description="Names of the containers whose entry point is outside the allowed list.",
    )
    containers: tuple[ModelContainerEntryPoint, ...] = Field(default=())


__all__: list[str] = ["ModelLaneEntryPointCensus"]
