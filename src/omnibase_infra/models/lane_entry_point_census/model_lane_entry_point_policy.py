# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The declared entry-point policy a census run grades against."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.lane_entry_point_census.model_entry_point_family import (
    ModelEntryPointFamily,
)


class ModelLaneEntryPointPolicy(BaseModel):
    """Which containers are runtime-image containers and which commands they may run.

    Declared in data (``deploy/lane-census/runtime-entry-points.yaml``); the
    handler holds none of these values.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    schema_version: str = Field(..., min_length=1)
    runtime_image_patterns: tuple[str, ...] = Field(
        ...,
        min_length=1,
        description="Regexes on the image reference of a runtime-image container.",
    )
    launcher_prefix: tuple[str, ...] = Field(
        default=(),
        description="Wrapper tokens the image entrypoint places before the command; stripped before grading.",
    )
    launcher_marks_runtime_image: bool = Field(
        default=False,
        description="A container whose entrypoint starts with the launcher prefix is a runtime-image container whatever its image reference reads.",
    )
    allowed_commands: tuple[str, ...] = Field(
        ..., min_length=1, description="Entry points a runtime-image container may run."
    )
    entry_point_families: tuple[ModelEntryPointFamily, ...] = Field(
        default=(), description="Entry-point labels the table reports as one row."
    )


__all__: list[str] = ["ModelLaneEntryPointPolicy"]
