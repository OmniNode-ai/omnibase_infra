# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Bootstrap input for runtime lane resolution (OMN-19747)."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ModelRuntimeLaneResolutionRequest(BaseModel):
    """The process environment the caller injects, and an optional home override."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)
    environ: dict[str, str] = Field(repr=False)
    home: Path | None = None
