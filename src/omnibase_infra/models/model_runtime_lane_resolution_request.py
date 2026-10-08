# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Bootstrap input for runtime lane resolution (OMN-19747)."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ModelRuntimeLaneResolutionRequest(BaseModel):
    """Use process bootstrap facts, or explicit facts for an isolated caller."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)
    environ: dict[str, str] | None = Field(default=None, repr=False)
    home: Path | None = None
