# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One start-first replacement of a dev runtime container (OMN-22151)."""

from pydantic import BaseModel, ConfigDict, Field


class ModelRuntimeStartFirstRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    service: str = Field(min_length=1)
    old_container: str = Field(min_length=1)
    ready_url: str = Field(min_length=1)
    bound_seconds: float = Field(gt=0)
    poll_seconds: float = Field(default=5, gt=0)
