# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runner TMPDIR ownership policy."""

from pydantic import BaseModel, ConfigDict, Field


class ModelRunnerTmpPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    uid: int = Field(default=1001, ge=0)
    mode: str = "1777"
    timeout_seconds: int = Field(default=60, gt=0)
