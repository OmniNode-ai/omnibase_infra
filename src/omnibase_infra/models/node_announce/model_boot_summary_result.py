# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Result of writing the boot summary line (OMN-26561)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelBootSummaryResult(BaseModel):
    """The line that was written."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    line: str = Field(description="The boot summary line, as logged")
