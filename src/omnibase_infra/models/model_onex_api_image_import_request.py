# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""onex-api pin import check: what is asked of the candidate image (OMN-23039)."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ModelOnexApiImageImportRequest(BaseModel):
    """One candidate image, and the pin to restore if it cannot be delivered."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    image: str = Field(min_length=1, description="The candidate image reference.")
    env_file: Path = Field(description="The operator env file holding ONEX_API_IMAGE.")
    previous_pin: str = Field(
        description="The pin to write back; empty when the repoint recorded none."
    )
    timeout_seconds: int = Field(
        default=120, gt=0, description="Bound on the throwaway container."
    )
