# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Semantic version model extracted from contract YAML (OMN-7653)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelContractVersion(BaseModel):
    """Semantic version extracted from contract YAML."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    major: int = Field(..., description="Major version")
    minor: int = Field(..., description="Minor version")
    patch: int = Field(..., description="Patch version")

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"
