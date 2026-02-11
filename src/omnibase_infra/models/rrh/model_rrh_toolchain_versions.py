# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Build-tool version snapshot for RRH validation."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelRRHToolchainVersions(BaseModel):
    """Build-tool version snapshot.

    Version strings are empty when the tool is not detected.

    Attributes:
        pre_commit: pre-commit version string.
        ruff: ruff version string.
        pytest: pytest version string.
        mypy: mypy version string.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    pre_commit: str = Field(default="", description="pre-commit version.")
    ruff: str = Field(default="", description="ruff version.")
    pytest: str = Field(default="", description="pytest version.")
    mypy: str = Field(default="", description="mypy version.")


__all__: list[str] = ["ModelRRHToolchainVersions"]
