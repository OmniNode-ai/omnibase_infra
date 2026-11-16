# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Sibling-clone pin result model — output of the pin resolution handler.

Output model for ``node_sibling_clone_pin_effect``. Carries the resolved version,
whether it was derived from git, and the git HEAD SHA.

Ticket: OMN-23947
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelSiblingClonePinResult(BaseModel):
    """Output model for the sibling-clone pin resolution handler.

    Attributes:
        package: The package name.
        version: The resolved version string.
        version_derived: True when the version was derived from git (dynamic
            versioning via ``git describe``), False when read from
            ``pyproject.toml``.
        git_sha: The git HEAD SHA of the clone.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    package: str = Field(
        ...,
        min_length=1,
        description="Package name.",
    )
    version: str = Field(
        ...,
        min_length=1,
        description="Resolved version string.",
    )
    version_derived: bool = Field(
        ...,
        description="True when the version was derived from git describe.",
    )
    git_sha: str = Field(
        ...,
        min_length=1,
        description="Git HEAD SHA of the clone.",
    )


__all__: list[str] = ["ModelSiblingClonePinResult"]
