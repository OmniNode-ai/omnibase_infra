# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Sibling-clone pin request model — inputs for the pin resolution handler.

Input model for the ``node_sibling_clone_pin_effect`` EFFECT node. The handler
reads the package's ``pyproject.toml`` and runs ``git`` in the given repo root,
so the request carries the package name and the filesystem path to the clone.

Ticket: OMN-23947
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ModelSiblingClonePinRequest(BaseModel):
    """Inputs for the sibling-clone pin resolution handler.

    Attributes:
        package: The package name (used in error messages).
        repo_root: Filesystem path to the sibling clone's repository root.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    package: str = Field(
        ...,
        min_length=1,
        description="Package name (used in error messages).",
    )
    repo_root: Path = Field(
        ...,
        description="Filesystem path to the sibling clone's repository root.",
    )


__all__: list[str] = ["ModelSiblingClonePinRequest"]
