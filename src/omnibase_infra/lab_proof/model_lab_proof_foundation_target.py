# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""The package a foundation_override profile installs over the runtime's pin.

Ticket: OMN-19572
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelLabProofFoundationTarget(BaseModel):
    """Which distribution and import package a foundation PR head replaces."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    distribution: str = Field(pattern=r"^[a-z][a-z0-9-]+$")
    import_package: str = Field(pattern=r"^[a-z][a-z0-9_]+$")
    source_root: str = Field(
        default="src",
        pattern=r"^[A-Za-z0-9_./-]+$",
        description="Directory in the repository that holds import_package.",
    )


__all__ = ["ModelLabProofFoundationTarget"]
