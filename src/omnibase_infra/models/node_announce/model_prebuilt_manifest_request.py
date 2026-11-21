# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Request to load the prebuilt node manifest (OMN-26361)."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ModelPrebuiltManifestRequest(BaseModel):
    """Read the prebuilt file at ``path`` and check its stamp against the installed packages."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: Path = Field(description="Where the image build wrote the prebuilt file")
    workload: str = Field(
        default="main", description="Announcement workload the booting runtime runs as"
    )
