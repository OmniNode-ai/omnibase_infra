# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Outcome of checking and materializing a workspace's runtime config."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelWorkspaceRuntimeConfigMaterialization(BaseModel):
    """Report success or a one-line reason the config could not be refreshed."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    ok: bool = Field(description="Whether the runtime config was refreshed.")
    sha: str = Field(default="", description="Source commit on a successful check.")
    detail: str = Field(default="", description="One-line explanation of failure.")
