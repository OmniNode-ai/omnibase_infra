# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed model for a single feature flag entry in a service contract YAML."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelFeatureFlag(BaseModel):
    """A single feature flag entry as declared in a service contract YAML."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(..., description="Flag name and env var key")
    env_var: str = Field(
        ..., description="Environment variable that controls this flag"
    )
    default: str = Field(
        ..., description="Default value as a string (e.g. 'false', 'true')"
    )
    description: str = Field(
        default="", description="Human-readable description of this flag"
    )
