# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Request for the cloud-mode lab-address guard (OMN-26709)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelCloudModeLabAddressRequest(BaseModel):
    """The process environment the guard judges, as sorted (name, value) pairs."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    environment: tuple[tuple[str, str], ...] = Field(default=())
