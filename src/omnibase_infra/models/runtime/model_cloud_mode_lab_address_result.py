# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Result of the cloud-mode lab-address guard (OMN-26709)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ModelCloudModeLabAddressResult(BaseModel):
    """The guard passed; ``cloud_mode`` says whether the check applied."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    cloud_mode: bool
