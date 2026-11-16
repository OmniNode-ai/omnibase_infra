# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""An agent unit whose launcher would start it without required settings (OMN-24677)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ModelUnitSettingsGap(BaseModel):
    """The unit, the env files its launcher sources, and the settings none assigns."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    unit: str
    env_files: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
