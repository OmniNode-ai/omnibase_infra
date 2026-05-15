# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Result model for overlay environment injection."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelOverlayEnvInjectionResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    injected_keys: tuple[str, ...] = Field(default_factory=tuple)
    skipped_existing_keys: tuple[str, ...] = Field(default_factory=tuple)
