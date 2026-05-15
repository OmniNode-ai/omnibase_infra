# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Result model for env injection after overlay resolution."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelOverlayEnvInjectionResult(BaseModel):
    """Records which keys were injected into os.environ and which were skipped."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    injected_keys: tuple[str, ...] = Field(
        default_factory=tuple, description="Keys that were written to os.environ."
    )
    skipped_existing_keys: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Keys skipped because they were already set in os.environ.",
    )


__all__ = ["ModelOverlayEnvInjectionResult"]
