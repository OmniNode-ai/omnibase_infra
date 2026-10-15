# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""What one observation round read from one surface."""

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.models.config_overlay.model_declared_state_observation import (
    ModelDeclaredStateObservation,
)


class ModelDeclaredStateSnapshot(BaseModel):
    """The surface generation and the per-item observations read at it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    generation: str = Field(min_length=1)
    observations: tuple[ModelDeclaredStateObservation, ...]


__all__ = ["ModelDeclaredStateSnapshot"]
