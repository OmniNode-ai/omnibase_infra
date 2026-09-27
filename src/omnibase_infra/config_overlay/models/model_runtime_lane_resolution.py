# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The lane a runtime resolved at startup, with its provenance (OMN-19747)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.enums.enum_config_overlay_source import EnumConfigOverlaySource
from omnibase_core.models.config_overlay import (
    ModelConfigOverlayScope,
    ModelRuntimeLaneDeclaration,
)

__all__ = ["ModelRuntimeLaneResolution"]


class ModelRuntimeLaneResolution(BaseModel):
    """A resolved ``runtime.lane`` declaration and where it was read from."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    declaration: ModelRuntimeLaneDeclaration = Field(
        ..., description="The lane and its roles, as the overlay declares them."
    )
    scope: ModelConfigOverlayScope = Field(
        ..., description="The (environment, lane) scope the document was read at."
    )
    source: EnumConfigOverlaySource = Field(
        ..., description="The one overlay source this deployment reads."
    )
    location: str = Field(
        ..., min_length=1, description="The exact place the document was read from."
    )
    sha256: str = Field(
        ...,
        pattern=r"^[0-9a-f]{64}$",
        description="sha256 of the document bytes as read, for replay.",
    )

    def log_line(self) -> str:
        """One line naming lane, roles, source and sha256 for the startup log."""
        roles = ",".join(role.value for role in self.declaration.roles)
        return (
            f"lane={self.declaration.lane_id} roles={roles} "
            f"source={self.source.value} sha256={self.sha256} "
            f"environment={self.scope.environment} location={self.location}"
        )
