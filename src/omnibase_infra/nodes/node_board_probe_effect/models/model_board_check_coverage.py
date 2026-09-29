# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One board criterion's declared executable coverage."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnibase_infra.nodes.node_board_probe_effect.models.enum_board_check_surface_class import (
    EnumBoardCheckSurfaceClass,
)


class ModelBoardCheckCoverage(BaseModel):
    """Cloud/post-release coverage must explain its deferred surface."""

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)
    check: str = Field(min_length=1, alias="check_id")
    surface_class: EnumBoardCheckSurfaceClass
    criteria: tuple[str, ...] = Field(min_length=1)
    reason: str | None = None

    @model_validator(mode="after")
    def validate_reason(self) -> "ModelBoardCheckCoverage":
        deferred = self.surface_class in (
            EnumBoardCheckSurfaceClass.CLOUD,
            EnumBoardCheckSurfaceClass.POST_RELEASE,
        )
        if deferred and (not self.reason or not self.reason.strip()):
            raise ValueError("cloud or post_release coverage requires a reason")
        if not deferred and self.reason is not None:
            raise ValueError("reason is only allowed for cloud or post_release")
        if any(not criterion.strip() for criterion in self.criteria):
            raise ValueError("criteria must not contain empty names")
        return self
