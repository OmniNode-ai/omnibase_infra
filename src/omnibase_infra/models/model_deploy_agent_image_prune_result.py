# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Image cleanup plan and host installation readback (OMN-19998)."""

from pydantic import BaseModel, ConfigDict, Field


class ModelDeployAgentImagePruneResult(BaseModel):
    """Nominal image bytes are not a claim about reclaimed shared layers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    success: bool
    error_message: str = ""
    remove_image_ids: tuple[str, ...] = ()
    kept_reasons: dict[str, str] = Field(default_factory=dict)
    remove_nominal_bytes: int = 0
    cutoff_epoch: int = 0
    stdout: str = ""
    installed_revision: str = ""
