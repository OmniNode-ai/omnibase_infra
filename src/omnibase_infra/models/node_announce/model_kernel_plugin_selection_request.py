# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Request to select the kernel plugins a lane enables from the node catalog (OMN-21591)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.node_announce.model_package_announce_result import (
    ModelPackageAnnounceResult,
)


class ModelKernelPluginSelectionRequest(BaseModel):
    """The catalog boot step 0 built and the plugin names the lane overlay enables."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    announced: ModelPackageAnnounceResult | None = Field(
        description="Result of boot step 0; None when step 0 failed and built no catalog"
    )
    enabled: tuple[str, ...] = Field(
        description="Names from the lane overlay's runtime.kernel_plugins, in order"
    )
