# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Rendered Compose volume entry."""

from pydantic import BaseModel, ConfigDict

from omnibase_infra.enums.enum_compose_mount_type import EnumComposeMountType


class ModelRunnerTmpVolume(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    type: EnumComposeMountType
    source: str = ""
    target: str
