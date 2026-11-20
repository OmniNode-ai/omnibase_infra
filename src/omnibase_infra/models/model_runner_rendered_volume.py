# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One long-form volume of docker compose config."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from omnibase_infra.enums.enum_runner_mount_type import EnumRunnerMountType


class ModelRunnerRenderedVolume(BaseModel):
    """One long-form volume of ``docker compose config --format json``."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    type: EnumRunnerMountType
    source: str = ""
    target: str
    read_only: bool = False
