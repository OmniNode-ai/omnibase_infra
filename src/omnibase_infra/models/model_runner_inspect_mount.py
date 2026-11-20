# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One entry of docker inspect Mounts."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelRunnerInspectMount(BaseModel):
    """One entry of ``docker inspect`` ``.Mounts``."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    kind: str = Field(alias="Type")
    name: str = Field(default="", alias="Name")
    source: str = Field(default="", alias="Source")
    target: str = Field(alias="Destination")
    writable: bool = Field(default=True, alias="RW")
