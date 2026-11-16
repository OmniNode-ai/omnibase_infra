# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""What the dev deploy agent hands the pin service guard before a build (OMN-23368)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.model_dev_lane_code_ref_pin import ModelDevLaneCodeRefPin


class ModelDevLanePinGuardRequest(BaseModel):
    """The deploy's service list, the compose it will run against and the agent's context."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    targets: tuple[str, ...]
    compose_files: tuple[str, ...] = Field(min_length=1)
    declared_services: frozenset[str]
    env: dict[str, str]
    agent_dir: str = Field(min_length=1)
    git_timeout_seconds: float = Field(gt=0)
    now: datetime
    pin: ModelDevLaneCodeRefPin | None = None
