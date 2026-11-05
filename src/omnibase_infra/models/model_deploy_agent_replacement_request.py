# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A host-local replacement requested by a deploy-agent bus command."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class ModelDeployAgentReplacementRequest(BaseModel):
    """Preflight one lane, or gate and supervise one compose recreate."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    operation: Literal["preflight", "recreate"] = "preflight"
    lane: str
    compose_project: str
    argv: tuple[str, ...] = ()
    services: tuple[str, ...] = ()
    timeout: int | None = None
    env: dict[str, str] | None = None
