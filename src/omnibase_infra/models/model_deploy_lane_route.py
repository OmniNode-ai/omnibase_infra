# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One row of the deploy lane routing table (OMN-26485)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ModelDeployLaneRoute(BaseModel):
    """A ``(runtime_lane, requester_repository) -> instance`` row."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    runtime_lane: str
    requester_repository: str
    instance: str
