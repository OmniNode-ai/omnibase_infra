# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One deploy-agent instance as the routing key check reads it (OMN-26485)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ModelDeployLaneInstance(BaseModel):
    """An instance of ``config/deploy_lane_routing.yaml``: only its availability is read.

    The table's other fields (hostnames, consumer group, flags, lane and verify blocks)
    belong to the deploy agent and the CI resolver and are ignored here.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    available: bool = True
