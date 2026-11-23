# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The routing table and the key checks it is judged against (OMN-26485)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.model_deploy_lane_routing_table import (
    ModelDeployLaneRoutingTable,
)


class ModelRoutingKeyCheckRequest(BaseModel):
    """What the key check reads.

    ``instance_key_checks`` maps an instance name to the ``hmac_key_check`` its agent's
    health answer carried (an absent name or an empty value is unread).
    ``runtime_key_check`` is the lane runtime's own. When ``runtime_lane`` is set the
    result also carries the routing decision for a command of that lane requested by
    ``requested_by`` (``gha/<repo>/...``).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    table: ModelDeployLaneRoutingTable
    runtime_key_check: str = Field(min_length=1)
    instance_key_checks: dict[str, str] = Field(default_factory=dict)
    runtime_lane: str = ""
    requested_by: str = ""
