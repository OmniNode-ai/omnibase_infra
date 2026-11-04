# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Inputs of one entry-point census run."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.lane_entry_point_census.model_lane_entry_point_policy import (
    ModelLaneEntryPointPolicy,
)


class ModelLaneEntryPointCensusRequest(BaseModel):
    """Where the policy, the lane manifest and the container inspection come from."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    operation: Literal["lane_entry_point_census.run"] = "lane_entry_point_census.run"
    policy_path: str = Field(
        default="", description="The entry-point policy data file for a host-local run."
    )
    policy: ModelLaneEntryPointPolicy | None = Field(
        default=None, description="DI policy overlay carried with a bus command."
    )
    manifest_path: str = Field(
        default="",
        description="Lane manifest; empty names each lane by its compose project.",
    )
    inspect_json_path: str = Field(
        default="",
        description="A recorded 'docker inspect' array; empty reads the docker daemon.",
    )
    inspection: tuple[dict[str, object], ...] | None = Field(
        default=None,
        description="Inspection carried over the bus; requires no host filesystem or Docker client in the runtime.",
    )
    docker_host: str = Field(
        default="",
        description="Daemon address for the live read; empty uses the local daemon.",
    )
    docker_timeout_seconds: int | None = Field(
        default=None,
        gt=0,
        description="Time allowed to each read-only docker call of a live read; unset waits.",
    )


__all__: list[str] = ["ModelLaneEntryPointCensusRequest"]
