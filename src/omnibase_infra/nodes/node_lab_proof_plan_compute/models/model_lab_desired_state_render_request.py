# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Immutable source artifacts for a desired-state render (OMN-19413)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_lab_proof_plan_compute.models.model_lab_target_ref import (
    ModelLabTargetRef,
)


class ModelLabDesiredStateRenderRequest(BaseModel):
    """Recorded inputs; no observed Docker state or timestamps become desired state.

    The adapter reads manifest/lock/fleet at target_ref.commit and obtains both
    compose artifacts with the deploy's file order, project, environment and
    working directory. broker_yaml is the mounted bootstrap profile at that ref;
    explicit one-shot cluster-config declarations in compose override it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    operation: Literal["lab_desired_state.render"] = "lab_desired_state.render"
    target_ref: ModelLabTargetRef
    host: str = Field(min_length=1)
    lane: str = Field(min_length=1)
    surface_kind: Literal["compose_lane", "runner_fleet", "host"] = "compose_lane"
    manifest_yaml: str
    lock_toml: str
    compose_json: str
    compose_hashes: str
    broker_yaml: str = "{}"
    fleet_yaml: str = "{}"
    runner_host_address: str | None = None
    runner_workdir: str | None = None
