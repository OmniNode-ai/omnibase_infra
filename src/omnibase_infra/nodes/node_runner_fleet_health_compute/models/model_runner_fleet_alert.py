# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed runner-fleet health alert (OMN-17427)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_runner_fleet_health_compute.models.enum_runner_fleet_alert_kind import (
    EnumRunnerFleetAlertKind,
)


class ModelRunnerFleetAlert(BaseModel):
    """Observed runner or queue condition requiring operator attention."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    kind: EnumRunnerFleetAlertKind = Field(..., description="Alert condition.")
    subject: str = Field(
        ..., description="Runner name for stuck/offline alerts, repo for queue alerts."
    )
    repo: str = Field(default="", description="org/repo associated with the alert.")
    labels: tuple[str, ...] = Field(default=(), description="Requested runner labels.")
    workflow: str = Field(default="", description="Workflow display name.")
    job: str = Field(default="", description="Job display name.")
    run_id: int | None = Field(default=None, description="Associated workflow run ID.")
    observed_seconds: float | None = Field(
        default=None, description="Observed job or queue age in seconds."
    )
    threshold_seconds: float | None = Field(
        default=None, description="Age threshold that triggered the alert."
    )
    queued_count: int = Field(
        default=0, description="Queued jobs in the repo/labels group."
    )
    detail: str = Field(default="", description="Supporting observation details.")


__all__ = ["ModelRunnerFleetAlert"]
