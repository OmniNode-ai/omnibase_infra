# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""A freshness-bounded lab-load reading.

Ticket: OMN-18412
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_ci_runner_route_compute.models.model_ci_runner_lab_host import (
    ModelCIRunnerLabHost,
)


class ModelCIRunnerLabObservation(BaseModel):
    """A freshness-bounded lab-load reading.

    The route job runs on GitHub-hosted compute and cannot reach the lab, so
    this reading arrives asynchronously and carries its own age. Stale is
    unknown, and unknown routes hosted.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    ok: bool = Field(description="Whether a usable lab record was read.")
    error: str = Field(
        default="",
        description="Named failure class when ok is false (no_record, "
        "unreadable_record, stale, no_hosts, malformed_host).",
    )
    age_seconds: int | None = Field(
        default=None, ge=0, description="Age of the reading at decision time."
    )
    hosts: tuple[ModelCIRunnerLabHost, ...] = Field(
        default=(), description="Per-host readings."
    )


__all__ = ["ModelCIRunnerLabObservation"]
