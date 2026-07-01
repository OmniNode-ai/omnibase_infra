# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""ModelBrokerDiskWatermarkOutput — probe results.

Ticket: OMN-13009
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_broker_disk_watermark_compute.models.enum_disk_severity import (
    EnumDiskSeverity,
)
from omnibase_infra.nodes.node_broker_disk_watermark_compute.models.model_broker_disk_check import (
    ModelBrokerDiskCheck,
)


class ModelBrokerDiskWatermarkOutput(BaseModel):
    """Output of the broker disk watermark compute node.

    ``checks`` is one entry per probed path (data-root first, then each named
    broker volume). The overall severity is the maximum severity across all
    checks: if any single check is P0, ``max_severity`` is P0; if any is WARN,
    it is WARN; otherwise CLEAN.

    ``p0_labels`` and ``warn_labels`` are convenience fields containing the
    human-readable labels of breaching checks — ready to paste into a Linear
    ticket title or Slack alert.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID = Field(..., description="Trace correlation ID from input.")
    checks: tuple[ModelBrokerDiskCheck, ...] = Field(
        ...,
        description="Probe result per path, data-root first then broker volumes.",
    )
    max_severity: EnumDiskSeverity = Field(
        ...,
        description="Worst severity across all checks.",
    )
    p0_labels: tuple[str, ...] = Field(
        default=(),
        description="Labels of P0-classified checks (for alert text).",
    )
    warn_labels: tuple[str, ...] = Field(
        default=(),
        description="Labels of WARN-classified checks (for alert text).",
    )
    docker_root_dir: str = Field(
        default="",
        description="Docker Root Dir path resolved from docker info (empty if unavailable).",
    )
    probe_error: str = Field(
        default="",
        description="Non-empty when docker info or disk_usage failed; checks may be partial.",
    )


__all__ = ["ModelBrokerDiskWatermarkOutput"]
