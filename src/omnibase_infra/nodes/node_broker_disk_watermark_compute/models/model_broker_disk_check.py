# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""ModelBrokerDiskCheck — result for a single probed path.

Ticket: OMN-13009
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_broker_disk_watermark_compute.models.enum_disk_severity import (
    EnumDiskSeverity,
)


class ModelBrokerDiskCheck(BaseModel):
    """Disk probe result for a single path (data-root or broker volume).

    ``path`` is the filesystem path that was probed (e.g. ``/data`` for the
    docker data-root, or the volume mount point for a named broker volume).

    ``label`` is a human-readable identifier used in alert text (e.g.
    ``"docker-data-root"``, ``"redpanda-stability-test"``).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str = Field(
        ..., description="Human-readable identifier for the probed target."
    )
    path: str = Field(..., description="Filesystem path that was probed.")
    total_bytes: int = Field(
        ..., description="Total capacity of the filesystem in bytes."
    )
    used_bytes: int = Field(..., description="Bytes used on the filesystem.")
    free_bytes: int = Field(..., description="Bytes free on the filesystem.")
    usage_pct: float = Field(
        ...,
        description="Fraction used: used_bytes / total_bytes (0.0-1.0).",
    )
    severity: EnumDiskSeverity = Field(
        ...,
        description="CLEAN / WARN / P0 classification against watermarks.",
    )
    min_free_bytes_floor: int = Field(
        default=10_485_760,
        description="Redpanda storage_min_free_bytes floor (default 10 MiB). "
        "When free_bytes <= this value the probe is classified P0 regardless of usage_pct.",
    )
    below_min_free_floor: bool = Field(
        default=False,
        description="True when free_bytes <= min_free_bytes_floor. "
        "Independently escalates severity to P0.",
    )


__all__ = ["ModelBrokerDiskCheck"]
