# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""ModelBrokerDiskWatermarkInput — input for the watermark probe.

Ticket: OMN-13009
"""

from __future__ import annotations

from collections.abc import Callable
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class ModelBrokerDiskWatermarkInput(BaseModel):
    """Input for the broker disk watermark compute node.

    ``broker_volume_names`` is the list of Docker volume names to probe in
    addition to the docker data-root. Typical values per lane:
        - ``omnibase-infra-stability-test_redpanda-data``
        - ``omnibase-infra-prod_redpanda-data``
        - ``omnibase-infra-judge_redpanda-data``

    ``warn_threshold`` and ``p0_threshold`` are fractional usage values (0-1).
    The defaults (0.85 / 0.95) match the incident analysis in OMN-13009.

    ``min_free_bytes_floor`` cross-checks that remaining free space exceeds
    redpanda ``storage_min_free_bytes`` (default 10 MiB = 10 485 760 bytes).
    Any path at or below this floor is immediately classified P0 regardless of
    usage_pct.

    ``docker_info_runner`` and ``disk_usage_runner`` are injected seams so unit
    tests never invoke real subprocesses. In production both are ``None`` and
    the handler falls back to ``subprocess.run`` / ``shutil.disk_usage``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    correlation_id: UUID = Field(
        default_factory=uuid4, description="Trace correlation ID."
    )
    broker_volume_names: tuple[str, ...] = Field(
        default=(),
        description="Docker volume names to probe (in addition to docker data-root).",
    )
    warn_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Fractional usage at which a WARN alert fires (default 0.85).",
    )
    p0_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Fractional usage at which a P0 alert fires (default 0.95).",
    )
    min_free_bytes_floor: int = Field(
        default=10_485_760,
        gt=0,
        description="Minimum free bytes floor (redpanda storage_min_free_bytes). "
        "Paths at or below this floor are classified P0 (default 10 MiB).",
    )
    docker_info_runner: Callable[[], dict[str, object]] | None = Field(
        default=None,
        description="Injected seam: callable returning parsed docker info JSON. "
        "None → real subprocess.run(['docker', 'info', '--format', '{{json .}}']).",
        exclude=True,
    )
    disk_usage_runner: Callable[[str], tuple[int, int, int]] | None = Field(
        default=None,
        description="Injected seam: callable(path) -> (total, used, free) bytes. "
        "None → real shutil.disk_usage(path).",
        exclude=True,
    )


__all__ = ["ModelBrokerDiskWatermarkInput"]
