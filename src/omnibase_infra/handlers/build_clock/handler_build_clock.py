# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""HandlerBuildClock: minutes from container start to first healthy /health (OMN-26316).

Pure compute. Docker keeps no history of it and the journal gives only bounds, so the
deploy agent reads both stamps per build and this handler returns the clock and the
verdict against the limit.
"""

from __future__ import annotations

from datetime import UTC, datetime

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.model_build_clock import ModelBuildClock
from omnibase_infra.models.model_build_clock_request import ModelBuildClockRequest


def _parse_docker_time(value: str) -> datetime | None:
    """Docker's StartedAt (RFC 3339, up to nine fractional digits) as aware UTC."""
    seconds, dot, fraction = value.strip().removesuffix("Z").partition(".")
    if dot:
        digits = "".join(ch for ch in fraction if ch.isdigit())
        seconds = f"{seconds}.{digits[:6].ljust(6, '0')}"
    try:
        parsed = datetime.fromisoformat(seconds)
    except ValueError:
        return None
    return (
        parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)
    )


class HandlerBuildClock:
    """Turn a container start and a first healthy reading into the clock and verdict."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.COMPUTE

    def handle(self, request: ModelBuildClockRequest) -> ModelBuildClock:
        started = _parse_docker_time(request.container_started_at)
        healthy = request.first_healthy_at
        if started is None:
            return ModelBuildClock(
                container_started_at=None,
                first_healthy_at=healthy,
                minutes_to_healthy=0.0,
                limit_minutes=request.limit_minutes,
                exceeded=False,
                verdict="unread",
                detail=f"container start is not a Docker time: {request.container_started_at!r}",
            )
        minutes = round((healthy - started).total_seconds() / 60, 2)
        exceeded = minutes > request.limit_minutes
        limit = f"{request.limit_minutes:g}"
        detail = (
            f"healthy {minutes} min after its container start, past the {limit} min limit"
            if exceeded
            else f"healthy {minutes} min after its container start, within {limit} min"
        )
        return ModelBuildClock(
            container_started_at=started,
            first_healthy_at=healthy,
            minutes_to_healthy=minutes,
            limit_minutes=request.limit_minutes,
            exceeded=exceeded,
            verdict="fail" if exceeded else "pass",
            detail=detail,
        )
