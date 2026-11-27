# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One dev build's time to health and whether dev went back to an older build (OMN-26703).

THE GAP
-------
Goal item 1 of 2026-11-26 asks, for every dev build, how long the runtime took to answer
health from its start, whether that is under five minutes, and whether dev went back to an
older build than it ran. The deploy agent's build clock (OMN-26316) judges the first of
these inside a deploy job and leaves it on the job's health checks, so a build the agent did
not run (a hand load) has no reading and a reader has no event to read on the bus.

THE DECISION
------------
Pure: no clock, no I/O, no repository read. The caller reads the container, the image's
revision label and the git order of the two revisions. Returns one ``ModelDevBuildHealth``
once the container's health has settled (``healthy`` or ``unhealthy``) and ``None`` while it
is ``starting`` or unread, so the event is published once, after the answer.

* ``health_seconds``: first passing health check minus ``StartedAt``; ``None`` when no
  passing check was read, in which case ``under_limit`` is ``None`` too (unread is never a
  pass).
* ``rolled_back``: only when ``revision_order`` is ``back`` (the running revision is an
  ancestor of the previous build's). ``unrelated`` and ``unread`` are never counted as a
  rollback and never hide one: the event names the order it could prove.
* ``no_rollback_since``: the start of this build when it went back or no clock exists yet,
  else the clock the previous build left.
"""

from __future__ import annotations

from uuid import NAMESPACE_URL, uuid5

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.model_dev_build_health import ModelDevBuildHealth
from omnibase_infra.models.model_dev_build_health_request import (
    ModelDevBuildHealthRequest,
)

_KEY_NAMESPACE = "onex:dev-build-health"
_SETTLED = frozenset({"healthy", "unhealthy"})


class HandlerDevBuildHealth:
    """Judge one dev container start's health clock and revision order (pure)."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.COMPUTE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.COMPUTE

    def handle(self, request: ModelDevBuildHealthRequest) -> ModelDevBuildHealth | None:
        reading = request.reading
        if reading.health_status not in _SETTLED:
            return None
        health_seconds: float | None = None
        under_limit: bool | None = None
        healthy_at = reading.first_healthy_at
        if healthy_at is not None and healthy_at >= reading.started_at:
            health_seconds = round((healthy_at - reading.started_at).total_seconds(), 3)
            under_limit = health_seconds <= request.limit_seconds
        rolled_back = request.revision_order == "back"
        since = (
            reading.started_at
            if rolled_back or request.previous_no_rollback_since is None
            else request.previous_no_rollback_since
        )
        return ModelDevBuildHealth(
            event_id=uuid5(
                NAMESPACE_URL,
                f"{_KEY_NAMESPACE}:{reading.container_digest}:{reading.started_at.isoformat()}",
            ),
            runtime_lane="dev",
            container=reading.container,
            started_at=reading.started_at,
            first_healthy_at=healthy_at,
            health_seconds=health_seconds,
            limit_seconds=request.limit_seconds,
            under_limit=under_limit,
            health_status=reading.health_status,
            image_revision=reading.image_revision,
            previous_image_revision=request.previous_image_revision,
            revision_order=request.revision_order,
            rolled_back=rolled_back,
            no_rollback_since=since,
        )
