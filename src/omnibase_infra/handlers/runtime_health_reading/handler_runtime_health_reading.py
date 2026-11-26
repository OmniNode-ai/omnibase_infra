# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""HandlerRuntimeHealthReading: judge one runtime /health reading (OMN-18640, OMN-20864).

Pure compute, moved out of the deploy agent's executor. The deploy agent probes
``/health`` and this handler returns why the reading is not a pass, and whether the
runtime answered running with its monitor degraded.
"""

from __future__ import annotations

import json

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.model_answering_degraded import ModelAnsweringDegraded
from omnibase_infra.models.model_runtime_health_judgement import (
    ModelRuntimeHealthJudgement,
)
from omnibase_infra.models.model_runtime_health_reading import (
    ModelRuntimeHealthReading,
)


def _runtime_health_reason(result: ModelRuntimeHealthReading) -> str:
    """Return why a runtime /health response is not a pass, or "" when it is.

    OMN-18640 AC8. The predicate this replaced returned a bare bool, so the
    executor knew a probe had failed and could not say ANYTHING about why. A
    verdict that fails the job has to name its own reason -- the reason is what
    reaches ``errors`` on the terminal event and what a reader of the job
    record has instead of going to the host.

    Every branch below is a distinct fact about the runtime and they take
    different next steps: curl could not reach it at all, it answered something
    that is not a health document, or it answered a health document that says
    it is not ready.
    """
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[:200]
        return f"curl exited {result.returncode}: {detail or 'no output'}"
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return f"health response is not JSON: {result.stdout.strip()[:200]!r}"
    if not isinstance(payload, dict):
        return f"health response is not a JSON object: {type(payload).__name__}"
    details = payload.get("details")
    if not isinstance(details, dict):
        return "health response carries no details object"
    status = payload.get("status")
    is_running = details.get("is_running")
    prefetch = details.get("config_prefetch_status")
    if status == "healthy" and is_running is True and prefetch in {"ok", "skipped"}:
        return ""
    return (
        f"health response is not ready: status={status!r} "
        f"is_running={is_running!r} config_prefetch_status={prefetch!r}"
    )


def _answering_degraded(
    result: ModelRuntimeHealthReading,
) -> ModelAnsweringDegraded | None:
    """The monitor-degraded reading of a runtime that is up, or ``None``.

    OMN-20864. MEASURED on the lab dev lane 2026-10-11: job 72e74ec6's
    runtime answered ``status=degraded is_running=True`` on every probe; its
    health monitor (``onex.evt.omnibase-infra.runtime-health-check.v1``) named
    ``projection_dlq_saturation`` on every cycle from 13:28Z, the same finding
    the container it replaced had carried since at least 12:41Z. The agent
    force-recreated it at 13:50Z, the recreate timed out, and the lane was down
    until 19:41Z.

    The runtime's own contract (``ServiceHealth``: degraded and operational is
    HTTP 200, "automatic restarts may mask recurring issues") is that such a
    runtime stays in service. So this is NARROW, and everything outside it
    keeps the OMN-18640 recreate: the probe answered JSON, ``status`` is
    ``degraded``, ``details.is_running`` is true, the runtime is attached and
    not starting, the event bus is not reported unhealthy, and the monitor
    block names at least one non-``HEALTHY`` dimension. A dead port, a
    timeout, an ``unhealthy`` runtime, a stopped one, one whose bus is down,
    or a degraded one whose monitor names nothing is not this.
    """
    if result.returncode != 0:
        return None
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    details = payload.get("details")
    if not isinstance(details, dict):
        return None
    if (
        payload.get("status") != "degraded"
        or details.get("is_running") is not True
        or details.get("runtime_attached") is False
        or details.get("startup_in_progress") is True
        or details.get("event_bus_healthy") is False
    ):
        return None
    monitor = details.get("runtime_health")
    if not isinstance(monitor, dict):
        return None
    raw_dimensions = monitor.get("dimensions")
    dimensions = frozenset(
        str(dimension["name"])
        for dimension in (raw_dimensions if isinstance(raw_dimensions, list) else [])
        if isinstance(dimension, dict)
        and dimension.get("name")
        and dimension.get("status") != "HEALTHY"
    )
    if not dimensions:
        return None
    return ModelAnsweringDegraded(
        observed_at=str(monitor.get("observed_at") or ""),
        dimensions=dimensions,
    )


class HandlerRuntimeHealthReading:
    """Judge a /health reading: its failure reason and any degraded monitor."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.COMPUTE

    def handle(self, request: ModelRuntimeHealthReading) -> ModelRuntimeHealthJudgement:
        return ModelRuntimeHealthJudgement(
            reason=_runtime_health_reason(request),
            answering_degraded=_answering_degraded(request),
        )
