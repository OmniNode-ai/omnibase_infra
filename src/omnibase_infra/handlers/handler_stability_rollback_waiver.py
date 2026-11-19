# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Admit a stability runtime deploy that has no retained rollback point (OMN-24562).

THE FAILURE
-----------
The deploy agent keeps an in-memory rollback point, captured from a lane whose
runtime targets all answer healthy, and refused every non-dev, non-prod runtime or
full deploy that had none. A lane that is already down has nothing healthy to
retain, so the guard refused the one deploy that could recover it. Measured
2026-11-15: the stability main container had crash-looped since 13:25Z because the
pre-fold ``runtime-effects`` container still held its host port, and the deploy
that retires that container runs after the guard.

THE WAIVER
----------
Read from the guard's own probe of every runtime target, never from a caller:

* every target passes: ``healthy``. A missing point is a defect (image unreadable,
  no unique container). The deploy is refused.
* every target failed outright, none timed out, and Docker reads every target
  container past its declared start window and not serving (``unhealthy``,
  ``absent`` or ``not-running``): ``unhealthy``. There is no healthy state to
  restore, so the deploy runs and the agent records that it ran without a point.
* a target passes, or a failing target is Docker-healthy: ``degraded``. The lane is
  still serving. Refused.
* a failing target Docker still reads ``starting`` (inside its declared start
  period): ``starting``. Refused.
* any probe timeout, probe error, unreadable Docker state or no target at all:
  ``unknown``. The lane may be serving. Refused.

Every lane the guard covers may be waived, which today is only the stability lane.
Prod and dev are not guarded here and are never probed.

The standalone deploy agent loads this module from its own checkout
(``executor.load_dev_boot_guard``) without the request and result models: it calls
:func:`admit_rollback_point`, which passes in the probe, the lane's targets and the
capture. The node path (``handle``) has the models and takes readings already read.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory

if TYPE_CHECKING:
    from omnibase_infra.models.model_stability_rollback_waiver_request import (
        ModelStabilityRollbackWaiverRequest,
    )
    from omnibase_infra.models.model_stability_rollback_waiver_result import (
        ModelStabilityRollbackWaiverResult,
    )

logger = logging.getLogger(__name__)

LaneReading = Literal["healthy", "unhealthy", "degraded", "starting", "unknown"]

# Docker states (runtime_health_wait.sh runtime_health_state) of a container that is
# past its declared start window and not serving.
_DOWN_STATES = frozenset({"unhealthy", "absent", "not-running"})

_GUARDED_SCOPES = frozenset({"runtime", "full"})
_UNGUARDED_LANES = frozenset({"prod", "dev"})


# Called as ``probe(service=..., port=...)``; returns the check and any raised error.
RuntimeProbe = Callable[..., tuple[Any, BaseException | None]]


def read_docker_state(container: str) -> str:
    """Docker's ``runtime_health_state`` of ``container``; ``""`` when unreadable.

    ``starting`` is Docker inside the container's declared start period;
    ``unhealthy`` is the healthcheck failing past it. The reading is the deploy
    agent's own (``deploy_agent.executor._runtime_health_helper``, its helper file
    and its probe bound), taken from the agent process that loaded this module
    through ``executor.load_dev_boot_guard``. Outside that process nothing is
    readable, so the lane reads ``unknown`` and the waiver is refused.
    """
    executor = sys.modules.get("deploy_agent.executor")
    reader = getattr(executor, "_runtime_health_helper", None)
    if reader is None:
        return ""
    return str(reader("runtime_health_state", container))


def classify_readings(
    readings: Sequence[tuple[str, str]],
) -> tuple[LaneReading, str]:
    """Reduce per-target ``(status, detail)`` pairs; fail closed to ``unknown``."""
    if not readings:
        return "unknown", "lane declares no runtime targets"
    for status, detail in readings:
        if status not in ("pass", "fail", "serving", "starting"):
            return "unknown", detail or f"probe status {status}"
    statuses = {status for status, _ in readings}
    details = "; ".join(detail for status, detail in readings if status != "pass")
    if statuses == {"pass"}:
        return "healthy", ""
    if "starting" in statuses:
        return "starting", details
    if statuses != {"fail"}:
        return "degraded", details or "a runtime target is serving"
    return "unhealthy", details


def take_readings(
    targets: Sequence[tuple[str, int]],
    probe: RuntimeProbe,
    state: Callable[[str], str],
) -> list[tuple[str, str]]:
    """Probe every target with the guard's own probe: ``(status, detail)`` each.

    A failed probe counts as ``fail`` only when Docker (``state``, the
    ``runtime_health_state`` reading of the container) reads it past its declared
    start window and not serving; ``starting`` reads ``starting``, ``healthy``
    reads ``serving`` and any other state reads ``unknown``. A timeout reads
    ``timeout`` and a probe error reads ``error``; neither is a definitive answer,
    so :func:`classify_readings` reads the lane ``unknown``.
    """
    readings: list[tuple[str, str]] = []
    try:
        for container, port in targets:
            check, timed_out = probe(service=container, port=port)
            detail = f"{container}: {check.detail or 'probe failed'}"
            if timed_out is not None:
                readings.append(("timeout", f"{container}: probe timed out"))
            elif check.status != "fail":
                readings.append((check.status, detail))
            else:
                docker = state(container)
                status = {"starting": "starting", "healthy": "serving"}.get(
                    docker, "fail" if docker in _DOWN_STATES else "unknown"
                )
                readings.append((status, f"{detail} (docker {docker or 'unread'})"))
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        readings.append(("error", f"probe error: {exc}"))
    return readings


def decide_waiver(
    *, lane: str, scope: str, point_retained: bool, readings: Sequence[tuple[str, str]]
) -> tuple[bool, bool, LaneReading | None, str, str]:
    """``(admitted, waived, reading, detail, refusal)`` for a deploy."""
    if scope not in _GUARDED_SCOPES or lane in _UNGUARDED_LANES or point_retained:
        return True, False, None, "", ""
    refusal = f"Refusing {lane} {scope} deploy: no healthy, retained rollback point"
    reading, detail = classify_readings(readings)
    if reading != "unhealthy":
        return (
            False,
            False,
            reading,
            detail,
            f"{refusal} (lane reads {reading}: {detail})",
        )
    return True, True, reading, detail, ""


def admit_rollback_point[Point](
    *,
    lane: str,
    scope: str,
    correlation_id: UUID,
    record_dir: Path,
    targets: Sequence[tuple[str, int]],
    probe: RuntimeProbe,
    capture: Callable[[], Point | None],
    state: Callable[[str], str] | None = None,
) -> Point | None:
    """The point a deploy runs under, or ``None`` when none is needed or it is waived.

    Raises ``RuntimeError`` when a point is required and the lane is not read down.
    """
    if scope not in _GUARDED_SCOPES or lane in _UNGUARDED_LANES:
        return None
    point = capture()
    if point is not None:
        return point
    readings = take_readings(targets, probe, state or read_docker_state)
    admitted, waived, reading, detail, refusal = decide_waiver(
        lane=lane, scope=scope, point_retained=False, readings=readings
    )
    logger.warning(
        "No rollback point for %s: guard probe reads %s (%s)", lane, reading, detail
    )
    if not admitted:
        raise RuntimeError(refusal)
    if waived:
        record_dir.mkdir(parents=True, exist_ok=True)
        (record_dir / f"{correlation_id}.json").write_text(
            json.dumps(
                {
                    "correlation_id": str(correlation_id),
                    "lane": lane,
                    "scope": scope,
                    "reading": reading,
                    "detail": detail,
                    "at": datetime.now(UTC).isoformat(),
                }
            ),
            encoding="utf-8",
        )
    return None


class HandlerStabilityRollbackWaiver:
    """The node's single entry point; the deploy agent calls :func:`admit_rollback_point`."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    def handle(
        self, request: ModelStabilityRollbackWaiverRequest
    ) -> ModelStabilityRollbackWaiverResult:
        """Admit, waive or refuse a deploy from the readings already taken."""
        # The standalone deploy agent loads this module without the request and
        # result models; the node path has them.
        from omnibase_infra.models.model_stability_rollback_waiver_result import (
            ModelStabilityRollbackWaiverResult,
        )

        admitted, waived, reading, detail, refusal = decide_waiver(
            lane=request.lane,
            scope=request.scope,
            point_retained=request.point_retained,
            readings=request.readings,
        )
        return ModelStabilityRollbackWaiverResult(
            admitted=admitted,
            waived=waived,
            reading=reading or "",
            detail=detail,
            refusal=refusal,
        )
