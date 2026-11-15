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
* a target failed outright and none timed out: ``unhealthy``. There is no healthy
  state to restore, so the deploy runs and the agent records that it ran without
  a point.
* any probe timeout, probe error or no target at all: ``unknown``. The lane may be
  serving. Refused.

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
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol
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

LaneReading = Literal["healthy", "unhealthy", "unknown"]

_GUARDED_SCOPES = frozenset({"runtime", "full"})
_UNGUARDED_LANES = frozenset({"prod", "dev"})


class _HealthCheck(Protocol):
    @property
    def status(self) -> str: ...

    @property
    def detail(self) -> str: ...


class _RuntimeProbe(Protocol):
    def __call__(
        self, *, service: str, port: int
    ) -> tuple[_HealthCheck, BaseException | None]: ...


def classify_readings(
    readings: Sequence[tuple[str, str]],
) -> tuple[LaneReading, str]:
    """Reduce per-target ``(status, detail)`` pairs; fail closed to ``unknown``."""
    if not readings:
        return "unknown", "lane declares no runtime targets"
    for status, detail in readings:
        if status not in ("pass", "fail"):
            return "unknown", detail or f"probe status {status}"
    failures = [
        detail or "probe failed" for status, detail in readings if status == "fail"
    ]
    if failures:
        return "unhealthy", "; ".join(failures)
    return "healthy", ""


def take_readings(
    targets: Sequence[tuple[str, int]], probe: _RuntimeProbe
) -> list[tuple[str, str]]:
    """Probe every target with the guard's own probe: ``(status, detail)`` each.

    A timeout reads ``timeout`` and a probe error reads ``error``; neither is a
    definitive answer, so :func:`classify_readings` reads the lane ``unknown``.
    """
    readings: list[tuple[str, str]] = []
    try:
        for container, port in targets:
            check, timed_out = probe(service=container, port=port)
            if timed_out is not None:
                readings.append(("timeout", f"{container}: probe timed out"))
            else:
                readings.append(
                    (check.status, f"{container}: {check.detail or 'probe failed'}")
                )
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
    probe: _RuntimeProbe,
    capture: Callable[[], Point | None],
) -> Point | None:
    """The point a deploy runs under, or ``None`` when none is needed or it is waived.

    Raises ``RuntimeError`` when a point is required and the lane is not read down.
    """
    if scope not in _GUARDED_SCOPES or lane in _UNGUARDED_LANES:
        return None
    point = capture()
    if point is not None:
        return point
    readings = take_readings(targets, probe)
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
