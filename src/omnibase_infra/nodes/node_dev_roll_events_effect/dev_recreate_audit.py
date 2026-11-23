# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The reader side of the one-recreate-an-hour rule (OMN-25912).

The deploy agent's window gate covers the idle converge and the idle restore, and a lane's
rollback command or the container's own restart policy never asks it. This handler reads
what those starts left on the bus (``dev_runtime_started``, ``dev_roll_manifest`` and
``dev_recreate_announced``, plus the container's own StartedAt and RestartCount) and names
every start that broke the declared rule, one ``dev_recreate_violation`` event per case:

* ``two_starts_in_window``: a window of the declared policy holds two starts.
* ``off_window_start``: a start after the policy's start grace whose requester is not a
  pinned fix prefix of the policy.
* ``unannounced_start``: a start with no immediate announcement before it.
* ``start_without_job_record``: a container start no ``dev_runtime_started`` event matches;
  the RestartCount rides in the event.

Every bound (the window, its grace, the pinned prefixes, the announcement lookback, the
start match tolerance) arrives in the request, read from the declared policy and the node
contract. The handler is pure: no clock, no I/O.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from uuid import NAMESPACE_URL, UUID, uuid5

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.enums.enum_dev_recreate_violation_kind import (
    EnumDevRecreateViolationKind,
)
from omnibase_infra.handlers.dev_recreate_window.handler_dev_recreate_window import (
    window_start,
)
from omnibase_infra.models.model_dev_recreate_announcement import (
    ModelDevRecreateAnnouncement,
)
from omnibase_infra.models.model_dev_recreate_audit_request import (
    ModelDevRecreateAuditRequest,
)
from omnibase_infra.models.model_dev_recreate_audit_result import (
    ModelDevRecreateAuditResult,
)
from omnibase_infra.models.model_dev_recreate_policy import ModelDevRecreatePolicy
from omnibase_infra.models.model_dev_recreate_violation import (
    ModelDevRecreateViolation,
    ModelDevRecreateViolationRow,
)
from omnibase_infra.protocols.protocol_dev_roll_events import ProtocolDevRollEvents

logger = logging.getLogger(__name__)

_KEY_NAMESPACE = "onex:dev-recreate-violation"


@dataclass(frozen=True)
class RecreateRecord:
    job_id: str
    at: datetime
    requested_by: str
    announced: bool


def _violation_id(kind: EnumDevRecreateViolationKind, key: str) -> UUID:
    return uuid5(NAMESPACE_URL, f"{_KEY_NAMESPACE}:{kind.value}:{key}")


def _stamp(at: datetime) -> str:
    return at.isoformat()


class DevRecreateAuditReader:
    """Read one window of dev recreate events and name each rule a start broke (pure)."""

    @property
    def handler_type(self) -> EnumHandlerType:
        """Return the pure compute role."""
        return EnumHandlerType.COMPUTE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Return the deterministic compute category."""
        return EnumHandlerTypeCategory.COMPUTE

    def handle(
        self, request: ModelDevRecreateAuditRequest
    ) -> ModelDevRecreateAuditResult:
        recreates = self._recreates(request)
        violations = [
            *self._two_in_window(request.policy, recreates),
            *self._off_window(request.policy, recreates),
            *self._unannounced(request.policy, recreates),
            *self._without_job(request),
        ]
        violations.sort(key=lambda found: (found.at, found.kind.value))
        return ModelDevRecreateAuditResult(
            recreates_read=len(recreates), violations=tuple(violations)
        )

    def _recreates(self, request: ModelDevRecreateAuditRequest) -> list[RecreateRecord]:
        """Each start with its recreate time, requester and whether an announcement preceded it.

        The recreate time is the manifest's ``accepted_at`` (what the window gate and the job
        records call the recreate), and the start event's ``started_at`` when no manifest
        names the job or the job started twice. The requester is the one the matched
        announcement names (the gate's own ask), else the manifest's issuer.
        """
        manifests = {manifest.job_id: manifest for manifest in request.manifests}
        lookback = timedelta(seconds=request.announcement_lookback_seconds)
        unused = sorted(
            (a for a in request.announcements if a.immediate),
            key=lambda a: a.announced_at,
        )
        seen: set[str] = set()
        timed: list[tuple[datetime, str, str]] = []
        for start in sorted(request.starts, key=lambda s: s.started_at):
            manifest = manifests.get(start.job_id)
            first = start.job_id not in seen
            seen.add(start.job_id)
            at = manifest.accepted_at if manifest and first else start.started_at
            timed.append((at, start.job_id, manifest.issuer if manifest else ""))
        recreates: list[RecreateRecord] = []
        for at, job_id, issuer in sorted(timed):
            match = self._latest_before(unused, at, lookback)
            if match is not None:
                unused.remove(match)
            recreates.append(
                RecreateRecord(
                    job_id=job_id,
                    at=at,
                    requested_by=match.requested_by if match else issuer,
                    announced=match is not None,
                )
            )
        return recreates

    @staticmethod
    def _latest_before(
        announcements: list[ModelDevRecreateAnnouncement],
        at: datetime,
        lookback: timedelta,
    ) -> ModelDevRecreateAnnouncement | None:
        candidates = [a for a in announcements if at - lookback <= a.announced_at <= at]
        return candidates[-1] if candidates else None

    @staticmethod
    def _row(recreate: RecreateRecord) -> ModelDevRecreateViolationRow:
        return ModelDevRecreateViolationRow(
            job_id=recreate.job_id,
            at=recreate.at,
            requested_by=recreate.requested_by,
            announced=recreate.announced,
        )

    def _two_in_window(
        self, policy: ModelDevRecreatePolicy, recreates: list[RecreateRecord]
    ) -> list[ModelDevRecreateViolation]:
        by_window: dict[datetime, list[RecreateRecord]] = defaultdict(list)
        for recreate in recreates:
            by_window[window_start(policy, recreate.at)].append(recreate)
        return [
            ModelDevRecreateViolation(
                event_id=_violation_id(
                    EnumDevRecreateViolationKind.TWO_STARTS_IN_WINDOW, _stamp(start)
                ),
                runtime_lane="dev",
                kind=EnumDevRecreateViolationKind.TWO_STARTS_IN_WINDOW,
                at=rows[1].at,
                window_start=start,
                rows=tuple(self._row(row) for row in rows),
                detail=f"{len(rows)} starts in the window that opened {_stamp(start)}",
            )
            for start, rows in sorted(by_window.items())
            if len(rows) > 1
        ]

    def _off_window(
        self, policy: ModelDevRecreatePolicy, recreates: list[RecreateRecord]
    ) -> list[ModelDevRecreateViolation]:
        grace = timedelta(seconds=policy.start_grace_seconds)
        found: list[ModelDevRecreateViolation] = []
        for recreate in recreates:
            start = window_start(policy, recreate.at)
            if recreate.at - start <= grace or recreate.requested_by.startswith(
                policy.pinned_fix_requester_prefixes
            ):
                continue
            found.append(
                ModelDevRecreateViolation(
                    event_id=_violation_id(
                        EnumDevRecreateViolationKind.OFF_WINDOW_START, recreate.job_id
                    ),
                    runtime_lane="dev",
                    kind=EnumDevRecreateViolationKind.OFF_WINDOW_START,
                    at=recreate.at,
                    window_start=start,
                    rows=(self._row(recreate),),
                    detail=(
                        f"start {_stamp(recreate.at)} is {recreate.at - start} after the "
                        f"window start by {recreate.requested_by or 'an unknown requester'}, "
                        "not a pinned fix"
                    ),
                )
            )
        return found

    def _unannounced(
        self, policy: ModelDevRecreatePolicy, recreates: list[RecreateRecord]
    ) -> list[ModelDevRecreateViolation]:
        return [
            ModelDevRecreateViolation(
                event_id=_violation_id(
                    EnumDevRecreateViolationKind.UNANNOUNCED_START, recreate.job_id
                ),
                runtime_lane="dev",
                kind=EnumDevRecreateViolationKind.UNANNOUNCED_START,
                at=recreate.at,
                window_start=window_start(policy, recreate.at),
                rows=(self._row(recreate),),
                detail=(
                    f"start {_stamp(recreate.at)} by "
                    f"{recreate.requested_by or 'an unknown requester'} has no earlier "
                    "immediate announcement"
                ),
            )
            for recreate in recreates
            if not recreate.announced
        ]

    @staticmethod
    def _without_job(
        request: ModelDevRecreateAuditRequest,
    ) -> list[ModelDevRecreateViolation]:
        tolerance = timedelta(seconds=request.start_match_tolerance_seconds)
        found: list[ModelDevRecreateViolation] = []
        for seen in request.container_observations:
            if any(
                abs(start.started_at - seen.started_at) <= tolerance
                for start in request.starts
            ):
                continue
            found.append(
                ModelDevRecreateViolation(
                    event_id=_violation_id(
                        EnumDevRecreateViolationKind.START_WITHOUT_JOB_RECORD,
                        f"{seen.container}:{_stamp(seen.started_at)}",
                    ),
                    runtime_lane="dev",
                    kind=EnumDevRecreateViolationKind.START_WITHOUT_JOB_RECORD,
                    at=seen.started_at,
                    window_start=window_start(request.policy, seen.started_at),
                    restart_count=seen.restart_count,
                    detail=(
                        f"container {seen.container} started {_stamp(seen.started_at)} "
                        f"(RestartCount {seen.restart_count}, exit code {seen.exit_code}) "
                        "with no dev_runtime_started event"
                    ),
                )
            )
        return found


def audit_and_publish(
    handler: DevRecreateAuditReader,
    request: ModelDevRecreateAuditRequest,
    events: ProtocolDevRollEvents,
) -> ModelDevRecreateAuditResult:
    """Read the window, publish one violation event per case, and return what was found."""
    result = handler.handle(request)
    for violation in result.violations:
        if not events.recreate_violation(violation):
            logger.error(
                "dev recreate violation %s was not accepted by the bus "
                "friction_type=dev_event_publish_failed",
                violation.event_id,
            )
    return result
