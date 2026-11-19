# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev runtime recreates at most once a window, on the window's start, announced first (OMN-25912).

On 2026-11-19 the dev runtime was started at 04:33:39Z, 04:54:06Z, 05:12:28Z and 05:38:11Z:
two merge-driven recreates and two ``docker restart`` calls of the same container by the
branch install reapply, each 7 to 14 minutes to ready. This handler is the one decision
every agent-local recreate path asks (the idle converge and the idle restore), extending the
roll window gate's batching rather than adding a second mechanism.

Every value is policy: the window length, its offset, the start grace and the requester
prefixes of a pinned fix come from ``ModelDevRecreatePolicy`` (rendered into
``docker/runtime-policy.env`` from ``contracts/services/runtime_policy.contract.yaml``).
Nothing here names an hour, a length or a requester.

The decision is pure: ``handle(request) -> decision``, no clock, no I/O. ``DevRecreateWindowGate``
holds the one piece of state (the last recreate) and refuses any recreate whose announcement
was not accepted by the bus first.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from omnibase_infra.enums.enum_dev_recreate_verdict import (
    EnumDevRecreateReason,
    EnumDevRecreateVerdict,
)
from omnibase_infra.models.model_dev_recreate_window import (
    ModelDevRecreateAnnouncement,
    ModelDevRecreateDecision,
    ModelDevRecreatePolicy,
    ModelDevRecreateRequest,
)

logger = logging.getLogger(__name__)

RUNTIME_LANE = "dev"


class RecreateNotAnnouncedError(RuntimeError):
    """A recreate was asked for and its announcement was not accepted by the bus."""


def _window_start(policy: ModelDevRecreatePolicy, now: datetime) -> datetime:
    """The start of the window ``now`` falls in: the policy's offset plus whole windows."""
    elapsed = int(now.timestamp()) - policy.offset_seconds
    return datetime.fromtimestamp(
        elapsed // policy.window_seconds * policy.window_seconds
        + policy.offset_seconds,
        tz=UTC,
    )


class HandlerDevRecreateWindow:
    """Decide whether the dev runtime may be recreated now (contract-bound, pure)."""

    def handle(self, request: ModelDevRecreateRequest) -> ModelDevRecreateDecision:
        policy = request.policy
        now = request.now
        if request.requested_by.startswith(policy.pinned_fix_requester_prefixes):
            return ModelDevRecreateDecision(
                verdict=EnumDevRecreateVerdict.RECREATE,
                reason=EnumDevRecreateReason.PINNED_FIX,
                next_recreate_at=now,
            )
        start = _window_start(policy, now)
        following = start + timedelta(seconds=policy.window_seconds)
        if request.last_recreate_at is not None and request.last_recreate_at >= start:
            return ModelDevRecreateDecision(
                verdict=EnumDevRecreateVerdict.HOLD,
                reason=EnumDevRecreateReason.WINDOW_SPENT,
                next_recreate_at=following,
            )
        if now - start > timedelta(seconds=policy.start_grace_seconds):
            return ModelDevRecreateDecision(
                verdict=EnumDevRecreateVerdict.HOLD,
                reason=EnumDevRecreateReason.WINDOW_MISSED,
                next_recreate_at=following,
            )
        return ModelDevRecreateDecision(
            verdict=EnumDevRecreateVerdict.RECREATE,
            reason=EnumDevRecreateReason.WINDOW_OPEN,
            next_recreate_at=now,
        )


class DevRecreateWindowGate:
    """The agent's stateful gate: window decision, announcement first, then the recreate.

    ``announce`` returns True only when the bus accepted the event. A recreate whose
    announcement was not accepted is refused, so no recreate has no announcement before it.
    A held ask announces the next recreate once per ``next_recreate_at``.
    """

    def __init__(
        self,
        policy: ModelDevRecreatePolicy,
        announce: Callable[[ModelDevRecreateAnnouncement], bool],
        *,
        last_recreate_at: datetime | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        announced_for: datetime | None = None,
    ) -> None:
        self._clock = clock
        self._policy = policy
        self._announce = announce
        self._handler = HandlerDevRecreateWindow()
        self.last_recreate_at = last_recreate_at
        self._last_announced_for: datetime | None = announced_for

    @property
    def announced_for(self) -> datetime | None:
        """The next recreate time last announced, for a caller that persists it."""
        return self._last_announced_for

    def permit(self, requested_by: str, now: datetime) -> bool:
        """Whether ``requested_by`` may recreate the dev runtime at ``now``.

        True means the recreate was announced and is recorded as started; the caller
        must run it. False means hold; the next recreate was announced once. Raises
        ``RecreateNotAnnouncedError`` when a recreate's announcement was not accepted.
        """
        decision = self._handler.handle(
            ModelDevRecreateRequest(
                policy=self._policy,
                now=now,
                requested_by=requested_by,
                last_recreate_at=self.last_recreate_at,
            )
        )
        immediate = decision.verdict is EnumDevRecreateVerdict.RECREATE
        if immediate or decision.next_recreate_at != self._last_announced_for:
            accepted = self._publish(decision, requested_by, now, immediate)
            if immediate and not accepted:
                logger.error(
                    "dev recreate refused: its announcement was not accepted "
                    "friction_type=dev_recreate_not_announced requested_by=%s",
                    requested_by,
                )
                raise RecreateNotAnnouncedError(requested_by)
            if accepted:
                self._last_announced_for = decision.next_recreate_at
        if immediate:
            self.last_recreate_at = now
        return immediate

    def allow(self, requested_by: str, observed_last: datetime | None = None) -> bool:
        """The agent's idle paths: permit now, a refused announcement reads as hold.

        ``observed_last`` is the newest recreate the job records show, so a recreate this
        gate did not permit (a merge's job) still spends its window.
        """
        if observed_last is not None and (
            self.last_recreate_at is None or observed_last > self.last_recreate_at
        ):
            self.last_recreate_at = observed_last
        try:
            return self.permit(requested_by, self._clock())
        except RecreateNotAnnouncedError:
            return False

    def run_recreate(
        self, requested_by: str, now: datetime, recreate: Callable[[], None]
    ) -> bool:
        """Announce, then recreate. Returns False on hold; raises when not announced."""
        if not self.permit(requested_by, now):
            return False
        recreate()
        return True

    def _publish(
        self,
        decision: ModelDevRecreateDecision,
        requested_by: str,
        now: datetime,
        immediate: bool,
    ) -> bool:
        try:
            return bool(
                self._announce(
                    ModelDevRecreateAnnouncement(
                        event_id=uuid4(),
                        runtime_lane=RUNTIME_LANE,
                        requested_by=requested_by,
                        announced_at=now,
                        next_recreate_at=decision.next_recreate_at,
                        reason=decision.reason,
                        immediate=immediate,
                    )
                )
            )
        except Exception:
            logger.exception(
                "dev recreate announcement could not be published "
                "friction_type=dev_event_publish_failed"
            )
            return False
