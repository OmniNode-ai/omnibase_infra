# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Strict ``/health`` verdict shared by the lane refresh health-gates [OMN-17563].

Both ``verify_stability_refresh.py`` (OMN-14873) and ``verify_dev_refresh.py``
(OMN-14889) previously carried their own copy of::

    status = str(payload.get("status", "")).lower()
    details_healthy = bool(payload.get("details", {}).get("healthy", False))
    if status == "healthy" or details_healthy:
        return True, ...

That reads healthy from two independent sources and ORs them, so the *weaker*
source wins. Measured pre-fix behaviour of both copies (OMN-17563) -- every one
of these was read as healthy:

* ``{"status": "degraded", "details": {"healthy": true}}``
* ``{"status": "unhealthy", "details": {"healthy": true}}``
* ``{"details": {"healthy": true}}`` -- no ``status`` field at all

The last two are worse than the ticket described: ``.get("status", "")``
reduced a missing status to the empty string, and the ``or details_healthy``
arm then carried the verdict on its own.

The observed consequence: ``refresh_stability_lane.sh`` wrote
``"overall": "PASS"`` for the stability lane at 2026-09-02T14:58Z, and ten
minutes later both runtime containers were ``unhealthy`` (FailingStreak 14/12,
RestartCount 0, same digest) under the lane's own
``onex-container-healthcheck --degraded-policy fail`` probe. The stability
lane's refresh receipt is where the ``stability-proven`` premise of a live
prod-promotion grant is resolved, so this failed open in the one direction
that matters.

**The rule this module encodes: the top-level ``status`` is the whole
verdict.** Nothing nested can promote a non-healthy status, and a body whose
status is missing, empty, or not a string is UNKNOWN health, which fails
closed. That is not a heuristic:

* ``ModelHealthCheckResponse.status`` is a closed
  ``Literal["healthy", "degraded", "unhealthy"]``, so ``status`` is total.
* OMN-15217's ``fold_runtime_verdict_into_status`` already folds a
  DEGRADED/CRITICAL ``ServiceRuntimeHealthMonitor`` verdict *down* into that
  top-level field, so ``status`` is the runtime's semantic verdict and not
  merely a process-liveness bit.

``details.healthy`` is still read, but only to render the failure detail --
seeing ``status='degraded' details.healthy=true`` in a receipt is what tells a
later reader which of the two disagreeing sources was believed.

Deliberately stdlib-only (no pydantic): the callers are standalone scripts
that ``refresh_*_lane.sh`` may run under a bare ``python3`` when the repo venv
is absent, so a third-party import here would break the deploy path.

Residual, stated rather than papered over: this is strictly weaker than
``onex-container-healthcheck --require-verdict``, which additionally fails
closed on an ABSENT or STALE monitor verdict block. Those bodies still report
``status: healthy`` and this gate accepts them -- and so does the lane's own
probe, which does NOT pass ``--require-verdict``. That blind window is ~300s
wide after every recreate (``RUNTIME_HEALTH_CHECK_INTERVAL``), which is when
the refresh gate runs, and it -- not the OR fixed here -- is what actually
produced the ``20260902T145841Z`` false PASS. Tracked as OMN-17624, not
silently implied by this module. The probe's separate fail-open on an absent
or unrecognised ``status`` is OMN-17623.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass

# Recorded verbatim in the refresh receipt (OMN-17563 AC-3) so a future reader
# can tell a strict PASS from a lenient one without re-deriving it. Bump the
# value -- never redefine it in place -- if the policy itself ever changes.
HEALTH_POLICY_STATUS_ONLY_STRICT = "status_only_strict.v1"

#: OMN-17624. Sign-off additionally requires a fresh ``details.runtime_health``
#: verdict. ``status_only_strict.v1`` cannot observe a DEGRADED lane inside the
#: monitor's first-verdict window, which is precisely when a refresh gate runs.
HEALTH_POLICY_VERDICT_REQUIRED = "verdict_required.v1"

#: Seconds of slack added to the derived wait so a probe landing fractionally
#: before the first emission does not read as a permanent absence.
VERDICT_WAIT_MARGIN_SECONDS = 60.0

#: How often the gate re-probes while waiting. Bounded attempts at a fixed
#: interval, mirroring ``check_consumer_group_with_retry`` -- never a poll
#: without a ceiling.
VERDICT_WAIT_POLL_SECONDS = 15.0

#: Each probe is an HTTP fetch with this timeout, spent BEFORE the poll sleep.
#: Worst-case wall clock is attempts * (poll + this), not attempts * poll.
VERDICT_PROBE_TIMEOUT_SECONDS = 10.0

#: Multiplier on the monitor's check interval past which a verdict is stale.
#: Two missed cycles is tolerance for a slow lane; three is a stopped monitor.
#: This has a DEFAULT because a freshness rule nobody passes enforces nothing:
#: a monitor that emits one verdict then dies leaves a frozen runtime_health
#: that an opt-in check would accept forever, which is strictly worse than the
#: blind window this ticket closes.
VERDICT_STALE_AFTER_INTERVALS = 3.0

#: The reason string that -- alone -- justifies waiting. Every other failure
#: is knowable on the first probe; retrying it burns the window for nothing.
REASON_VERDICT_ABSENT = "verdict_absent"

#: The only ``status`` value that means healthy.
HEALTH_STATUS_HEALTHY = "healthy"


class HealthPayloadError(ValueError):
    """A ``/health`` body that cannot be read as a :class:`HealthPayload`.

    Raised rather than defaulted. A body we cannot parse is UNKNOWN health, and
    unknown is not healthy -- the whole defect this module closes came from a
    ``.get(..., default)`` chain turning "absent" into "fine".
    """


@dataclass(frozen=True)
class HealthPayload:
    """Typed view of a runtime ``/health`` JSON body.

    Attributes:
        status: The top-level ``status``, stripped and lowercased. Always a
            non-empty string -- :func:`parse_health_payload` refuses to build
            this object without one.
        details_healthy: ``details.healthy`` when present as a real bool, else
            ``None`` (absent, non-dict ``details``, or a non-bool value).
            Diagnostic only; never a verdict input.
    """

    status: str
    details_healthy: bool | None

    @property
    def healthy(self) -> bool:
        """True iff the top-level ``status`` says healthy."""
        return self.status == HEALTH_STATUS_HEALTHY

    def describe(self) -> str:
        """Render both sources, so a disagreement is legible in the receipt."""
        if self.details_healthy is None:
            nested = "absent"
        else:
            nested = "true" if self.details_healthy else "false"
        return f"status={self.status!r} details.healthy={nested}"


@dataclass(frozen=True)
class HealthVerdict:
    """The health-gate's verdict on one ``/health`` probe.

    Attributes:
        ok: Whether the probe proves health. False for every non-healthy
            status, every unreadable body, and every transport failure.
        policy: Which policy produced ``ok``. Persisted into the receipt.
        status: The observed status, or ``None`` when no status could be read
            (malformed body, or the endpoint was never reached).
        details_healthy: The nested flag, for the disagreement record only.
        detail: Human-readable reason, printed by the gate and stored in the
            receipt's ``health_detail``.
    """

    ok: bool
    policy: str
    status: str | None
    details_healthy: bool | None
    detail: str
    #: Machine-readable cause, mirrored from ``evaluate_health_response``
    #: (``verdict_absent`` / ``verdict_stale`` / ``status_unreadable``).
    #: ``detail`` is prose for humans; retry logic must never parse prose.
    reason: str | None = None


def parse_health_payload(raw: bytes | str) -> HealthPayload:
    """Parse a ``/health`` body into a :class:`HealthPayload`.

    Args:
        raw: The raw response body.

    Returns:
        The parsed payload.

    Raises:
        HealthPayloadError: The body is not JSON, is not a JSON object, or
            carries no usable top-level ``status``.
    """
    try:
        decoded = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise HealthPayloadError(f"health payload not valid JSON: {exc}") from exc

    if not isinstance(decoded, dict):
        raise HealthPayloadError(
            f"health payload is not a JSON object (got {type(decoded).__name__})"
        )

    if "status" not in decoded:
        raise HealthPayloadError(
            "health payload carries no top-level 'status' -- health is unknown, "
            "not proven"
        )

    status = decoded["status"]
    if not isinstance(status, str) or not status.strip():
        raise HealthPayloadError(
            f"health payload 'status' is not a non-empty string: {status!r}"
        )

    return HealthPayload(
        status=status.strip().lower(),
        details_healthy=_nested_healthy_flag(decoded),
    )


def _nested_healthy_flag(payload: dict[str, object]) -> bool | None:
    """Read ``details.healthy`` when it is genuinely a bool, else ``None``.

    Never defaults to ``False``: "the flag says false" and "there is no flag"
    are different facts and the receipt should be able to say which it saw.
    """
    if "details" not in payload:
        return None
    details = payload["details"]
    if not isinstance(details, dict) or "healthy" not in details:
        return None
    flag = details["healthy"]
    return flag if isinstance(flag, bool) else None


@dataclass(frozen=True)
class VerdictWaitBound:
    """A bounded wait for the monitor's first verdict, with its arithmetic kept.

    OMN-17624 AC3 requires the receipt to record which bound was applied. A
    number without its derivation is unreviewable, so the inputs travel with
    the result.
    """

    attempts: int
    interval_seconds: float
    total_seconds: float
    worst_case_seconds: float
    first_visible_verdict_seconds: float
    check_interval_seconds: float
    boot_grace_seconds: float
    probe_timeout_seconds: float

    def describe(self) -> str:
        return (
            f"verdict_wait: {self.attempts} attempts x "
            f"{self.interval_seconds:g}s = {int(self.total_seconds)}s sleep, "
            f"worst case {int(self.worst_case_seconds)}s wall clock "
            f"(each probe may spend up to {self.probe_timeout_seconds:g}s "
            f"before its sleep); first visible verdict "
            f"~{int(self.first_visible_verdict_seconds)}s from "
            f"check_interval={self.check_interval_seconds:g}s, "
            f"boot_grace={self.boot_grace_seconds:g}s"
        )


def derive_verdict_wait_bound(
    *,
    check_interval_seconds: float,
    boot_grace_seconds: float,
    margin_seconds: float = VERDICT_WAIT_MARGIN_SECONDS,
    poll_seconds: float = VERDICT_WAIT_POLL_SECONDS,
    probe_timeout_seconds: float = VERDICT_PROBE_TIMEOUT_SECONDS,
) -> VerdictWaitBound:
    """Derive how long to wait for a first ``details.runtime_health`` verdict.

    ``ServiceRuntimeHealthMonitor._loop`` sleeps one ``check_interval`` BEFORE
    its first check, so checks land at 1x, 2x, 3x the interval -- never at t=0.
    ``_emit`` then suppresses publication while ``elapsed < boot_grace``.

    The first VISIBLE verdict is therefore the first check strictly after the
    grace window. "Strictly after" is deliberate: a check landing exactly on
    the boundary passes ``elapsed < grace`` by a hair of monotonic clock, and a
    gate that bets on that tie fails intermittently for a reason nobody can
    reproduce.

    Hardcoding 390s would be correct only for the default 300s/120s pair. A
    lane that shortens its interval would get a bound that outlives the signal
    it waits for, which is the same false-PASS shape in a new costume.
    """
    if check_interval_seconds <= 0:
        raise ValueError(
            f"check_interval_seconds must be positive, got {check_interval_seconds}"
        )
    if boot_grace_seconds < 0:
        raise ValueError(
            f"boot_grace_seconds must be non-negative, got {boot_grace_seconds}"
        )
    if poll_seconds <= 0:
        raise ValueError(f"poll_seconds must be positive, got {poll_seconds}")

    checks_within_grace = math.floor(boot_grace_seconds / check_interval_seconds)
    first_visible = (checks_within_grace + 1) * check_interval_seconds
    total = first_visible + margin_seconds
    attempts = max(1, math.ceil(total / poll_seconds))
    return VerdictWaitBound(
        attempts=attempts,
        interval_seconds=poll_seconds,
        total_seconds=attempts * poll_seconds,
        worst_case_seconds=attempts * (poll_seconds + probe_timeout_seconds),
        first_visible_verdict_seconds=first_visible,
        check_interval_seconds=check_interval_seconds,
        boot_grace_seconds=boot_grace_seconds,
        probe_timeout_seconds=probe_timeout_seconds,
    )


def default_max_verdict_age(check_interval_seconds: float) -> float:
    """Freshness ceiling derived from the monitor's own cadence.

    Exists so freshness has a value when no caller supplies one. An opt-in
    freshness check is not a freshness check: a monitor that publishes once
    and then crashes serves the same verdict forever, and an unbounded gate
    accepts it forever.
    """
    return check_interval_seconds * VERDICT_STALE_AFTER_INTERVALS


def wait_for_verdict(
    probe: Callable[[], HealthVerdict],
    *,
    bound: VerdictWaitBound,
    sleep_fn: object | None = None,
) -> tuple[HealthVerdict, str]:
    """Re-probe until a monitor verdict exists, for a bounded window.

    Shared by both refresh gates. The per-lane HTTP fetch stays in each gate
    (they already each own a ``check_health``), but the WAIT POLICY lives here
    once -- duplicating a fetch is cheap, duplicating the rule that decides
    whether a lane is stability-proven is not.

    Mirrors ``check_consumer_group_with_retry``: bounded attempts at a fixed
    interval, never an unbounded poll, and the LAST observed result is
    returned. A lane that never produces a verdict inside the window is a
    genuine, accurately-reported finding -- not a reason to pass.

    Returns the verdict and the human-readable bound for the receipt (AC3).
    """
    sleep = sleep_fn or time.sleep
    verdict = HealthVerdict(
        ok=False,
        policy=HEALTH_POLICY_VERDICT_REQUIRED,
        status=None,
        details_healthy=None,
        detail="health never probed",
    )
    for attempt in range(1, bound.attempts + 1):
        verdict = probe()
        if verdict.ok:
            return verdict, f"{bound.describe()} satisfied on attempt {attempt}"
        if verdict.reason != REASON_VERDICT_ABSENT:
            # Waiting only helps a verdict that has not been published YET.
            # A stale verdict, an unreadable body or a dead endpoint is just
            # as true on attempt 24 as on attempt 1; retrying it converts a
            # fast correct failure into a multi-minute stall on every refresh.
            return verdict, (
                f"{bound.describe()} not waited: failure is terminal "
                f"(reason={verdict.reason!r}) on attempt {attempt}"
            )
        if attempt < bound.attempts:
            sleep(bound.interval_seconds)  # type: ignore[operator]
    return verdict, f"{bound.describe()} exhausted without a verdict"


def evaluate_health_body(
    raw: bytes | str,
    *,
    require_verdict: bool = False,
    max_verdict_age_seconds: float | None = None,
) -> HealthVerdict:
    """Strict verdict for a fetched ``/health`` body. Fails closed.

    OMN-17624: with ``require_verdict=True`` a body carrying no fresh
    ``details.runtime_health`` is refused rather than accepted. Absence is NOT
    treated as "monitor disabled, skip" -- ``service_kernel`` starts the
    monitor only ``if use_kafka`` and swallows a start failure with
    ``except Exception: ... runtime_health_monitor = None``, so a crashed
    monitor and an intentionally absent one look identical at this boundary.
    Skipping on absence would rebuild the blind spot this closes. A
    monitor-less profile opts out explicitly at the call site, on the record.

    Freshness is delegated to ``evaluate_health_response`` rather than
    reimplemented -- it already owns ``max_verdict_age_seconds`` and is a pure
    function (no clock, no I/O, no environment), so this stays hermetic.
    """
    try:
        payload = parse_health_payload(raw)
    except HealthPayloadError as exc:
        return HealthVerdict(
            ok=False,
            policy=(
                HEALTH_POLICY_VERDICT_REQUIRED
                if require_verdict
                else HEALTH_POLICY_STATUS_ONLY_STRICT
            ),
            status=None,
            details_healthy=None,
            detail=str(exc),
        )

    if not require_verdict:
        return HealthVerdict(
            ok=payload.healthy,
            policy=HEALTH_POLICY_STATUS_ONLY_STRICT,
            status=payload.status,
            details_healthy=payload.details_healthy,
            detail=payload.describe(),
        )

    from omnibase_infra.runtime.health.container_healthcheck import (
        evaluate_health_response,
    )

    # Re-decoded rather than threaded through HealthPayload: that dataclass is
    # compared by equality in existing tests, so widening it would edit a test
    # whose subject is not this change. parse_health_payload already proved the
    # body decodes, so this cannot raise.
    decoded_body = json.loads(raw)
    container_verdict = evaluate_health_response(
        http_status=200,
        payload=decoded_body,
        require_verdict=True,
        max_verdict_age_seconds=max_verdict_age_seconds,
    )
    ok = container_verdict.verdict == "PASS"
    return HealthVerdict(
        ok=ok,
        policy=HEALTH_POLICY_VERDICT_REQUIRED,
        status=payload.status,
        details_healthy=payload.details_healthy,
        detail=(
            payload.describe()
            if ok
            else f"{payload.describe()} (verdict gate: {container_verdict.reason})"
        ),
        reason=None if ok else container_verdict.reason,
    )


def unreachable_verdict(detail: str) -> HealthVerdict:
    """Verdict for a probe that never produced a body (transport / non-200)."""
    return HealthVerdict(
        ok=False,
        policy=HEALTH_POLICY_STATUS_ONLY_STRICT,
        status=None,
        details_healthy=None,
        detail=detail,
    )
