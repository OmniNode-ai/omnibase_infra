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
from dataclasses import dataclass

# Recorded verbatim in the refresh receipt (OMN-17563 AC-3) so a future reader
# can tell a strict PASS from a lenient one without re-deriving it. Bump the
# value -- never redefine it in place -- if the policy itself ever changes.
HEALTH_POLICY_STATUS_ONLY_STRICT = "status_only_strict.v1"

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


def evaluate_health_body(raw: bytes | str) -> HealthVerdict:
    """Strict verdict for a fetched ``/health`` body. Fails closed."""
    try:
        payload = parse_health_payload(raw)
    except HealthPayloadError as exc:
        return HealthVerdict(
            ok=False,
            policy=HEALTH_POLICY_STATUS_ONLY_STRICT,
            status=None,
            details_healthy=None,
            detail=str(exc),
        )
    return HealthVerdict(
        ok=payload.healthy,
        policy=HEALTH_POLICY_STATUS_ONLY_STRICT,
        status=payload.status,
        details_healthy=payload.details_healthy,
        detail=payload.describe(),
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
