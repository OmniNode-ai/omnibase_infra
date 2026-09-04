# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Semantic container healthcheck for runtime lanes (OMN-15217).

The pre-OMN-15217 container healthcheck was::

    test: ["CMD", "curl", "-sf", "<runtime-loopback>:8085/health"]

``curl -sf`` asserts one thing: the response code is < 400. That is a liveness
probe wearing a health probe's name. Observed on the stability lane
2026-07-27T12:58Z: ``docker ps`` reported ``Up 3 hours (healthy)`` for
``omninode-stability-test-runtime`` while that same runtime logged
``Runtime health check: status=DEGRADED contracts=296 errors=4`` every five
minutes. A promotion gate, an operator, or a proof packet citing that green is
citing a false signal — which is what blocked the OMN-15181 prod bootstrap.

This module reads the runtime's *semantic* health instead: the
``details.runtime_health`` block published by
:mod:`omnibase_infra.runtime.health.runtime_health_block`, which carries the
``ServiceRuntimeHealthMonitor`` verdict.

Usage as a container healthcheck (compose)::

    healthcheck:
      test: ["CMD", "python", "/usr/local/bin/onex-container-healthcheck",
             "--degraded-policy", "fail"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 1800s

``Dockerfile.runtime`` installs this file at
``/usr/local/bin/onex-container-healthcheck``. It is invoked as a *file*, not as
``python -m omnibase_infra...``: importing the package chain costs ~6.8s inside
the runtime image against a 10s probe timeout, while this stdlib-only module
starts in ~0.12s (both measured in-container 2026-07-27). That is why nothing
here imports from ``omnibase_infra`` — a unit test pins the property.

``start_period`` and ``retries`` are deliberately preserved from the shallow
check: contract discovery and Kafka group joins take many minutes on a cold
runtime, and the monitor's first verdict lands one check interval after boot.
Within ``start_period`` Docker reports ``starting``, so a not-yet-computed
verdict never flaps a booting container.

Usage as a proof reader (gates, promotion checks)::

    python -m omnibase_infra.runtime.health.container_healthcheck \\
        --url <lane-health-url> --require-verdict --json

(off the container's critical path, so the import cost is irrelevant there)

``--require-verdict`` fails closed when no monitor verdict is present: for a
liveness probe an absent verdict is "not known yet" (pass, do not restart), but
for a *proof* consumer an absent verdict is "cannot prove healthy" (fail).
Same evaluator, different policy, one code path.

An **unreadable top-level status** is treated differently from an absent verdict,
and the restart-loop question that distinction raises was decided as follows
(OMN-17623). A body whose ``status`` is absent, empty, null, non-string, or
outside ``healthy``/``degraded``/``unhealthy`` fails closed as
``status_unreadable`` on *every* policy, not only under ``--require-verdict``:

* ``ServiceHealth._handle_health`` types its status
  ``Literal["healthy", "degraded", "unhealthy"]``, every branch assigns one of
  the three, and ``fold_runtime_verdict_into_status`` returns the same Literal —
  so no ONEX runtime can serve such a body. A booting runtime serves
  ``unhealthy``/``degraded``, never ``starting``. The condition means the probe
  is not talking to a runtime health endpoint at all.
* This module already fails closed on the strictly *more* broken inputs — an
  unreachable endpoint (``probe_unreachable``) and a body that is absent or not
  a JSON object (``payload_missing``). Passing the less-broken case in between
  was the inconsistency, not a deliberate leniency.
* Docker suppresses probe failures for the entire ``start_period`` (120s here,
  1200-1800s on the dev lane) and then requires 3-5 *consecutive* failures, so
  this cannot flap a container over one malformed response.

That is why it is unconditional, unlike ``verdict_absent`` / ``verdict_stale``
below: those cover the monitor's optional enrichment, which legitimately has not
landed one interval after boot. The top-level status is the health contract
itself and is present from the first response.

Exit codes: ``0`` healthy, ``1`` not healthy (Docker's unhealthy signal).
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Literal

DEGRADED_POLICY_FAIL = "fail"
DEGRADED_POLICY_WARN = "warn"

_DEFAULT_TIMEOUT_SECONDS = 5.0
_DEFAULT_PORT = 8085
_MAX_RESPONSE_BYTES = 4 * 1024 * 1024

# Mirrors runtime_health_block.RUNTIME_HEALTH_DETAIL_KEY. Duplicated as a plain
# literal (with a seam test pinning the two together) so the healthcheck process
# imports nothing but the standard library — it runs on every probe interval
# inside a container with a 10s timeout, and package import cost is not a budget
# a health probe should spend.
RUNTIME_HEALTH_DETAIL_KEY = "runtime_health"

# The closed set ``ServiceHealth._handle_health`` can serve. Its local is typed
# ``Literal["healthy", "degraded", "unhealthy"]`` and every branch assigns one of
# the three, as does ``fold_runtime_verdict_into_status``, so a status outside
# this set did not come from an ONEX runtime. Mirrored as plain literals for the
# stdlib-only reason in ``RUNTIME_HEALTH_DETAIL_KEY`` above.
_RECOGNISED_STATUSES = frozenset({"healthy", "degraded", "unhealthy"})

# A status echoed into container logs is bounded: the body is read up to
# _MAX_RESPONSE_BYTES and a wrong endpoint can put anything in that field.
_MAX_ECHOED_STATUS_CHARS = 40

EXIT_HEALTHY = 0
EXIT_UNHEALTHY = 1

_VerdictLiteral = Literal["PASS", "FAIL"]


class ContainerHealthVerdict:
    """Result of evaluating a ``/health`` response.

    Attributes:
        verdict: ``PASS`` or ``FAIL``.
        reason: Stable machine-greppable reason code (e.g. ``runtime_degraded``).
        detail: Human-readable detail for container logs / ``docker inspect``.
    """

    __slots__ = ("detail", "reason", "verdict")

    def __init__(self, verdict: _VerdictLiteral, reason: str, detail: str = "") -> None:
        self.verdict = verdict
        self.reason = reason
        self.detail = detail

    @property
    def exit_code(self) -> int:
        """Process exit code — ``0`` for PASS, ``1`` for FAIL."""
        return EXIT_HEALTHY if self.verdict == "PASS" else EXIT_UNHEALTHY

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable view (for ``--json``)."""
        return {
            "verdict": self.verdict,
            "reason": self.reason,
            "detail": self.detail,
            "exit_code": self.exit_code,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"ContainerHealthVerdict(verdict={self.verdict!r}, "
            f"reason={self.reason!r}, detail={self.detail!r})"
        )


def evaluate_health_response(
    *,
    http_status: int | None,
    payload: Mapping[str, object] | None,
    degraded_policy: str = DEGRADED_POLICY_FAIL,
    require_verdict: bool = False,
    max_verdict_age_seconds: float | None = None,
) -> ContainerHealthVerdict:
    """Evaluate a ``/health`` response into a container health verdict.

    Pure function — no I/O, no clock, no environment reads — so the full verdict
    table is testable hermetically against recorded payloads.

    Args:
        http_status: HTTP status code, or ``None`` if the endpoint was
            unreachable / the response was not parseable as JSON.
        payload: Decoded ``/health`` JSON body, or ``None``.
        degraded_policy: ``fail`` (default) treats a DEGRADED runtime as
            unhealthy; ``warn`` reports it but still exits 0.
        require_verdict: When ``True``, a missing monitor verdict fails closed.
        max_verdict_age_seconds: When set, a verdict older than this is treated
            as stale. Stale fails closed only under ``require_verdict``, because
            a liveness probe must not restart a container merely for a monitor
            cycle that has not landed yet.

    Returns:
        A :class:`ContainerHealthVerdict`.
    """
    if http_status is None:
        return ContainerHealthVerdict(
            "FAIL", "probe_unreachable", "health endpoint unreachable or unparseable"
        )
    if http_status >= 400:
        return ContainerHealthVerdict(
            "FAIL", "http_error", f"health endpoint returned HTTP {http_status}"
        )
    if payload is None:
        return ContainerHealthVerdict(
            "FAIL", "payload_missing", "health endpoint returned no JSON body"
        )

    fail_on_degraded = degraded_policy != DEGRADED_POLICY_WARN

    top_status = str(payload.get("status", "")).lower()

    # OMN-17623: an unrecognised status is unknown health, not proven health.
    # Unconditional rather than gated behind require_verdict — see the module
    # docstring for the restart-loop reasoning that settles it.
    if top_status not in _RECOGNISED_STATUSES:
        observed = repr(payload.get("status"))[:_MAX_ECHOED_STATUS_CHARS]
        return ContainerHealthVerdict(
            "FAIL",
            "status_unreadable",
            f"health payload status {observed} is not one of "
            f"healthy/degraded/unhealthy — health is unknown, not proven",
        )

    if top_status == "unhealthy":
        return ContainerHealthVerdict(
            "FAIL", "runtime_unhealthy", "health payload reports status=unhealthy"
        )

    details_raw = payload.get("details")
    details: Mapping[str, object] = (
        details_raw if isinstance(details_raw, Mapping) else {}
    )

    # --- Semantic verdict from ServiceRuntimeHealthMonitor -------------------
    verdict_raw = details.get(RUNTIME_HEALTH_DETAIL_KEY)
    verdict_block: Mapping[str, object] | None = (
        verdict_raw if isinstance(verdict_raw, Mapping) else None
    )

    if verdict_block is None:
        if require_verdict:
            return ContainerHealthVerdict(
                "FAIL",
                "verdict_absent",
                "health payload carries no runtime_health verdict — health is "
                "unknown, not proven",
            )
    else:
        verdict_status = str(verdict_block.get("status", "")).upper()

        if max_verdict_age_seconds is not None:
            age = _coerce_float(verdict_block.get("age_seconds"))
            if require_verdict and (age is None or age > max_verdict_age_seconds):
                age_text = "unknown" if age is None else f"{age:.0f}s"
                return ContainerHealthVerdict(
                    "FAIL",
                    "verdict_stale",
                    f"runtime_health verdict age {age_text} exceeds "
                    f"{max_verdict_age_seconds:.0f}s",
                )

        if verdict_status == "CRITICAL":
            return ContainerHealthVerdict(
                "FAIL",
                "runtime_critical",
                _describe_dimensions(verdict_block, "CRITICAL")
                or "runtime health monitor reports CRITICAL",
            )
        if verdict_status == "DEGRADED":
            detail = (
                _describe_dimensions(verdict_block, "DEGRADED")
                or "runtime health monitor reports DEGRADED"
            )
            if fail_on_degraded:
                return ContainerHealthVerdict("FAIL", "runtime_degraded", detail)
            return ContainerHealthVerdict("PASS", "runtime_degraded_warn", detail)

    # --- Process-level degradation (handlers failed to instantiate) ----------
    if top_status == "degraded" or bool(details.get("degraded", False)):
        detail = "health payload reports a degraded runtime process"
        if fail_on_degraded:
            return ContainerHealthVerdict("FAIL", "process_degraded", detail)
        return ContainerHealthVerdict("PASS", "process_degraded_warn", detail)

    return ContainerHealthVerdict("PASS", "healthy", "runtime healthy")


def _describe_dimensions(
    verdict_block: Mapping[str, object], status: str
) -> str | None:
    """Summarize the dimensions matching ``status`` for the failure detail."""
    dimensions = verdict_block.get("dimensions")
    if not isinstance(dimensions, Sequence) or isinstance(dimensions, str | bytes):
        return None
    parts: list[str] = []
    for dimension in dimensions:
        if not isinstance(dimension, Mapping):
            continue
        if str(dimension.get("status", "")).upper() != status:
            continue
        name = str(dimension.get("name", "unknown"))
        detail = str(dimension.get("detail", "")).strip()
        parts.append(f"{name}: {detail}" if detail else name)
    if not parts:
        return None
    return f"runtime health {status} — " + "; ".join(parts)


def _coerce_float(value: object) -> float | None:
    """Best-effort float coercion for values arriving from JSON."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def fetch_health(
    url: str, timeout_seconds: float
) -> tuple[int | None, Mapping[str, object] | None]:
    """Fetch and decode the health endpoint.

    Returns:
        ``(http_status, payload)``. ``http_status`` is ``None`` when the
        endpoint could not be reached at all; ``payload`` is ``None`` when the
        body was absent or not a JSON object.
    """
    try:
        # Localhost-only container probe; the URL comes from the container's own
        # configuration, never from a request.
        with urllib.request.urlopen(url, timeout=timeout_seconds) as response:  # noqa: S310
            status = int(response.status)
            body = response.read(_MAX_RESPONSE_BYTES)
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read(_MAX_RESPONSE_BYTES)
        except Exception:  # noqa: BLE001 — boundary: error body is best-effort
            body = b""
        return int(exc.code), _decode(body)
    except Exception:  # noqa: BLE001 — boundary: any transport failure is unreachable
        return None, None

    return status, _decode(body)


def _decode(body: bytes) -> Mapping[str, object] | None:
    """Decode a JSON object body, or ``None`` when it is not one."""
    if not body:
        return None
    try:
        decoded = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    return decoded if isinstance(decoded, dict) else None


# Default probe target: this container's own health endpoint on the loopback
# interface. A container healthcheck checks the process it runs beside — there
# is no routing authority for "myself", and resolving a service address here
# would probe a *different* replica and report the wrong container's health.
# Lanes that move the listener off 8085 (ONEX_HTTP_PORT) must pass --url in the
# compose healthcheck; this process reads no environment (the check-env-reads
# gate keeps configuration in the overlay, and a healthcheck's configuration
# belongs on its command line where `docker inspect` can show it).
DEFAULT_HEALTH_URL = f"http://localhost:{_DEFAULT_PORT}/health"  # url-authority-ok: container self-probe on loopback


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser (exposed for tests)."""
    parser = argparse.ArgumentParser(
        prog="container_healthcheck",
        description=(
            "Semantic container healthcheck — consumes the runtime's own health "
            "verdict instead of trusting an HTTP 200 (OMN-15217)."
        ),
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_HEALTH_URL,
        help=f"Health endpoint URL (default: {DEFAULT_HEALTH_URL})",
    )
    parser.add_argument(
        "--degraded-policy",
        choices=[DEGRADED_POLICY_FAIL, DEGRADED_POLICY_WARN],
        default=DEGRADED_POLICY_FAIL,
        help="How to treat a DEGRADED runtime verdict (default: fail)",
    )
    parser.add_argument(
        "--require-verdict",
        action="store_true",
        default=False,
        help=(
            "Fail closed when no runtime_health verdict is present or it is "
            "stale — use for proof/promotion readers, not for liveness probes"
        ),
    )
    parser.add_argument(
        "--max-verdict-age-seconds",
        type=float,
        default=None,
        help="Treat a verdict older than this as stale (default: no age limit)",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=_DEFAULT_TIMEOUT_SECONDS,
        help="HTTP timeout in seconds (default: 5)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the verdict as JSON instead of a single human-readable line",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint. Returns the process exit code."""
    args = build_parser().parse_args(argv)

    http_status, payload = fetch_health(args.url, args.timeout_seconds)
    verdict = evaluate_health_response(
        http_status=http_status,
        payload=payload,
        degraded_policy=args.degraded_policy,
        require_verdict=args.require_verdict,
        max_verdict_age_seconds=args.max_verdict_age_seconds,
    )

    # Written to stdout (not logged): Docker captures healthcheck output into
    # the container's health log, which is where `docker inspect` surfaces the
    # reason a container went unhealthy.
    if args.json:
        line = json.dumps(verdict.as_dict(), sort_keys=True)
    else:
        line = f"{verdict.verdict} [{verdict.reason}] {verdict.detail}".rstrip()
    sys.stdout.write(line + "\n")
    return verdict.exit_code


if __name__ == "__main__":  # pragma: no cover - process entrypoint
    sys.exit(main())


__all__: list[str] = [
    "DEGRADED_POLICY_FAIL",
    "DEGRADED_POLICY_WARN",
    "DEFAULT_HEALTH_URL",
    "EXIT_HEALTHY",
    "EXIT_UNHEALTHY",
    "RUNTIME_HEALTH_DETAIL_KEY",
    "ContainerHealthVerdict",
    "build_parser",
    "evaluate_health_response",
    "fetch_health",
    "main",
]
