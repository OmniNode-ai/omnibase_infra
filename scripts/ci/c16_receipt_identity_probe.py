# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-19181 -- the C16 producer: receipt identity, SKIP never PASS, typed death.

WHAT THIS IS
    A probe that drives beta criterion C16 against a deployed lane's
    tenant-bearing API and grades its three named probes against expectations
    declared HERE, in the grader. Its EXIT CODE is the criterion's verdict --
    the shape C15 (chain-canary.yml) and C11
    (chain-canary-c11-negative-paths.yml) already set -- and it writes a JSON
    record so a reader can see the three rows without reading a log.

    ``0``  all three probes graded PASS
    ``1``  at least one probe graded FAIL or SKIP; the record names which and why
    ``2``  the probe could not run at all (unresolvable credential or DSN, a
           ledger role that bypasses row-level security, an unreadable replay
           file). Distinct from ``1`` on purpose: "I could not run" is not "the
           platform is wrong", and collapsing them is how a configuration
           failure gets reported as a product failure.

WHAT C16 SAYS, AND WHICH PROBE ANSWERS EACH CLAUSE
    "Receipt identity equals route; SKIP never PASS; a failed run never
    reports ok=true; every dead run emits a typed terminal failure cause that
    the caller can distinguish from still running, including a config-class
    dispatch failure." Two delegations are submitted per invocation:

    * the HEALTHY run -- the chain canary's own liveness prompt and task class,
      which the lane serves every two hours;
    * the DYING run -- identical except for a task class no consumer accepts.
      The gateway declares ``task_type`` as a bare string (the class universe
      is registry-owned), so the submission is accepted with 202 and the run
      dies one hop downstream, at the consumer's closed class set. That is a
      config-class dispatch failure produced by a CALLER mistake, not a fault
      injected into the lane: it breaks nothing and needs no lane mutation.

    R-DELEG-12  receipt identity equals route. The HEALTHY run's receipt
                (``GET /v1/workflows/{id}/receipt``) names ``route``,
                ``provider`` and ``terminal_model_used``; the route the chain
                actually TOOK is read independently, from the event ledger's
                last ``routing-decision`` hop and its terminal hop for the same
                correlation id. Every key must be present and non-empty on both
                sides and equal. The comparison is across two surfaces on
                purpose: the status and receipt routes read one database row,
                so comparing them to each other would be a tautology.
    R-DELEG-11  SKIP never PASS; a failed run never reports success. The
                HEALTHY run, if completed, must carry a NON-EMPTY list of
                quality-rule evaluations, every one of which states a boolean
                verdict, every blocking one ``True``. A completed run with no
                evaluated rule -- or a rule whose ``passed`` is null -- is a
                verifier that skipped reading as a pass. The DYING run must
                never read ``completed`` on the status or the receipt.
    R-DELEG-26  every dead run emits a typed terminal failure cause the caller
                can tell apart from still running. The DYING run must reach
                ``failed`` inside the budget (``published`` at the deadline is
                indistinguishable from running), and its receipt must carry a
                ``terminal_failure_class`` and ``terminal_failure_code`` in the
                gateway's typed grammar, agreeing with the status endpoint.

SKIP IS A RECOGNISED OUTCOME, AND IT IS RED
    A probe whose subject never materialised -- the healthy run did not
    complete, so there is no route to compare; the gateway refused the dying
    submission, so there is no dead run -- grades ``SKIP`` with a reason, and
    ``SKIP`` fails the verdict exactly as ``FAIL`` does. On the one criterion
    whose own subject is "SKIP never PASS", a grader that let a skipped probe
    through would refute the criterion in its own plumbing.

WHY A REPLAY MODE EXISTS
    ``--replay`` grades recorded observations instead of making requests. That
    is what makes the grading falsifiable offline, in the ordinary test suite,
    including the cases a live run cannot produce on demand (a receipt naming a
    route the chain did not take; a dead run whose cause is untyped).

NO CREDENTIAL REACHES ARGV, A LOG OR THE RECORD
    The API key and the ledger DSN are read from the environment by NAME.
    ``/proc`` is world-readable, so a value on a command line is readable by
    every process on the host; the record is an uploaded artifact, so a value
    in it would be a real exposure. The record also carries no tenant id, no
    endpoint URL and no response text: only the keys it compares.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

HEADER_NAME: Final[str] = "x-api-key"
WORKFLOW_TYPE: Final[str] = "delegate-skill"

# The healthy run is the chain canary's own liveness shape, so a green here is
# evidence about the delegation the lane already serves every two hours rather
# than about a bespoke probe-only path.
HEALTHY_PAYLOAD: Final[dict[str, Any]] = {
    "prompt": (
        "Reply with the single word: alive. This is an automated C16 "
        "receipt-identity probe of the delegation chain."
    ),
    "task_type": "test",
    "source": "external-client",
    "max_tokens": 32,
}

# The dying run differs from the healthy one in exactly one field.
DYING_TASK_TYPE: Final[str] = "omn19181_not_a_task_class"
DYING_PAYLOAD: Final[dict[str, Any]] = {**HEALTHY_PAYLOAD, "task_type": DYING_TASK_TYPE}

PROBES: Final[tuple[str, ...]] = ("R-DELEG-11", "R-DELEG-12", "R-DELEG-26")
PASS: Final[str] = "PASS"
FAIL: Final[str] = "FAIL"
SKIP: Final[str] = "SKIP"

TERMINAL_STATUSES: Final[frozenset[str]] = frozenset({"completed", "failed"})

ROUTING_DECISION_TOPIC: Final[str] = "onex.evt.omnibase-infra.routing-decision.v1"
LEDGER_TERMINAL_TOPICS: Final[frozenset[str]] = frozenset(
    {"onex.evt.omnimarket.delegate-skill-completed.v1"}
)

# The gateway's typed failure grammar, read from the one construction site that
# builds it (onex-api workflow_failure_attribution._ATTRIBUTION_GRAMMAR): a
# class with a conventional Error/Exception suffix and a canonical ONEX_ code.
FAILURE_CLASS_RE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(Error|Exception)$"
)
FAILURE_CODE_RE: Final[re.Pattern[str]] = re.compile(r"^ONEX_[A-Z0-9_]+$")

RECORD_VERSION: Final[int] = 1

EXIT_OK: Final[int] = 0
EXIT_FINDINGS: Final[int] = 1
EXIT_INPUT: Final[int] = 2


class ProbeInputError(RuntimeError):
    """The probe could not run. Exit 2, never exit 1."""


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunObservation:
    """One submitted delegation, as the gateway reported it."""

    submit_status: int | None
    status: Mapping[str, Any] | None = None
    receipt: Mapping[str, Any] | None = None
    error: str | None = None


@dataclass(frozen=True)
class LedgerObservation:
    """The same run's hops, read from the event ledger by correlation id."""

    routing_decisions: tuple[Mapping[str, Any], ...] = ()
    terminal: Mapping[str, Any] | None = None
    error: str | None = None


@dataclass(frozen=True)
class Observations:
    healthy: RunObservation
    healthy_ledger: LedgerObservation
    dying: RunObservation


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------


@dataclass
class ProbeResult:
    probe: str
    outcome: str
    reasons: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "probe": self.probe,
            "outcome": self.outcome,
            "reasons": list(self.reasons),
            "evidence": dict(self.evidence),
        }


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _status_of(body: Mapping[str, Any] | None) -> str | None:
    return _text(body.get("status")) if body is not None else None


def _payload_of(envelope: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = envelope.get("payload")
    return payload if isinstance(payload, Mapping) else {}


def grade_identity(healthy: RunObservation, ledger: LedgerObservation) -> ProbeResult:
    """R-DELEG-12: the receipt names the route the chain actually took."""
    result = ProbeResult("R-DELEG-12", FAIL)
    receipt_status = _status_of(healthy.receipt)
    if healthy.receipt is None or receipt_status != "completed":
        result.outcome = SKIP
        result.reasons.append(
            "the healthy run produced no completed receipt "
            f"(submit={healthy.submit_status}, receipt status={receipt_status!r}"
            f"{', error=' + healthy.error if healthy.error else ''}), so there "
            "is no route on the receipt side to compare. SKIP is not PASS"
        )
        return result
    if ledger.error is not None:
        result.outcome = SKIP
        result.reasons.append(
            f"the event ledger could not be read ({ledger.error}), so the route "
            "the chain took has no independent observation. SKIP is not PASS"
        )
        return result
    if not ledger.routing_decisions or ledger.terminal is None:
        result.reasons.append(
            "the event ledger carries "
            f"{len(ledger.routing_decisions)} routing-decision hop(s) and "
            f"{'a' if ledger.terminal is not None else 'no'} terminal hop for "
            "this correlation id; a completed receipt whose route the ledger "
            "cannot corroborate names a route nobody can show was taken"
        )
        return result

    decision = _payload_of(ledger.routing_decisions[-1])
    terminal = _payload_of(ledger.terminal)
    receipt = healthy.receipt
    # Each key: the receipt's value, then every independent observation of it.
    comparisons: dict[str, tuple[Any, tuple[tuple[str, Any], ...]]] = {
        "route": (
            receipt.get("route"),
            (("routing_decision.route", decision.get("route")),),
        ),
        "provider": (
            receipt.get("provider"),
            (
                ("routing_decision.provider", decision.get("provider")),
                ("terminal.provider", terminal.get("provider")),
            ),
        ),
        "model": (
            receipt.get("terminal_model_used"),
            (
                ("routing_decision.selected_model", decision.get("selected_model")),
                ("terminal.model_name", terminal.get("model_name")),
            ),
        ),
    }
    for key, (claimed, observed) in comparisons.items():
        result.evidence[f"receipt.{key}"] = claimed
        if _text(claimed) is None:
            result.reasons.append(
                f"the receipt carries no usable {key} ({claimed!r}); an absent "
                "key must not drop out of the comparison -- two blanks are equal "
                "and are not an identity"
            )
            continue
        for label, value in observed:
            result.evidence[label] = value
            if _text(value) is None:
                result.reasons.append(
                    f"{label} is {value!r}, so the receipt's {key}={claimed!r} "
                    "has nothing to be compared against"
                )
            elif value != claimed:
                result.reasons.append(
                    f"the receipt names {key}={claimed!r} but {label}={value!r}"
                )
    if not result.reasons:
        result.outcome = PASS
    return result


def grade_skip_never_pass(
    healthy: RunObservation, dying: RunObservation
) -> ProbeResult:
    """R-DELEG-11: no skipped verdict reads as a pass, no failure as success."""
    result = ProbeResult("R-DELEG-11", FAIL)
    receipt_status = _status_of(healthy.receipt)
    status_status = _status_of(healthy.status)
    result.evidence["healthy.receipt.status"] = receipt_status
    result.evidence["healthy.status.status"] = status_status
    result.evidence["dying.receipt.status"] = _status_of(dying.receipt)
    result.evidence["dying.status.status"] = _status_of(dying.status)

    # The dying arm first: it is gradeable whether or not the healthy one is.
    for label, body in (("status", dying.status), ("receipt", dying.receipt)):
        if _status_of(body) == "completed":
            result.reasons.append(
                f"the dying run's {label} reports 'completed' for a task class "
                f"no consumer accepts ({DYING_TASK_TYPE!r}): a failed run "
                "reported success"
            )

    if healthy.receipt is None or receipt_status != "completed":
        if not result.reasons:
            result.outcome = SKIP
            result.reasons.append(
                "the healthy run produced no completed receipt "
                f"(receipt status={receipt_status!r}), so there is no quality "
                "verdict to hold to SKIP-never-PASS. SKIP is not PASS"
            )
        return result

    if status_status != receipt_status:
        result.reasons.append(
            f"the status endpoint reads {status_status!r} while the receipt "
            f"reads {receipt_status!r} for the same run"
        )
    rules = healthy.receipt.get("rule_evaluations")
    if not isinstance(rules, list) or not rules:
        result.reasons.append(
            "the healthy run reads 'completed' with no evaluated quality rule "
            f"(rule_evaluations={rules!r}): a verdict nobody reached, reported "
            "as a pass"
        )
    else:
        result.evidence["healthy.rules_evaluated"] = len(rules)
        for index, rule in enumerate(rules):
            if not isinstance(rule, Mapping):
                result.reasons.append(f"rule_evaluations[{index}] is not an object")
                continue
            name = rule.get("rule", f"#{index}")
            passed = rule.get("passed")
            if not isinstance(passed, bool):
                result.reasons.append(
                    f"rule {name!r} carries passed={passed!r} on a completed run: "
                    "a skipped rule counted toward a pass"
                )
            elif not passed and rule.get("enforcement") == "blocking":
                result.reasons.append(
                    f"blocking rule {name!r} did not pass, yet the run reads "
                    "'completed'"
                )
    if not result.reasons:
        result.outcome = PASS
    return result


def grade_typed_death(dying: RunObservation) -> ProbeResult:
    """R-DELEG-26: a dead run ends typed, and distinguishably from running."""
    result = ProbeResult("R-DELEG-26", FAIL)
    result.evidence["dying.submit_status"] = dying.submit_status
    if dying.submit_status != 202:
        result.outcome = SKIP
        result.reasons.append(
            f"the dying submission was answered {dying.submit_status} rather than "
            "accepted, so no run existed to die. The dying arm must be re-derived "
            "from what the gateway now refuses at ingress; SKIP is not PASS"
        )
        return result

    status_status = _status_of(dying.status)
    result.evidence["dying.status.status"] = status_status
    if status_status not in TERMINAL_STATUSES:
        result.reasons.append(
            f"the dying run still reads {status_status!r} at the deadline"
            f"{' (' + dying.error + ')' if dying.error else ''}: a run that died "
            "is indistinguishable from one still running"
        )
        return result
    if status_status != "failed":
        result.reasons.append(f"the dying run reached {status_status!r}, not 'failed'")
        return result
    if dying.receipt is None:
        result.reasons.append(
            "the dying run reads 'failed' but no receipt could be fetched for it"
        )
        return result

    for key, pattern in (
        ("terminal_failure_class", FAILURE_CLASS_RE),
        ("terminal_failure_code", FAILURE_CODE_RE),
    ):
        on_receipt = dying.receipt.get(key)
        on_status = dying.status.get(key) if dying.status is not None else None
        result.evidence[f"receipt.{key}"] = on_receipt
        if not isinstance(on_receipt, str) or not pattern.match(on_receipt):
            result.reasons.append(
                f"the dead run's receipt carries {key}={on_receipt!r}, which is "
                "not a typed cause: the caller learns that it failed and not why"
            )
        elif on_status != on_receipt:
            result.reasons.append(
                f"the status endpoint carries {key}={on_status!r} while the "
                f"receipt carries {on_receipt!r}"
            )
    if not result.reasons:
        result.outcome = PASS
    return result


@dataclass
class Record:
    results: list[ProbeResult]

    @property
    def failures(self) -> list[str]:
        return [
            f"{r.probe} {r.outcome}: {'; '.join(r.reasons)}"
            for r in self.results
            if r.outcome != PASS
        ]

    @property
    def verdict(self) -> str:
        return "pass" if not self.failures else "fail"

    @property
    def exit_code(self) -> int:
        return EXIT_OK if not self.failures else EXIT_FINDINGS

    @property
    def detail(self) -> str:
        if not self.failures:
            return (
                "the receipt named the route the ledger shows was taken, the "
                "completed run's every quality rule reached a boolean verdict, "
                "and the dead run ended 'failed' with a typed cause"
            )
        return " | ".join(self.failures)


def grade(observations: Observations) -> Record:
    results = [
        grade_skip_never_pass(observations.healthy, observations.dying),
        grade_identity(observations.healthy, observations.healthy_ledger),
        grade_typed_death(observations.dying),
    ]
    graded = tuple(r.probe for r in results)
    if graded != PROBES:  # pragma: no cover - a structural invariant
        raise AssertionError(f"graded {graded}, the criterion names {PROBES}")
    return Record(results)


# ---------------------------------------------------------------------------
# Live half
# ---------------------------------------------------------------------------


def resolve_env(env_name: str, environ: Mapping[str, str], what: str) -> str:
    """Resolve a secret by NAME. Absent is exit 2, not a verdict."""
    if not env_name:
        raise ProbeInputError(
            f"no environment variable NAME was given for the {what}; this probe "
            "never takes a secret VALUE on its command line"
        )
    value = environ.get(env_name, "")
    if not value:
        raise ProbeInputError(
            f"the {what} environment variable {env_name!r} is empty or unset. "
            "This is a configuration failure and is reported as one"
        )
    return value


def _maybe_json(raw: str) -> Any:
    try:
        return json.loads(raw)
    except ValueError:
        return raw


def _http(
    method: str,
    url: str,
    *,
    api_key: str,
    body: Mapping[str, Any] | None,
    timeout: float,
) -> tuple[int | None, Any, str | None]:
    if not url.startswith(("http://", "https://")):
        raise ProbeInputError("refusing a non-HTTP base URL")
    headers = {"accept": "application/json", HEADER_NAME: api_key}
    data: bytes | None = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["content-type"] = "application/json"
    request = urllib.request.Request(  # noqa: S310 - scheme checked directly above
        url, data=data, headers=headers, method=method
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            raw = response.read().decode("utf-8", errors="replace")
            return response.status, _maybe_json(raw), None
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        return exc.code, _maybe_json(raw), None
    except (urllib.error.URLError, OSError, ValueError) as exc:
        # The exception TYPE, never its message: a URLError can echo the
        # request, and every request here carries a credential.
        return None, None, type(exc).__name__


@dataclass
class _Submitted:
    submit_status: int | None
    workflow_id: str | None
    correlation_id: str
    error: str | None


def _submit(
    base_url: str, payload: Mapping[str, Any], *, api_key: str, timeout: float
) -> _Submitted:
    correlation_id = str(uuid.uuid4())
    status, body, error = _http(
        "POST",
        f"{base_url}/v1/workflows",
        api_key=api_key,
        body={
            "workflow_type": WORKFLOW_TYPE,
            "correlation_id": correlation_id,
            "payload": dict(payload),
        },
        timeout=timeout,
    )
    workflow_id = body.get("workflow_id") if isinstance(body, dict) else None
    if status != 202 and error is None:
        error = f"submit answered {status}"
    return _Submitted(status, _text(workflow_id), correlation_id, error)


def _observe_run(
    base_url: str,
    submitted: _Submitted,
    *,
    api_key: str,
    deadline: float,
    runner_identity: str,
    timeout: float,
    poll_seconds: float,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> RunObservation:
    if submitted.workflow_id is None:
        return RunObservation(submitted.submit_status, error=submitted.error)
    status_url = f"{base_url}/v1/workflows/{submitted.workflow_id}/status"
    last: Mapping[str, Any] | None = None
    error: str | None = None
    while True:
        code, body, transport = _http(
            "GET", status_url, api_key=api_key, body=None, timeout=timeout
        )
        if code == 200 and isinstance(body, dict):
            last, error = body, None
            if _status_of(body) in TERMINAL_STATUSES:
                break
        else:
            error = transport or f"status answered {code}"
        if clock() >= deadline:
            error = error or "budget spent before a terminal status"
            return RunObservation(submitted.submit_status, status=last, error=error)
        sleep(poll_seconds)
    query = urllib.parse.urlencode({"runner_identity": runner_identity})
    code, body, transport = _http(
        "GET",
        f"{base_url}/v1/workflows/{submitted.workflow_id}/receipt?{query}",
        api_key=api_key,
        body=None,
        timeout=timeout,
    )
    if code == 200 and isinstance(body, dict):
        return RunObservation(submitted.submit_status, status=last, receipt=body)
    return RunObservation(
        submitted.submit_status,
        status=last,
        error=transport or f"receipt answered {code}",
    )


def _read_ledger(
    dsn: str,
    correlation_id: str,
    *,
    wait_seconds: float,
    poll_seconds: float,
) -> LedgerObservation:
    """Read the run's routing and terminal hops from the event ledger.

    Refuses a role carrying SUPERUSER or BYPASSRLS, asked of the server rather
    than inferred from the DSN: such a role reads past row-level security, so a
    green read through it cannot tell a readable ledger from one readable only
    by an identity nothing else has (the chain canary's OMN-18060 refusal).
    """
    import psycopg2  # imported here so --replay needs no database driver

    try:
        connection = psycopg2.connect(dsn, connect_timeout=10)
    except psycopg2.Error as exc:
        return LedgerObservation(error=f"connect failed: {type(exc).__name__}")
    try:
        connection.set_session(readonly=True, autocommit=True)
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT rolsuper, rolbypassrls FROM pg_roles "
                "WHERE rolname = current_user"
            )
            role = cursor.fetchone()
            if role is None or role[0] or role[1]:
                raise ProbeInputError(
                    "the ledger DSN authenticates as a role that is SUPERUSER, "
                    "BYPASSRLS, or has no pg_roles row; refusing to read through it"
                )
        deadline = time.monotonic() + wait_seconds
        while True:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT topic, convert_from(event_value, 'UTF8') "
                    "FROM public.event_ledger WHERE correlation_id = %s::uuid "
                    "ORDER BY ledger_written_at, kafka_offset",
                    (correlation_id,),
                )
                rows = cursor.fetchall()
            decisions: list[Mapping[str, Any]] = []
            terminal: Mapping[str, Any] | None = None
            for topic, raw in rows:
                envelope = _maybe_json(raw)
                if not isinstance(envelope, dict):
                    continue
                if topic == ROUTING_DECISION_TOPIC:
                    decisions.append(envelope)
                elif topic in LEDGER_TERMINAL_TOPICS:
                    terminal = envelope
            if terminal is not None or time.monotonic() >= deadline:
                return LedgerObservation(tuple(decisions), terminal)
            time.sleep(poll_seconds)
    except psycopg2.Error as exc:
        return LedgerObservation(error=f"read failed: {type(exc).__name__}")
    finally:
        connection.close()


def observe_live(
    *,
    base_url: str,
    api_key: str,
    ledger_dsn: str,
    budget_seconds: float,
    runner_identity: str,
    timeout: float = 20.0,
    poll_seconds: float = 3.0,
) -> Observations:
    base_url = base_url.rstrip("/")
    healthy = _submit(base_url, HEALTHY_PAYLOAD, api_key=api_key, timeout=timeout)
    dying = _submit(base_url, DYING_PAYLOAD, api_key=api_key, timeout=timeout)
    deadline = time.monotonic() + budget_seconds
    kwargs: dict[str, Any] = {
        "api_key": api_key,
        "deadline": deadline,
        "runner_identity": runner_identity,
        "timeout": timeout,
        "poll_seconds": poll_seconds,
    }
    healthy_run = _observe_run(base_url, healthy, **kwargs)
    dying_run = _observe_run(base_url, dying, **kwargs)
    ledger = (
        _read_ledger(
            ledger_dsn,
            healthy.correlation_id,
            wait_seconds=60.0,
            poll_seconds=poll_seconds,
        )
        if _status_of(healthy_run.receipt) == "completed"
        else LedgerObservation(error="not read: the healthy run did not complete")
    )
    return Observations(healthy_run, ledger, dying_run)


# ---------------------------------------------------------------------------
# Replay half
# ---------------------------------------------------------------------------


def _run_from(raw: Any, name: str) -> RunObservation:
    if not isinstance(raw, dict):
        raise ProbeInputError(f"replay observation {name!r} is not an object")
    submit = raw.get("submit_status")
    if submit is not None and not isinstance(submit, int):
        raise ProbeInputError(
            f"replay observation {name!r} has a non-integer submit_status"
        )
    for key in ("status", "receipt"):
        if raw.get(key) is not None and not isinstance(raw.get(key), dict):
            raise ProbeInputError(f"replay observation {name!r}.{key} is not an object")
    return RunObservation(
        submit, raw.get("status"), raw.get("receipt"), raw.get("error")
    )


def observations_from_replay(payload: Mapping[str, Any]) -> Observations:
    raw = payload.get("observations")
    if not isinstance(raw, dict):
        raise ProbeInputError("replay payload has no 'observations' mapping")
    for name in ("healthy", "healthy_ledger", "dying"):
        if name not in raw:
            raise ProbeInputError(f"replay payload has no {name!r} observation")
    ledger = raw["healthy_ledger"]
    if not isinstance(ledger, dict):
        raise ProbeInputError("replay observation 'healthy_ledger' is not an object")
    decisions = ledger.get("routing_decisions") or []
    if not isinstance(decisions, list) or not all(
        isinstance(d, dict) for d in decisions
    ):
        raise ProbeInputError("replay 'routing_decisions' is not a list of objects")
    return Observations(
        healthy=_run_from(raw["healthy"], "healthy"),
        healthy_ledger=LedgerObservation(
            tuple(decisions), ledger.get("terminal"), ledger.get("error")
        ),
        dying=_run_from(raw["dying"], "dying"),
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def record_dict(record: Record, *, base_url: str, as_of: str) -> dict[str, Any]:
    return {
        "version": RECORD_VERSION,
        "criterion": "C16",
        "ticket": "OMN-19181",
        "as_of": as_of,
        "base_url": base_url,
        "dying_task_type": DYING_TASK_TYPE,
        "verdict": record.verdict,
        "detail": record.detail,
        "probes": [r.to_dict() for r in record.results],
    }


def _render_summary(record: Record, *, base_url: str, as_of: str) -> str:
    lines = [
        "## C16 -- receipt identity, SKIP never PASS, typed death (OMN-19181)",
        "",
        f"**Verdict: `{record.verdict}`** against `{base_url}` at `{as_of}`.",
        "",
        "| Probe | Outcome | Reasons |",
        "| --- | --- | --- |",
    ]
    for r in record.results:
        reasons = "; ".join(r.reasons).replace("|", "\\|") or "—"
        lines.append(f"| {r.probe} | `{r.outcome}` | {reasons} |")
    lines += ["", "SKIP fails the verdict exactly as FAIL does.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--base-url", default="", help="Tenant-bearing API base URL.")
    parser.add_argument(
        "--credential-env",
        default="",
        help="NAME of the environment variable holding the API key. Never a value.",
    )
    parser.add_argument(
        "--ledger-dsn-env",
        default="",
        help="NAME of the environment variable holding the event-ledger DSN. Never a value.",
    )
    parser.add_argument("--budget-seconds", type=float, default=240.0)
    parser.add_argument(
        "--runner-identity",
        default="",
        help="Who is verifying (e.g. the Actions run id); the receipt's verifier field.",
    )
    parser.add_argument(
        "--replay",
        default="",
        help="Grade recorded observations from this JSON file instead of making requests.",
    )
    parser.add_argument("--record", default="", help="Write the JSON record here.")
    parser.add_argument("--summary", default="", help="Append a markdown summary here.")
    args = parser.parse_args(argv)

    as_of = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    base_url = args.base_url or "<replay>"
    try:
        if args.replay:
            payload = json.loads(Path(args.replay).read_text(encoding="utf-8"))
            observations = observations_from_replay(payload)
        else:
            if not args.base_url:
                raise ProbeInputError("--base-url is required unless --replay is given")
            if not args.runner_identity.strip():
                raise ProbeInputError(
                    "--runner-identity is required and must be non-empty"
                )
            api_key = resolve_env(args.credential_env, os.environ, "API key")
            ledger_dsn = resolve_env(args.ledger_dsn_env, os.environ, "ledger DSN")
            print(f"base url:        {base_url}")
            print(f"credential var:  {args.credential_env}  (a NAME; never printed)")
            print(f"ledger dsn var:  {args.ledger_dsn_env}  (a NAME; never printed)")
            print(f"dying task type: {DYING_TASK_TYPE}")
            observations = observe_live(
                base_url=args.base_url,
                api_key=api_key,
                ledger_dsn=ledger_dsn,
                budget_seconds=args.budget_seconds,
                runner_identity=args.runner_identity,
            )
        record = grade(observations)
    except (ProbeInputError, OSError, ValueError) as exc:
        message = f"C16 probe could not run: {exc}"
        print(f"::error::{message}", file=sys.stderr)
        if args.record:
            Path(args.record).write_text(
                json.dumps(
                    {
                        "version": RECORD_VERSION,
                        "criterion": "C16",
                        "ticket": "OMN-19181",
                        "as_of": as_of,
                        "base_url": base_url,
                        "verdict": "could_not_run",
                        "detail": message,
                        "probes": [],
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        return EXIT_INPUT

    if args.record:
        Path(args.record).write_text(
            json.dumps(
                record_dict(record, base_url=base_url, as_of=as_of),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as handle:
            handle.write(_render_summary(record, base_url=base_url, as_of=as_of))
    for r in record.results:
        print(f"{r.probe:<11} {r.outcome:<4} {'; '.join(r.reasons)}")
    if record.failures:
        print(f"::error::C16 RED -- {record.detail}", file=sys.stderr)
    else:
        print(f"C16 GREEN -- {record.detail}")
    return record.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
