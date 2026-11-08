# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Grade the C16 pinned-backend probe (OMN-23176, scenario R7).

Three delegations are pinned, one per role of the config, with the delegate
CLI's ``--backend-id``. For each, the accepted attempt's backend and host and
the terminal's cost must equal the ``delegation_events`` row for the same
correlation id. A backend the lane lacks, a missing row, or an unreadable table
is red; the graded set is the config's, never what answered.
"""

from __future__ import annotations

import json
import re
import subprocess
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import yaml

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.model_c16_pinned_backend_request import (
    ModelC16PinnedBackendRequest,
)
from omnibase_infra.models.model_c16_pinned_backend_result import (
    ModelC16PinnedBackendResult,
)

PASS: Final[str] = "PASS"
FAIL: Final[str] = "FAIL"
SKIP: Final[str] = "SKIP"


class ProbeInputError(RuntimeError):
    """The probe could not run: unreadable config or replay."""


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


PINNED_PROBE: Final[str] = "R7-PINNED"
PINNED_ROLES: Final[tuple[str, ...]] = ("local", "free_cloud", "paid")
ROW_QUERY_PLACEHOLDER: Final[str] = "{correlation_id}"
COST_TOLERANCE: Final[float] = 1e-9


@dataclass(frozen=True)
class PinnedRun:
    """One delegation pinned to a named backend, as its receipt reports it.

    ``accepted`` is the receipt's accepted attempt (backend_id, host, cost_usd)
    or ``None`` when the pin got no answer, which includes a backend the lane
    does not declare. ``terminal_cost_usd`` is the receipt's own total cost.
    """

    role: str
    backend_id: str
    correlation_id: str = ""
    accepted: Mapping[str, Any] | None = None
    terminal_cost_usd: float | None = None
    error: str | None = None


@dataclass(frozen=True)
class DelegationRow:
    """The ``delegation_events`` row for one correlation id."""

    backend_id: str | None = None
    host: str | None = None
    cost_usd: float | None = None


@dataclass(frozen=True)
class PinnedObservations:
    runs: tuple[PinnedRun, ...]
    rows: Mapping[str, DelegationRow | None]
    rows_error: str | None = None


def load_pinned_config(path: Path) -> dict[str, str]:
    """Read the role -> backend id map. Every role is required, none invented."""
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ProbeInputError(f"pinned-backend config unreadable: {exc}") from exc
    named = document.get("pinned_backends") if isinstance(document, dict) else None
    if not isinstance(named, dict):
        raise ProbeInputError("pinned-backend config has no 'pinned_backends' map")
    unknown = sorted(set(named) - set(PINNED_ROLES))
    missing = [role for role in PINNED_ROLES if _text(named.get(role)) is None]
    if unknown or missing:
        raise ProbeInputError(
            f"pinned-backend config must name exactly {PINNED_ROLES}; "
            f"missing {missing}, unknown {unknown}"
        )
    return {role: str(named[role]).strip() for role in PINNED_ROLES}


def load_row_query(path: Path) -> str:
    """Read the delegation_events query template from the config (one placeholder)."""
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ProbeInputError(f"pinned-backend config unreadable: {exc}") from exc
    query = document.get("row_query") if isinstance(document, dict) else None
    if not isinstance(query, str) or query.count(ROW_QUERY_PLACEHOLDER) != 1:
        raise ProbeInputError(
            f"config 'row_query' must carry exactly one {ROW_QUERY_PLACEHOLDER}"
        )
    return query


def pinned_run_from_receipt(
    role: str, backend_id: str, receipt: Mapping[str, Any] | None, error: str | None
) -> PinnedRun:
    """Reduce a delegate receipt to the fields the comparison reads."""
    if receipt is None:
        return PinnedRun(role, backend_id, error=error or "no receipt")
    terminal = receipt.get("terminal_payload")
    if not isinstance(terminal, Mapping):
        terminal = {
            "status": receipt.get("status"),
            "attempts": receipt.get("attempts"),
            "metrics": {"cost_usd": receipt.get("terminal_cost_usd")},
        }
    attempts = terminal.get("attempts")
    accepted = [
        a
        for a in (attempts if isinstance(attempts, list) else [])
        if isinstance(a, Mapping) and a.get("acceptance_decision") == "accept"
    ]
    metrics = terminal.get("metrics")
    cost = metrics.get("cost_usd") if isinstance(metrics, Mapping) else None
    if cost is None:
        cost = receipt.get("terminal_cost_usd")
    return PinnedRun(
        role,
        backend_id,
        correlation_id=str(receipt.get("correlation_id") or ""),
        accepted=accepted[-1] if accepted else None,
        terminal_cost_usd=float(cost) if isinstance(cost, (int, float)) else None,
        error=error,
    )


def grade_pinned_backends(
    config: Mapping[str, str], observations: PinnedObservations
) -> ProbeResult:
    """R7: each pinned delegation's receipt agrees with its delegation_events row.

    The graded set is the CONFIG's set. A role with no run, a backend the lane
    did not answer for, or a row that cannot be read is red; the set is never
    shrunk to what answered.
    """
    result = ProbeResult(PINNED_PROBE, FAIL)
    by_role = {run.role: run for run in observations.runs}
    for role in PINNED_ROLES:
        backend = config[role]
        run = by_role.get(role)
        if run is None:
            result.reasons.append(
                f"{role}: no pinned delegation was made for {backend!r}"
            )
            continue
        if run.backend_id != backend:
            result.reasons.append(
                f"{role}: the run was pinned to {run.backend_id!r}, the config names {backend!r}"
            )
            continue
        accepted = run.accepted
        if accepted is None:
            result.reasons.append(
                f"{role}: backend {backend!r} produced no accepted attempt "
                f"(missing from the lane or refused: {run.error or 'no error recorded'})"
            )
            continue
        result.evidence[f"{role}.receipt.backend_id"] = accepted.get("backend_id")
        result.evidence[f"{role}.receipt.host"] = accepted.get("host")
        if accepted.get("backend_id") != backend:
            result.reasons.append(
                f"{role}: pinned to {backend!r} but the accepted attempt came from "
                f"{accepted.get('backend_id')!r}"
            )
        if observations.rows_error is not None:
            result.reasons.append(
                f"{role}: delegation_events could not be read ({observations.rows_error}); SKIP is not PASS"
            )
            continue
        row = observations.rows.get(run.correlation_id)
        if row is None:
            result.reasons.append(
                f"{role}: no delegation_events row for correlation id {run.correlation_id!r}"
            )
            continue
        result.evidence[f"{role}.row.backend_id"] = row.backend_id
        result.evidence[f"{role}.row.host"] = row.host
        for key, claimed, stored in (
            ("backend_id", accepted.get("backend_id"), row.backend_id),
            ("host", accepted.get("host"), row.host),
        ):
            if _text(claimed) is None or _text(stored) is None:
                result.reasons.append(
                    f"{role}: {key} is {claimed!r} on the receipt and {stored!r} on the "
                    "row; a blank is not an identity"
                )
            elif claimed != stored:
                result.reasons.append(
                    f"{role}: the receipt names {key}={claimed!r}, the row {stored!r}"
                )
        receipt_cost = run.terminal_cost_usd
        if receipt_cost is None or row.cost_usd is None:
            result.reasons.append(
                f"{role}: cost_usd is {receipt_cost!r} on the receipt and {row.cost_usd!r} on the row"
            )
        elif abs(receipt_cost - row.cost_usd) > COST_TOLERANCE:
            result.reasons.append(
                f"{role}: the receipt cost_usd={receipt_cost!r}, the row {row.cost_usd!r}"
            )
    if not result.reasons:
        result.outcome = PASS
    elif all("SKIP is not PASS" in reason for reason in result.reasons):
        result.outcome = SKIP
    return result


def pinned_observations_from_replay(payload: Mapping[str, Any]) -> PinnedObservations:
    """Build observations from recorded runs and rows (offline grading)."""
    raw = payload.get("pinned")
    if not isinstance(raw, dict) or not isinstance(raw.get("runs"), list):
        raise ProbeInputError("replay payload has no 'pinned.runs' list")
    runs = tuple(
        pinned_run_from_receipt(
            str(item.get("role", "")),
            str(item.get("backend_id", "")),
            item.get("receipt"),
            item.get("error"),
        )
        for item in raw["runs"]
        if isinstance(item, dict)
    )
    rows_raw = raw.get("rows", {})
    if not isinstance(rows_raw, dict):
        raise ProbeInputError("replay 'pinned.rows' is not an object")
    rows = {
        key: (
            DelegationRow(
                value.get("backend_id"), value.get("host"), value.get("cost_usd")
            )
            if isinstance(value, dict)
            else None
        )
        for key, value in rows_raw.items()
    }
    error = raw.get("rows_error")
    return PinnedObservations(runs, rows, error if isinstance(error, str) else None)


def observe_pinned_live(
    config: Mapping[str, str],
    *,
    delegate_argv: list[str],
    psql_argv: list[str],
    run_command: Callable[[list[str], float], tuple[int, str]],
    row_query: str,
    timeout: float,
) -> PinnedObservations:
    """Pin one delegation per configured role and read each one's row.

    ``delegate_argv`` is the delegate CLI prefix up to (not including) the
    ``--backend-id`` flag; ``psql_argv`` is a read-only SQL client prefix (empty
    when the lane has no reader, which grades SKIP). ``run_command`` is the one
    place a process starts.
    """
    runs: list[PinnedRun] = []
    for role in PINNED_ROLES:
        backend = config[role]
        code, out = run_command([*delegate_argv, "--backend-id", backend], timeout)
        receipt: Mapping[str, Any] | None = None
        match = re.search(r"full receipt (\S+receipt\.json)", out)
        error = None if code == 0 else f"delegate exited {code}"
        if match:
            try:
                loaded = json.loads(Path(match.group(1)).read_text(encoding="utf-8"))
                inner = loaded.get("receipt", {}).get("result", {})
                receipt = {
                    "correlation_id": loaded.get("correlation_id"),
                    "terminal_payload": (inner.get("terminal_payload") or {}).get(
                        "payload"
                    ),
                }
            except (OSError, ValueError, AttributeError) as exc:
                error = f"receipt unreadable: {type(exc).__name__}"
        elif error is None:
            error = "no receipt path in the delegate output"
        runs.append(pinned_run_from_receipt(role, backend, receipt, error))
    if not psql_argv:
        return PinnedObservations(
            tuple(runs), {}, "no delegation_events reader is configured"
        )
    rows: dict[str, DelegationRow | None] = {}
    for run in runs:
        if not run.correlation_id:
            continue
        correlation = str(uuid.UUID(run.correlation_id))
        # ``correlation`` is a canonical uuid string (uuid.UUID above), the only
        # value put into the configured query.
        sql = row_query.replace(ROW_QUERY_PLACEHOLDER, correlation)
        code, out = run_command([*psql_argv, "-At", "-F", "|", "-c", sql], timeout)
        if code != 0:
            return PinnedObservations(tuple(runs), {}, f"the reader exited {code}")
        line = out.strip().splitlines()[-1] if out.strip() else ""
        parts = line.split("|") if line else []
        rows[run.correlation_id] = (
            DelegationRow(
                parts[0] or None,
                parts[1] or None,
                float(parts[2]) if len(parts) > 2 and parts[2] else None,
            )
            if len(parts) == 3
            else None
        )
    return PinnedObservations(tuple(runs), rows)


def run_command(argv: list[str], timeout: float) -> tuple[int, str]:
    """Start one process: the only place this probe does."""
    try:
        done = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 124, type(exc).__name__
    return done.returncode, done.stdout + done.stderr


class HandlerC16PinnedBackend:
    """Grade the pinned-backend probe; only a PASS reads success."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def handle(
        self, request: ModelC16PinnedBackendRequest
    ) -> ModelC16PinnedBackendResult:
        try:
            config = load_pinned_config(Path(request.config_path))
            if request.replay_path:
                payload = json.loads(
                    Path(request.replay_path).read_text(encoding="utf-8")
                )
                observations = pinned_observations_from_replay(payload)
            else:
                if not request.delegate_argv:
                    raise ProbeInputError("delegate_argv is required for a live run")
                observations = observe_pinned_live(
                    config,
                    delegate_argv=list(request.delegate_argv),
                    psql_argv=list(request.psql_argv),
                    run_command=run_command,
                    row_query=load_row_query(Path(request.config_path)),
                    timeout=request.timeout_seconds,
                )
            result = grade_pinned_backends(config, observations)
        except (ProbeInputError, OSError, ValueError) as exc:
            message = f"C16 pinned-backend probe could not run: {exc}"
            return ModelC16PinnedBackendResult(
                success=False, outcome="COULD_NOT_RUN", error_message=message
            )
        if request.record_path:
            Path(request.record_path).write_text(
                json.dumps(
                    {"ticket": "OMN-23176", "probes": [result.to_dict()]},
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        reasons = tuple(result.reasons)
        return ModelC16PinnedBackendResult(
            success=result.outcome == PASS,
            outcome=result.outcome,
            reasons=reasons,
            evidence=dict(result.evidence),
            error_message=None if result.outcome == PASS else "; ".join(reasons),
        )
