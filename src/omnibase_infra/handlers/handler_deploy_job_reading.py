# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A deploy job's terminal outcome as a ledger reading row (OMN-24617).

THE FAILURE
-----------
A failed deploy-agent job left no reading anyone acted on. The stability follow job on
the lab dev host failed from 09:58Z on 2026-11-15 and no ledger row named it for about
four hours; the dev deploy that failed at 10:30Z was found only when a reading lane went
looking. The job record and the terminal bus event both existed, and neither is a ledger
row.

THE ROUTE
---------
For a job that reached a terminal status on a lane the contract does not exclude, this
handler builds one ``STATUS`` row in the shape of the reading writer's rows
(``node_reading_writer_effect``: ``src=reading``, a ``verdict``, a ``reason``) and appends
it through the append command that writer's overlay already names: rows on stdin,
``--lane``, ``--host``, ``--request-id`` and ``--rows-file -`` on the command line, one JSON
receipt on stdout. It declares no topic; the bus append command owns the ledger's topics.

* ``success`` reads WORKING and names no failing step.
* ``failed`` reads NOT_WORKING and names the failing step in ``reason``
  (``job-failed:<step>``) and in ``failing_step``.
* A failed job also writes one ALARM row (``src=alarm``, ``alarm=deploy-job-failed``) that
  names its owner and the action the owner takes, so a failure is a row someone is told
  about and not only a reading that must be looked for (OMN-24562).
* A code-ref pin inside the contract's ``pin_alarm_window_hours`` of its expiry writes one
  ALARM row (``alarm=pin-near-expiry``) with the same owner line, through :meth:`handle_pin`.
* ``superseded``, ``accepted`` and ``in_progress`` write nothing: a superseded command
  never ran and a job still running has no terminal outcome.
* A lane in the contract's ``excluded_lanes`` (prod) writes nothing.

A reading informs and refuses nothing: :func:`publish` never raises, and a row that cannot
be appended is logged and reported in the result, never allowed to disturb the job's own
terminal publish. The names, bounds and route variables are the node contract's.
"""

from __future__ import annotations

import logging
import re
import shlex
import subprocess
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import JsonValue

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.model_deploy_job_reading_request import (
    ModelDeployJobReadingRequest,
)
from omnibase_infra.models.model_deploy_job_reading_result import (
    ModelDeployJobReadingResult,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

_UNKNOWN = "unknown"
_TOKEN_UNSAFE = re.compile(r"[^A-Za-z0-9._:-]+")


def _text(settings: Mapping[str, JsonValue], key: str) -> str:
    value = settings[key]
    if not isinstance(value, str):
        raise RuntimeError(f"the deploy job reading contract's {key} is not text")
    return value


def _names(settings: Mapping[str, JsonValue], key: str) -> tuple[str, ...]:
    value = settings[key]
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise RuntimeError(f"the deploy job reading contract's {key} is not a list")
    return tuple(str(v) for v in value)


def _stamp(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _token(text: str) -> str:
    return _TOKEN_UNSAFE.sub("-", text).strip("-") or _UNKNOWN


def _request_id(request: ModelDeployJobReadingRequest) -> str:
    """One id per job, so a retry of the append is a duplicate and writes nothing twice."""
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"onex:deploy-job-reading:{request.correlation_id}",
        )
    )


def _pin_request_id(branch: str, expires_at: datetime, now: datetime) -> str:
    """One id per pin and clock hour, for the one row ``build_pin_alarm_row`` builds in that hour.

    The row is identical for every tick of the hour, so a retry within the hour is a
    duplicate to the append; the next clock hour is a new id and a new row.
    """
    bucket = now.astimezone(UTC).strftime("%Y%m%dT%H")
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"onex:deploy-pin-alarm:{branch}:{_stamp(expires_at)}:{bucket}",
        )
    )


def _failing_step(
    settings: Mapping[str, JsonValue], request: ModelDeployJobReadingRequest
) -> str:
    """The first phase that failed, else the first that did not succeed, else ``unknown``."""
    order = _names(settings, "phase_order")
    for phase in order:
        if request.phase_results.get(phase) == "failed":
            return phase
    # A job refused between phases marks none failed; it stopped where the first phase
    # that did not succeed was due.
    for phase in order:
        if request.phase_results.get(phase) != "success":
            return phase
    if request.not_started:
        return _token(f"not-started-{request.not_started}")
    return _UNKNOWN


def _alarm_head(
    settings: Mapping[str, JsonValue], alarm: str, moment: datetime
) -> list[str]:
    """The cells every alarm row opens with, then the owner line cells."""
    return [
        f"{_stamp(moment)} | STATUS",
        f"lane={_text(settings, 'row_lane')}",
        "src=alarm",
        f"actor={_text(settings, 'row_actor')}",
        "model=none",
        f"alarm={alarm}",
    ]


def _append_env(
    settings: Mapping[str, JsonValue], environ: Mapping[str, str]
) -> dict[str, str]:
    """The append command's environment: the agent's, with its prefixed broker login unprefixed."""
    env = dict(environ)
    prefix = env.get(_text(settings, "sasl_prefix_env"), "")
    for name in _names(settings, "sasl_login_names"):
        if prefix and name not in env and f"{prefix}{name}" in env:
            env[name] = env[f"{prefix}{name}"]
    return env


def append_ledger_row(
    settings: Mapping[str, JsonValue],
    environ: Mapping[str, str],
    row: str,
    request_id: str,
) -> str:
    """Append ``row`` through the node's bus append command; ``""`` when it took it, else why not.

    Every row this node writes (job readings, alarms, build decisions) goes through here:
    rows on stdin, ``--lane``, ``--host``, ``--request-id`` and ``--rows-file -`` on the
    command line, bounded by the contract's ``append_timeout_seconds``.
    """
    command_env = _text(settings, "append_command_env")
    host_env = _text(settings, "host_name_env")
    command = environ.get(command_env, "").strip()
    host = environ.get(host_env, "").strip()
    if not command or not host:
        return f"{command_env} and {host_env} must both be set"
    argv: Sequence[str] = [
        *shlex.split(command),
        "--lane",
        _text(settings, "row_lane"),
        "--host",
        host,
        "--request-id",
        request_id,
        "--rows-file",
        "-",
    ]
    timeout = settings["append_timeout_seconds"]
    env = _append_env(settings, environ)
    try:
        done = subprocess.run(
            argv,
            input=f"{row} | req={request_id}\n",
            capture_output=True,
            text=True,
            env=env,
            timeout=float(timeout) if isinstance(timeout, int | float) else None,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"append-command-failed:{type(exc).__name__}"
    if done.returncode != 0:
        tail = done.stderr.strip()[-300:].replace("\n", " | ")
        return f"append-command-exit-{done.returncode}: {tail}".rstrip(": ")
    return ""


class HandlerDeployJobReading:
    """Build a terminal deploy job's reading row and append it to the ledger."""

    def __init__(
        self, *, settings: Mapping[str, JsonValue], environ: Mapping[str, str]
    ) -> None:
        self._settings = settings
        self._environ = environ

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    def build_row(self, request: ModelDeployJobReadingRequest) -> str | None:
        """The reading row of a terminal job on a covered lane, or ``None`` when it has none."""
        if request.runtime_lane in _names(self._settings, "excluded_lanes"):
            return None
        if request.job_outcome not in ("success", "failed"):
            return None
        failed = request.job_outcome == "failed"
        step = _failing_step(self._settings, request) if failed else ""
        completed = request.completed_at or request.published_at
        cells = [
            f"lane={_text(self._settings, 'row_lane')}",
            "src=reading",
            f"actor={_text(self._settings, 'row_actor')}",
            "model=none",
            f"piece={_text(self._settings, 'piece')}",
            f"name={_text(self._settings, 'piece_name')}",
            f"verdict={'NOT_WORKING' if failed else 'WORKING'}",
            "outcome=read",
            f"reason=job-failed:{step}" if failed else "reason=job-success",
            f"deploy_lane={_token(request.runtime_lane)}",
            f"job={request.correlation_id}",
            f"kind={_token(request.scope)}",
        ]
        if failed:
            cells.append(f"failing_step={step}")
            if request.first_error:
                width = self._settings["error_cell_chars"]
                limit = width if isinstance(width, int) else 0
                cells.append(f"error={_token(request.first_error)[:limit]}")
        cells += [
            f"completed_at={_stamp(completed)}",
            f"probed_at={_stamp(request.published_at)}",
            f"published_at={_stamp(request.published_at)}",
            f"max_age={self._settings['max_age_seconds']}",
            f"key={request.correlation_id}",
        ]
        return f"{_stamp(request.published_at)} | STATUS | " + " | ".join(cells)

    def build_alarm_row(self, request: ModelDeployJobReadingRequest) -> str | None:
        """The alarm row of a failed job on a covered lane, ``None`` for any other job."""
        if request.runtime_lane in _names(self._settings, "excluded_lanes"):
            return None
        if request.job_outcome != "failed":
            return None
        cells = _alarm_head(self._settings, "deploy-job-failed", request.published_at)
        cells += [
            f"deploy_lane={_token(request.runtime_lane)}",
            f"job={request.correlation_id}",
            f"kind={_token(request.scope)}",
            f"failing_step={_failing_step(self._settings, request)}",
            f"owner={_text(self._settings, 'alarm_owner')}",
            f"owner_line={_text(self._settings, 'alarm_job_owner_line')}",
        ]
        return " | ".join(cells)

    def build_pin_alarm_row(
        self,
        *,
        branch: str,
        pr: int,
        ticket: str,
        registered_by: str,
        expires_at: datetime,
        now: datetime,
    ) -> str | None:
        """The alarm row of a live code-ref pin inside the window before its expiry, else ``None``.

        The row is a function of the pin and the clock hour of ``now`` (its stamp and
        ``hours_left`` are those of the hour's start), so every tick of one hour builds
        the identical row: the work-ledger append answers duplicate for it, where a row
        that changed under one request id would be refused.
        """
        window = self._settings["pin_alarm_window_hours"]
        hours = float(window) if isinstance(window, int | float) else 0.0
        if (expires_at - now).total_seconds() / 3600 > hours:
            return None
        bucket = now.astimezone(UTC).replace(minute=0, second=0, microsecond=0)
        left = (expires_at - bucket).total_seconds() / 3600
        cells = _alarm_head(self._settings, "pin-near-expiry", bucket)
        cells += [
            f"pin_branch={_token(branch)}",
            f"pin_pr={pr}",
            f"pin_ticket={_token(ticket)}",
            f"pin_owner={_token(registered_by)}",
            f"expires_at={_stamp(expires_at)}",
            f"hours_left={max(left, 0.0):.1f}",
            f"owner={_text(self._settings, 'alarm_owner')}",
            f"owner_line={_text(self._settings, 'alarm_pin_owner_line')}",
        ]
        return " | ".join(cells)

    def build_adoption_revert_alarm_row(
        self, *, adopted_ref: str, previous_ref: str, now: datetime
    ) -> str:
        """The alarm row of an agent that restarted in a storm after adopting ``adopted_ref``
        and returned to ``previous_ref`` (OMN-26494); it names both refs."""
        cells = _alarm_head(self._settings, "agent-adoption-reverted", now)
        cells += [
            f"adopted_ref={_token(adopted_ref)}",
            f"previous_ref={_token(previous_ref)}",
            f"owner={_text(self._settings, 'alarm_owner')}",
            f"owner_line={_text(self._settings, 'alarm_adoption_owner_line')}",
        ]
        return " | ".join(cells)

    def handle_adoption_revert(
        self, *, adopted_ref: str, previous_ref: str, now: datetime
    ) -> ModelDeployJobReadingResult:
        """Append the revert's alarm row; the request id is fixed by the two refs, so a repeat
        of the same storm's revert is a duplicate to the append and writes nothing twice."""
        row = self.build_adoption_revert_alarm_row(
            adopted_ref=adopted_ref, previous_ref=previous_ref, now=now
        )
        request_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"onex:deploy-adoption-revert:{adopted_ref}:{previous_ref}",
            )
        )
        error = append_ledger_row(self._settings, self._environ, row, request_id)
        outcome = "unappended" if error else "appended"
        return ModelDeployJobReadingResult(
            outcome=outcome,
            row=row,
            append_key=request_id,
            detail=error,
            alarm_row=row,
            alarm_outcome=outcome,
        )

    def pin_state_dir(self) -> Path | None:
        """The agent's state directory, from the variable the contract names; ``None`` when unset."""
        value = self._environ.get(
            _text(self._settings, "pin_state_dir_env"), ""
        ).strip()
        return Path(value) if value else None

    def handle_pin(
        self,
        *,
        branch: str,
        pr: int,
        ticket: str,
        registered_by: str,
        expires_at: datetime,
        now: datetime,
    ) -> ModelDeployJobReadingResult:
        """Append the pin's alarm row when it is inside the window; ``skipped`` when it is not."""
        row = self.build_pin_alarm_row(
            branch=branch,
            pr=pr,
            ticket=ticket,
            registered_by=registered_by,
            expires_at=expires_at,
            now=now,
        )
        if row is None:
            return ModelDeployJobReadingResult(outcome="skipped")
        request_id = _pin_request_id(branch, expires_at, now)
        error = append_ledger_row(self._settings, self._environ, row, request_id)
        if error:
            logger.warning("pin alarm for %s not appended: %s", branch, error)
            return ModelDeployJobReadingResult(
                outcome="unappended",
                row=row,
                append_key=request_id,
                detail=error,
                alarm_row=row,
                alarm_outcome="unappended",
            )
        return ModelDeployJobReadingResult(
            outcome="appended",
            row=row,
            append_key=request_id,
            alarm_row=row,
            alarm_outcome="appended",
        )

    def handle(
        self, request: ModelDeployJobReadingRequest
    ) -> ModelDeployJobReadingResult:
        """Build the job's reading row (and a failed job's alarm row) and append them."""
        row = self.build_row(request)
        if row is None:
            return ModelDeployJobReadingResult(outcome="skipped")
        request_id = _request_id(request)
        error = append_ledger_row(self._settings, self._environ, row, request_id)
        alarm_row = self.build_alarm_row(request) or ""
        alarm_outcome: Literal["none", "appended", "unappended"] = "none"
        if alarm_row:
            alarm_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"onex:deploy-job-alarm:{request.correlation_id}",
                )
            )
            alarm_error = append_ledger_row(
                self._settings, self._environ, alarm_row, alarm_id
            )
            alarm_outcome = "unappended" if alarm_error else "appended"
            if alarm_error:
                logger.warning(
                    "job alarm for %s not appended: %s",
                    request.correlation_id,
                    alarm_error,
                )
                error = error or alarm_error
        if error:
            logger.warning(
                "job reading for %s not appended: %s", request.correlation_id, error
            )
            return ModelDeployJobReadingResult(
                outcome="unappended",
                row=row,
                append_key=request_id,
                detail=error,
                alarm_row=alarm_row,
                alarm_outcome=alarm_outcome,
            )
        return ModelDeployJobReadingResult(
            outcome="appended",
            row=row,
            append_key=request_id,
            alarm_row=alarm_row,
            alarm_outcome=alarm_outcome,
        )
