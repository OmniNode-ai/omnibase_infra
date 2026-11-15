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
from typing import TYPE_CHECKING

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

    def failing_step(self, request: ModelDeployJobReadingRequest) -> str:
        """The first deploy phase that failed, else why no phase ran, else ``unknown``."""
        for phase in _names(self._settings, "phase_order"):
            if request.phase_results.get(phase) == "failed":
                return phase
        if request.not_started:
            return _token(f"not-started-{request.not_started}")
        return _UNKNOWN

    def request_id(self, request: ModelDeployJobReadingRequest) -> str:
        """One id per job, so a retry of the append is a duplicate and writes nothing twice."""
        return str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"onex:deploy-job-reading:{request.correlation_id}",
            )
        )

    def build_row(self, request: ModelDeployJobReadingRequest) -> str | None:
        """The reading row of a terminal job on a covered lane, or ``None`` when it has none."""
        if request.runtime_lane in _names(self._settings, "excluded_lanes"):
            return None
        if request.job_outcome not in ("success", "failed"):
            return None
        failed = request.job_outcome == "failed"
        step = self.failing_step(request) if failed else ""
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
        cells += [
            f"completed_at={_stamp(completed)}",
            f"probed_at={_stamp(request.published_at)}",
            f"published_at={_stamp(request.published_at)}",
            f"max_age={self._settings['max_age_seconds']}",
            f"key={request.correlation_id}",
        ]
        return f"{_stamp(request.published_at)} | STATUS | " + " | ".join(cells)

    def handle(
        self, request: ModelDeployJobReadingRequest
    ) -> ModelDeployJobReadingResult:
        """Build the job's reading row and append it through the reading writer's append route."""
        row = self.build_row(request)
        if row is None:
            return ModelDeployJobReadingResult(outcome="skipped")
        request_id = self.request_id(request)
        error = self._append(row, request_id)
        if error:
            logger.warning(
                "job reading for %s not appended: %s", request.correlation_id, error
            )
            return ModelDeployJobReadingResult(
                outcome="unappended", row=row, append_key=request_id, detail=error
            )
        return ModelDeployJobReadingResult(
            outcome="appended", row=row, append_key=request_id
        )

    def _append(self, row: str, request_id: str) -> str:
        """Append ``row``; an empty string when the append command took it, else why not."""
        command_env = _text(self._settings, "append_command_env")
        host_env = _text(self._settings, "host_name_env")
        command = self._environ.get(command_env, "").strip()
        host = self._environ.get(host_env, "").strip()
        if not command or not host:
            return f"{command_env} and {host_env} must both be set"
        argv: Sequence[str] = [
            *shlex.split(command),
            "--lane",
            _text(self._settings, "row_lane"),
            "--host",
            host,
            "--request-id",
            request_id,
            "--rows-file",
            "-",
        ]
        timeout = self._settings["append_timeout_seconds"]
        try:
            done = subprocess.run(
                argv,
                input=f"{row} | req={request_id}\n",
                capture_output=True,
                text=True,
                timeout=float(timeout) if isinstance(timeout, int | float) else None,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return f"append-command-failed:{type(exc).__name__}"
        if done.returncode != 0:
            tail = done.stderr.strip()[-300:].replace("\n", " | ")
            return f"append-command-exit-{done.returncode}: {tail}".rstrip(": ")
        return ""
