# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One dev merge sha's build decision as a ledger row (OMN-26431).

THE FAILURE
-----------
After the dev merges of 2026-11-22 01:11Z to 01:30Z no lane could say, without GitHub
reads, whether a build had been requested for a merge sha. The deploy agent decides that
for each command it receives and each idle tick it takes (queued with a job id, deferred
by a gate, or refused) and left no record of the decision on the bus or the ledger.

THE ROUTE
---------
This handler builds one ``STATUS`` row for a decision (``src=build-decision``) and appends
it through the same bus append command the job reading uses, handed in as ``append`` by the
host composition root (the handler imports no other handler); it declares no topic. The row carries the AC1 cells of the rebuild trigger's own row
(``trigger_sha=``, ``class=``, ``correlation=`` where one exists) and the agent's answer:

* ``result=queued | job=<job id>``
* ``result=deferred | reason=<gate> | earliest_start=<UTC stamp or unknown>``
* ``result=refused | reason=<verdict>``
* a ``NON_RUNTIME`` sha names no result, job, reason or earliest start: no build follows.

A reader greps ``trigger_sha=<sha>`` and ``src=build-decision``. The request id of the
append is a function of the repository, sha, result and reason, so a repeated decision is a
duplicate to the ledger as well as to the agent's own record of what it wrote.

The row is a record and never a gate: the append is bounded by the contract's
``append_timeout_seconds``, a failed append is returned in the result and logged, and
nothing here raises for it.
"""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Callable, Mapping
from datetime import UTC, datetime

from pydantic import JsonValue

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.model_deploy_job_reading_result import (
    ModelDeployJobReadingResult,
)
from omnibase_infra.models.model_dev_merge_build_decision_request import (
    ModelDevMergeBuildDecisionRequest,
)

logger = logging.getLogger(__name__)

_UNKNOWN = "unknown"
_TOKEN_UNSAFE = re.compile(r"[^A-Za-z0-9._:-]+")


def _text(settings: Mapping[str, JsonValue], key: str) -> str:
    value = settings[key]
    if not isinstance(value, str):
        raise RuntimeError(f"the deploy job reading contract's {key} is not text")
    return value


def _token(text: str) -> str:
    return _TOKEN_UNSAFE.sub("-", text).strip("-") or _UNKNOWN


def _stamp(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def decision_key(request: ModelDevMergeBuildDecisionRequest) -> str:
    """The identity of one decision: repository, sha, result and reason (never the clock)."""
    return ":".join(
        (
            request.repository,
            request.trigger_sha,
            request.result or request.runtime_class,
            _token(request.reason) if request.reason else "",
        )
    )


class HandlerDevMergeBuildDecision:
    """Build a dev merge sha's build-decision row and append it to the ledger."""

    def __init__(
        self,
        *,
        settings: Mapping[str, JsonValue],
        append: Callable[[str, str], str],
    ) -> None:
        """``append(row, request_id)`` returns ``""`` when the ledger took the row, else why not."""
        self._settings = settings
        self._append = append

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    def build_row(self, request: ModelDevMergeBuildDecisionRequest) -> str:
        """The decision's row. A ``NON_RUNTIME`` sha carries no build cell."""
        cells = [
            f"lane={_text(self._settings, 'row_lane')}",
            f"src={_text(self._settings, 'build_decision_src')}",
            f"actor={_text(self._settings, 'build_decision_actor')}",
            "model=none",
            f"repo={_token(request.repository)}",
            f"trigger_sha={request.trigger_sha}",
            f"class={request.runtime_class}",
        ]
        if request.runtime_class != "NON_RUNTIME" and request.result:
            cells.append(f"result={request.result}")
            if request.job:
                cells.append(f"job={_token(request.job)}")
            if request.reason:
                cells.append(f"reason={_token(request.reason)}")
            if request.result == "deferred":
                earliest = (
                    _stamp(request.earliest_start)
                    if request.earliest_start
                    else _UNKNOWN
                )
                cells.append(f"earliest_start={earliest}")
        if request.correlation:
            cells.append(f"correlation={_token(request.correlation)}")
        cells.append(f"decided_at={_stamp(request.decided_at)}")
        return f"{_stamp(request.decided_at)} | STATUS | " + " | ".join(cells)

    def handle(
        self, request: ModelDevMergeBuildDecisionRequest
    ) -> ModelDeployJobReadingResult:
        """Append the decision's row; the result says whether the ledger took it."""
        row = self.build_row(request)
        request_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"onex:dev-merge-build-decision:{decision_key(request)}",
            )
        )
        error = self._append(row, request_id)
        if error:
            logger.warning(
                "build decision row for %s %s not appended: %s "
                "friction_type=build_decision_row_unappended",
                request.repository,
                request.trigger_sha[:12],
                error,
            )
            return ModelDeployJobReadingResult(
                outcome="unappended", row=row, append_key=request_id, detail=error
            )
        return ModelDeployJobReadingResult(
            outcome="appended", row=row, append_key=request_id
        )
