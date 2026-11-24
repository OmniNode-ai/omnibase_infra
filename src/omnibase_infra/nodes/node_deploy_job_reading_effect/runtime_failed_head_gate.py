# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The deploy agent's idle builds skip a dev head whose last roll failed (OMN-26493).

The deploy agent loads this by file path (``load_dev_boot_guard``), like the node's other
host composition roots, and imports nothing from ``deploy_agent``: the agent hands in
itself and the ledger writer, and every read goes through its own idle seams.

WHY THIS EXISTS
---------------
Between 2026-11-16 and 2026-11-23 seven failed dev jobs rebuilt an omnibase_infra head that
an earlier job had already failed to boot (7b25f0ff three times: 5623655f, 8648de6f,
a9ecbe18). Idle restore (``agent/idle-restore``, once an hour on a lane that is down) and
idle converge (``agent/idle-converge``) build the dev head at the tracking ref, so a head that
failed a deterministic check fails the same way again, and each failed roll recreates the
runtime and leaves it unhealthy until the guard restores the rollback image (median 57
minutes over 29 rolls).

THE RULE
--------
An idle build is refused when the newest terminal dev job ended ``failed``, recorded the
infra commit it built (the same read as ``runtime_post_failure_converge``), that commit is
the dev head the build would stage, and its error names one of the verdict classes in
:class:`EnumFailedHeadVerdictClass` (``[runtime_degraded]``, ``[http_error]``,
``[probe_unreachable]``, ``BOOT FAILED``, ``compose up timed out``). A newer head is a
different head, so it is built. A failure with no recorded class, no recorded source or a
passed job after it refuses nothing. A request a lane or the operator signed never reaches
this rule: only ``agent/idle-restore`` and ``agent/idle-converge`` do.

Idle converge of omnimarket is judged by the infra head too, because the image is always
built from the infra tracking ref; the job record keeps no omnimarket ref, so a failure an
omnimarket move would fix is not told apart and waits for the next infra head or a signed
request.

ONE LEDGER ROW PER HEAD, AND HOW IT SURVIVES A RESTART
------------------------------------------------------
The need is derived on every check from the agent's job records, which are the durable
record. The first refusal of a head writes one build decision row (``src=build-decision``,
``result=refused``, ``reason=failed_head``, ``job=<failed job>``, ``verdict_class=<class>``)
through ``runtime_deploy_job_reading._emit``, whose key (repository, head, result, reason) is
held in the agent's state directory and marked before the append: the second refusal, the
other idle path's refusal and a refusal after a restart find the key and write nothing.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Final

from omnibase_infra.enums.enum_failed_head_verdict_class import (
    EnumFailedHeadVerdictClass,
)
from omnibase_infra.models.model_failed_head_refusal import ModelFailedHeadRefusal
from omnibase_infra.models.model_job_record import ModelJobRecord
from omnibase_infra.nodes.node_deploy_job_reading_effect.runtime_post_failure_converge import (
    failed_dev_job,
    latest_terminal_dev_job,
    same_revision,
)
from omnibase_infra.protocols.protocol_command import ProtocolCommand
from omnibase_infra.protocols.protocol_post_failure_host import (
    ProtocolPostFailureHost,
)

logger = logging.getLogger(__name__)

#: The ``requested_by`` of the two idle builds (``idle_converge.IDLE_RESTORE_REQUESTER`` and
#: ``IDLE_CONVERGE_REQUESTER``); the agent's tests hold the two spellings together.
IDLE_REQUESTERS: Final = frozenset({"agent/idle-restore", "agent/idle-converge"})

#: The repository whose dev head every idle build stages at the tracking ref.
INFRA_REPOSITORY: Final = "omnibase_infra"

_VERDICT_CLASSES: Final = (
    (
        EnumFailedHeadVerdictClass.RUNTIME_DEGRADED,
        re.compile(r"\[runtime_degraded\]"),
    ),
    (EnumFailedHeadVerdictClass.HTTP_ERROR, re.compile(r"\[http_error\]")),
    (
        EnumFailedHeadVerdictClass.PROBE_UNREACHABLE,
        re.compile(r"\[probe_unreachable\]"),
    ),
    (EnumFailedHeadVerdictClass.BOOT_FAILED, re.compile(r"BOOT FAILED")),
    (
        EnumFailedHeadVerdictClass.COMPOSE_UP_TIMED_OUT,
        re.compile(r"compose up timed out"),
    ),
)


def verdict_class(errors: tuple[str, ...]) -> EnumFailedHeadVerdictClass | None:
    """The first recorded verdict class a failed job's errors name, or ``None``."""
    text = "\n".join(errors)
    return next(
        (found for found, pattern in _VERDICT_CLASSES if pattern.search(text)), None
    )


def refusal(state_dir: Path, head: str) -> ModelFailedHeadRefusal | None:
    """Why an idle build of ``head`` is refused, read from the job records, or ``None``."""
    job: ModelJobRecord | None = latest_terminal_dev_job(state_dir)
    failed = failed_dev_job(job)
    source = failed.sources.get(INFRA_REPOSITORY) if failed is not None else None
    found = verdict_class(job.errors) if job is not None else None
    if (
        job is None
        or source is None
        or found is None
        or not same_revision(head, source)
    ):
        return None
    return ModelFailedHeadRefusal(
        head=head, failed_job=job.correlation_id, verdict_class=found
    )


class FailedHeadGate:
    """The rule's place in the agent's two idle build paths.

    Holds only the host, the ledger writer and the heads already logged; the need comes
    from the job records, so a restart of the agent loses nothing it needs.
    """

    def __init__(
        self,
        host: ProtocolPostFailureHost,
        record: Callable[[ModelFailedHeadRefusal, datetime], object],
    ) -> None:
        self._host = host
        self._record = record
        self._logged: set[str] = set()

    def refuses(self, cmd: ProtocolCommand) -> bool:
        """Whether the idle build ``cmd`` is refused; writes the head's one ledger row."""
        if cmd.requested_by not in IDLE_REQUESTERS:
            return False
        host = self._host
        head = host._idle_read_head_ref(INFRA_REPOSITORY)
        if head is None:
            return False
        found = refusal(host.job_store.state_dir, head)
        if found is None:
            return False
        self._record(found, host._idle_converge_now())
        if head not in self._logged:
            self._logged.add(head)
            logger.warning(
                "%s refused: %s %s failed job %s with %s; waits for a newer head "
                "or a signed request friction_type=failed_head_idle_build",
                cmd.requested_by,
                INFRA_REPOSITORY,
                head[:12],
                found.failed_job,
                found.verdict_class.value,
            )
        return True
