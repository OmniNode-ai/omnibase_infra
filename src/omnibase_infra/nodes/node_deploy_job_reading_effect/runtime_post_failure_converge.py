# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""After a failed or recovered dev roll the agent rebuilds the newer dev head once (OMN-26429).

The deploy agent loads this by file path (``load_dev_boot_guard``), like the other host
composition roots of this node, and shares nothing with it but the job records it reads.
It imports nothing from ``deploy_agent``: the agent hands in itself and the command
factory, and every read goes through the agent's own idle seams.

WHY THIS EXISTS
---------------
Job ca1533c4 (``agent/idle-restore``, dev lane) built omnibase_infra 0f7f2b73 and
failed at the runtime phase; the boot guard rolled the lane back to the two-day-old
image and the job ended ``failed`` at 02:08Z. The fixes for that failure had merged
while it ran. Nothing started a rebuild afterwards: idle converge refuses omnibase_infra
by design (``not_routed_elsewhere``: the table sends those rebuilds to this instance)
and omnimarket inside C15's window, and idle restore saw a running runtime. The lane
stayed on the old image until a person noticed.

THE RULE
--------
When the newest terminal dev job ended ``failed`` (a boot-guard recovery ends inside
that job: the record turns ``failed`` only after it) and a repository's dev head is not
the source that job built, run ONE rebuild of the newer head as
``agent/post-failure-converge``: an ordinary dev job through the agent's own accept and
``_execute_command`` path, so the single-flight lock, the lane lock, the load gate, the
recreate window and the FROZEN flag all still apply.

It is narrower than idle converge and never replaces it. It does not fire with no
failed job behind it, while a job is in flight, on a frozen lane, for a prod or
stability job, for a failure that records no source (a rebuild of an unknown source
cannot be shown to differ), or when the head is the failed job's source (that build
would fail the same way). Idle converge's ``not_routed_elsewhere`` refusal is untouched.

STATE, AND HOW IT SURVIVES A RESTART
------------------------------------
The need is derived on every check from the job records in the agent's jobs directory,
which are the durable record: the newest terminal dev job, its recorded source, and the
head read now. Nothing waits in memory, so:

* a probe window is a DELAY: the failed record is still the newest terminal job at the
  first check after the window, and the rebuild starts then;
* a head that moves while waiting is simply the head read at the next check (newest
  wins, one enqueue);
* a restart of the agent loses only :attr:`PostFailureConverge.attempted`; the rebuild
  it started has its own job record, which is then the newest dev job, so the failed
  record no longer answers the question. A rebuild that itself failed carries its own
  source: the same head refuses (``up_to_date``), a newer head is one more rebuild
  (once per head), and an unrecorded source refuses (``source_unknown``), so a failure
  that records nothing cannot loop.

PROBE WINDOWS
-------------
C15 and C16 are read through the agent's ``_idle_probe_blocker`` seam, which is idle
converge's own ``probe_blocking`` over the one schedule parser with the same lead
(converge ceiling plus margin). An unreadable schedule is a refusal, never a clear
window.

THE RECORDED SOURCE
-------------------
A job record keeps no built source of its own. The boot guard's failure text carries the
infra commit the job built (``commit=<40 hex>``), which is what this reads; the lineage
decision's ``build_ref`` is read first when a record has one. The omnimarket ref a build
staged is only in the journal, so an omnimarket-only move is not seen by this rule until
the job record carries it.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Collection
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Final, Protocol
from uuid import UUID, uuid4

from pydantic import ValidationError

from omnibase_infra.enums.enum_job_status import EnumJobStatus
from omnibase_infra.enums.enum_post_failure_verdict import EnumPostFailureVerdict
from omnibase_infra.models.model_failed_dev_job import ModelFailedDevJob
from omnibase_infra.models.model_job_command import ModelJobCommand
from omnibase_infra.models.model_job_lineage import ModelJobLineage
from omnibase_infra.models.model_job_record import ModelJobRecord
from omnibase_infra.models.model_post_failure_decision import ModelPostFailureDecision
from omnibase_infra.models.model_post_failure_inputs import ModelPostFailureInputs
from omnibase_infra.protocols.protocol_command import ProtocolCommand
from omnibase_infra.protocols.protocol_job_store import ProtocolJobStore
from omnibase_infra.protocols.protocol_lane_flags import ProtocolLaneFlags
from omnibase_infra.protocols.protocol_post_failure_host import ProtocolPostFailureHost

logger = logging.getLogger(__name__)

#: ``requested_by`` of the rebuild, so its job record and terminal event are never
#: mistaken for a merge's, an idle converge's or an idle restore's.
POST_FAILURE_CONVERGE_REQUESTER: Final = "agent/post-failure-converge"

#: The runtime lane whose failures this rule answers, as a job record spells it.
DEV_LANE: Final = "dev"

#: The idle branch's check interval (``idle_converge.CHECK_INTERVAL_SECONDS``).
CHECK_INTERVAL_SECONDS: Final = 60

#: The commit the boot guard writes into a failure (``commit=<40 hex>: <reason>``).
_COMMIT_TOKEN: Final = re.compile(r"\bcommit=([0-9a-f]{40})\b")
_SHORT_REF_MIN: Final = 12


def same_revision(head: str, source: str) -> bool:
    """Whether two refs name one commit; a recorded short ref matches by prefix."""
    if head == source:
        return True
    shorter, longer = sorted((head, source), key=len)
    return len(shorter) >= _SHORT_REF_MIN and longer.startswith(shorter)


def failed_job_sources(job: ModelJobRecord) -> dict[str, str]:
    """The commits a failed job built, as its own record names them.

    The infra commit comes from the lineage decision's ``build_ref`` when the record
    has one, else from the ``commit=<sha>`` the boot guard writes into a failure. A
    record that names none yields no source.
    """
    if job.lineage is not None and _COMMIT_TOKEN.fullmatch(
        f"commit={job.lineage.build_ref}"
    ):
        return {"omnibase_infra": job.lineage.build_ref}
    for error in job.errors:
        if match := _COMMIT_TOKEN.search(error):
            return {"omnibase_infra": match.group(1)}
    return {}


def failed_dev_job(job: ModelJobRecord | None) -> ModelFailedDevJob | None:
    """``job`` as a failed dev job, or ``None`` when it passed or ran another lane."""
    if (
        job is None
        or job.status != EnumJobStatus.FAILED
        or job.command.runtime_lane != DEV_LANE
    ):
        return None
    return ModelFailedDevJob(
        correlation_id=job.correlation_id,
        requested_by=job.command.requested_by,
        sources=failed_job_sources(job),
    )


def read_job_record(path: Path) -> ModelJobRecord | None:
    """One job record, or ``None`` when the file is unreadable or not a job record."""
    try:
        return ModelJobRecord.model_validate_json(path.read_text())
    except (OSError, ValidationError):
        return None


def latest_terminal_dev_job(state_dir: Path) -> ModelJobRecord | None:
    """The dev-lane job that ended last (``success`` or ``failed``), or ``None``."""
    ended: list[tuple[datetime, ModelJobRecord]] = []
    for path in state_dir.glob("*.json"):
        job = read_job_record(path)
        if (
            job is not None
            and job.completed_at is not None
            and job.status in (EnumJobStatus.SUCCESS, EnumJobStatus.FAILED)
            and job.command.runtime_lane == DEV_LANE
        ):
            ended.append((job.completed_at, job))
    return max(ended, key=lambda row: row[0], default=(None, None))[1]


def decide(inputs: ModelPostFailureInputs) -> ModelPostFailureDecision:
    """Rebuild, or the first reason not to. Pure: every read happened already."""
    verdict = EnumPostFailureVerdict

    def refuse(
        refusal: EnumPostFailureVerdict, detail: str
    ) -> ModelPostFailureDecision:
        return ModelPostFailureDecision(verdict=refusal, detail=detail)

    if inputs.frozen_reason is not None:
        return refuse(verdict.FROZEN, inputs.frozen_reason)
    if inputs.job_active:
        return refuse(verdict.JOB_ACTIVE, "a job is in flight")
    failed = inputs.failed_job
    if failed is None:
        return refuse(verdict.NO_FAILED_JOB, "the newest terminal dev job did not fail")
    if not failed.sources:
        return refuse(
            verdict.SOURCE_UNKNOWN,
            f"job {failed.correlation_id} failed and records no source it built",
        )
    behind = {
        repo: head
        for repo, source in failed.sources.items()
        if (head := inputs.heads.get(repo)) is not None
        and not same_revision(head, source)
    }
    if not behind:
        if any(inputs.heads.get(repo) is None for repo in failed.sources):
            return refuse(verdict.HEAD_UNREADABLE, "a dev head could not be read")
        return refuse(
            verdict.UP_TO_DATE,
            f"every dev head is the source job {failed.correlation_id} built",
        )
    keys = tuple(sorted(f"{repo}@{head}" for repo, head in behind.items()))
    if set(keys) <= inputs.attempted:
        return refuse(
            verdict.ALREADY_ATTEMPTED,
            f"a rebuild of {', '.join(keys)} was already started",
        )
    if inputs.windows_error is not None:
        return refuse(verdict.WINDOWS_UNREADABLE, inputs.windows_error)
    if inputs.probe_blocker is not None:
        return refuse(verdict.PROBE_WINDOW, inputs.probe_blocker)
    moved = ", ".join(
        f"{repo} {failed.sources[repo][:12]} -> {head[:12]}"
        for repo, head in sorted(behind.items())
    )
    return ModelPostFailureDecision(
        verdict=verdict.CONVERGE,
        detail=f"job {failed.correlation_id} failed; dev head moved: {moved}",
        heads=behind,
        attempt_keys=keys,
    )


class PostFailureConverge:
    """The rule's state and its place in the poll loop's idle branch.

    Holds only the per-process attempted set, the throttle and the last verdict; every
    read goes through the agent's own idle seams, looked up at call time.
    """

    def __init__(
        self,
        host: ProtocolPostFailureHost,
        converge_command: Callable[[], ProtocolCommand],
    ) -> None:
        self._host = host
        self._converge_command = converge_command
        self.attempted: set[str] = set()
        self.last_check: float | None = None
        self.last_verdict: EnumPostFailureVerdict | None = None

    def command(self) -> ProtocolCommand:
        """The one rebuild: the idle converge's dev workspace command, named apart."""
        return self._converge_command().model_copy(
            update={
                "correlation_id": uuid4(),
                "requested_by": POST_FAILURE_CONVERGE_REQUESTER,
            }
        )

    def check(self) -> ModelPostFailureDecision | None:
        """One check per interval; starts the one rebuild when the rule says so."""
        host = self._host
        if host._router is None or DEV_LANE not in host._allowed_lanes:
            return None
        tick = time.monotonic()
        if (
            self.last_check is not None
            and tick - self.last_check < CHECK_INTERVAL_SECONDS
        ):
            return None
        self.last_check = tick
        decision = decide(self._inputs())
        if decision.verdict not in (
            self.last_verdict,
            EnumPostFailureVerdict.CONVERGE,
        ):
            logger.info(
                "post-failure converge: %s (%s)",
                decision.verdict.value,
                decision.detail,
            )
        self.last_verdict = decision.verdict
        if decision.verdict is not EnumPostFailureVerdict.CONVERGE:
            return decision
        cmd = self.command()
        # A deferred rebuild is not marked attempted: the need stays and runs once the
        # load gate and the recreate window admit it.
        if not host._idle_converge_gate_open(cmd.requested_by):
            return decision
        self.attempted.update(decision.attempt_keys)
        logger.info(
            "post-failure converge: running %s as %s: %s",
            cmd.correlation_id,
            cmd.requested_by,
            decision.detail,
        )
        host.job_store.accept(cmd.correlation_id, cmd.model_dump(mode="json"))
        host._execute_command(cmd)
        return decision

    def _inputs(self) -> ModelPostFailureInputs:
        host = self._host
        now = host._idle_converge_now()
        job_active = host.job_store.has_active_job()
        failed = failed_dev_job(latest_terminal_dev_job(host.job_store.state_dir))
        heads: dict[str, str | None] = {}
        probe_blocker = windows_error = None
        if failed is not None and failed.sources and not job_active:
            heads = {repo: host._idle_read_head_ref(repo) for repo in failed.sources}
            try:
                probe_blocker = host._idle_probe_blocker(now)
            except Exception as exc:
                logger.exception("post-failure converge: probe schedule unreadable")
                windows_error = (
                    f"probe schedule unreadable: {type(exc).__name__}: {exc}"
                )
        return ModelPostFailureInputs(
            now=now,
            job_active=job_active,
            failed_job=failed,
            heads=heads,
            probe_blocker=probe_blocker,
            windows_error=windows_error,
            attempted=frozenset(self.attempted),
            frozen_reason=getattr(host, "_lane_flags", _NO_FLAGS).frozen_reason(now),
        )


class NoFlags:
    """The lane flags of an agent that carries none: nothing is frozen."""

    def frozen_reason(self, now: datetime) -> str | None:
        return None


_NO_FLAGS: Final = NoFlags()
