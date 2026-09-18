# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Durable job state with structured recovery."""

from __future__ import annotations

import os
import tempfile
import time
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field

from deploy_agent.events import DEPLOY_PHASE_ORDER, Phase, PhaseStatus


class EnumJobSettlingStage(StrEnum):
    """Work this job is still doing AFTER its verdict was written (OMN-18636 AC5).

    The deploy phases settle the compose lane's verdict, and then the agent keeps
    working on the same job: the k3s lab-overlay apply (OMN-18200), the onex-api
    pin delivery (OMN-18572), the cloud-migrate repair build on the failing path
    (OMN-18545) and the terminal bus publish all run after
    ``JobStore.complete``.

    That ordering is deliberate and is NOT what this enum changes. The compose
    lane converged on its own merits, and a lab-overlay failure must not report a
    lane that IS running the merged sha as broken -- so the lab verdict travels
    in its own receipt rather than folding into this job's status. What was
    missing is that a reader could not TELL. Job ``79d743db`` was logged
    "completed successfully" at 2026-09-17T19:56:03.781Z and the same thread ran
    four ``k3s ctr images import`` invocations until 19:59:35Z, delivered the pin
    at 19:59:35.719Z, published at 19:59:36Z and rejoined an evicted consumer
    group at 19:59:41Z. For 3m38s the record read ``success`` while the agent was
    still executing that job's work, and the diagnosis had to write the sentence
    this enum exists to make false: "no job is in progress" per the job store is
    not "the agent is idle".

    Naming the stage rather than carrying a bare boolean costs nothing and tells
    a reader WHICH host mutation is in flight, which is the next question after
    "is it still working".
    """

    LAB_OVERLAY = "lab_overlay"
    ONEX_API_PIN = "onex_api_pin"
    REPAIR_BUILD = "repair_build"
    PUBLISH = "publish"


class JobState(BaseModel):
    correlation_id: UUID
    command: dict[str, Any]
    accepted_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    current_phase: Phase = Phase.PREFLIGHT
    phase_results: dict[Phase, PhaseStatus] = Field(default_factory=dict)
    status: Literal["accepted", "in_progress", "success", "failed"] = "accepted"
    errors: list[str] = Field(default_factory=list)
    result_publish_pending: bool = False
    completed_at: datetime | None = None
    #: OMN-18636 AC5. Post-terminal work still executing for this job, or
    #: ``None`` when there is none.
    #:
    #: Defaults to ``None`` so a record written by a previous version of this
    #: agent -- the state directory survives a restart -- reads as "not
    #: settling" rather than turning every job already on disk into a
    #: permanently-settling one at the moment of upgrade.
    settling_stage: EnumJobSettlingStage | None = None


def reconcile_terminal_phase_results(
    phase_results: dict[Phase, PhaseStatus],
) -> dict[Phase, PhaseStatus]:
    """Settle every deploy phase verdict at the moment a job goes terminal.

    OMN-18057. Command 23edaf62 raised out of ``Phase.RUNTIME`` and its job
    record -- and therefore its terminal event -- kept ``runtime: in_progress``
    beside a ``completed_at`` and a duration, with ``seed``/``verification``
    simply absent. The event asserted the deploy was over while refusing to say
    how it ended, and an absent phase is indistinguishable from a phase whose
    result was lost.

    Two rules, applied to the deploy phases only (``Phase.PUBLISH`` is the act
    of emitting the event and is settled by the caller afterwards):

    * a phase still marked IN_PROGRESS or PENDING when the job goes terminal
      FAILED -- it is the phase that raised;
    * a phase never reached is SKIPPED -- explicitly, so "not run" is a fact on
      the record rather than a gap in it.
    """
    settled = dict(phase_results)
    for phase in DEPLOY_PHASE_ORDER:
        current = settled.get(phase)
        if current is None:
            settled[phase] = PhaseStatus.SKIPPED
        elif current in (PhaseStatus.IN_PROGRESS, PhaseStatus.PENDING):
            settled[phase] = PhaseStatus.FAILED
    return settled


def describe_interruption(job: JobState) -> str:
    """Name the phase a crashed job was RUNNING, not the last one it started.

    OMN-18636 (from the ``deploy-agent-http-hang-diag-2105`` diagnosis, TERMINAL
    2026-09-17T22:10:00Z). ``recover_crashed_jobs`` read ``job.current_phase``,
    and that field does not mean what the error string claimed it did.

    ``update_phase`` writes ``current_phase`` on EVERY update, including the
    ``SUCCESS`` one at the end of a phase. So between two phases — after
    ``git`` succeeds and before ``compose_gen`` starts — ``current_phase`` still
    reads ``git``, and a process killed in that gap recorded "interrupted during
    phase git" about a phase that had COMPLETED. That sends the reader at the
    wrong step: the evidence says the git pull was the thing that died, when in
    fact nothing was running and the next phase had not begun.

    ``phase_results`` already carries the fact the string needs, because exactly
    one phase is ``IN_PROGRESS`` at a time and none is between phases. Three
    distinct answers, kept distinct because they take three different next
    steps:

    * a phase IS in progress — that phase was running, and it is where to look;
    * no phase is in progress but some have succeeded — the job died in the gap
      BETWEEN phases, and the last completed one bounds how far it got;
    * nothing has succeeded either — the job was accepted and died before its
      first phase started, which points at startup rather than at any phase.

    Call this BEFORE ``reconcile_terminal_phase_results``, which settles every
    ``IN_PROGRESS`` phase to ``FAILED`` and so erases the distinction this reads.
    """
    running = [
        phase
        for phase in DEPLOY_PHASE_ORDER
        if job.phase_results.get(phase) == PhaseStatus.IN_PROGRESS
    ]
    if running:
        return f"interrupted during phase {running[-1]}"
    completed = [
        phase
        for phase in DEPLOY_PHASE_ORDER
        if job.phase_results.get(phase) == PhaseStatus.SUCCESS
    ]
    if completed:
        return (
            "interrupted between phases; no phase was running, and the last "
            f"completed phase was {completed[-1]}"
        )
    return "interrupted before any phase started"


class JobStore:
    def __init__(
        self,
        state_dir: Path,
        max_completed_age_days: int = 7,
        max_failed_age_days: int = 30,
    ):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.max_completed_age_days = max_completed_age_days
        self.max_failed_age_days = max_failed_age_days

    def _job_path(self, correlation_id: UUID) -> Path:
        return self.state_dir / f"{correlation_id}.json"

    def _atomic_write(self, path: Path, data: str) -> None:
        fd, tmp = tempfile.mkstemp(dir=self.state_dir, suffix=".tmp")
        try:
            os.write(fd, data.encode())
            os.close(fd)
            Path(tmp).replace(path)
        except Exception:
            os.close(fd) if not os.get_inheritable(fd) else None
            tmp_path = Path(tmp)
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    def _save(self, job: JobState) -> None:
        self._atomic_write(
            self._job_path(job.correlation_id),
            job.model_dump_json(indent=2),
        )

    def accept(self, correlation_id: UUID, command: dict[str, Any]) -> JobState:
        job = JobState(correlation_id=correlation_id, command=command)
        self._save(job)
        return job

    def is_duplicate(self, correlation_id: UUID) -> bool:
        return self._job_path(correlation_id).exists()

    def has_active_job(self) -> bool:
        for path in self.state_dir.glob("*.json"):
            try:
                job = JobState.model_validate_json(path.read_text())
                if job.status in ("accepted", "in_progress"):
                    return True
            except Exception:  # noqa: BLE001
                continue
        return False

    def load(self, correlation_id: UUID) -> JobState | None:
        path = self._job_path(correlation_id)
        if not path.exists():
            return None
        return JobState.model_validate_json(path.read_text())

    def load_active(self) -> JobState | None:
        for path in self.state_dir.glob("*.json"):
            try:
                job = JobState.model_validate_json(path.read_text())
                if job.status in ("accepted", "in_progress"):
                    return job
            except Exception:  # noqa: BLE001
                continue
        return None

    def update_phase(
        self, correlation_id: UUID, phase: Phase, phase_status: PhaseStatus
    ) -> JobState:
        job = self.load(correlation_id)
        if job is None:
            raise ValueError(f"Job {correlation_id} not found")
        job.current_phase = phase
        job.phase_results[phase] = phase_status
        if phase_status == PhaseStatus.IN_PROGRESS:
            job.status = "in_progress"
        self._save(job)
        return job

    def complete(
        self,
        correlation_id: UUID,
        status: Literal["success", "failed"],
        errors: list[str] | None = None,
        settling_stage: EnumJobSettlingStage | None = None,
    ) -> JobState:
        """Write the job's terminal verdict, and what it is still doing.

        ``settling_stage`` is part of THIS write, not a second one after it
        (OMN-18636 AC5). A ``complete`` followed by a separate ``set_settling``
        would leave a window -- however short -- in which the record reads
        exactly as it did on 2026-09-17: terminal, with nothing saying that the
        agent is still executing that job's post-terminal work. A window is what
        the 19:56:03Z reader fell into, so there is not one.
        """
        job = self.load(correlation_id)
        if job is None:
            raise ValueError(f"Job {correlation_id} not found")
        job.status = status
        job.completed_at = datetime.now(UTC)
        job.phase_results = reconcile_terminal_phase_results(job.phase_results)
        job.settling_stage = settling_stage
        if errors:
            job.errors.extend(errors)
        self._save(job)
        return job

    def set_settling(
        self, correlation_id: UUID, stage: EnumJobSettlingStage
    ) -> JobState | None:
        """Name the post-terminal phase now executing for this job.

        Returns ``None`` for an unknown job rather than raising: every caller is
        on the post-terminal path, where the job's verdict is already durable and
        already published or owed to the bus, and an exception raised here would
        skip the terminal publish that follows. Losing the settling field is a
        degraded record; losing the publish is a lost result.
        """
        job = self.load(correlation_id)
        if job is None:
            return None
        job.settling_stage = stage
        self._save(job)
        return job

    def clear_settling(self, correlation_id: UUID) -> JobState | None:
        """Declare the job fully settled: nothing further runs for it.

        A field that is only ever set turns every finished job into a
        permanently-settling one, which distinguishes nothing.
        """
        job = self.load(correlation_id)
        if job is None:
            return None
        job.settling_stage = None
        self._save(job)
        return job

    def recover_crashed_jobs(self) -> list[JobState]:
        recovered = []
        for path in self.state_dir.glob("*.json"):
            try:
                job = JobState.model_validate_json(path.read_text())
            except Exception:  # noqa: BLE001
                continue
            if job.status in ("accepted", "in_progress"):
                # OMN-18636: read the interruption BEFORE reconciling. The
                # reconciliation settles every IN_PROGRESS phase to FAILED,
                # which is exactly the fact that distinguishes "this phase was
                # running" from "the job died between phases".
                interruption = describe_interruption(job)
                # Settle every deploy phase, not only the current one: a phase
                # the crashed process never reached is SKIPPED on the record
                # rather than absent from it (OMN-18057).
                job.phase_results = reconcile_terminal_phase_results(job.phase_results)
                job.status = "failed"
                job.completed_at = datetime.now(UTC)
                # OMN-18636 AC5: the process that was executing this job's
                # post-terminal work no longer exists. A settling stage left on
                # a recovered record asserts that a host mutation is in flight
                # in a dead process, which is a worse claim than none.
                job.settling_stage = None
                job.errors.append(interruption)
                job.result_publish_pending = True
                self._save(job)
                recovered.append(job)
        return recovered

    def get_pending_publish(self) -> list[JobState]:
        pending = []
        for path in self.state_dir.glob("*.json"):
            try:
                job = JobState.model_validate_json(path.read_text())
                if job.result_publish_pending:
                    pending.append(job)
            except Exception:  # noqa: BLE001
                continue
        return pending

    def mark_published(self, correlation_id: UUID) -> None:
        job = self.load(correlation_id)
        if job is None:
            return
        job.result_publish_pending = False
        self._save(job)

    def prune_completed(self) -> int:
        pruned = 0
        now = time.time()
        for path in self.state_dir.glob("*.json"):
            try:
                job = JobState.model_validate_json(path.read_text())
            except Exception:  # noqa: BLE001
                continue
            if job.completed_at is None:
                continue
            age_days = (now - job.completed_at.timestamp()) / 86400
            max_age = (
                self.max_failed_age_days
                if job.status == "failed"
                else self.max_completed_age_days
            )
            if age_days >= max_age:
                path.unlink()
                pruned += 1
        return pruned
