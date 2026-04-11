# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Durable job state with structured recovery."""

from __future__ import annotations

import os
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field

from deploy_agent.events import Phase, PhaseStatus


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
    ) -> JobState:
        job = self.load(correlation_id)
        if job is None:
            raise ValueError(f"Job {correlation_id} not found")
        job.status = status
        job.completed_at = datetime.now(UTC)
        if errors:
            job.errors.extend(errors)
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
                # Mark the current in-progress phase as failed
                if (
                    job.current_phase in job.phase_results
                    and job.phase_results[job.current_phase] == PhaseStatus.IN_PROGRESS
                ):
                    job.phase_results[job.current_phase] = PhaseStatus.FAILED
                job.status = "failed"
                job.completed_at = datetime.now(UTC)
                job.errors.append(f"interrupted during phase {job.current_phase}")
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
