# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runner-fleet job reader and alerts (OMN-17427).

The snapshot EFFECT reads each watched run's jobs (repo, workflow, job, labels,
runner, age) and the health COMPUTE turns them into typed alerts on the
verdict: a runner busy on one job past the job timeout, a runner offline, and
a (repo, labels) queue whose oldest queued job is past the queue-age bound.

The replay test runs the 2026-10-14 measurements (25 pool runners all busy,
none wedged, doc checks queued 34-45 minutes, two second-host runners exited) through
the COMPUTE handler and asserts which alerts fire. No test here makes a real
GitHub, SSH or Docker call.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_runner_fleet_health_compute.handlers.handler_runner_fleet_health_evaluate import (
    HandlerRunnerFleetHealthEvaluate,
)
from omnibase_infra.nodes.node_runner_fleet_health_compute.models.enum_runner_fleet_alert_kind import (
    EnumRunnerFleetAlertKind,
)
from omnibase_infra.nodes.node_runner_fleet_health_compute.models.model_runner_fleet_alert import (
    ModelRunnerFleetAlert,
)
from omnibase_infra.nodes.node_runner_fleet_health_compute.models.model_runner_fleet_health_evaluate_command import (
    ModelRunnerFleetHealthEvaluateCommand,
)
from omnibase_infra.nodes.node_runner_health_snapshot_effect.handlers.handler_runner_fleet_snapshot import (
    HandlerRunnerFleetSnapshot,
    parse_run_jobs,
)
from omnibase_infra.nodes.node_runner_health_snapshot_effect.models.model_runner_fleet_runner_fact import (
    ModelRunnerFleetRunnerFact,
)
from omnibase_infra.nodes.node_runner_health_snapshot_effect.models.model_runner_fleet_snapshot import (
    ModelRunnerFleetSnapshot,
)
from omnibase_infra.nodes.node_runner_health_snapshot_effect.models.model_runner_fleet_snapshot_gather_command import (
    ModelRunnerFleetSnapshotGatherCommand,
)
from omnibase_infra.nodes.node_runner_health_snapshot_effect.models.model_runner_job_fact import (
    ModelRunnerJobFact,
)
from omnibase_infra.observability.runner_health.model_runner_fleet_config import (
    ModelRunnerFleetConfig,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "runner_fleet_replay_2026_10_14.json"
_NOW = datetime(2026, 10, 14, 23, 34, tzinfo=UTC)


def _snapshot(**overrides: object) -> ModelRunnerFleetSnapshot:
    defaults: dict[str, object] = {
        "correlation_id": uuid4(),
        "collected_at": _NOW,
        "host": "runner-host-a",
        "expected_count": 2,
        "runners": (),
    }
    defaults.update(overrides)
    return ModelRunnerFleetSnapshot.model_validate(defaults)


def _job(**overrides: object) -> ModelRunnerJobFact:
    defaults: dict[str, object] = {
        "repo": "OmniNode-ai/repo-a",
        "run_id": 1,
        "job_id": 11,
        "workflow": "CI",
        "job": "pytest",
        "event": "pull_request",
        "status": "in_progress",
        "labels": ("self-hosted", "omnibase-ci"),
        "runner": "omninode-runner-1",
        "age_seconds": 60.0,
    }
    defaults.update(overrides)
    return ModelRunnerJobFact.model_validate(defaults)


async def _alerts(
    snapshot: ModelRunnerFleetSnapshot,
) -> tuple[ModelRunnerFleetAlert, ...]:
    verdict = await HandlerRunnerFleetHealthEvaluate().handle(
        ModelRunnerFleetHealthEvaluateCommand(
            correlation_id=snapshot.correlation_id, snapshot=snapshot
        )
    )
    return verdict.alerts


@pytest.mark.unit
class TestParseRunJobs:
    def test_in_progress_and_queued_jobs_become_facts_with_ages(self) -> None:
        payload = {
            "jobs": [
                {
                    "id": 501,
                    "name": "validate-plugin",
                    "status": "in_progress",
                    "created_at": "2026-10-14T23:00:00Z",
                    "started_at": "2026-10-14T23:12:00Z",
                    "labels": ["self-hosted", "omnibase-ci"],
                    "runner_name": "omninode-runner-4",
                    "workflow_name": "Validate Plugin",
                },
                {
                    "id": 502,
                    "name": "doc-lint",
                    "status": "queued",
                    "created_at": "2026-10-14T23:00:00Z",
                    "started_at": None,
                    "labels": ["self-hosted", "omnibase-ci"],
                    "runner_name": None,
                },
                {
                    "id": 503,
                    "name": "done",
                    "status": "completed",
                    "created_at": "2026-10-14T22:00:00Z",
                    "started_at": "2026-10-14T22:01:00Z",
                    "labels": ["self-hosted"],
                    "runner_name": "omninode-runner-2",
                },
            ]
        }
        facts = parse_run_jobs(
            "OmniNode-ai/repo-b",
            {"id": 77, "name": "Validate Plugin", "event": "merge_group"},
            payload,
            now=_NOW,
        )
        assert [f.job_id for f in facts] == [501, 502]
        running, queued = facts
        assert running.status == "in_progress"
        assert running.runner == "omninode-runner-4"
        assert running.age_seconds == 22 * 60
        assert running.event == "merge_group"
        assert running.run_id == 77
        assert running.labels == ("self-hosted", "omnibase-ci")
        assert queued.status == "queued"
        assert queued.runner == ""
        assert queued.age_seconds == 34 * 60
        assert queued.workflow == "Validate Plugin"

    def test_malformed_job_is_skipped_not_fabricated(self) -> None:
        payload = {"jobs": [{"id": 9, "status": "queued", "created_at": ""}]}
        assert parse_run_jobs("o/r", {"id": 1}, payload, now=_NOW) == ()


@pytest.mark.unit
class TestSnapshotReadsJobs:
    @pytest.mark.asyncio
    async def test_snapshot_carries_jobs_of_watched_runs(self) -> None:
        config = ModelRunnerFleetConfig(
            version="1.0",
            github_org="OmniNode-ai",
            runner_host="runner-host-a",
            runner_group="omnibase-ci",
            runner_name_prefix="omninode-runner",
            expected_count=1,
            watch_repos=("OmniNode-ai/docs-repo",),
        )
        created = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        jobs_payload = {
            "jobs": [
                {
                    "id": 900,
                    "name": "doc-lint",
                    "status": "queued",
                    "created_at": created,
                    "labels": ["self-hosted", "omnibase-ci"],
                    "runner_name": None,
                }
            ]
        }

        class _Proc:
            def __init__(self, stdout: bytes = b"") -> None:
                self._stdout = stdout
                self.returncode = 0

            async def communicate(self) -> tuple[bytes, bytes]:
                return self._stdout, b""

        async def _dispatch(*args: object, **_kwargs: object) -> _Proc:
            argv = [str(a) for a in args]
            if argv[:2] == ["gh", "api"] and argv[2].endswith("/jobs"):
                return _Proc(json.dumps(jobs_payload).encode())
            if argv[:2] == ["gh", "api"] and "/actions/runs" in argv[2]:
                run = {
                    "id": 42,
                    "name": "Doc checks",
                    "status": "queued",
                    "event": "pull_request",
                    "created_at": created,
                }
                return _Proc(json.dumps(run).encode())
            if argv[:2] == ["gh", "api"]:
                return _Proc(b"")
            if argv[:2] == ["gh", "run"]:
                return _Proc(b"[]")
            if argv[0] == "ssh" and "buildx" in argv[2]:
                return _Proc(b"OK\n")
            if argv[0] == "ssh" and "df -P" in argv[2]:
                return _Proc(b"37\n")
            return _Proc(b"")

        handler = HandlerRunnerFleetSnapshot(config=config)
        with patch("asyncio.create_subprocess_exec", AsyncMock(side_effect=_dispatch)):
            snapshot = await handler.handle(
                ModelRunnerFleetSnapshotGatherCommand(correlation_id=uuid4())
            )
        assert len(snapshot.jobs) == 1
        job = snapshot.jobs[0]
        assert (job.repo, job.run_id, job.job_id) == (
            "OmniNode-ai/docs-repo",
            42,
            900,
        )
        assert job.event == "pull_request"
        assert job.status == "queued"


@pytest.mark.unit
class TestRunnerFleetAlerts:
    @pytest.mark.asyncio
    async def test_runner_busy_past_job_timeout_is_stuck(self) -> None:
        snapshot = _snapshot(
            runners=(
                ModelRunnerFleetRunnerFact(
                    name="omninode-runner-1", github_status="online", github_busy=True
                ),
            ),
            jobs=(_job(age_seconds=6 * 3600 + 60),),
        )
        alerts = await _alerts(snapshot)
        stuck = [a for a in alerts if a.kind == EnumRunnerFleetAlertKind.RUNNER_STUCK]
        assert [a.subject for a in stuck] == ["omninode-runner-1"]
        assert stuck[0].repo == "OmniNode-ai/repo-a"
        assert stuck[0].observed_seconds == 6 * 3600 + 60

    @pytest.mark.asyncio
    async def test_runner_on_a_short_job_is_not_stuck(self) -> None:
        snapshot = _snapshot(jobs=(_job(age_seconds=22 * 60),))
        assert await _alerts(snapshot) == ()

    @pytest.mark.asyncio
    async def test_offline_runner_alerts(self) -> None:
        snapshot = _snapshot(
            runners=(
                ModelRunnerFleetRunnerFact(
                    name="omninode-runner-2",
                    github_status="offline",
                    github_busy=False,
                    docker_status="exited",
                ),
            )
        )
        alerts = await _alerts(snapshot)
        assert [(a.kind, a.subject) for a in alerts] == [
            (EnumRunnerFleetAlertKind.RUNNER_OFFLINE, "omninode-runner-2")
        ]

    @pytest.mark.asyncio
    async def test_offline_is_not_claimed_when_github_source_failed(self) -> None:
        snapshot = _snapshot(
            github_source_ok=False,
            runners=(
                ModelRunnerFleetRunnerFact(
                    name="omninode-runner-2",
                    github_status="offline",
                    github_busy=False,
                    docker_status="running",
                ),
            ),
        )
        assert await _alerts(snapshot) == ()

    @pytest.mark.asyncio
    async def test_queue_age_alert_per_repo_and_labels(self) -> None:
        snapshot = _snapshot(
            jobs=(
                _job(
                    repo="OmniNode-ai/docs-repo",
                    status="queued",
                    runner="",
                    age_seconds=31 * 60,
                ),
                _job(
                    repo="OmniNode-ai/docs-repo",
                    job_id=12,
                    status="queued",
                    runner="",
                    age_seconds=5 * 60,
                ),
                _job(
                    repo="OmniNode-ai/omnibase_infra",
                    status="queued",
                    runner="",
                    age_seconds=29 * 60,
                ),
            )
        )
        alerts = await _alerts(snapshot)
        assert len(alerts) == 1
        alert = alerts[0]
        assert alert.kind == EnumRunnerFleetAlertKind.QUEUE_AGE_EXCEEDED
        assert alert.repo == "OmniNode-ai/docs-repo"
        assert alert.labels == ("omnibase-ci", "self-hosted")
        assert alert.observed_seconds == 31 * 60
        assert alert.queued_count == 2


@pytest.mark.unit
class TestReplay20261014:
    @pytest.mark.asyncio
    async def test_replayed_night_fires_queue_and_offline_alerts_and_no_stuck(
        self,
    ) -> None:
        snapshot = ModelRunnerFleetSnapshot.model_validate(
            json.loads(_FIXTURE.read_text())["snapshot"]
        )
        alerts = await _alerts(snapshot)
        by_kind: dict[EnumRunnerFleetAlertKind, list[str]] = {}
        for alert in alerts:
            by_kind.setdefault(alert.kind, []).append(alert.subject)

        assert sorted(by_kind[EnumRunnerFleetAlertKind.QUEUE_AGE_EXCEEDED]) == [
            "OmniNode-ai/docs-repo",
            "OmniNode-ai/repo-a",
            "OmniNode-ai/repo-b",
        ]
        assert sorted(by_kind[EnumRunnerFleetAlertKind.RUNNER_OFFLINE]) == [
            "host2-ci-runner-10",
            "host2-ci-runner-6",
        ]
        assert EnumRunnerFleetAlertKind.RUNNER_STUCK not in by_kind
        doc_alert = next(a for a in alerts if a.subject == "OmniNode-ai/docs-repo")
        assert doc_alert.observed_seconds == pytest.approx(44.9 * 60)
