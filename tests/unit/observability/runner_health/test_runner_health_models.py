# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for runner health models."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from omnibase_infra.observability.runner_health.enum_runner_health_state import (
    EnumRunnerHealthState,
)
from omnibase_infra.observability.runner_health.model_runner_health_alert import (
    ModelRunnerHealthAlert,
)
from omnibase_infra.observability.runner_health.model_runner_health_snapshot import (
    ModelRunnerHealthSnapshot,
)
from omnibase_infra.observability.runner_health.model_runner_status import (
    ModelRunnerStatus,
)


@pytest.mark.unit
class TestModelRunnerStatus:
    def test_healthy_runner(self) -> None:
        status = ModelRunnerStatus(
            name="omninode-runner-1",
            github_status="online",
            github_busy=False,
            docker_status="healthy",
            docker_uptime="Up 2 hours (healthy)",
            state=EnumRunnerHealthState.HEALTHY,
        )
        assert status.name == "omninode-runner-1"
        assert status.state == EnumRunnerHealthState.HEALTHY

    def test_mismatch_runner(self) -> None:
        status = ModelRunnerStatus(
            name="omninode-runner-3",
            github_status="offline",
            github_busy=False,
            docker_status="healthy",
            docker_uptime="Up 1 hour (healthy)",
            state=EnumRunnerHealthState.GITHUB_OFFLINE,
        )
        assert status.state == EnumRunnerHealthState.GITHUB_OFFLINE

    def test_frozen(self) -> None:
        status = ModelRunnerStatus(
            name="omninode-runner-1",
            github_status="online",
            github_busy=False,
            docker_status="healthy",
            docker_uptime="Up 2 hours",
            state=EnumRunnerHealthState.HEALTHY,
        )
        with pytest.raises(Exception):
            status.name = "changed"  # type: ignore[misc]


@pytest.mark.unit
class TestModelRunnerHealthSnapshot:
    def test_from_statuses(self) -> None:
        now = datetime.now(tz=UTC)
        statuses = (
            ModelRunnerStatus(
                name="omninode-runner-1",
                github_status="online",
                github_busy=False,
                docker_status="healthy",
                docker_uptime="Up 2h",
                state=EnumRunnerHealthState.HEALTHY,
            ),
        )
        snapshot = ModelRunnerHealthSnapshot(
            correlation_id=uuid4(),
            collected_at=now,
            runners=statuses,
            expected_runners=10,
            observed_runners=1,
            healthy_count=1,
            degraded_count=0,
            host="192.168.86.201",
            host_disk_percent=38.0,
        )
        assert snapshot.healthy_count == 1
        assert snapshot.expected_runners == 10
        assert snapshot.observed_runners == 1

    def test_frozen(self) -> None:
        now = datetime.now(tz=UTC)
        snapshot = ModelRunnerHealthSnapshot(
            correlation_id=uuid4(),
            collected_at=now,
            runners=(),
            expected_runners=10,
            observed_runners=0,
            healthy_count=0,
            degraded_count=0,
            host="192.168.86.201",
            host_disk_percent=25.0,
        )
        with pytest.raises(Exception):
            snapshot.expected_runners = 5  # type: ignore[misc]

    def test_source_failure_tracking(self) -> None:
        now = datetime.now(tz=UTC)
        snapshot = ModelRunnerHealthSnapshot(
            correlation_id=uuid4(),
            collected_at=now,
            runners=(),
            expected_runners=10,
            observed_runners=0,
            healthy_count=0,
            degraded_count=0,
            github_source_ok=False,
            docker_source_ok=True,
            source_errors=("GitHub API returned exit code 1",),
            host="192.168.86.201",
            host_disk_percent=25.0,
        )
        assert not snapshot.github_source_ok
        assert snapshot.source_errors[0].startswith("GitHub")


@pytest.mark.unit
class TestModelRunnerHealthAlert:
    def test_to_slack_message(self) -> None:
        alert = ModelRunnerHealthAlert(
            correlation_id=uuid4(),
            alert_type="runner_health_degraded",
            degraded_runners=(
                ModelRunnerStatus(
                    name="omninode-runner-3",
                    github_status="offline",
                    github_busy=False,
                    docker_status="healthy",
                    docker_uptime="Up 1h",
                    state=EnumRunnerHealthState.GITHUB_OFFLINE,
                ),
            ),
            total_runners=10,
            healthy_count=9,
            host="192.168.86.201",
        )
        msg = alert.to_slack_message()
        assert "omninode-runner-3" in msg
        assert "9/10" in msg

    def test_frozen(self) -> None:
        alert = ModelRunnerHealthAlert(
            correlation_id=uuid4(),
            total_runners=10,
            healthy_count=10,
            host="192.168.86.201",
        )
        with pytest.raises(Exception):
            alert.total_runners = 5  # type: ignore[misc]
