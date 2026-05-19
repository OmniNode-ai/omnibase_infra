# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration checks for runner health model public API surface (OMN-11268)."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from omnibase_infra.observability.runner_health import (
    EnumRunnerHealthState,
    ModelRunnerHealthAlert,
    ModelRunnerHealthSnapshot,
    ModelRunnerStatus,
)


@pytest.mark.integration
def test_runner_health_package_exports_all_required_symbols() -> None:
    """All four public symbols must be importable from the package __init__."""
    assert EnumRunnerHealthState
    assert ModelRunnerStatus
    assert ModelRunnerHealthSnapshot
    assert ModelRunnerHealthAlert


@pytest.mark.integration
def test_enum_runner_health_state_has_required_members() -> None:
    required = {
        "HEALTHY",
        "GITHUB_OFFLINE",
        "DOCKER_UNHEALTHY",
        "OOM_KILLED",
        "CRASH_LOOPING",
        "STALE_REGISTRATION",
        "MISSING",
    }
    actual = {m.name for m in EnumRunnerHealthState}
    assert required <= actual, f"Missing members: {required - actual}"


@pytest.mark.integration
def test_alert_to_slack_message_covers_all_degraded_states() -> None:
    degraded_states = [
        s for s in EnumRunnerHealthState if s != EnumRunnerHealthState.HEALTHY
    ]
    runners = tuple(
        ModelRunnerStatus(
            name=f"runner-{s.value}",
            github_status="offline",
            github_busy=False,
            docker_status="unhealthy",
            docker_uptime="",
            state=s,
            error=f"error for {s.value}",
        )
        for s in degraded_states
    )
    alert = ModelRunnerHealthAlert(
        correlation_id=uuid4(),
        degraded_runners=runners,
        total_runners=len(runners),
        healthy_count=0,
        host="192.168.86.201",
    )
    msg = alert.to_slack_message()
    for r in runners:
        assert r.name in msg, f"Runner {r.name} missing from Slack message"
        assert r.state.value in msg, f"State {r.state.value} missing from Slack message"


@pytest.mark.integration
def test_snapshot_roundtrip_via_model_dump() -> None:
    runner = ModelRunnerStatus(
        name="omninode-runner-1",
        github_status="online",
        github_busy=False,
        docker_status="healthy",
        docker_uptime="Up 3h",
        state=EnumRunnerHealthState.HEALTHY,
    )
    snap = ModelRunnerHealthSnapshot(
        correlation_id=uuid4(),
        collected_at=datetime.now(tz=UTC),
        runners=(runner,),
        expected_runners=4,
        observed_runners=1,
        healthy_count=1,
        degraded_count=0,
        host="192.168.86.201",
    )
    dumped = snap.model_dump(mode="json")
    assert dumped["healthy_count"] == 1
    assert dumped["runners"][0]["name"] == "omninode-runner-1"
