# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A build answers health within 5 minutes of its container start, measured per build.

OMN-26316 goal item 1. Nothing recorded container start to first healthy per build:
docker keeps no history and the journal gives only bounds. Verification now reads
``State.StartedAt`` and the time of the first passing ``/health`` and records both,
and the minutes between, on the executor (job record and dev-runtime-started event
carry them); a build past 5 minutes fails verification.

Exemplar: ``test_verify_honours_declared_start_omn19374.py`` -- same real bash helpers
over a fake ``docker``; only curl, compose and the clocks are faked.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, tzinfo
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from deploy_agent.executor import DeployExecutor, VerificationFailedError

from tests.unit import test_verify_honours_declared_start_omn19374 as declared

# Fixtures of the declared-start test, rebound by name so this module uses them.
fake_bin = declared.fake_bin
_real_helpers_no_sleep = declared._real_helpers_no_sleep
_StartingLane = declared._StartingLane

START = datetime(2026, 11, 21, 9, 10, 40, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _wall_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wall time advances with the fake monotonic clock the helper fixture drives."""
    from deploy_agent import executor as executor_mod

    class _FakeDatetime(datetime):
        @classmethod
        def now(cls, tz: tzinfo | None = None) -> datetime:
            return START + timedelta(seconds=executor_mod.time.monotonic())

    monkeypatch.setattr(executor_mod, "datetime", _FakeDatetime)


def _docker_env() -> dict[str, str]:
    return {
        "FAKE_HEALTHCHECK": declared.DECLARED_RUNTIME_HEALTHCHECK,
        "FAKE_STATE": "running starting",
        "FAKE_STARTED_AT": "2026-11-21T09:10:40.123456789Z",
    }


@pytest.mark.unit
def test_known_good_build_records_its_clock(fake_bin: Path) -> None:
    """Probes fail for ~14 s then pass: minutes is recorded and under the limit."""
    lane = _StartingLane(fake_bin=fake_bin, runtime_fails=3, docker_env=_docker_env())
    executor = DeployExecutor()
    with patch("deploy_agent.executor._run", side_effect=lane):
        checks = executor.verify(on_phase_update=declared._noop_phase_update)
    (clock,) = [c for c in checks if c.service == "build-clock"]
    assert clock.status == "pass"
    assert "min after its container start" in clock.detail
    assert all(c.status == "pass" for c in checks if c.service != "docker")


@pytest.mark.unit
def test_known_bad_build_answering_after_six_minutes_fails(fake_bin: Path) -> None:
    """A fake health that answers 6 minutes after the container start fails the build."""
    from deploy_agent import executor as executor_mod

    lane = _StartingLane(fake_bin=fake_bin, runtime_fails=1, docker_env=_docker_env())
    executor = DeployExecutor()
    real_probe = executor._probe_runtime_health

    def slow_first_probe(**kwargs: Any) -> Any:
        if lane.runtime_probes == 1:
            executor_mod._verify_recreate_sleep(360)
        return real_probe(**kwargs)

    with patch("deploy_agent.executor._run", side_effect=lane):
        with patch.object(executor, "_probe_runtime_health", slow_first_probe):
            with pytest.raises(VerificationFailedError):
                executor.verify(on_phase_update=declared._noop_phase_update)
    failed = [c for c in executor.health_checks if c.status == "fail"]
    assert [c.service for c in failed] == ["build-clock"]
    assert "5" in failed[0].detail
