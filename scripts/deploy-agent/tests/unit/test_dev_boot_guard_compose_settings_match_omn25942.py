# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A verified dev roll is not rolled back because the clone moved after ``up`` (OMN-25942).

Replays dev job 20cf29c6 from its recorded readings (fixture
``fixtures/omn25942/dev_job_20cf29c6_compose_settings_differ.json``): every phase read
success, the runtime answered healthy at 08:24:09Z, then the post-roll capture refused
with ``compose settings differ from the healthy container`` and the guard restored the
previous image. The deploy-source clone had been moved by the canonical clone sync
after the job pinned it, so the capture rendered other compose files than ``up`` had.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest
from deploy_agent import executor as executor_mod
from deploy_agent.executor import DeployExecutor

from tests.unit import test_dev_boot_guard_omn21279 as boot_guard_tests

pytestmark = [pytest.mark.unit, pytest.mark.dev_boot_guard]

docker = boot_guard_tests.docker

FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "omn25942"
    / "dev_job_20cf29c6_compose_settings_differ.json"
)
RECORDED = json.loads(FIXTURE.read_text())
JOB = RECORDED["job_record"]["correlation_id"]
JOB_COMMIT = RECORDED["job_record"]["git_ref"]


def _moved_clone_hashes(monkeypatch: pytest.MonkeyPatch) -> None:
    """What ``compose config --hash`` prints once the clone holds other compose files."""
    run = executor_mod._run

    def moved(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        result = run(argv, **kwargs)
        if "config" in argv and "--hash" in argv:
            result.stdout = "\n".join(
                f"{s} moved-{s}" for s in boot_guard_tests.SERVICES
            )
        return result

    monkeypatch.setattr(executor_mod, "_run", moved)


def test_recorded_job_shows_a_verified_roll_and_a_moved_clone() -> None:
    record = RECORDED["job_record"]
    assert all(result == "success" for result in record["phase_results"].values())
    assert "compose settings differ" in record["errors"][0]
    pinned_at = RECORDED["deploy_source_clone_reflog"]["entries_utc"][0]["at"]
    moves = [
        entry["at"]
        for entry in RECORDED["deploy_source_clone_reflog"]["entries_utc"][1:]
        if entry["at"] < RECORDED["incident"]["at"]
    ]
    assert moves and all(at > pinned_at for at in moves)


def test_compose_settings_match_failure_does_not_roll_back_a_verified_roll(
    tmp_path: Path,
    docker: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = DeployExecutor()
    executor.verify = Mock(return_value=[])
    guard = executor_mod.DevBootGuard(executor, tmp_path)
    guard.prepare()
    mark_failed = Mock()
    commands_before = len(docker["commands"])

    with guard.protect(JOB, JOB_COMMIT, mark_failed):
        docker["containers"] = boot_guard_tests._containers()
        _moved_clone_hashes(monkeypatch)

    mark_failed.assert_not_called()
    assert not list(tmp_path.rglob("incident-*.json"))
    assert not any("up" in argv for argv in docker["commands"][commands_before:]), (
        "a verified roll must not be restored to the previous image"
    )
    assert guard.point.current is not None
    assert guard.point.current.revision == boot_guard_tests.GOOD_COMMIT
    (drift_path,) = tmp_path.rglob(f"settings-drift-{JOB}.json")
    drift = json.loads(drift_path.read_text())
    assert drift["commit"] == JOB_COMMIT
    assert drift["service"] == "omninode-runtime"
    assert drift["container_config_hash"] == "hash-omninode-runtime"
    assert drift["rendered_config_hash"] == "moved-omninode-runtime"
    assert drift["kept_recovery_revision"] == boot_guard_tests.GOOD_COMMIT


def test_compose_settings_match_still_refuses_to_capture_on_drift(
    tmp_path: Path,
    docker: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard = executor_mod.DevBootGuard(DeployExecutor(), tmp_path)
    _moved_clone_hashes(monkeypatch)
    with pytest.raises(
        boot_guard_tests.guard_mod.DevBootSettingsDriftError,
        match="omninode-runtime: compose settings differ from the healthy container",
    ):
        guard._capture()


def test_compose_settings_match_failed_roll_is_still_restored(
    tmp_path: Path,
    docker: dict,
) -> None:
    executor = DeployExecutor()
    executor.verify = Mock(return_value=[])
    guard = executor_mod.DevBootGuard(executor, tmp_path)
    guard.prepare()
    mark_failed = Mock()
    with pytest.raises(boot_guard_tests.guard_mod.DevBootRecoveryError):
        with guard.protect(JOB, JOB_COMMIT, mark_failed):
            docker["containers"] = boot_guard_tests._containers(healthy=False)
            raise RuntimeError("compose up failed")
    mark_failed.assert_called_once()
    assert any("up" in argv for argv in docker["commands"])
