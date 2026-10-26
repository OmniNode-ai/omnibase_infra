# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Replay the three exhausted dev runtimes from job c37f6142 (2026-10-25)."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock
from uuid import uuid4

import pytest
from deploy_agent import executor as executor_mod
from deploy_agent.events import EnumRuntimeLane, ModelRebuildRequested, Scope
from deploy_agent.executor import DeployExecutor

from tests.unit.test_auto_rollback import _agent

guard_mod = executor_mod.load_dev_boot_guard()

pytestmark = [pytest.mark.unit, pytest.mark.dev_boot_guard]
BAD_COMMIT = "67735dc2a39458a6a1cff5d029c8470151cf17ef"
GOOD_COMMIT = "a" * 40
BOOT_ERROR = (
    "[entrypoint] BOOT FAILED 5/5: exit 1: ProtocolConfigurationError "
    "ONEX_CORE_041_INVALID_CONFIGURATION: BIFROST_BASE_CONTRACT_PATH is unset "
    "or blank; there is no default."
)
SERVICES = ("omninode-runtime", "runtime-effects", "runtime-worker")


def _containers(*, healthy: bool = True) -> list[dict]:
    project = executor_mod.lane_config_for(EnumRuntimeLane.DEV).compose_project
    return [
        {
            "Id": f"container-{service}",
            "Name": f"/{project}-{service}-1",
            "Image": "sha256:" + str(index + 1) * 64,
            "State": {
                "Status": "running",
                "Health": {"Status": "healthy" if healthy else "starting"},
            },
            "Config": {
                "Image": f"runtime-{service}:latest",
                "Healthcheck": {
                    "StartPeriod": 10_000_000_000,
                    "Interval": 1_000_000_000,
                    "Timeout": 1_000_000_000,
                    "Retries": 5,
                },
                "Labels": {
                    "com.docker.compose.project": project,
                    "com.docker.compose.service": service,
                    "com.docker.compose.oneoff": "False",
                    "com.docker.compose.config-hash": f"hash-{service}",
                    "org.opencontainers.image.revision": GOOD_COMMIT,
                },
            },
        }
        for index, service in enumerate(SERVICES)
    ]


@pytest.fixture
def docker(monkeypatch: pytest.MonkeyPatch) -> dict:
    state = {"containers": _containers(), "logs": "", "commands": []}
    project = executor_mod.lane_config_for(EnumRuntimeLane.DEV).compose_project
    state["compose"] = {
        "name": project,
        "services": {
            service: {
                "image": "runtime:latest",
                "environment": {"BIFROST_BASE_CONTRACT_PATH": "/old/base.yaml"},
                "depends_on": {"postgres": {"condition": "service_healthy"}},
                "networks": {"default": None},
                "volumes": [
                    {"type": "volume", "source": "data", "target": "/app/data"}
                ],
            }
            for service in SERVICES
        },
        "networks": {"default": {"name": f"{project}_default"}},
        "volumes": {"data": {"name": f"{project}_data"}},
    }

    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        state["commands"].append(argv)
        output = ""
        if argv[:2] == ["docker", "ps"]:
            output = "\n".join(c["Id"] for c in state["containers"])
        elif argv[:2] == ["docker", "inspect"]:
            output = json.dumps(state["containers"])
        elif argv[:2] == ["docker", "logs"]:
            output = state["logs"]
        elif "config" in argv:
            output = (
                "\n".join(f"{s} hash-{s}" for s in SERVICES)
                if "--hash" in argv
                else json.dumps(state["compose"])
            )
        elif "up" in argv:
            state["containers"] = _containers()
            state["logs"] = ""
        return subprocess.CompletedProcess(argv, 0, output, "")

    monkeypatch.setattr(executor_mod, "_run", run)
    state["deliver_reports"] = guard_mod.deliver_reports
    monkeypatch.setattr(guard_mod, "deliver_reports", Mock())
    return state


def test_recorded_boot_exhaustion_fails_without_spending_the_start_window(
    tmp_path: Path, docker: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    docker["containers"] = _containers(healthy=False)
    docker["logs"] = (
        BOOT_ERROR + "\n[entrypoint] RETRIES EXHAUSTED: 5 consecutive failing boots"
    )
    sleep = Mock(side_effect=AssertionError("exhausted boots must fail immediately"))
    monkeypatch.setattr(guard_mod.time, "sleep", sleep)
    guard = executor_mod.DevBootGuard(DeployExecutor(), tmp_path)
    with pytest.raises(
        guard_mod.DevBootFailedError, match="BIFROST_BASE_CONTRACT_PATH"
    ):
        guard.wait()
    sleep.assert_not_called()


def test_all_three_containers_must_be_docker_healthy(
    tmp_path: Path, docker: dict
) -> None:
    docker["containers"][2]["State"]["Health"]["Status"] = "unhealthy"
    with pytest.raises(guard_mod.DevBootFailedError, match="runtime-worker"):
        executor_mod.DevBootGuard(DeployExecutor(), tmp_path).wait()


def test_starting_containers_get_their_declared_window(
    tmp_path: Path, docker: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    docker["containers"] = _containers(healthy=False)
    tick = [0.0]
    monkeypatch.setattr(guard_mod.time, "monotonic", lambda: tick[0])

    def sleep(seconds: float) -> None:
        tick[0] += seconds
        if tick[0] >= 10:
            docker["containers"] = _containers()

    monkeypatch.setattr(guard_mod.time, "sleep", sleep)
    executor_mod.DevBootGuard(DeployExecutor(), tmp_path).wait()
    assert tick[0] == 10


def test_window_expires_even_when_docker_keeps_reporting_starting(
    tmp_path: Path, docker: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    docker["containers"] = _containers(healthy=False)
    tick = [0.0]
    monkeypatch.setattr(guard_mod.time, "monotonic", lambda: tick[0])
    monkeypatch.setattr(
        guard_mod.time, "sleep", lambda seconds: tick.__setitem__(0, tick[0] + seconds)
    )
    with pytest.raises(guard_mod.DevBootFailedError, match="window"):
        executor_mod.DevBootGuard(DeployExecutor(), tmp_path).wait()
    assert tick[0] == 20


@pytest.mark.parametrize("failure", ["compose", "probe-timeout", "boots"])
def test_failed_deploy_redeploys_retained_images_and_old_settings_and_stays_failed(
    tmp_path: Path, docker: dict, failure: str
) -> None:
    executor = DeployExecutor()
    executor.verify = Mock(return_value=[])
    guard = executor_mod.DevBootGuard(executor, tmp_path)
    guard.prepare()
    saved = copy.deepcopy(docker["compose"])
    docker["compose"]["services"]["omninode-runtime"]["environment"] = {}
    mark_failed = Mock()
    with pytest.raises(guard_mod.DevBootRecoveryError, match=BAD_COMMIT):
        with guard.protect("c37f6142", BAD_COMMIT, mark_failed):
            docker["containers"] = _containers(healthy=False)
            docker["logs"] = BOOT_ERROR
            if failure == "compose":
                raise RuntimeError("compose up failed")
            if failure == "probe-timeout":
                raise subprocess.TimeoutExpired("curl", 10)
            guard.wait()
    mark_failed.assert_called_once()
    up = next(argv for argv in docker["commands"] if "up" in argv)
    assert up[:2] == ["docker", "compose"]
    assert up[up.index("-p") + 1] == saved["name"]
    assert "--no-build" in up and "--no-deps" in up
    assert up[up.index("--pull") + 1] == "never"
    assert set(up[-3:]) == set(SERVICES)
    recovered = json.loads(Path(up[up.index("-f") + 1]).read_text())
    assert recovered["services"]["omninode-runtime"]["environment"] == {
        "BIFROST_BASE_CONTRACT_PATH": "/old/base.yaml"
    }
    assert all("boot-good" in s["image"] for s in recovered["services"].values())
    assert recovered["networks"]["default"]["external"] is True
    assert recovered["volumes"]["data"]["external"] is True
    assert not any(argv[0] == "git" for argv in docker["commands"])
    incidents = list(tmp_path.rglob("incident-*.json"))
    assert len(incidents) == 1
    incident = json.loads(incidents[0].read_text())
    assert incident["commit"] == BAD_COMMIT
    assert incident["restored_revision"] == GOOD_COMMIT
    assert "BIFROST_BASE_CONTRACT_PATH" in incident["error"]
    assert incident["recovery_verified"] is True
    executor.verify.assert_called_once()


def test_last_healthy_snapshot_survives_agent_restart(
    tmp_path: Path, docker: dict
) -> None:
    first = executor_mod.DevBootGuard(DeployExecutor(), tmp_path)
    first.prepare()
    docker["containers"] = _containers(healthy=False)
    second = executor_mod.DevBootGuard(DeployExecutor(), tmp_path)
    second.prepare()
    assert second.point is not None and second.point.revision == GOOD_COMMIT
    assert (second.snapshot_path.stat().st_mode & 0o777) == 0o400


def test_no_healthy_history_refuses_before_replacing_any_container(
    tmp_path: Path, docker: dict
) -> None:
    docker["containers"] = _containers(healthy=False)
    with pytest.raises(guard_mod.DevBootFailedError, match="last healthy"):
        executor_mod.DevBootGuard(DeployExecutor(), tmp_path).prepare()
    assert not any("up" in argv for argv in docker["commands"])


def test_recovery_also_restores_a_container_left_absent_by_compose(
    tmp_path: Path, docker: dict
) -> None:
    executor = DeployExecutor()
    executor.verify = Mock(return_value=[])
    guard = executor_mod.DevBootGuard(executor, tmp_path)
    guard.prepare()
    with pytest.raises(guard_mod.DevBootRecoveryError, match="verified=True"):
        with guard.protect("missing-container", BAD_COMMIT, Mock()):
            docker["containers"].pop()
            raise RuntimeError("compose failed after removing the worker")
    assert len(docker["containers"]) == 3
    assert any("up" in argv for argv in docker["commands"])


def test_predeploy_refusal_keeps_the_original_healthy_family_running(
    tmp_path: Path, docker: dict
) -> None:
    guard = executor_mod.DevBootGuard(DeployExecutor(), tmp_path)
    guard.prepare()
    with pytest.raises(guard_mod.DevBootRecoveryError):
        with guard.protect("bad-candidate", BAD_COMMIT, Mock()):
            raise guard_mod.DevBootFailedError(
                "BIFROST_BASE_CONTRACT_PATH is unset or blank"
            )
    assert not any("up" in argv for argv in docker["commands"])


def test_new_container_generation_on_same_image_still_restores_old_settings(
    tmp_path: Path, docker: dict
) -> None:
    executor = DeployExecutor()
    executor.verify = Mock(return_value=[])
    guard = executor_mod.DevBootGuard(executor, tmp_path)
    guard.prepare()
    with pytest.raises(guard_mod.DevBootRecoveryError, match="verified=True"):
        with guard.protect("same-image-new-config", BAD_COMMIT, Mock()):
            for container in docker["containers"]:
                container["Id"] = "candidate-" + container["Id"]
            raise RuntimeError(
                "candidate HTTP health failed after config-only recreate"
            )
    assert any("up" in argv for argv in docker["commands"])


def test_failed_recovery_is_reported_without_losing_the_original_boot_cause(
    tmp_path: Path, docker: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    guard = executor_mod.DevBootGuard(DeployExecutor(), tmp_path)
    guard.prepare()
    original = executor_mod._run

    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if "up" in argv:
            return subprocess.CompletedProcess(argv, 1, "", "docker unavailable")
        return original(argv, **kwargs)

    monkeypatch.setattr(executor_mod, "_run", run)
    with pytest.raises(guard_mod.DevBootRecoveryError, match="verified=False"):
        with guard.protect("failed-recovery", BAD_COMMIT, Mock()):
            docker["containers"] = _containers(healthy=False)
            docker["logs"] = BOOT_ERROR
            guard.wait()
    incident = json.loads(next(tmp_path.rglob("incident-*.json")).read_text())
    assert "BIFROST_BASE_CONTRACT_PATH" in incident["error"]
    assert incident["recovery_verified"] is False
    assert incident["ledger_pending"] is True and incident["exception_pending"] is True


def test_canonical_compose_literal_dollars_are_preserved_on_recovery(
    tmp_path: Path, docker: dict
) -> None:
    # `docker compose config --format json` escapes literals for its own
    # round trip. Double-escaping them again changes the runtime's value.
    value = "literal-$$BOOT_GUARD_UNSET_1735"
    docker["compose"]["services"]["omninode-runtime"]["environment"]["VALUE"] = value
    executor = DeployExecutor()
    executor.verify = Mock(return_value=[])
    guard = executor_mod.DevBootGuard(executor, tmp_path)
    guard.prepare()
    with pytest.raises(guard_mod.DevBootRecoveryError):
        with guard.protect("literal-env", BAD_COMMIT, Mock()):
            docker["containers"] = _containers(healthy=False)
            raise RuntimeError("failed candidate")
    recovered = json.loads((guard.root / "recovery-compose.json").read_text())
    assert recovered["services"]["omninode-runtime"]["environment"]["VALUE"] == value


def test_boot_counter_is_detected_after_earlier_attempts_in_the_log(
    tmp_path: Path, docker: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    docker["containers"] = _containers(healthy=False)
    docker["logs"] = BOOT_ERROR.replace("5/5", "1/5") + "\n" + BOOT_ERROR
    monkeypatch.setattr(
        guard_mod.time,
        "sleep",
        Mock(side_effect=AssertionError("boot budget exhausted")),
    )
    with pytest.raises(guard_mod.DevBootFailedError, match="BOOT FAILED 5/5"):
        executor_mod.DevBootGuard(DeployExecutor(), tmp_path).wait()


def test_setting_error_logged_separately_from_the_counter_reaches_the_incident(
    tmp_path: Path, docker: dict
) -> None:
    executor = DeployExecutor()
    executor.verify = Mock(return_value=[])
    guard = executor_mod.DevBootGuard(executor, tmp_path)
    guard.prepare()
    with pytest.raises(guard_mod.DevBootRecoveryError):
        with guard.protect("separate-cause", BAD_COMMIT, Mock()):
            docker["containers"] = _containers(healthy=False)
            docker["logs"] = (
                "ProtocolConfigurationError: BIFROST_BASE_CONTRACT_PATH is unset or blank\n"
                "[entrypoint] BOOT FAILED 5/5: exit 1: preflight exited 1"
            )
            guard.wait()
    incident = json.loads(next(tmp_path.rglob("incident-*.json")).read_text())
    assert "BIFROST_BASE_CONTRACT_PATH is unset or blank" in incident["error"]


@pytest.mark.parametrize("missing", ["unset", "package-file", "active-package"])
def test_candidate_preflight_replays_unset_and_missing_package_settings(
    tmp_path: Path, missing: str
) -> None:
    env = {**os.environ, "BIFROST_CONTRACT_PATH": str(tmp_path / "rendered.yaml")}
    env.pop("BIFROST_BASE_CONTRACT_PATH", None)
    env.pop("ONEX_ACTIVE_RUNTIME_PACKAGES", None)
    expected = "BIFROST_BASE_CONTRACT_PATH"
    if missing == "package-file":
        env["BIFROST_BASE_CONTRACT_PATH"] = str(
            tmp_path / "site-packages/missing_package/base.yaml"
        )
    elif missing == "active-package":
        path = tmp_path / "base.yaml"
        path.write_text("{}")
        env["BIFROST_BASE_CONTRACT_PATH"] = str(path)
        env["ONEX_ACTIVE_RUNTIME_PACKAGES"] = "required_package_absent_omn21279"
        expected = "ONEX_ACTIVE_RUNTIME_PACKAGES"
    done = subprocess.run(
        [sys.executable, "-c", executor_mod.IMAGE_PREFLIGHT_SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert done.returncode != 0
    assert expected in done.stderr


def test_candidate_with_existing_contract_and_package_passes(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text("{}")
    env = {
        **os.environ,
        "BIFROST_CONTRACT_PATH": "/output.yaml",
        "BIFROST_BASE_CONTRACT_PATH": str(base),
        "ONEX_ACTIVE_RUNTIME_PACKAGES": "json",
    }
    done = subprocess.run(
        [sys.executable, "-c", executor_mod.IMAGE_PREFLIGHT_SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert done.returncode == 0, done.stderr


@pytest.mark.parametrize("failure", ["compose", "boots", "probe-timeout"])
def test_agent_real_dev_flow_recovers_before_publishing_failed_job(
    tmp_path: Path, docker: dict, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    cmd = ModelRebuildRequested(
        correlation_id=uuid4(),
        requested_by="recorded-incident-replay",
        scope=Scope.RUNTIME,
        runtime_lane=EnumRuntimeLane.DEV,
    )
    agent, executor = _agent(tmp_path, monkeypatch, cmd)
    monkeypatch.setattr(
        "deploy_agent.agent.capture_rollback_point", Mock(return_value=None)
    )
    executor.git_pull.return_value = BAD_COMMIT
    executor.dev_boot_config_hashes.side_effect = (
        DeployExecutor().dev_boot_config_hashes
    )
    executor.verify_dev_boot_recovery.side_effect = lambda **kw: executor.verify(
        **kw, lane=EnumRuntimeLane.DEV, recover_runtime=False
    )

    def verify(**kwargs: object) -> list:
        if kwargs.get("recover_runtime") is False:
            kwargs["on_phase_update"](
                executor_mod.Phase.VERIFICATION, executor_mod.PhaseStatus.IN_PROGRESS
            )
            recovering = agent.job_store.load(cmd.correlation_id)
            assert recovering is not None and recovering.status == "failed"
        return []

    executor.verify.side_effect = verify

    def rebuild(*args: object, **kwargs: object) -> list[str]:
        docker["containers"] = _containers(healthy=False)
        docker["logs"] = BOOT_ERROR
        if failure == "compose":
            raise RuntimeError("compose up left the dev runtimes down")
        if failure == "probe-timeout":
            docker["containers"] = _containers()
            docker["containers"][0]["Image"] = "sha256:" + "9" * 64

            def timeout_then_recover(**kw: object) -> list:
                if kw.get("recover_runtime") is False:
                    return verify(**kw)
                raise subprocess.TimeoutExpired("curl", 10)

            executor.verify.side_effect = timeout_then_recover
        return list(SERVICES)

    executor.rebuild_scope.side_effect = rebuild
    agent._run_deploy(cmd)
    job = agent.job_store.load(cmd.correlation_id)
    assert job is not None and job.status == "failed"
    assert all(guard_mod._healthy(c) for c in docker["containers"])
    assert any("up" in argv for argv in docker["commands"])
    assert BAD_COMMIT in job.errors[0]
    assert f"forward redeploy revision={GOOD_COMMIT} verified=True" in " ".join(
        job.errors
    )


@pytest.mark.parametrize("lane", [EnumRuntimeLane.PROD, EnumRuntimeLane.STABILITY_TEST])
def test_nondev_jobs_never_construct_the_dev_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lane: EnumRuntimeLane
) -> None:
    cmd = ModelRebuildRequested(
        correlation_id=uuid4(),
        requested_by="lane-fence-test",
        scope=Scope.RUNTIME,
        runtime_lane=lane,
        image_digest="sha256:" + "f" * 64 if lane is EnumRuntimeLane.PROD else None,
    )
    agent, _ = _agent(tmp_path, monkeypatch, cmd)
    guard = Mock(side_effect=AssertionError("non-dev job reached dev boot guard"))
    monkeypatch.setattr("deploy_agent.agent.DevBootGuard", guard)
    agent._run_deploy(cmd)
    guard.assert_not_called()
    job = agent.job_store.load(cmd.correlation_id)
    assert job is not None and job.status == "success"


def test_executor_preflight_runs_before_first_runtime_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(argv)
        return subprocess.CompletedProcess(
            argv,
            1,
            "",
            "BIFROST_BASE_CONTRACT_PATH is unset or blank; there is no default",
        )

    monkeypatch.setattr(executor_mod, "_run", run)
    executor = DeployExecutor()
    with pytest.raises(
        guard_mod.DevBootFailedError, match="BIFROST_BASE_CONTRACT_PATH"
    ):
        executor._compose_up(
            executor_mod.Phase.RUNTIME,
            Scope.RUNTIME,
            list(SERVICES),
            Mock(),
            lane=EnumRuntimeLane.DEV,
        )
    assert len(commands) == 1
    assert "run" in commands[0] and "--no-deps" in commands[0]
    assert "up" not in commands[0]
    assert "--entrypoint" in commands[0]


@pytest.mark.parametrize("lane", [EnumRuntimeLane.PROD, EnumRuntimeLane.STABILITY_TEST])
def test_candidate_preflight_has_no_nondev_effects(
    monkeypatch: pytest.MonkeyPatch, lane: EnumRuntimeLane
) -> None:
    run = Mock(side_effect=AssertionError("non-dev image was probed"))
    monkeypatch.setattr(executor_mod, "_run", run)
    executor_mod.preflight_dev_images(lane, list(SERVICES))
    run.assert_not_called()


def test_reporting_outage_keeps_status_and_exception_pending_for_retry(
    tmp_path: Path, docker: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Use the real delivery method here, with both external transports isolated.
    monkeypatch.setattr(guard_mod, "deliver_reports", docker["deliver_reports"])
    root = tmp_path / "reports"
    path = root / "incident-c37f6142.json"
    incident = {
        "job": "c37f6142",
        "commit": BAD_COMMIT,
        "error": BOOT_ERROR,
        "project": "omnibase-infra",
        "restored_revision": GOOD_COMMIT,
        "recovery_verified": True,
        "at": "2026-10-25T15:46:43Z",
        "ledger_pending": True,
        "exception_pending": True,
    }
    executor_mod._atomic_write_private_json(path, incident)
    monkeypatch.setenv("ONEX_LEDGER_WRITE_VIA", "bus")
    monkeypatch.setenv("DEPLOY_AGENT_REPORT_PLUGIN_DIR", str(tmp_path / "plugin"))
    run = Mock(return_value=subprocess.CompletedProcess([], 75, "", "bus unavailable"))
    monkeypatch.setattr(executor_mod, "_run", run)
    executor_mod.deliver_dev_boot_reports(root)
    assert json.loads(path.read_text())["ledger_pending"] is True
    assert json.loads(path.read_text())["exception_pending"] is True
    run.side_effect = [
        subprocess.CompletedProcess([], 0, "OK line=42 via=bus status=accepted", ""),
        subprocess.CompletedProcess(
            [], 0, "EXCEPTION-QUEUE appended id=boot-failed", ""
        ),
    ]
    executor_mod.deliver_dev_boot_reports(root)
    stored = json.loads(path.read_text())
    assert stored["ledger_pending"] is False and stored["exception_pending"] is False
    assert BAD_COMMIT in str(run.call_args_list)
    assert "BIFROST_BASE_CONTRACT_PATH" in str(run.call_args_list)
    status_rows = [
        call.args[0][-1] for call in run.call_args_list if "--append" in call.args[0]
    ]
    assert len(status_rows) == 2 and status_rows[0] == status_rows[1]
    calls = run.call_count
    executor_mod.deliver_dev_boot_reports(root)
    assert run.call_count == calls
