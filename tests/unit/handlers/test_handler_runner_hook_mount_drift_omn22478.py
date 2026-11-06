# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Control for the tracked runner hook mount drift check (OMN-22478).

The recording is the runner host read on 2026-11-06 at about 15:50Z. Eleven
containers (runner-1 to runner-10 and the deploy runner) execute the hook
directory mounted under the hook directory path, whose copies equal the host
copies, while the image-baked ``/usr/local/bin/runner-job-started.sh`` they never
run hashes differently (85c02074...). ``omninode-runner-18`` is a container with
single-file bind mounts pinned to old inodes, executing ``/usr/local/bin``
copies that really differ from the host.

The retired host-only script hashed ``/usr/local/bin`` everywhere, so it flagged
all eleven current containers. ``legacy_findings`` below is that rule; it is the
failing control that reproduces the 11 false findings the tracked check must not
report, and the check must still report the real runner-18 drift.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from omnibase_infra.handlers.handler_runner_hook_mount_drift import (
    HandlerRunnerHookMountDrift,
    ProtocolRunnerDocker,
    executed_paths,
    slack_notify,
)
from omnibase_infra.models.model_runner_hook_mount_drift_request import (
    ModelRunnerHookMountDriftRequest,
)
from omnibase_infra.nodes.node_deploy_runners.runtime_hook_mount_drift import main

pytestmark = pytest.mark.unit

HOOK_DIR = "/opt/onex-runner-hooks"
SCRIPTS = ("runner-job-started.sh", "entrypoint.sh", "healthcheck.sh")
MATCH = r"^omninode-(runner-|deploy-runner$|verify-runner-)"

# Recorded hashes (full sha256 as read on the host).
HOST = {
    "runner-job-started.sh": "3f6f39f7f5cc2bc372439c72b99039dd9dc07034b8e4f544d980e6aacfcfe41b",
    "entrypoint.sh": "5e75d99f1b48a3db8fce1ae4e458e7bfb48d06fdafdce14dccd5099f5632c903",
    "healthcheck.sh": "e204f17d25b8d422d64b5cfd6ab519dc001a59874fd03061ba7005f5ba1ca252",
}
BAKED_JOB_STARTED = "85c0207464ac00b002d75a7ec4b2774336950aacf6d392ecba67d3764b189b85"
RUNNER_18_COPIES = {
    "/usr/local/bin/runner-job-started.sh": "f4f1dae0054cf6bb212249b3865be0fc8aac051a6774cc2ab734af94e53a3e17",
    "/usr/local/bin/entrypoint.sh": "64937cb98cf38a2cb35f124ef58b8eeae2b9db123c74e8dedc73ae0bd3ab65e3",
    "/usr/local/bin/healthcheck.sh": "9cd1f2f29f42975a763f8b9cd337bd9a9bf5997f475aadfb843714e2b787d54b",
}

MOUNTED_INSPECT: dict[str, object] = {
    "Config": {
        "Env": [
            "PATH=/usr/bin",
            f"ACTIONS_RUNNER_HOOK_JOB_STARTED={HOOK_DIR}/runner-job-started.sh",
        ],
        "Entrypoint": [f"{HOOK_DIR}/entrypoint.sh"],
        "Cmd": None,
        "Healthcheck": {
            "Test": [
                "CMD-SHELL",
                f"{HOOK_DIR}/healthcheck.sh && {HOOK_DIR}/model-review-healthcheck.sh",
            ]
        },
    }
}
# The entrypoint is a bash -c wrapper that execs the mounted one (verify runners).
WRAPPED_INSPECT: dict[str, object] = {
    "Config": {
        "Env": [f"ACTIONS_RUNNER_HOOK_JOB_STARTED={HOOK_DIR}/runner-job-started.sh"],
        "Entrypoint": ["/bin/bash", "-c", f"set -eu\nexec {HOOK_DIR}/entrypoint.sh\n"],
        "Healthcheck": {"Test": ["CMD-SHELL", f"{HOOK_DIR}/healthcheck.sh"]},
    }
}
RUNNER_18_INSPECT: dict[str, object] = {
    "Config": {
        "Env": ["ACTIONS_RUNNER_HOOK_JOB_STARTED=/usr/local/bin/runner-job-started.sh"],
        "Entrypoint": ["/usr/local/bin/entrypoint.sh"],
        "Cmd": None,
        "Healthcheck": {"Test": ["CMD-SHELL", "/usr/local/bin/healthcheck.sh"]},
    }
}


class FakeContainer:
    """The container as the handler sees it: it resolves the docker port."""

    def __init__(self, docker: object) -> None:
        self._docker = docker

    def get_service(self, interface: type) -> object:
        assert interface is ProtocolRunnerDocker
        return self._docker


class RecordedDocker:
    """Replays the recorded host state."""

    def __init__(
        self,
        containers: dict[str, dict[str, object]],
        files: dict[str, dict[str, str]],
    ) -> None:
        self.containers = containers
        self.files = files

    def container_names(self) -> list[str]:
        return sorted(self.containers)

    def inspect(self, name: str) -> dict[str, object]:
        return self.containers[name]

    def sha256(self, name: str, path: str) -> str | None:
        return self.files[name].get(path)


def recorded_host() -> RecordedDocker:
    names = [f"omninode-runner-{n}" for n in range(1, 11)] + ["omninode-deploy-runner"]
    containers: dict[str, dict[str, object]] = dict.fromkeys(names, MOUNTED_INSPECT)
    containers["omninode-verify-runner-1"] = WRAPPED_INSPECT
    containers["omninode-runner-18"] = RUNNER_18_INSPECT
    mounted = {f"{HOOK_DIR}/{s}": digest for s, digest in HOST.items()}
    mounted["/usr/local/bin/runner-job-started.sh"] = BAKED_JOB_STARTED
    files = {n: dict(mounted) for n in containers}
    files["omninode-runner-18"] = dict(RUNNER_18_COPIES)
    return RecordedDocker(containers, files)


def without_runner_18(docker: RecordedDocker) -> RecordedDocker:
    del docker.containers["omninode-runner-18"]
    del docker.files["omninode-runner-18"]
    return docker


def serve(docker: RecordedDocker, host_dir: Path) -> ModelRunnerHookMountDriftRequest:
    """Write real host files and rewrite the recorded host hashes to theirs.

    The recording carries hashes, not bytes, so the temp host copies get stand-in
    bytes and every recorded hash equal to a host hash is swapped for the stand-in's.
    """
    stand_in = {s: hashlib.sha256(s.encode()).hexdigest() for s in SCRIPTS}
    for script in SCRIPTS:
        (host_dir / script).write_bytes(script.encode())
    swap = {recorded: stand_in[s] for s, recorded in HOST.items()}
    for files in docker.files.values():
        for path, digest in list(files.items()):
            files[path] = swap.get(digest, digest)
    return ModelRunnerHookMountDriftRequest(
        host_hook_dir=str(host_dir), container_match=MATCH
    )


def legacy_findings(docker: RecordedDocker) -> list[str]:
    """The retired script's rule: hash /usr/local/bin/<script> in every container."""
    out = []
    for name in docker.container_names():
        if not name.startswith(("omninode-runner-", "omninode-deploy-runner")):
            continue
        for script in SCRIPTS:
            got = docker.sha256(name, f"/usr/local/bin/{script}")
            if got is not None and got != HOST[script]:
                out.append(f"{name} {script}")
    return out


def test_control_the_retired_rule_flags_the_eleven_current_containers() -> None:
    """Failing control: the old path rule reads the baked copy and flags current runners."""
    flagged = [f for f in legacy_findings(recorded_host()) if "runner-18" not in f]
    assert len(flagged) == 11
    assert all(f.endswith("runner-job-started.sh") for f in flagged)


def test_executed_paths_read_what_each_container_runs() -> None:
    assert executed_paths(MOUNTED_INSPECT, SCRIPTS) == {
        "runner-job-started.sh": f"{HOOK_DIR}/runner-job-started.sh",
        "entrypoint.sh": f"{HOOK_DIR}/entrypoint.sh",
        "healthcheck.sh": f"{HOOK_DIR}/healthcheck.sh",
    }
    wrapped = executed_paths(WRAPPED_INSPECT, SCRIPTS)
    assert wrapped["entrypoint.sh"] == f"{HOOK_DIR}/entrypoint.sh"
    assert executed_paths(RUNNER_18_INSPECT, SCRIPTS) == {
        s: f"/usr/local/bin/{s}" for s in SCRIPTS
    }


def test_the_model_review_healthcheck_is_not_read_as_the_healthcheck() -> None:
    inspected: dict[str, object] = {
        "Config": {
            "Healthcheck": {
                "Test": ["CMD-SHELL", f"{HOOK_DIR}/model-review-healthcheck.sh"]
            }
        }
    }
    assert executed_paths(inspected, ("healthcheck.sh",)) == {"healthcheck.sh": None}


def test_recorded_state_reports_only_the_real_runner_18_drift(tmp_path: Path) -> None:
    docker = recorded_host()
    result = HandlerRunnerHookMountDrift(FakeContainer(docker)).handle(
        serve(docker, tmp_path)
    )
    assert result.outcome == "drift"
    assert {(f.container, f.script) for f in result.findings} == {
        ("omninode-runner-18", s) for s in SCRIPTS
    }
    assert result.checked == 13 * 3
    assert all(f.reason == "differs" for f in result.findings)


def test_golden_a_fleet_whose_executed_hooks_equal_the_host_copies_is_ok(
    tmp_path: Path,
) -> None:
    docker = without_runner_18(recorded_host())
    result = HandlerRunnerHookMountDrift(FakeContainer(docker)).handle(
        serve(docker, tmp_path)
    )
    assert (result.outcome, result.findings, result.checked) == ("ok", (), 12 * 3)


def test_injected_break_a_changed_host_hook_flags_every_container_on_the_old_copy(
    tmp_path: Path,
) -> None:
    """The check is not a rubber stamp: a host edit nobody rolled out is drift."""
    docker = recorded_host()
    request = serve(docker, tmp_path)
    (tmp_path / "runner-job-started.sh").write_bytes(b"edited on the host only")
    result = HandlerRunnerHookMountDrift(FakeContainer(docker)).handle(request)
    flagged = {
        f.container for f in result.findings if f.script == "runner-job-started.sh"
    }
    assert len(flagged) == 13


def test_error_missing_host_script_is_an_error_never_ok(tmp_path: Path) -> None:
    request = ModelRunnerHookMountDriftRequest(
        host_hook_dir=str(tmp_path), container_match=MATCH
    )
    result = HandlerRunnerHookMountDrift(FakeContainer(recorded_host())).handle(request)
    assert result.outcome == "error"
    assert "host script missing" in (result.error_message or "")


def test_error_no_matching_container_is_an_error_never_ok(tmp_path: Path) -> None:
    docker = recorded_host()
    serve(docker, tmp_path)
    request = ModelRunnerHookMountDriftRequest(
        host_hook_dir=str(tmp_path), container_match="^nothing$"
    )
    assert (
        HandlerRunnerHookMountDrift(FakeContainer(docker)).handle(request).outcome
        == "error"
    )


def test_error_unreadable_docker_is_an_error_never_ok(tmp_path: Path) -> None:
    class Broken(RecordedDocker):
        def container_names(self) -> list[str]:
            raise OSError("docker ps exited 1")

    request = serve(recorded_host(), tmp_path)
    result = HandlerRunnerHookMountDrift(FakeContainer(Broken({}, {}))).handle(request)
    assert result.outcome == "error"
    assert "docker ps" in (result.error_message or "")


def test_an_unreadable_executed_file_is_a_finding(tmp_path: Path) -> None:
    docker = recorded_host()
    del docker.files["omninode-runner-3"][f"{HOOK_DIR}/healthcheck.sh"]
    result = HandlerRunnerHookMountDrift(FakeContainer(docker)).handle(
        serve(docker, tmp_path)
    )
    unreadable = [f for f in result.findings if f.container == "omninode-runner-3"]
    assert [(f.script, f.reason) for f in unreadable] == [
        ("healthcheck.sh", "unreadable")
    ]


def test_a_container_declaring_no_hook_is_not_checked_rather_than_passed(
    tmp_path: Path,
) -> None:
    docker = recorded_host()
    docker.containers["omninode-runner-2"] = {
        "Config": {"Env": [], "Entrypoint": ["/bin/sh"]}
    }
    result = HandlerRunnerHookMountDrift(FakeContainer(docker)).handle(
        serve(docker, tmp_path)
    )
    assert "omninode-runner-2 runner-job-started.sh" in result.not_checked


def _cli(host_dir: Path, state: Path, docker: RecordedDocker, sent: list[str]) -> int:
    return main(
        [
            "--hook-dir",
            str(host_dir),
            "--container-match",
            MATCH,
            "--state-file",
            str(state),
            "--docker-timeout",
            "5",
            "--alert-env-file",
            str(host_dir / "absent.env"),
            "--notify-timeout",
            "5",
        ],
        docker=docker,
        notify=sent.append,
    )


def test_cli_alerts_on_a_change_in_the_finding_count_only(tmp_path: Path) -> None:
    docker = recorded_host()
    serve(docker, tmp_path)
    state, sent = tmp_path / "state", []
    assert _cli(tmp_path, state, docker, sent) == 1
    assert len(sent) == 1
    assert "[HOOK MOUNT DRIFT]" in sent[0]
    assert "omninode-runner-18" in sent[0]
    assert _cli(tmp_path, state, docker, sent) == 1
    assert len(sent) == 1, "the same count again must not alert again"


def test_cli_clean_fleet_exits_zero_and_sends_nothing(tmp_path: Path) -> None:
    docker = without_runner_18(recorded_host())
    serve(docker, tmp_path)
    sent: list[str] = []
    assert _cli(tmp_path, tmp_path / "state", docker, sent) == 0
    assert sent == []


def test_cli_unreadable_host_exits_two(tmp_path: Path) -> None:
    sent: list[str] = []
    assert _cli(tmp_path, tmp_path / "state", recorded_host(), sent) == 2
    assert sent == []


def test_slack_notify_is_silent_without_the_env_file(tmp_path: Path) -> None:
    slack_notify("text", tmp_path / "absent.env", 5)
