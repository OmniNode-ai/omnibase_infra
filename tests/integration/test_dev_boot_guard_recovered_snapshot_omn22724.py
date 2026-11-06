# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A failed job after a recovery leaves the recovered dev family running (OMN-22724).

Jobs 209869f7 and 54914064 on 2026-11-06: the first failed after replacing the
runtime and recovered from the retained images (new container ids, a compose
config hash the regular model never reproduces); the second failed before any
roll and, reading the snapshot's pre-recovery ids as a changed family,
force-recreated all three services.

This drives the real boot guard through both jobs with a real ``docker``
executable on ``PATH``: every ``docker`` call is a subprocess, the snapshot and
the recovery compose file are real files, and nothing is patched in the
executor. The executable is a small stand-in that keeps the family's state in a
JSON file, so the test needs no Docker daemon.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path
from typing import Any

import pytest
from deploy_agent import executor as executor_mod
from deploy_agent.events import EnumRuntimeLane
from deploy_agent.executor import DeployExecutor

pytestmark = pytest.mark.integration

guard_mod = executor_mod.load_dev_boot_guard()

SERVICES = ("omninode-runtime", "runtime-effects", "runtime-worker")
GOOD_COMMIT = "a" * 40
BAD_COMMIT = "b" * 40

_FAKE_DOCKER = """\
#!{python}
import json, os, sys

state_path = os.environ["FAKE_DOCKER_STATE"]
state = json.load(open(state_path))
argv = sys.argv[1:]
with open(state_path + ".log", "a") as log:
    log.write(json.dumps(argv) + "\\n")
if argv[:1] == ["ps"]:
    print("\\n".join(c["Id"] for c in state["containers"]))
elif argv[:1] == ["inspect"]:
    print(json.dumps(state["containers"]))
elif argv[:1] == ["logs"]:
    print(state["logs"])
elif "config" in argv:
    if "--hash" in argv:
        print("\\n".join(s + " hash-" + s for s in state["services"]))
    else:
        print(json.dumps(state["compose"]))
elif "up" in argv:
    # A recovery from the retained images: new ids, and a config hash that the
    # regular compose model never reproduces.
    for container in state["containers"]:
        container["Id"] = "recovered-" + container["Id"]
        container["Config"]["Labels"]["com.docker.compose.config-hash"] = "recovery"
        container["State"]["Health"]["Status"] = "healthy"
    state["logs"] = ""
json.dump(state, open(state_path, "w"))
"""


def _containers(project: str, *, healthy: bool) -> list[dict[str, Any]]:
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


def _compose(project: str) -> dict[str, Any]:
    return {
        "name": project,
        "services": {
            service: {
                "image": "runtime:latest",
                "environment": {"BIFROST_BASE_CONTRACT_PATH": "/old/base.yaml"},
                "networks": {"default": None},
            }
            for service in SERVICES
        },
        "networks": {"default": {"name": f"{project}_default"}},
    }


def test_a_second_failed_job_leaves_the_recovered_family_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = executor_mod.lane_config_for(EnumRuntimeLane.DEV).compose_project
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(_FAKE_DOCKER.format(python=sys.executable), encoding="utf-8")
    docker.chmod(docker.stat().st_mode | stat.S_IXUSR)
    state_path = tmp_path / "docker-state.json"

    def write_state(containers: list[dict[str, Any]]) -> None:
        state_path.write_text(
            json.dumps(
                {
                    "containers": containers,
                    "logs": "",
                    "services": list(SERVICES),
                    "compose": _compose(project),
                }
            ),
            encoding="utf-8",
        )

    def read_state() -> dict[str, Any]:
        return json.loads(state_path.read_text(encoding="utf-8"))

    def up_calls() -> int:
        log = Path(str(state_path) + ".log")
        if not log.exists():
            return 0
        return sum(
            "up" in json.loads(line) for line in log.read_text().splitlines() if line
        )

    write_state(_containers(project, healthy=True))
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_DOCKER_STATE", str(state_path))
    monkeypatch.setattr(guard_mod, "deliver_reports", lambda *args, **kwargs: None)
    executor = DeployExecutor()
    monkeypatch.setattr(executor, "verify", lambda *args, **kwargs: [])
    root = tmp_path / "guard"

    first = executor_mod.DevBootGuard(executor, root)
    first.prepare()
    with pytest.raises(guard_mod.DevBootRecoveryError, match="verified=True"):
        with first.protect("209869f7", BAD_COMMIT, lambda *args, **kwargs: None):
            write_state(_containers(project, healthy=False))
            raise RuntimeError("runtime roll stopped at omninode-runtime")
    assert up_calls() == 1
    recovered = sorted(c["Id"] for c in read_state()["containers"])
    assert all(i.startswith("recovered-") for i in recovered)

    # The next job runs in a fresh agent process: it starts from the file on disk.
    second = executor_mod.DevBootGuard(executor, root)
    second.prepare()
    held = second.point.current
    assert held is not None and sorted(held.container_ids) == recovered
    with pytest.raises(guard_mod.DevBootRecoveryError, match="verified=True"):
        with second.protect("54914064", BAD_COMMIT, lambda *args, **kwargs: None):
            raise RuntimeError(
                "candidate boot probe timed out; running containers kept"
            )

    assert up_calls() == 1, "the healthy recovered family was recreated again"
    incident = json.loads((second.root / "incident-54914064.json").read_text())
    assert incident["recovery_skipped_healthy"] is True
    assert sorted(c["Id"] for c in read_state()["containers"]) == recovered
