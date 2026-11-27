# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A verified dev roll stays up when the clone moved after ``up`` (OMN-25942).

Job 20cf29c6: the roll was replaced, verified and Docker healthy, then the capture
after it rendered other compose files than ``up`` read and the guard restored the
previous image. This drives the real boot guard with a real ``docker`` executable on
``PATH`` (a stand-in that keeps the family's state in a JSON file) whose
``compose config --hash`` prints other hashes after the roll, and asserts that no
``up`` follows, no incident is written and the drift is recorded.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pytest
from deploy_agent import executor as executor_mod
from deploy_agent.events import EnumRuntimeLane
from deploy_agent.executor import DeployExecutor

from tests.integration import test_dev_boot_guard_recovered_snapshot_omn22724 as base

pytestmark = pytest.mark.integration

_FAKE_DOCKER = base._FAKE_DOCKER.replace(
    's + " hash-" + s', 's + " " + state.get("hash_prefix", "hash-") + s'
)


def test_a_verified_roll_is_kept_when_the_rendered_compose_hash_moved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = executor_mod.lane_config_for(EnumRuntimeLane.DEV).compose_project
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(_FAKE_DOCKER.format(python=sys.executable), encoding="utf-8")
    docker.chmod(docker.stat().st_mode | stat.S_IXUSR)
    state_path = tmp_path / "docker-state.json"

    def write_state(hash_prefix: str) -> None:
        state_path.write_text(
            json.dumps(
                {
                    "containers": base._containers(project, healthy=True),
                    "logs": "",
                    "services": list(base.SERVICES),
                    "compose": base._compose(project),
                    "hash_prefix": hash_prefix,
                }
            ),
            encoding="utf-8",
        )

    def up_calls() -> int:
        log = Path(str(state_path) + ".log")
        if not log.exists():
            return 0
        return sum(
            "up" in json.loads(line) for line in log.read_text().splitlines() if line
        )

    write_state("hash-")
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_DOCKER_STATE", str(state_path))
    monkeypatch.setattr(base.guard_mod, "deliver_reports", lambda *a, **k: None)
    executor = DeployExecutor()
    monkeypatch.setattr(executor, "verify", lambda *args, **kwargs: [])
    # The post-roll ledger proof has its own tests; this one drives compose drift.
    monkeypatch.setattr(base.guard_mod, "_prove_work_ledger_answers", lambda *_: None)
    root = tmp_path / "guard"

    guard = executor_mod.DevBootGuard(executor, root)
    guard.prepare()
    failed: list[object] = []
    with guard.protect("20cf29c6", base.BAD_COMMIT, lambda *a, **k: failed.append(a)):
        write_state("moved-")

    assert not failed
    assert up_calls() == 0, "a verified roll was restored to the previous image"
    assert not list(root.rglob("incident-*.json"))
    (drift_path,) = root.rglob("settings-drift-20cf29c6.json")
    drift = json.loads(drift_path.read_text())
    assert drift["service"] == "omninode-runtime"
    assert drift["container_config_hash"] == "hash-omninode-runtime"
    assert drift["rendered_config_hash"] == "moved-omninode-runtime"
    assert guard.point.current is not None
    assert guard.point.current.revision == base.GOOD_COMMIT
