# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Exercise runner bind preparation across the deploy handler's SSH boundary."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from omnibase_infra.handlers.handler_deploy_runners import HandlerDeployRunners
from omnibase_infra.models.model_deploy_runners_request import ModelDeployRunnersRequest
from tests.unit.scripts.test_deploy_runners_tmp_dirs_omn19998 import HOST_TOOL

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("persist_changes", [True, False], ids=["prepare", "readback"])
async def test_secondary_deploy_prepares_and_verifies_pause_bind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, persist_changes: bool
) -> None:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name in ("ssh", "docker", "sudo", "stat"):
        tool = bindir / name
        tool.write_text(HOST_TOOL, encoding="utf-8")
        tool.chmod(0o755)

    compose_file = tmp_path / "compose.yml"
    compose_file.write_text("services: {}\n", encoding="utf-8")
    tmpdir = "/srv/runners/tmp"
    pause = "/srv/runners/disk-admission-pause"
    keys = "/srv/runners/keys"
    runner_home = "/home/runner"
    state = tmp_path / "host.json"
    calls = tmp_path / "calls.txt"
    service = {
        "environment": {"TMPDIR": f"{runner_home}/work-tmp"},
        "volumes": [
            {"type": "bind", "source": tmpdir, "target": f"{runner_home}/work-tmp"},
            {
                "type": "bind",
                "source": pause,
                "target": f"{runner_home}/.onex-disk-admission-pause",
            },
            {"type": "bind", "source": keys, "target": f"{runner_home}/keys"},
        ],
    }
    state.write_text(
        json.dumps(
            {
                "compose": {"services": {"runner-1": service, "runner-2": service}},
                "dirs": {
                    tmpdir: {"uid": 1001, "mode": "1777"},
                    pause: {"uid": 0, "mode": "755"},
                    keys: {"uid": 0, "mode": "700"},
                },
                "ignore_install": not persist_changes,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("TMP_HOST_STATE", str(state))
    monkeypatch.setenv("TMP_HOST_CALLS", str(calls))

    result = await HandlerDeployRunners().handle(
        ModelDeployRunnersRequest(
            host="fixture-host",
            host_dir=str(tmp_path),
            compose_file=str(compose_file),
            prefix="runner",
            count=2,
            secondary=True,
        )
    )

    directories = json.loads(state.read_text(encoding="utf-8"))["dirs"]
    assert directories[keys] == {"uid": 0, "mode": "700"}
    assert directories[tmpdir] == {"uid": 1001, "mode": "1777"}
    operations = calls.read_text(encoding="utf-8").splitlines()
    assert operations.count(f"sudo -n install -d -o 1001 -m 755 -- {pause}") == 1
    assert not any(keys in line for line in operations)
    if persist_changes:
        assert result.success
        assert set(result.prepared) == {tmpdir, pause}
        assert directories[pause] == {"uid": 1001, "mode": "755"}
    else:
        assert not result.success
        assert result.prepared == ()
        assert result.error_message == (
            f"RUNNER-TMP-VERIFY host=fixture-host path={pause}: "
            "owner or mode readback failed."
        )
        assert directories[pause] == {"uid": 0, "mode": "755"}
