# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Prove stale units are observed and repaired at job boundaries (OMN-20037).

The tracked PATH fix left the installed unit stale for two days. Exercise the
real file replacement through self-update, including command preservation when
reload fails, and prove the read-only HTTP surface without opening sockets.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import make_mocked_request
from deploy_agent.agent import _unit_drift_report
from deploy_agent.events import EnumSelfUpdateBoundary
from deploy_agent.executor import DeployExecutor
from deploy_agent.health import create_health_app
from deploy_agent.job_state import JobStore
from deploy_agent.unit_drift import (
    EnumUnitStatus,
    ModelUnitEntry,
    check_units,
    load_manifest,
    own_unit_name_from_cgroup,
    sync_own_unit,
)
from pydantic import ValidationError

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "text",
    [
        "units: []\nunknown: true\n",
        "units: [{name: x.service}]\n",
        "units: [{name: x, tracked: x, installed: x, hosts: [], unknown: true}]\n",
    ],
)
def test_manifest_rejects_unknown_or_missing_fields(tmp_path: Path, text: str) -> None:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(text)
    with pytest.raises(ValidationError):
        load_manifest(manifest)


def test_rendered_bytes_use_given_home_and_cannot_be_installed(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    (tmp_path / "render.sh").write_text('printf "%s\\n%s\\n" "$HOME" "$1"')
    installed = home / "x.plist"
    installed.write_text(f"{home}\n{home}/lane\n")
    entry = ModelUnitEntry(
        name="x.plist",
        tracked="template",
        installed="~/x.plist",
        hosts=["TestHost.local"],
        render_script="render.sh",
        render_args=["{home}/lane"],
    )
    results = check_units([entry], repo_root=tmp_path, hostname="TESTHOST", home=home)
    assert results[0].status is EnumUnitStatus.OK
    installed.write_text("stale")
    with pytest.raises(ValueError, match="render"):
        sync_own_unit(
            [entry],
            unit_name="x.plist",
            repo_root=tmp_path,
            hostname="testhost",
            home=home,
            runner=lambda cmd: 0,
            stamp="s",
        )
    assert installed.read_text() == "stale"
    assert list(home.iterdir()) == [installed]


def test_cgroup_uses_innermost_service_only_from_unified_line() -> None:
    assert (
        own_unit_name_from_cgroup(
            "1:name=systemd:/wrong.service\n"
            "0::/user@1000.service/app.slice/agent.service/worker\n"
        )
        == "agent.service"
    )
    assert own_unit_name_from_cgroup("1:name=systemd:/wrong.service\n") is None


def _unit_tree(tmp_path: Path) -> tuple[Path, Path, Path]:
    agent_dir = tmp_path / "repo" / "scripts" / "deploy-agent"
    agent_dir.mkdir(parents=True)
    repo = agent_dir.parents[1]
    (repo / "deploy").mkdir()
    (repo / "tracked.service").write_text("PATH=new\n")
    home = tmp_path / "home"
    home.mkdir()
    installed = home / "agent.service"
    installed.write_text("PATH=stale\n")
    (repo / "deploy" / "unit-drift-manifest.yaml").write_text(
        "units:\n  - name: agent.service\n    tracked: tracked.service\n"
        "    installed: ~/agent.service\n    hosts: [labhost]\n"
    )
    return agent_dir, home, installed


@pytest.mark.parametrize("pull", [False, True])
@pytest.mark.parametrize("reload_exit", [0, 1])
def test_self_update_syncs_before_code_check_and_preserves_command_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pull: bool,
    reload_exit: int,
    caplog: pytest.LogCaptureFixture,
) -> None:
    agent_dir, home, installed = _unit_tree(tmp_path)
    monkeypatch.setenv("DEPLOY_AGENT_DIR", str(agent_dir))
    monkeypatch.setenv("DEPLOY_AGENT_TRACKING_REF", "dev")
    monkeypatch.delenv("DEPLOY_AGENT_NO_SELF_UPDATE", raising=False)
    actions: list[str] = []
    read_text = Path.read_text

    def read_cgroup(path: Path, *args: object, **kwargs: object) -> str:
        if path == Path("/proc/self/cgroup"):
            return "0::/user@1000.service/app.slice/agent.service\n"
        return read_text(path)  # Manifest reads need no special arguments.

    def run(cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
        stdout = ""
        returncode = 0
        if "rev-parse" in cmd:
            stdout = "old" if pull and cmd[-1] == "HEAD" else "current"
        if "pull" in cmd:
            actions.append("pull")
        if "daemon-reload" in cmd:
            assert installed.read_bytes() == b"PATH=new\n"
            actions.append("reload")
            returncode = reload_exit
        return subprocess.CompletedProcess(cmd, returncode, stdout, "")

    with (
        patch("deploy_agent.executor.sys.platform", "linux"),
        patch("deploy_agent.executor.loaded_code_sha", return_value="current"),
        patch("deploy_agent.executor.socket.gethostname", return_value="labhost"),
        patch.object(Path, "home", return_value=home),
        patch.object(Path, "read_text", read_cgroup),
        patch("deploy_agent.executor._run", side_effect=run),
        patch("deploy_agent.executor.subprocess.run") as restart,
        patch("deploy_agent.executor.os.execv") as reexec,
    ):
        DeployExecutor().self_update(
            boundary=EnumSelfUpdateBoundary.PRE_ACCEPT,
            on_before_reexec=lambda: actions.append("rewind"),
        )
    assert installed.read_bytes() == b"PATH=new\n"
    backups = list(home.glob("agent.service.bak-unit-drift-*"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == b"PATH=stale\n"
    expected = (["pull"] if pull else []) + ["reload"]
    if reload_exit == 0:
        assert actions == [*expected, "rewind"]
        restart.assert_called_once_with(
            ["systemctl", "--user", "restart", "agent.service"],
            check=True,
            timeout=60,
        )
    else:
        assert actions == expected
        restart.assert_not_called()
        assert "daemon-reload" in caplog.text
    reexec.assert_not_called()


def test_health_provider_uses_agent_repo_and_reports_read_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_dir, home, installed = _unit_tree(tmp_path)
    monkeypatch.setenv("DEPLOY_AGENT_DIR", str(agent_dir))
    with (
        patch("deploy_agent.agent.socket.gethostname", return_value="labhost"),
        patch.object(Path, "home", return_value=home),
    ):
        assert _unit_drift_report()["drift"] is True
        assert installed.read_bytes() == b"PATH=stale\n"
        (agent_dir.parents[1] / "deploy" / "unit-drift-manifest.yaml").unlink()
        result = _unit_drift_report()
    assert result["drift"] is None
    assert result["reason"]


@pytest.mark.parametrize("with_provider", [True, False])
async def test_unit_drift_route_without_socket(
    tmp_path: Path, with_provider: bool
) -> None:
    payload: dict[str, object] = {"drift": True, "units": []}
    app = create_health_app(
        JobStore(state_dir=tmp_path / "jobs"),
        lambda: "idle",
        get_unit_drift=(lambda: payload) if with_provider else None,
    )
    request = make_mocked_request("GET", "/unit-drift", app=app)
    route = await app.router.resolve(request)
    response = await route.handler(request)
    assert isinstance(response, web.Response)
    assert response.status == 200
    assert response.text is not None
    body = json.loads(response.text)
    if with_provider:
        assert body == payload
    else:
        assert body["drift"] is None
        assert body["reason"]
