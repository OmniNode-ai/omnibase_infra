# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Installed unit files are compared to the tracked copies (OMN-20037).

omnibase_infra#4125 (OMN-19522) fixed the tracked deploy-agent-dev.service PATH,
but the copy installed on the lab host predated it, so every job failed for two
days. ``deploy_agent.unit_drift`` reads a declared manifest, compares tracked and
installed bytes, and reports OK / DRIFT / MISSING. It only reads, except
``sync_own_unit``, which the agent's self-update calls to re-install its own unit.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from aiohttp.test_utils import TestClient, TestServer
from deploy_agent import unit_drift
from deploy_agent.agent import _unit_drift_report
from deploy_agent.health import create_health_app
from deploy_agent.job_state import JobStore
from deploy_agent.unit_drift import (
    EnumUnitStatus,
    ModelUnitEntry,
    check_units,
    has_drift,
    load_manifest,
    own_unit_name_from_cgroup,
    sync_own_unit,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_HOST = "labhost"


def _entry(
    name: str = "x.service", *, installed: str = "~/units/x.service"
) -> ModelUnitEntry:
    return ModelUnitEntry(
        name=name, tracked=f"tracked/{name}", installed=installed, hosts=[_HOST]
    )


def _tree(tmp_path: Path, tracked: str, installed: str | None) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    (repo / "tracked").mkdir(parents=True)
    (repo / "tracked" / "x.service").write_text(tracked)
    home = tmp_path / "home"
    (home / "units").mkdir(parents=True)
    if installed is not None:
        (home / "units" / "x.service").write_text(installed)
    return repo, home


@pytest.mark.unit
def test_identical_copy_is_ok_and_a_differing_copy_is_drift(tmp_path: Path) -> None:
    repo, home = _tree(tmp_path, "PATH=a\n", "PATH=a\n")
    ok = check_units([_entry()], repo_root=repo, hostname=_HOST, home=home)
    assert [r.status for r in ok] == [EnumUnitStatus.OK]
    assert not has_drift(ok)

    (home / "units" / "x.service").write_text("PATH=stale\n")
    drift = check_units([_entry()], repo_root=repo, hostname=_HOST, home=home)
    assert [r.status for r in drift] == [EnumUnitStatus.DRIFT]
    assert has_drift(drift)


@pytest.mark.unit
def test_absent_installed_copy_on_a_declared_host_is_missing(tmp_path: Path) -> None:
    repo, home = _tree(tmp_path, "a\n", None)
    results = check_units([_entry()], repo_root=repo, hostname=_HOST, home=home)
    assert [r.status for r in results] == [EnumUnitStatus.MISSING]
    assert has_drift(results)


@pytest.mark.unit
def test_an_entry_for_another_host_is_not_applicable_and_not_drift(
    tmp_path: Path,
) -> None:
    repo, home = _tree(tmp_path, "a\n", "different\n")
    results = check_units([_entry()], repo_root=repo, hostname="other", home=home)
    assert [r.status for r in results] == [EnumUnitStatus.NOT_APPLICABLE]
    assert not has_drift(results)


@pytest.mark.unit
def test_installed_override_points_an_entry_at_a_backup_file(tmp_path: Path) -> None:
    repo, home = _tree(tmp_path, "new\n", "new\n")
    backup = tmp_path / "x.service.bak-pre-fix"
    backup.write_text("old\n")
    results = check_units(
        [_entry()],
        repo_root=repo,
        hostname=_HOST,
        home=home,
        overrides={"x.service": backup},
    )
    assert results[0].status is EnumUnitStatus.DRIFT
    assert results[0].installed_path == backup


@pytest.mark.unit
def test_the_shipped_manifest_loads_and_every_tracked_file_exists() -> None:
    entries = load_manifest(_REPO_ROOT / "deploy" / "unit-drift-manifest.yaml")
    assert entries, "manifest must declare units"
    names = [e.name for e in entries]
    assert "deploy-agent-dev.service" in names
    for entry in entries:
        assert (_REPO_ROOT / entry.tracked).is_file(), entry.tracked


@pytest.mark.unit
def test_cli_exits_nonzero_and_prints_a_drift_line(tmp_path: Path) -> None:
    repo, home = _tree(tmp_path, "new\n", "old\n")
    manifest = repo / "manifest.yaml"
    manifest.write_text(
        "units:\n"
        "  - name: x.service\n"
        "    tracked: tracked/x.service\n"
        "    installed: ~/units/x.service\n"
        f"    hosts: [{_HOST}]\n"
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "deploy_agent.unit_drift",
            "--manifest",
            str(manifest),
            "--repo-root",
            str(repo),
            "--hostname",
            _HOST,
            "--home",
            str(home),
            "--json",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
        check=False,
    )
    assert proc.returncode == 1, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["drift"] is True
    assert payload["units"][0]["status"] == "DRIFT"


@pytest.mark.unit
def test_own_unit_name_is_read_from_the_cgroup_path() -> None:
    cgroup = (
        "0::/user.slice/user-1000.slice/user@1000.service/app.slice/"
        "deploy-agent-dev.service\n"
    )
    assert own_unit_name_from_cgroup(cgroup) == "deploy-agent-dev.service"
    assert own_unit_name_from_cgroup("0::/init.scope\n") is None


@pytest.mark.unit
def test_self_update_sync_installs_the_tracked_unit_keeps_a_backup_and_reloads(
    tmp_path: Path,
) -> None:
    repo, home = _tree(tmp_path, "PATH=new\n", "PATH=stale\n")
    calls: list[list[str]] = []

    def runner(cmd: list[str]) -> int:
        calls.append(cmd)
        return 0

    changed = sync_own_unit(
        [_entry()],
        unit_name="x.service",
        repo_root=repo,
        hostname=_HOST,
        home=home,
        runner=runner,
        stamp="20260929T000000Z",
    )
    assert changed is True
    installed = home / "units" / "x.service"
    assert installed.read_text() == "PATH=new\n"
    backup = home / "units" / "x.service.bak-unit-drift-20260929T000000Z"
    assert backup.read_text() == "PATH=stale\n"
    assert ["systemctl", "--user", "daemon-reload"] in calls


@pytest.mark.unit
def test_self_update_sync_does_nothing_when_the_copies_match_or_unit_unknown(
    tmp_path: Path,
) -> None:
    repo, home = _tree(tmp_path, "same\n", "same\n")
    calls: list[list[str]] = []

    def runner(cmd: list[str]) -> int:
        calls.append(cmd)
        return 0

    kwargs = {
        "repo_root": repo,
        "hostname": _HOST,
        "home": home,
        "runner": runner,
        "stamp": "s",
    }
    assert sync_own_unit([_entry()], unit_name="x.service", **kwargs) is False
    assert sync_own_unit([_entry()], unit_name=None, **kwargs) is False
    assert sync_own_unit([_entry()], unit_name="other.service", **kwargs) is False
    assert calls == []


@pytest.mark.unit
def test_self_update_sync_does_not_restart_when_the_reload_fails(
    tmp_path: Path,
) -> None:
    repo, home = _tree(tmp_path, "new\n", "old\n")
    with pytest.raises(RuntimeError, match="daemon-reload"):
        sync_own_unit(
            [_entry()],
            unit_name="x.service",
            repo_root=repo,
            hostname=_HOST,
            home=home,
            runner=lambda cmd: 1,
            stamp="s",
        )


@pytest.mark.unit
async def test_route_serves_the_drift_report(tmp_path: Path) -> None:
    store = JobStore(state_dir=tmp_path / "jobs")
    report = {"drift": True, "units": [{"name": "x.service", "status": "DRIFT"}]}
    app = create_health_app(store, lambda: "idle", get_unit_drift=lambda: report)
    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/unit-drift")
        assert resp.status == 200
        assert await resp.json() == report


@pytest.mark.unit
async def test_route_without_a_provider_says_indeterminate_not_clean(
    tmp_path: Path,
) -> None:
    store = JobStore(state_dir=tmp_path / "jobs")
    app = create_health_app(store, lambda: "idle")
    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/unit-drift")
        body = await resp.json()
        assert body["drift"] is None
        assert unit_drift.INDETERMINATE_REASON in body["reason"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "script",
    [
        "disk-gc.sh",
        "docker-volume-gc.sh",
        "disk-watermark-check.sh",
        "buildx-orphan-sweep.sh",
        "runner-disk-admission-restore.sh",
        "onex-build-loop-trigger.sh",
        "monitor_logs.py",
    ],
)
def test_scheduled_scripts_detect_stale_installed_bytes(
    tmp_path: Path, script: str
) -> None:
    entries = load_manifest(_REPO_ROOT / "deploy" / "unit-drift-manifest.yaml")
    entry = next((entry for entry in entries if entry.name == script), None)
    assert entry is not None, f"scheduled script is unobserved: {script}"
    assert entry.tracked == f"scripts/{script}"
    assert "omninode-pc" in entry.hosts
    expected = (_REPO_ROOT / entry.tracked).read_bytes()
    installed = tmp_path / script
    installed.write_bytes(expected)
    kwargs = {
        "repo_root": _REPO_ROOT,
        "hostname": "omninode-pc",
        "home": tmp_path,
        "overrides": {script: installed},
    }
    matching = check_units([entry], **kwargs)
    assert matching[0].status is EnumUnitStatus.OK
    assert not has_drift(matching)
    installed.write_bytes(expected + b"\n# stale installed revision\n")
    stale = check_units([entry], **kwargs)
    assert stale[0].status is EnumUnitStatus.DRIFT
    assert has_drift(stale)


@pytest.mark.unit
def test_registry_bound_script_uses_declared_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, home = _tree(tmp_path, "new\n", None)
    registry = tmp_path / "registry"
    registry.mkdir()
    installed = registry / "installed.sh"
    installed.write_text("new\n")
    monkeypatch.setenv("UNIT_DRIFT_REGISTRY_ROOT", str(registry))
    monkeypatch.setenv("OMNI_HOME", str(tmp_path / "deploy-source"))
    entry = _entry(installed="{registry_root}/installed.sh")
    results = check_units([entry], repo_root=repo, hostname=_HOST, home=home)
    assert results[0].installed_path == installed
    assert results[0].status is EnumUnitStatus.OK
    installed.write_text("old\n")
    assert (
        check_units([entry], repo_root=repo, hostname=_HOST, home=home)[0].status
        is EnumUnitStatus.DRIFT
    )


@pytest.mark.unit
def test_registry_bound_script_refuses_an_undeclared_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, home = _tree(tmp_path, "new\n", None)
    monkeypatch.delenv("UNIT_DRIFT_REGISTRY_ROOT", raising=False)
    with pytest.raises(ValueError, match="UNIT_DRIFT_REGISTRY_ROOT"):
        check_units(
            [_entry(installed="{registry_root}/installed.sh")],
            repo_root=repo,
            hostname=_HOST,
            home=home,
        )


@pytest.mark.unit
def test_registry_binding_is_not_required_for_another_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, home = _tree(tmp_path, "new\n", None)
    monkeypatch.delenv("UNIT_DRIFT_REGISTRY_ROOT", raising=False)
    results = check_units(
        [_entry(installed="{registry_root}/installed.sh")],
        repo_root=repo,
        hostname="other",
        home=home,
    )
    assert results[0].status is EnumUnitStatus.NOT_APPLICABLE
    assert not has_drift(results)


@pytest.mark.unit
@pytest.mark.parametrize("bound", [True, False])
def test_health_provider_resolves_registry_scripts_and_reports_missing_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bound: bool
) -> None:
    repo, home = _tree(tmp_path, "new\n", None)
    agent_dir = repo / "scripts" / "deploy-agent"
    agent_dir.mkdir(parents=True)
    (repo / "deploy").mkdir()
    (repo / "deploy" / "unit-drift-manifest.yaml").write_text(
        "units:\n  - name: installed.sh\n    tracked: tracked/x.service\n"
        "    installed: '{registry_root}/installed.sh'\n    hosts: [labhost]\n"
    )
    registry = tmp_path / "registry"
    registry.mkdir()
    (registry / "installed.sh").write_text("old\n")
    monkeypatch.setenv("DEPLOY_AGENT_DIR", str(agent_dir))
    monkeypatch.setenv("OMNI_HOME", str(tmp_path / "deploy-source"))
    if bound:
        monkeypatch.setenv("UNIT_DRIFT_REGISTRY_ROOT", str(registry))
    else:
        monkeypatch.delenv("UNIT_DRIFT_REGISTRY_ROOT", raising=False)
    with (
        patch("deploy_agent.agent.socket.gethostname", return_value=_HOST),
        patch.object(Path, "home", return_value=home),
    ):
        payload = _unit_drift_report()
    if bound:
        assert payload["drift"] is True
        assert payload["units"][0]["installed_path"] == str(registry / "installed.sh")
        assert payload["units"][0]["status"] == "DRIFT"
    else:
        assert payload["drift"] is None
        assert "UNIT_DRIFT_REGISTRY_ROOT" in payload["reason"]


@pytest.mark.unit
def test_dev_agent_declares_and_protects_registry_binding() -> None:
    unit = (
        _REPO_ROOT / "scripts/deploy-agent/deploy/deploy-agent-dev.service"
    ).read_text()
    declarations = [
        line for line in unit.splitlines() if line.startswith("Environment=")
    ]
    root = next(
        line
        for line in declarations
        if line.startswith("Environment=UNIT_DRIFT_REGISTRY_ROOT=")
    )
    assert Path(root.split("=", 2)[2]).is_absolute()
    protected = next(
        line for line in declarations if "DEPLOY_AGENT_ENV_PROTECTED=" in line
    )
    assert "UNIT_DRIFT_REGISTRY_ROOT" in protected.split("=", 2)[2].rstrip('"').split()
    source = next(
        line for line in declarations if line.startswith("Environment=OMNI_HOME=")
    )
    assert root.split("=", 2)[2] != source.split("=", 2)[2]
