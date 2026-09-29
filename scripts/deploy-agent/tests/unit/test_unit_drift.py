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

import pytest
from aiohttp.test_utils import TestClient, TestServer
from deploy_agent import unit_drift
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
