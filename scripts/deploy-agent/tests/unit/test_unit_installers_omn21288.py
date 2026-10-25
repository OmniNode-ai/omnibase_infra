# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Repair lab-host unit drift with committed installers (OMN-21288).

The fixture carries the statuses of the lab-host drift read (lane unit-drift-201-1210):
the log monitor and git-mirror units were hand copies with no committed installer, and
monitor_logs.py was never deployed where the tracked unit runs it. Its hashes are
placeholders. Exercise installers against temporary files and a fake systemctl, so
verification checks deployed bytes and command ordering without touching the host.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml
from deploy_agent.unit_drift import (
    EnumUnitStatus,
    ModelUnitEntry,
    check_units,
    load_manifest,
    main,
    report,
)
from pydantic import ValidationError

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_MANIFEST = _REPO_ROOT / "deploy/unit-drift-manifest.yaml"
_FIXTURE = Path(__file__).parent / "fixtures/unit_drift_dev201_omn21288.json"
_LOG_INSTALLER = "deploy/disk-gc/install-host-maintenance.sh"
_MIRROR_INSTALLER = "deploy/maintenance/omninode-host-maintenance-sync.sh"
_MIRROR_NAMES = (
    "omninode-git-mirror-daemon.service",
    "omninode-git-mirror-refresh.service",
    "omninode-git-mirror-refresh.timer",
)


def _entries(*names: str) -> list[ModelUnitEntry]:
    return [entry for entry in load_manifest(_MANIFEST) if entry.name in names]


def _fake_systemctl(tmp_path: Path) -> tuple[dict[str, str], Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "systemctl.log"
    fake = bin_dir / "systemctl"
    fake.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$SYSTEMCTL_LOG"\n')
    fake.chmod(0o755)
    return {
        **scrub_git_location_env(),
        "HOME": str(tmp_path / "home"),
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "SYSTEMCTL_LOG": str(log),
    }, log


def _run(
    script: Path, env: dict[str, str], *args: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(script), *args],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _log_monitor_tree(tmp_path: Path, exec_start: str) -> Path:
    repo = tmp_path / "repo"
    for relative in (
        _LOG_INSTALLER,
        "deploy/systemd/onex-log-monitor.service",
        "scripts/monitor_logs.py",
        "scripts/monitor_alert_contract.yaml",
    ):
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_REPO_ROOT / relative, target)
    unit = repo / "deploy/systemd/onex-log-monitor.service"
    unit.write_text(
        "\n".join(
            f"ExecStart={exec_start}" if line.startswith("ExecStart=") else line
            for line in unit.read_text().splitlines()
        )
        + "\n"
    )
    return repo


@pytest.mark.parametrize(
    "name",
    [
        unit["name"]
        for unit in json.loads(_FIXTURE.read_text())["units"]
        if unit["status"] in ("DRIFT", "MISSING")
        and unit["name"] != "deploy-agent.service"
    ],
)
def test_dev201_drift_has_a_committed_installer(name: str) -> None:
    (entry,) = _entries(name)
    assert entry.installer, name
    installer = _REPO_ROOT / entry.installer
    assert installer.is_file()
    assert installer.read_text().startswith("#!/usr/bin/env bash\n")
    assert Path(entry.tracked).name in installer.read_text()


def test_deploy_agent_units_remain_without_installers() -> None:
    entries = [
        entry
        for entry in load_manifest(_MANIFEST)
        if entry.name.startswith("deploy-agent") and entry.name.endswith(".service")
    ]
    assert any(entry.name == "deploy-agent.service" for entry in entries)
    assert all(entry.installer is None for entry in entries)


@pytest.mark.parametrize(
    ("stem", "installer"),
    [
        ("onex-disk-gc", "install-disk-gc.sh"),
        ("onex-buildx-orphan-sweep", "install-buildx-orphan-sweep.sh"),
        ("onex-runner-disk-guard", "install-runner-disk-guard.sh"),
        ("onex-worktree-prune", "install-worktree-prune.sh"),
    ],
)
def test_disk_maintenance_units_declare_their_installers(
    stem: str, installer: str
) -> None:
    entries = _entries(f"{stem}.service", f"{stem}.timer")
    assert len(entries) == 2
    assert all(entry.installer == f"deploy/disk-gc/{installer}" for entry in entries)


def test_log_monitor_installer_deploys_unit_script_and_contract(tmp_path: Path) -> None:
    env, log = _fake_systemctl(tmp_path)
    script = tmp_path / "deployed/scripts/monitor_logs.py"
    # The last matching token is the deployment target, even with trailing flags.
    repo = _log_monitor_tree(
        tmp_path,
        f"/usr/bin/python3 /unused/monitor_logs.py {script} --verbose",
    )
    result = _run(repo / _LOG_INSTALLER, env, "--log-monitor")
    assert result.returncode == 0, result.stderr
    home = Path(env["HOME"])
    installed_unit = home / ".config/systemd/user/onex-log-monitor.service"
    assert (
        installed_unit.read_bytes()
        == (repo / "deploy/systemd/onex-log-monitor.service").read_bytes()
    )
    for name in ("monitor_logs.py", "monitor_alert_contract.yaml"):
        assert (script.parent / name).read_bytes() == (
            repo / "scripts" / name
        ).read_bytes()
    entries = _entries("onex-log-monitor.service", "monitor_logs.py")
    host = entries[0].hosts[0]
    results = check_units(
        entries,
        repo_root=repo,
        hostname=host,
        home=home,
        overrides={
            "onex-log-monitor.service": installed_unit,
            "monitor_logs.py": script,
        },
    )
    assert [result.status for result in results] == [EnumUnitStatus.OK] * 2
    assert not (home / ".config/systemd/user/onex-disk-gc.service").exists()
    assert log.read_text().splitlines() == [
        "--user daemon-reload",
        "--user enable onex-log-monitor.service",
        "--user restart onex-log-monitor.service",
    ]


@pytest.mark.parametrize(
    "exec_start",
    ["/usr/bin/python3 unrelated.py", "/usr/bin/python3 relative/monitor_logs.py"],
)
def test_log_monitor_refuses_missing_or_relative_script_destination(
    tmp_path: Path, exec_start: str
) -> None:
    env, log = _fake_systemctl(tmp_path)
    repo = _log_monitor_tree(tmp_path, exec_start)
    result = _run(repo / _LOG_INSTALLER, env, "--log-monitor")
    assert result.returncode == 2
    assert "monitor_logs.py" in result.stderr
    assert not Path(env["HOME"]).exists()
    assert not log.exists()


def _maintenance_manifest_rows() -> list[tuple[str, str, str]]:
    text = (_REPO_ROOT / _MIRROR_INSTALLER).read_text()
    block = text.split("MANIFEST=(", 1)[1].split("\n)", 1)[0]
    return re.findall(r'^\s*"([^|]+)\|([^|]+)\|([0-7]{4})"\s*$', block, re.MULTILINE)


def test_git_mirror_installer_manifest_matches_drift_manifest() -> None:
    entries = [
        entry
        for entry in load_manifest(_MANIFEST)
        if entry.installer == _MIRROR_INSTALLER
    ]
    assert {entry.name for entry in entries} == set(_MIRROR_NAMES)
    rows = _maintenance_manifest_rows()
    for entry in entries:
        matches = [row for row in rows if row[0] == entry.tracked]
        assert matches == [(entry.tracked, entry.installed, "0644")]


def test_git_mirror_converge_only_activates_timer(tmp_path: Path) -> None:
    env, log = _fake_systemctl(tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    entries = _entries(*_MIRROR_NAMES)
    for entry in entries:
        target = repo / entry.tracked
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_REPO_ROOT / entry.tracked, target)
    git_env = {
        **scrub_git_location_env(),
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_SYSTEM": "/dev/null",
    }
    for args in (
        "init --quiet -b dev",
        "add -A",
        "-c user.name=test -c user.email=test@example.com "
        "-c core.hooksPath=/dev/null commit --quiet --no-gpg-sign -m seed",
        "update-ref refs/remotes/origin/dev HEAD",
    ):
        subprocess.run(
            ["git", *args.split()],
            cwd=repo,
            env=git_env,
            check=True,
            capture_output=True,
        )
    unit_dir = tmp_path / "etc/systemd/system"
    unit_dir.mkdir(parents=True)
    (unit_dir / entries[0].name).write_text("stale daemon unit\n")
    manifest = tmp_path / "manifest.txt"
    manifest.write_text(
        "".join(f"{entry.tracked}|{unit_dir / entry.name}|0644\n" for entry in entries)
    )
    env.update(
        OMNINODE_INFRA_REPO_ROOT=str(repo),
        OMNINODE_MAINTENANCE_SYNC_MANIFEST=str(manifest),
        OMNINODE_MAINTENANCE_SYNC_SKIP_FETCH="1",
        OMNINODE_MAINTENANCE_SYNC_SYSTEMCTL=str(tmp_path / "bin/systemctl"),
        OMNINODE_MAINTENANCE_SYNC_RETIRED_CRON=str(tmp_path / "absent-cron"),
        OMNINODE_ALERT_ENV_FILE=str(tmp_path / "absent.env"),
    )
    result = _run(_REPO_ROOT / _MIRROR_INSTALLER, env, "--converge")
    assert result.returncode == 0, result.stdout + result.stderr
    host = entries[0].hosts[0]
    results = check_units(
        entries,
        repo_root=repo,
        hostname=host,
        home=Path(env["HOME"]),
        overrides={entry.name: unit_dir / entry.name for entry in entries},
    )
    assert [result.status for result in results] == [EnumUnitStatus.OK] * 3
    assert log.read_text().splitlines() == [
        "daemon-reload",
        "enable --now omninode-git-mirror-refresh.timer",
        "is-active --quiet omninode-git-mirror-refresh.timer",
    ]


def test_manifest_unit_documentation_points_at_existing_repo_paths() -> None:
    prefix = "https://github.com/OmniNode-ai/omnibase_infra/"
    for entry in load_manifest(_MANIFEST):
        if not entry.tracked.endswith((".service", ".timer")):
            continue
        for line in (_REPO_ROOT / entry.tracked).read_text().splitlines():
            if not line.startswith("Documentation="):
                continue
            value = line.removeprefix("Documentation=")
            assert "knowledge-base:" not in value, entry.tracked
            for url in value.split():
                for kind in ("blob", "tree"):
                    target_prefix = f"{prefix}{kind}/dev/"
                    if url.startswith(target_prefix):
                        target = _REPO_ROOT / url.removeprefix(target_prefix)
                        assert (
                            target.is_file() if kind == "blob" else target.is_dir()
                        ), (
                            entry.tracked,
                            url,
                        )


@pytest.mark.parametrize("installer", ["/absolute/install.sh", "../install.sh", ""])
def test_installer_must_be_repo_relative(installer: str) -> None:
    with pytest.raises(ValidationError, match="repo-root-relative"):
        ModelUnitEntry(
            name="x.service",
            tracked="x.service",
            installed="~/x.service",
            hosts=["lab-host"],
            installer=installer,
        )


@pytest.mark.parametrize("status", list(EnumUnitStatus))
@pytest.mark.parametrize("installer", [_MIRROR_INSTALLER, None])
def test_results_and_text_report_include_repair_only_when_needed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    status: EnumUnitStatus,
    installer: str | None,
) -> None:
    (shipped_entry,) = _entries("omninode-git-mirror-daemon.service")
    entry = shipped_entry.model_copy(update={"installer": installer})
    installed = tmp_path / entry.name
    if status in (EnumUnitStatus.OK, EnumUnitStatus.DRIFT):
        installed.write_bytes(
            (_REPO_ROOT / entry.tracked).read_bytes()
            if status is EnumUnitStatus.OK
            else b"stale unit\n"
        )
    host = entry.hosts[0]
    hostname = "other-host" if status is EnumUnitStatus.NOT_APPLICABLE else host
    results = check_units(
        [entry],
        repo_root=_REPO_ROOT,
        hostname=hostname,
        home=tmp_path,
        overrides={entry.name: installed},
    )
    assert results[0].status is status
    repair = (
        shlex.join(["bash", installer, *entry.installer_args]) if installer else None
    )
    assert results[0].repair == repair
    assert report(results, hostname)["units"][0]["repair"] == repair
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(yaml.safe_dump({"units": [entry.model_dump()]}))
    assert main(
        [
            "--manifest",
            str(manifest),
            "--repo-root",
            str(_REPO_ROOT),
            "--hostname",
            hostname,
            "--home",
            str(tmp_path),
            "--installed-override",
            f"{entry.name}={installed}",
        ]
    ) == int(status in (EnumUnitStatus.DRIFT, EnumUnitStatus.MISSING))
    line = capsys.readouterr().out.splitlines()[0]
    expected = f"UNIT-DRIFT {status.value} {entry.name} installed={installed}"
    if installer and status in (EnumUnitStatus.DRIFT, EnumUnitStatus.MISSING):
        expected += f" repair={repair}"
    assert line == expected
