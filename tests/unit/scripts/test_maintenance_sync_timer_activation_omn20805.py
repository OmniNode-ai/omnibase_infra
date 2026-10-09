# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The maintenance sync enables the timers it installs and retires the legacy cron files (OMN-20805).

A unit file on disk schedules nothing, and a timer enabled beside a still-live
cron line runs the job twice. These tests drive the real sync script against a
throwaway git clone, a manifest of scratch paths and a stub ``systemctl``, so no
test touches ``/etc`` or a real systemd.

The properties:

* a converge that writes unit files reloads systemd once, enables and confirms
  every timer, and only then moves the legacy cron files aside;
* a timer that will not enable leaves the legacy cron files in place and fails
  the run, so a failed enable never leaves the host with no schedule;
* an in-sync tick writes nothing and reloads nothing;
* ``--check`` writes nothing and reddens on an inactive timer or a live legacy
  cron file.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SYNC_SCRIPT = REPO_ROOT / "deploy" / "maintenance" / "omninode-host-maintenance-sync.sh"

SERVICE_REL = "deploy/maintenance/systemd/omninode-demo.service"
TIMER_REL = "deploy/maintenance/systemd/omninode-demo.timer"
SERVICE_BODY = "[Service]\nType=oneshot\nExecStart=/bin/true\n"
TIMER_BODY = "[Timer]\nOnCalendar=*:0/5\n"

pytestmark = pytest.mark.unit


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        env={
            **scrub_git_location_env(),
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_SYSTEM": "/dev/null",
        },
    )


@pytest.fixture
def fake_clone(tmp_path: Path) -> Path:
    repo = tmp_path / "infra-clone"
    (repo / "deploy" / "maintenance" / "systemd").mkdir(parents=True)
    _git(repo.parent, "init", "--quiet", "-b", "dev", str(repo))
    _git(repo, "config", "user.email", "test@omninode.ai")
    _git(repo, "config", "user.name", "test")
    (repo / SERVICE_REL).write_text(SERVICE_BODY)
    (repo / TIMER_REL).write_text(TIMER_BODY)
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "--no-gpg-sign", "-m", "seed")
    _git(repo, "update-ref", "refs/remotes/origin/dev", "HEAD")
    return repo


class Host:
    """Scratch stand-ins for /etc/systemd/system, /etc/cron.d and systemctl."""

    def __init__(self, tmp_path: Path) -> None:
        self.unit_dir = tmp_path / "etc" / "systemd" / "system"
        self.unit_dir.mkdir(parents=True)
        self.cron_dir = tmp_path / "etc" / "cron.d"
        self.cron_dir.mkdir(parents=True)
        self.legacy = self.cron_dir / "omninode-demo"
        self.legacy.write_text("*/5 * * * * root /bin/true\n")
        self.calls = tmp_path / "systemctl.calls"
        self.active = tmp_path / "active"
        self.stub = tmp_path / "systemctl"
        # `enable --now` marks the timer active unless FAIL_ENABLE is set;
        # `is-active` answers from that marker.
        self.stub.write_text(
            "#!/usr/bin/env bash\n"
            f'echo "$*" >> {self.calls}\n'
            'case "$1" in\n'
            "  daemon-reload) exit 0 ;;\n"
            '  enable) [[ -n "${FAIL_ENABLE:-}" ]] && exit 1\n'
            f"          touch {self.active}; exit 0 ;;\n"
            f"  is-active) [[ -e {self.active} ]] ;;\n"
            "esac\n"
        )
        self.stub.chmod(0o755)
        self.manifest = tmp_path / "manifest.txt"
        self.manifest.write_text(
            f"{SERVICE_REL}|{self.unit_dir}/omninode-demo.service|0644\n"
            f"{TIMER_REL}|{self.unit_dir}/omninode-demo.timer|0644\n"
        )

    def systemctl_calls(self) -> list[str]:
        return self.calls.read_text().splitlines() if self.calls.exists() else []


@pytest.fixture
def host(tmp_path: Path) -> Host:
    return Host(tmp_path)


def _run(
    clone: Path, host: Host, tmp_path: Path, *args: str, fail_enable: bool = False
) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLACK_")}
    env.update(
        {
            "OMNINODE_INFRA_REPO_ROOT": str(clone),
            "OMNINODE_MAINTENANCE_SYNC_MANIFEST": str(host.manifest),
            "OMNINODE_MAINTENANCE_SYNC_SKIP_FETCH": "1",
            "OMNINODE_MAINTENANCE_SYNC_SYSTEMCTL": str(host.stub),
            "OMNINODE_MAINTENANCE_SYNC_RETIRED_CRON": str(host.legacy),
            "OMNINODE_ALERT_ENV_FILE": str(tmp_path / "absent.env"),
        }
    )
    if fail_enable:
        env["FAIL_ENABLE"] = "1"
    return subprocess.run(
        ["bash", str(SYNC_SCRIPT), *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )


def test_converge_installs_units_enables_timer_then_retires_legacy_cron(
    tmp_path: Path, fake_clone: Path, host: Host
) -> None:
    proc = _run(fake_clone, host, tmp_path, "--converge")

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (host.unit_dir / "omninode-demo.timer").read_text() == TIMER_BODY
    assert (host.unit_dir / "omninode-demo.service").read_text() == SERVICE_BODY
    calls = host.systemctl_calls()
    assert calls[0] == "daemon-reload"
    assert calls.count("daemon-reload") == 1
    assert "enable --now omninode-demo.timer" in calls
    assert not host.legacy.exists(), "the legacy cron file is still live"
    retired = list((host.cron_dir / "onex-retired").glob("omninode-demo.retired-*"))
    assert len(retired) == 1, "the legacy cron file must be moved aside, not deleted"
    assert "RETIRED|" in proc.stdout


def test_failed_enable_leaves_the_legacy_cron_file_and_fails(
    tmp_path: Path, fake_clone: Path, host: Host
) -> None:
    proc = _run(fake_clone, host, tmp_path, "--converge", fail_enable=True)

    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert host.legacy.exists(), (
        "a timer that would not enable must leave the old schedule running, "
        "not the host with none"
    )
    assert not (host.cron_dir / "onex-retired").exists()
    assert "ENABLE FAILED" in proc.stdout


def test_in_sync_tick_writes_and_reloads_nothing(
    tmp_path: Path, fake_clone: Path, host: Host
) -> None:
    first = _run(fake_clone, host, tmp_path, "--converge")
    assert first.returncode == 0, first.stdout + first.stderr
    host.calls.unlink()

    second = _run(fake_clone, host, tmp_path, "--converge")

    assert second.returncode == 0, second.stdout + second.stderr
    assert "converged=0" in second.stdout
    assert "daemon-reload" not in host.systemctl_calls()


def test_check_reddens_on_an_inactive_timer_and_a_live_legacy_cron_file(
    tmp_path: Path, fake_clone: Path, host: Host
) -> None:
    first = _run(fake_clone, host, tmp_path, "--converge")
    assert first.returncode == 0, first.stdout + first.stderr
    host.active.unlink()
    host.legacy.write_text("*/5 * * * * root /bin/true\n")

    proc = _run(fake_clone, host, tmp_path, "--check")

    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "timer is not active" in proc.stdout
    assert "legacy cron file is still live" in proc.stdout
    assert host.legacy.exists(), "--check must write nothing"


def test_check_is_green_when_timer_active_and_cron_retired(
    tmp_path: Path, fake_clone: Path, host: Host
) -> None:
    assert _run(fake_clone, host, tmp_path, "--converge").returncode == 0

    proc = _run(fake_clone, host, tmp_path, "--check")

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "timer active" in proc.stdout
