# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Systemd coordination tests for the lane-census timer drop-in (OMN-13011).

The ticket requires sharing the OMN-13008 timer unit rather than adding a second
one. These tests assert the coordination contract:
  1. The lane-census ExecStart is delivered as a drop-in for the SHARED
     onex-disk-gc.service (not a new .timer / .service).
  2. The drop-in invokes scripts/lane-census-check.sh.
  3. The installer refuses to install a second timer and depends on the base unit.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_LANE_CENSUS_DIR = _REPO / "deploy" / "lane-census"
_DROPIN = _LANE_CENSUS_DIR / "onex-disk-gc.service.d" / "20-lane-census.conf"
_INSTALLER = _LANE_CENSUS_DIR / "install-lane-census.sh"


def test_dropin_targets_shared_disk_gc_service() -> None:
    """The drop-in lives under onex-disk-gc.service.d (shared unit), not a new unit."""
    assert _DROPIN.exists(), f"missing drop-in: {_DROPIN}"
    assert _DROPIN.parent.name == "onex-disk-gc.service.d"
    body = _DROPIN.read_text(encoding="utf-8")
    assert "[Service]" in body
    assert "ExecStart=" in body


def test_dropin_invokes_lane_census_check() -> None:
    body = _DROPIN.read_text(encoding="utf-8")
    assert "scripts/lane-census-check.sh" in body


def test_no_second_timer_on_a_host_that_runs_disk_gc() -> None:
    """Coordination rule: never a second census timer beside onex-disk-gc.timer.

    OMN-19959 ships ONE standalone unit pair, under ``standalone/``, for a lab
    host that carries no onex-disk-gc.service at all (.202). It is not a second
    timer on any host: the installer refuses ``--standalone`` where the base
    unit exists (``test_installer_refuses_standalone_beside_disk_gc``). The
    top level of this directory still ships no timer and no service.
    """
    timers = list(_LANE_CENSUS_DIR.glob("*.timer"))
    assert not timers, (
        f"lane-census must share the onex-disk-gc.timer, not add its own: {timers}"
    )
    services = [
        p
        for p in _LANE_CENSUS_DIR.glob("*.service")
        if p.name != "onex-disk-gc.service"
    ]
    assert not services, f"unexpected standalone service unit(s): {services}"
    standalone = sorted(p.name for p in (_LANE_CENSUS_DIR / "standalone").iterdir())
    assert standalone == ["onex-lane-census.service", "onex-lane-census.timer"]


def test_installer_depends_on_base_unit() -> None:
    """Installer fails fast if the base onex-disk-gc.service is not installed."""
    body = _INSTALLER.read_text(encoding="utf-8")
    assert "onex-disk-gc.service" in body
    assert "install-disk-gc.sh" in body  # points operator at OMN-13008's installer
    assert "daemon-reload" in body


# ---------------------------------------------------------------------------
# OMN-19959 — the installer renders the broker container and the clone path
# ---------------------------------------------------------------------------


def _installer_env(tmp_path: Path) -> tuple[dict[str, str], Path]:
    """A fake HOME and a systemctl stub that records its calls."""
    home = tmp_path / "home"
    (home / ".config/systemd/user").mkdir(parents=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "systemctl.log"
    stub = bin_dir / "systemctl"
    stub.write_text(f'#!/usr/bin/env bash\necho "$*" >> "{calls}"\nexit 0\n')
    stub.chmod(0o755)
    env = {"HOME": str(home), "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    return env, home


def _install(env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/bin/bash", str(_INSTALLER), *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
        timeout=60,
    )


def test_installer_requires_a_broker_container(tmp_path: Path) -> None:
    env, home = _installer_env(tmp_path)
    (home / ".config/systemd/user/onex-disk-gc.service").write_text("[Service]\n")
    result = _install(env)
    assert result.returncode == 2, result.stderr
    assert "--broker-container" in result.stderr


def test_installer_renders_the_dropin_with_broker_and_clone(tmp_path: Path) -> None:
    env, home = _installer_env(tmp_path)
    units = home / ".config/systemd/user"
    (units / "onex-disk-gc.service").write_text("[Service]\n")
    result = _install(
        env, "--broker-container", "omnibase-infra-redpanda", "--repo-root", str(_REPO)
    )
    assert result.returncode == 0, result.stderr
    body = (units / "onex-disk-gc.service.d/20-lane-census.conf").read_text()
    assert "Environment=LANE_MEMORY_BROKER_CONTAINER=omnibase-infra-redpanda" in body
    assert f"ExecStart=-/bin/bash {_REPO}/scripts/lane-census-check.sh" in body
    assert "--snapshot" in body and "--memory" in body
    assert (
        "%h/Code/omni_home/omnibase_infra"
        not in body.split("[Service]", 1)[1].split("ExecStart", 1)[1]
    )
    assert not (units / "onex-lane-census.timer").exists()


def test_installer_refuses_standalone_beside_disk_gc(tmp_path: Path) -> None:
    env, home = _installer_env(tmp_path)
    (home / ".config/systemd/user/onex-disk-gc.service").write_text("[Service]\n")
    result = _install(env, "--standalone", "--broker-container", "b")
    assert result.returncode == 1
    assert "Install without --standalone" in result.stderr
    assert not (home / ".config/systemd/user/onex-lane-census.service").exists()


def test_installer_standalone_on_a_host_without_disk_gc(tmp_path: Path) -> None:
    env, home = _installer_env(tmp_path)
    units = home / ".config/systemd/user"
    result = _install(
        env,
        "--standalone",
        "--broker-container",
        "omnibase-infra-dev-202-redpanda",
        "--repo-root",
        str(_REPO),
    )
    assert result.returncode == 0, result.stderr
    service = (units / "onex-lane-census.service").read_text()
    assert (
        "Environment=LANE_MEMORY_BROKER_CONTAINER=omnibase-infra-dev-202-redpanda"
        in service
    )
    assert f"ExecStart=/bin/bash {_REPO}/scripts/lane-census-check.sh" in service
    assert (units / "onex-lane-census.timer").exists()
    assert not (units / "onex-disk-gc.service.d/20-lane-census.conf").exists()
    calls = (tmp_path / "systemctl.log").read_text()
    assert "--user enable --now onex-lane-census.timer" in calls
