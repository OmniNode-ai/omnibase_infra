# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Systemd coordination tests for the lane-census installer (OMN-13011, OMN-20933).

The ticket requires sharing the OMN-13008 timer unit rather than adding a second
one. The units themselves are lab configuration: since OMN-20933 they ship from
the operator's private config root (``$ONEX_WORKSPACE_CONFIG_ROOT/systemd/lane-census/``),
whose own tests pin their content. These tests assert the installer's half of the contract:
  1. It reads the units from the private config root, and refuses loudly when no
     root is configured or a unit is absent there. It never falls back to a copy
     in this repository.
  2. It depends on the base unit and refuses a second schedule beside it.
  3. It renders the clone path and the broker container into the installed unit.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.helpers.lane_census_fixtures import NEUTRAL_CONFIG_ROOT, NEUTRAL_MANIFEST

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_INSTALLER = _REPO / "deploy" / "lane-census" / "install-lane-census.sh"

# The clone path the private units name and the installer rewrites, read from
# the installer so the two cannot disagree.
_SHIPPED = re.search(
    r"^SHIPPED_REPO_ROOT='([^']+)'$",
    _INSTALLER.read_text(encoding="utf-8"),
    re.MULTILINE,
)
assert _SHIPPED is not None, "the installer declares no SHIPPED_REPO_ROOT"
_SHIPPED_REPO_ROOT = _SHIPPED.group(1)
_EXEC = (
    f"/bin/bash {_SHIPPED_REPO_ROOT}/scripts/lane-census-check.sh "
    "--snapshot %h/.local/state/onex/census-snapshot.json --memory"
)
# A line no unit in any real config root carries, so a rendered unit that holds
# it was read from the root the test supplied.
_MARKER = "# synthetic private unit (test_lane_census_timer_dropin)"


def _write_units(root: Path) -> Path:
    """A private config root holding the three units the installer reads."""
    units = root / "systemd" / "lane-census"
    (units / "onex-disk-gc.service.d").mkdir(parents=True)
    (units / "standalone").mkdir()
    (units / "onex-disk-gc.service.d" / "20-lane-census.conf").write_text(
        f"{_MARKER}\n[Service]\nExecStart=-{_EXEC}\n"
    )
    (units / "standalone" / "onex-lane-census.service").write_text(
        f"{_MARKER}\n[Unit]\nDescription=census\n\n[Service]\nType=oneshot\n"
        f"ExecStart={_EXEC}\n"
    )
    (units / "standalone" / "onex-lane-census.timer").write_text(
        f"{_MARKER}\n[Timer]\nOnCalendar=hourly\nUnit=onex-lane-census.service\n\n"
        "[Install]\nWantedBy=timers.target\n"
    )
    return units


def _write_manifest(root: Path) -> Path:
    """The neutral lane manifest, where the census reads it under the root (OMN-20934)."""
    manifest = root / NEUTRAL_MANIFEST.relative_to(NEUTRAL_CONFIG_ROOT)
    manifest.parent.mkdir(parents=True)
    shutil.copyfile(NEUTRAL_MANIFEST, manifest)
    return manifest


def test_installer_depends_on_base_unit() -> None:
    """Installer fails fast if the base onex-disk-gc.service is not installed."""
    body = _INSTALLER.read_text(encoding="utf-8")
    assert "onex-disk-gc.service" in body
    assert "install-disk-gc.sh" in body  # points operator at OMN-13008's installer
    assert "daemon-reload" in body


# ---------------------------------------------------------------------------
# OMN-19959 — the installer renders the broker container and the clone path
# ---------------------------------------------------------------------------


def _installer_env(
    tmp_path: Path, *, config_root: bool = True
) -> tuple[dict[str, str], Path]:
    """A fake HOME, a systemctl stub that records its calls and a private root.

    The environment is built from scratch, so neither ONEX_WORKSPACE_CONFIG_ROOT
    nor OMNI_HOME leaks in from the machine running the test.
    """
    home = tmp_path / "home"
    (home / ".config/systemd/user").mkdir(parents=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "systemctl.log"
    stub = bin_dir / "systemctl"
    stub.write_text(f'#!/usr/bin/env bash\necho "$*" >> "{calls}"\nexit 0\n')
    stub.chmod(0o755)
    env = {"HOME": str(home), "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    if config_root:
        root = tmp_path / "config-root"
        _write_units(root)
        _write_manifest(root)
        env["ONEX_WORKSPACE_CONFIG_ROOT"] = str(root)
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
    assert _MARKER in body, "the drop-in was not read from the private config root"
    assert "Environment=LANE_MEMORY_BROKER_CONTAINER=omnibase-infra-redpanda" in body
    assert (
        f"Environment=ONEX_WORKSPACE_CONFIG_ROOT={env['ONEX_WORKSPACE_CONFIG_ROOT']}"
        in body
    )
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
    assert _MARKER in service, "the unit was not read from the private config root"
    assert _MARKER in (units / "onex-lane-census.timer").read_text()
    assert (
        "Environment=LANE_MEMORY_BROKER_CONTAINER=omnibase-infra-dev-202-redpanda"
        in service
    )
    assert f"ExecStart=/bin/bash {_REPO}/scripts/lane-census-check.sh" in service
    assert (
        f"Environment=ONEX_WORKSPACE_CONFIG_ROOT={env['ONEX_WORKSPACE_CONFIG_ROOT']}"
        in service
    )
    assert (units / "onex-lane-census.timer").exists()
    assert not (units / "onex-disk-gc.service.d/20-lane-census.conf").exists()
    calls = (tmp_path / "systemctl.log").read_text()
    assert "--user enable --now onex-lane-census.timer" in calls


# ---------------------------------------------------------------------------
# OMN-20933 -- the units come from the private config root, or nowhere
# ---------------------------------------------------------------------------


def test_installer_refuses_without_a_config_root(tmp_path: Path) -> None:
    env, home = _installer_env(tmp_path, config_root=False)
    units = home / ".config/systemd/user"
    (units / "onex-disk-gc.service").write_text("[Service]\n")
    result = _install(env, "--broker-container", "b", "--repo-root", str(_REPO))
    assert result.returncode == 2, result.stderr
    assert "not configured" in result.stderr
    assert "ONEX_WORKSPACE_CONFIG_ROOT" in result.stderr
    assert not (units / "onex-disk-gc.service.d/20-lane-census.conf").exists()


def test_installer_refuses_when_the_root_holds_no_units(tmp_path: Path) -> None:
    env, home = _installer_env(tmp_path, config_root=False)
    empty = tmp_path / "empty-root"
    empty.mkdir()
    env["ONEX_WORKSPACE_CONFIG_ROOT"] = str(empty)
    result = _install(
        env, "--standalone", "--broker-container", "b", "--repo-root", str(_REPO)
    )
    assert result.returncode == 2, result.stderr
    assert f"{empty}/systemd/lane-census/" in result.stderr
    assert not (home / ".config/systemd/user/onex-lane-census.service").exists()


def test_the_workspace_root_alone_is_not_a_config_root(tmp_path: Path) -> None:
    """There is no implicit default root: OMNI_HOME names no unit source."""
    env, home = _installer_env(tmp_path, config_root=False)
    (tmp_path / "workspace").mkdir()
    env["OMNI_HOME"] = str(tmp_path / "workspace")
    units = home / ".config/systemd/user"
    (units / "onex-disk-gc.service").write_text("[Service]\n")
    result = _install(env, "--broker-container", "b", "--repo-root", str(_REPO))
    assert result.returncode == 2, result.stderr
    assert "ONEX_WORKSPACE_CONFIG_ROOT" in result.stderr
    assert not (units / "onex-disk-gc.service.d/20-lane-census.conf").exists()


def test_installer_ships_no_unit_beside_itself() -> None:
    """The installer has no copy in this repository to fall back to."""
    beside = _INSTALLER.parent
    assert not list(beside.rglob("*.service")), list(beside.rglob("*.service"))
    assert not list(beside.rglob("*.timer")), list(beside.rglob("*.timer"))
    assert not list(beside.rglob("*.conf")), list(beside.rglob("*.conf"))


# ---------------------------------------------------------------------------
# OMN-20934 -- the root also holds the lane manifest the census reads
# ---------------------------------------------------------------------------


def test_installer_requires_the_manifest_under_the_config_root(
    tmp_path: Path,
) -> None:
    """A root holding the units but no lane manifest installs nothing."""
    env, home = _installer_env(tmp_path, config_root=False)
    root = tmp_path / "units-only-root"
    _write_units(root)
    env["ONEX_WORKSPACE_CONFIG_ROOT"] = str(root)
    units = home / ".config/systemd/user"
    (units / "onex-disk-gc.service").write_text("[Service]\n")
    result = _install(
        env, "--broker-container", "omnibase-infra-redpanda", "--repo-root", str(_REPO)
    )
    assert result.returncode == 2, result.stderr
    assert "lane-manifest.yaml" in result.stderr
    assert "ONEX_WORKSPACE_CONFIG_ROOT" in result.stderr
    assert not (units / "onex-disk-gc.service.d/20-lane-census.conf").exists()
