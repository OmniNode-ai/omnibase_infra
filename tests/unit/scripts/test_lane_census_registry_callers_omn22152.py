# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Every caller of the lane census driver supplies the host registry (OMN-22152).

``lane-census-check.sh`` exits 3 when no host registry file is named. Before this
change nothing in the repository set ``LANE_CENSUS_REGISTRY`` for the three
callers that run the driver (the hosted refresh job, the hourly systemd unit and
the dev-lane refresh), so the first scheduled census and the next refresh after
the registry reader merged would both have exited 3. The driver now reads the
lane census node's overlay the way the node itself finds it,
``<root>/node_lane_census_effect/overlay.yaml`` under ``ONEX_SKILL_OVERLAY_ROOTS``,
when ``LANE_CENSUS_REGISTRY`` is unset; the hourly unit carries the registry's
path; the hosted jobs place the overlay and export both variables. The positive
control for each is the same caller with nothing supplied, which stays a loud
exit.

Falsifier: ``-k registry_callers`` fails at 2bbe82c65 and passes with the fix.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)
from tests.helpers.lane_census_fixtures import (
    NEUTRAL_CONFIG_ROOT,
    NEUTRAL_HOST_ID,
    NEUTRAL_HOSTNAME,
    NEUTRAL_MANIFEST,
    NEUTRAL_REGISTRY,
)

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_DRIVER = _REPO / "scripts" / "lane-census-check.sh"
_INSTALLER = _REPO / "deploy" / "lane-census" / "install-lane-census.sh"
_REFRESH = _REPO / "scripts" / "runtime_build" / "refresh_dev_lane.sh"
_OVERLAY_ACTION = _REPO / ".github" / "actions" / "fetch-census-overlay" / "action.yml"
_NODE_OVERLAY = Path("node_lane_census_effect") / "overlay.yaml"


def _overlay_root(tmp_path: Path) -> Path:
    """A node overlay root holding the neutral registry where the node looks."""
    root = tmp_path / "overlay-root"
    (root / _NODE_OVERLAY).parent.mkdir(parents=True)
    shutil.copyfile(NEUTRAL_REGISTRY, root / _NODE_OVERLAY)
    return root


def _stub_bin(tmp_path: Path, hostname: str) -> tuple[Path, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    marker = tmp_path / "engine-called.marker"
    for name, body in (
        ("docker", f'echo "$*" >> "{marker}"; exit 0'),
        ("rpk", "exit 0"),
        ("hostname", f'echo "{hostname}"'),
        ("systemctl", "exit 0"),
    ):
        path = bin_dir / name
        path.write_text(f"#!/usr/bin/env bash\n{body}\n")
        path.chmod(0o755)
    return bin_dir, marker


def _driver(
    tmp_path: Path, *args: str, roots: Path | None
) -> tuple[subprocess.CompletedProcess[str], Path]:
    """The driver with LANE_CENSUS_REGISTRY unset and ``roots`` as the overlay roots."""
    bin_dir, marker = _stub_bin(tmp_path, NEUTRAL_HOSTNAME)
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["HOME"] = str(tmp_path)
    env["LANE_CENSUS_DOCKER_SOCKET"] = "/nonexistent/registry-callers-test.sock"
    env["ONEX_WORKSPACE_CONFIG_ROOT"] = str(NEUTRAL_CONFIG_ROOT)
    for name in (
        "LANE_MANIFEST",
        "KAFKA_BOOTSTRAP_SERVERS",
        "LANE_CENSUS_HOST",
        "LANE_CENSUS_REGISTRY",
        "ONEX_SKILL_OVERLAY_ROOTS",
    ):
        env.pop(name, None)
    if roots is not None:
        env["ONEX_SKILL_OVERLAY_ROOTS"] = str(roots)
    proc = subprocess.run(
        ["bash", str(_DRIVER), *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )
    return proc, marker


# --- the driver, which every caller runs ----------------------------------------


def test_registry_callers_driver_reads_the_node_overlay_when_the_variable_is_unset(
    tmp_path: Path,
) -> None:
    proc, _ = _driver(tmp_path, "--json", "--dry-run", roots=_overlay_root(tmp_path))
    assert proc.returncode in (0, 30), (proc.returncode, proc.stderr)
    plan = json.loads(proc.stdout.splitlines()[0])
    assert plan["host"] == NEUTRAL_HOST_ID


def test_registry_callers_driver_with_no_registry_anywhere_still_exits_3_before_the_engine(
    tmp_path: Path,
) -> None:
    """Positive control: the default is a lookup, not a silent pass."""
    proc, marker = _driver(tmp_path, "--json", "--dry-run", roots=None)
    assert proc.returncode == 3, (proc.returncode, proc.stderr)
    assert "LANE_CENSUS_REGISTRY" in proc.stderr
    assert "ONEX_SKILL_OVERLAY_ROOTS" in proc.stderr
    assert not marker.exists(), "the engine was read"


def test_registry_callers_driver_ignores_a_root_that_holds_no_node_overlay(
    tmp_path: Path,
) -> None:
    empty = tmp_path / "empty-root"
    empty.mkdir()
    proc, marker = _driver(tmp_path, "--json", "--dry-run", roots=empty)
    assert proc.returncode == 3, (proc.returncode, proc.stderr)
    assert not marker.exists(), "the engine was read"


# --- the hourly unit installer ---------------------------------------------------


def _installer_env(
    tmp_path: Path, *, registry: Path | None
) -> tuple[dict[str, str], Path]:
    home = tmp_path / "home"
    (home / ".config/systemd/user").mkdir(parents=True)
    bin_dir, _ = _stub_bin(tmp_path, NEUTRAL_HOSTNAME)
    cfg = tmp_path / "config-root"
    units = cfg / "systemd" / "lane-census"
    (units / "onex-disk-gc.service.d").mkdir(parents=True)
    (units / "standalone").mkdir()
    (units / "onex-disk-gc.service.d" / "20-lane-census.conf").write_text(
        "[Service]\nExecStart=-/bin/true\n"
    )
    (units / "standalone" / "onex-lane-census.service").write_text(
        "[Unit]\nDescription=census\n\n[Service]\nType=oneshot\nExecStart=/bin/true\n"
    )
    (units / "standalone" / "onex-lane-census.timer").write_text(
        "[Timer]\nOnCalendar=hourly\n\n[Install]\nWantedBy=timers.target\n"
    )
    manifest = cfg / NEUTRAL_MANIFEST.relative_to(NEUTRAL_CONFIG_ROOT)
    manifest.parent.mkdir(parents=True)
    shutil.copyfile(NEUTRAL_MANIFEST, manifest)
    env = {
        "HOME": str(home),
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "ONEX_WORKSPACE_CONFIG_ROOT": str(cfg),
    }
    if registry is not None:
        env["LANE_CENSUS_REGISTRY"] = str(registry)
    return env, home


def _install(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "/bin/bash",
            str(_INSTALLER),
            "--standalone",
            "--broker-container",
            "b",
            "--repo-root",
            str(_REPO),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
        timeout=60,
    )


def test_registry_callers_installer_renders_the_registry_into_the_unit(
    tmp_path: Path,
) -> None:
    env, home = _installer_env(tmp_path, registry=NEUTRAL_REGISTRY)
    result = _install(env)
    assert result.returncode == 0, result.stderr
    service = (home / ".config/systemd/user/onex-lane-census.service").read_text()
    assert f"Environment=LANE_CENSUS_REGISTRY={NEUTRAL_REGISTRY}" in service
    assert "Environment=ONEX_WORKSPACE_CONFIG_ROOT=" in service


def test_registry_callers_installer_with_no_registry_installs_nothing(
    tmp_path: Path,
) -> None:
    env, home = _installer_env(tmp_path, registry=None)
    # A unit has no overlay roots, so roots alone are not a registry to render.
    env["ONEX_SKILL_OVERLAY_ROOTS"] = str(_overlay_root(tmp_path))
    result = _install(env)
    assert result.returncode == 2, (result.returncode, result.stderr)
    assert "LANE_CENSUS_REGISTRY" in result.stderr
    assert not (home / ".config/systemd/user/onex-lane-census.service").exists()
    assert not (home / ".config/systemd/user/onex-lane-census.timer").exists()


def test_registry_callers_installer_refuses_a_registry_that_is_not_a_file(
    tmp_path: Path,
) -> None:
    env, home = _installer_env(tmp_path, registry=tmp_path / "absent.yaml")
    result = _install(env)
    assert result.returncode == 2, (result.returncode, result.stderr)
    assert not (home / ".config/systemd/user/onex-lane-census.service").exists()


# --- the dev-lane refresh --------------------------------------------------------

_REFRESH_SHIM = r"""#!/usr/bin/env python3
import json, os, sys
rows = json.load(open(os.environ["SHIM_INVENTORY"]))
args = sys.argv[1:]
if args[:2] == ["ps", "-a"]:
    for r in rows:
        print("\t".join([r["name"], r["state"], r["status"], r["image"], r["labels"]]))
elif args[:2] == ["network", "ls"]:
    print(os.environ["SHIM_NETWORK"])
    print("bridge")
sys.exit(0)
"""


def _refresh(tmp_path: Path, *, roots: Path | None) -> subprocess.CompletedProcess[str]:
    """``refresh_dev_lane.sh --census-gate`` with LANE_CENSUS_REGISTRY unset."""
    manifest = yaml.safe_load(NEUTRAL_MANIFEST.read_text())
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    shim = bin_dir / "docker"
    shim.write_text(
        _REFRESH_SHIM.replace("#!/usr/bin/env python3", f"#!{sys.executable}")
    )
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    inventory = tmp_path / "inventory.json"
    inventory.write_text("[]")
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["HOME"] = str(tmp_path)
    env["OMNI_HOME"] = str(tmp_path)
    env["ONEX_WORKSPACE_CONFIG_ROOT"] = str(NEUTRAL_CONFIG_ROOT)
    env["SHIM_INVENTORY"] = str(inventory)
    env["SHIM_NETWORK"] = manifest["lanes"]["dev"]["network"]
    env["LANE_CENSUS_DOCKER_SOCKET"] = "/nonexistent/registry-callers-test.sock"
    env["LANE_CENSUS_HOST"] = NEUTRAL_HOSTNAME
    env["LANE_CENSUS_PYTHON"] = sys.executable
    env["ONEX_LANE_LOCK_DIR"] = str(tmp_path / "locks")
    for name in (
        "LANE_MANIFEST",
        "LANE_CENSUS_REGISTRY",
        "ONEX_SKILL_OVERLAY_ROOTS",
        "OMNIBASE_INFRA_COMPOSE_PROJECT",
        "LANE_MEMORY_BROKER_CONTAINER",
    ):
        env.pop(name, None)
    if roots is not None:
        env["ONEX_SKILL_OVERLAY_ROOTS"] = str(roots)
    return subprocess.run(
        ["bash", str(_REFRESH), "--census-gate"],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(_REPO),
        timeout=120,
        check=False,
    )


def test_registry_callers_refresh_census_gate_reads_the_node_overlay(
    tmp_path: Path,
) -> None:
    proc = _refresh(tmp_path, roots=_overlay_root(tmp_path))
    assert proc.returncode == 0, (proc.returncode, proc.stderr)
    assert "host registry is not configured" not in proc.stderr, proc.stderr


def test_registry_callers_refresh_census_gate_with_no_registry_exits_3(
    tmp_path: Path,
) -> None:
    """Positive control: the refresh's census does not read clean with no registry."""
    proc = _refresh(tmp_path, roots=None)
    assert proc.returncode == 3, (proc.returncode, proc.stderr)
    assert "LANE_CENSUS_REGISTRY" in proc.stderr


# --- the hosted callers ----------------------------------------------------------


def _place_step() -> str:
    """The shell of the overlay action's place step, as the runner would execute it."""
    steps = yaml.safe_load(_OVERLAY_ACTION.read_text())["runs"]["steps"]
    return next(step["run"] for step in steps if step.get("id") == "place")


def _run_place_step(tmp_path: Path, *, carries_registry: bool) -> dict[str, str]:
    """Run the place step over a fetched overlay checkout; return what it exported."""
    work = tmp_path / "work"
    census = work / ".lane-census-overlay-checkout" / "config" / "lab" / "lane-census"
    census.mkdir(parents=True)
    (census / "lane-manifest.yaml").write_text("lanes: {}\n")
    (census / "census-snapshot.json").write_text("{}\n")
    if carries_registry:
        node = work / ".lane-census-overlay-checkout" / _NODE_OVERLAY
        node.parent.mkdir(parents=True)
        shutil.copyfile(NEUTRAL_REGISTRY, node)
    checkout = work / ".lane-census-overlay-checkout"
    for git_args in (
        ["init", "-q"],
        ["add", "-A"],
        [
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.invalid",
            "commit",
            "-qm",
            "x",
        ],
    ):
        subprocess.run(
            ["git", "-C", str(checkout), *git_args],
            check=True,
            env=scrub_git_location_env(os.environ),
        )
    github_env = tmp_path / "github-env"
    github_env.touch()
    github_output = tmp_path / "github-output"
    github_output.touch()
    env = dict(os.environ)
    env.update(
        RUNNER_TEMP=str(tmp_path / "runner-temp"),
        GITHUB_ENV=str(github_env),
        GITHUB_OUTPUT=str(github_output),
        GITHUB_REPOSITORY_OWNER="example-owner",
        OVERLAY_NAME="example-overlay",
    )
    (tmp_path / "runner-temp").mkdir()
    proc = subprocess.run(
        ["bash", "-c", _place_step()],
        cwd=work,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return dict(
        line.split("=", 1) for line in github_env.read_text().splitlines() if line
    )


def test_registry_callers_overlay_action_exports_the_registry_the_overlay_carries(
    tmp_path: Path,
) -> None:
    """The parity job fetches the census through this action; it must place the registry."""
    exported = _run_place_step(tmp_path, carries_registry=True)
    root = exported["ONEX_WORKSPACE_CONFIG_ROOT"]
    assert exported["ONEX_SKILL_OVERLAY_ROOTS"] == root
    assert exported["LANE_CENSUS_REGISTRY"] == f"{root}/{_NODE_OVERLAY}"
    assert Path(exported["LANE_CENSUS_REGISTRY"]).is_file()


def test_registry_callers_overlay_action_exports_no_registry_when_the_overlay_has_none(
    tmp_path: Path,
) -> None:
    """Positive control: nothing is exported for a registry that is not there."""
    exported = _run_place_step(tmp_path, carries_registry=False)
    assert "ONEX_WORKSPACE_CONFIG_ROOT" in exported
    assert "LANE_CENSUS_REGISTRY" not in exported
    assert "ONEX_SKILL_OVERLAY_ROOTS" not in exported


def test_registry_callers_overlay_action_fetches_the_node_overlay() -> None:
    steps = yaml.safe_load(_OVERLAY_ACTION.read_text())["runs"]["steps"]
    checkout = next(s for s in steps if "sparse-checkout" in s.get("with", {}))
    assert str(_NODE_OVERLAY) in checkout["with"]["sparse-checkout"].split()
