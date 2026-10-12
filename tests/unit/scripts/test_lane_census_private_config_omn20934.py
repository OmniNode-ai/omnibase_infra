# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20934: the lane census is private config, read from the workspace config root.

Operator RULING 2026-10-10T21:57:38Z: no deployment or lab configuration lives
in a shipped repository. The lane manifest and the committed census named one
lab's hosts, lanes, addresses and people, and every automated refresh of the
census failed the public-repo hygiene gate on them. They now live in
``config/lab/lane-census`` under ``ONEX_WORKSPACE_CONFIG_ROOT`` (else the owner
the runtime resolves for ``OMNI_HOME``), and every reader resolves them there or
refuses, naming the variable.

AC1 falsifier: the repository carries neither file. AC2 falsifier: each reader,
with and without the config root.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)
from tests.helpers.lane_census_fixtures import (
    NEUTRAL_CONFIG_ROOT,
    NEUTRAL_MANIFEST,
    NEUTRAL_SNAPSHOT,
    use_neutral_lane_census,
)

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_SCRIPTS = _REPO / "scripts"
sys.path.insert(0, str(_SCRIPTS))

import lane_census_plan as plan


def _unset(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("ONEX_WORKSPACE_CONFIG_ROOT", "OMNI_HOME", "LANE_MANIFEST"):
        monkeypatch.delenv(name, raising=False)


def test_the_repository_ships_no_lane_census() -> None:
    """AC1: neither the manifest nor the committed census is in this repository."""
    tracked = subprocess.run(
        ["git", "ls-files", "--", "deploy/lane-census", "config/lab"],
        cwd=_REPO,
        env=scrub_git_location_env(os.environ),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert tracked, "positive control: the lane census installer is still tracked"
    assert not [
        path
        for path in tracked
        if path.endswith(("lane-manifest.yaml", "census-snapshot.json"))
    ], tracked


def test_the_config_root_names_the_census_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_neutral_lane_census(monkeypatch)
    assert plan.lane_census_dir() == NEUTRAL_MANIFEST.parent
    assert plan.default_manifest_path() == NEUTRAL_MANIFEST
    assert plan.default_snapshot_path() == NEUTRAL_SNAPSHOT


def test_omni_home_resolves_through_the_runtime_owner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With no explicit root, the runtime's own resolver answers for OMNI_HOME."""
    from omnibase_infra.runtime.service_kernel import workspace_runtime_config_root

    _unset(monkeypatch)
    registry = tmp_path / "registry"
    registry.mkdir()
    monkeypatch.setenv("OMNI_HOME", str(registry))
    expected = (
        workspace_runtime_config_root(registry) / "config" / "lab" / "lane-census"
    )
    assert plan.lane_census_dir() == expected


def test_lane_manifest_overrides_the_manifest_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _unset(monkeypatch)
    explicit = tmp_path / "m.yaml"
    monkeypatch.setenv("LANE_MANIFEST", str(explicit))
    assert plan.default_manifest_path() == explicit
    with pytest.raises(plan.LaneCensusConfigError):
        plan.default_snapshot_path()


def test_nothing_configured_is_a_loud_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    _unset(monkeypatch)
    with pytest.raises(plan.LaneCensusConfigError, match="ONEX_WORKSPACE_CONFIG_ROOT"):
        plan.default_manifest_path()


def test_a_root_without_the_files_is_a_loud_refusal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _unset(monkeypatch)
    monkeypatch.setenv("ONEX_WORKSPACE_CONFIG_ROOT", str(tmp_path))
    with pytest.raises(plan.LaneCensusConfigError, match="does not exist"):
        plan.default_manifest_path()


def _run(
    script: str, *args: str, root: Path | None
) -> subprocess.CompletedProcess[str]:
    env = {
        k: v for k, v in os.environ.items() if k not in {"OMNI_HOME", "LANE_MANIFEST"}
    }
    env.pop("ONEX_WORKSPACE_CONFIG_ROOT", None)
    if root is not None:
        env["ONEX_WORKSPACE_CONFIG_ROOT"] = str(root)
    return subprocess.run(
        [sys.executable, str(_SCRIPTS / script), *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
        timeout=120,
    )


@pytest.mark.parametrize(
    ("script", "args"),
    [
        ("check_lane_census_age.py", ("--max-age-days", "100000")),
        # The neutral lanes have no compose files; the lab's are checked against
        # the lab's manifest by the staleness job.
        ("check_lane_env_drift.py", ("--compose-dir", str(NEUTRAL_CONFIG_ROOT))),
        ("generate_claude_lane_block.py", ()),
    ],
)
def test_each_reader_resolves_the_root_and_refuses_without_it(
    script: str, args: tuple[str, ...]
) -> None:
    """AC2: the same reader, with and without ONEX_WORKSPACE_CONFIG_ROOT."""
    with_root = _run(script, *args, root=NEUTRAL_CONFIG_ROOT)
    assert with_root.returncode == 0, with_root.stderr
    without = _run(script, *args, root=None)
    assert without.returncode != 0
    assert "ONEX_WORKSPACE_CONFIG_ROOT" in without.stderr


def test_the_drift_checker_resolves_the_root_and_refuses_without_it() -> None:
    """The neutral pair reads; the checker's own verdict on it is its tests' job."""
    with_root = _run("check_lane_census_drift.py", "--json", root=NEUTRAL_CONFIG_ROOT)
    assert "could not read the census inputs" not in with_root.stderr
    json.loads(with_root.stdout)
    without = _run("check_lane_census_drift.py", root=None)
    assert without.returncode == 2
    assert "ONEX_WORKSPACE_CONFIG_ROOT" in without.stderr


def test_the_lab_alarm_config_names_the_manifest_under_the_root() -> None:
    config = json.loads((_REPO / "config" / "lab_alarm.json").read_text())
    assert config["lane_manifest"] == (
        "${ONEX_WORKSPACE_CONFIG_ROOT}/config/lab/lane-census/lane-manifest.yaml"
    )
