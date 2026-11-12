# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The lane census refresh workflow exports the host registry (OMN-23996).

``.github/workflows/lane-census-refresh.yml`` places the CI overlay outside the
workspace and, before the collector runs, must export the lane census node's
overlay (the host registry, OMN-22152) as ``ONEX_SKILL_OVERLAY_ROOTS`` and
``LANE_CENSUS_REGISTRY`` when the overlay repository carries it. Without that
the collector exits 3 with the not-configured error. The tests run the
workflow's place step in bash over a temp checkout and read what it wrote to
``GITHUB_ENV``.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[2]
_WORKFLOW = _REPO / ".github" / "workflows" / "lane-census-refresh.yml"
_FIXTURE_REGISTRY = (
    _REPO
    / "tests"
    / "fixtures"
    / "lane_census"
    / "node_lane_census_effect"
    / "overlay.yaml"
)


def _place_step() -> str:
    """The ``run`` text of the refresh workflow's place step."""
    data = yaml.safe_load(_WORKFLOW.read_text())
    steps = data["jobs"]["refresh"]["steps"]
    for step in steps:
        if step.get("name") == "Place the CI overlay outside the workspace":
            return step["run"]
    raise AssertionError("the place step is not in the refresh workflow")


def _checkout_step() -> dict[str, Any]:
    """The refresh workflow's overlay checkout step."""
    data = yaml.safe_load(_WORKFLOW.read_text())
    steps = data["jobs"]["refresh"]["steps"]
    for step in steps:
        if (
            step.get("name")
            == "Check out the CI overlay (app token so the bot PR triggers CI)"
        ):
            return step
    raise AssertionError("the overlay checkout step is not in the refresh workflow")


def _run_place_step(tmp_path: Path, *, carries_registry: bool) -> dict[str, str]:
    """Run the place step over a temp layout and return the exported env."""
    work = tmp_path / "work"
    ws = work / "ws"
    overlay = ws / ".ci-overlay-checkout"
    (overlay / "config" / "lab" / "lane-census").mkdir(parents=True)
    (overlay / "config" / "lab" / "lane-census" / "lane-manifest.yaml").write_text(
        "lanes: []\n"
    )
    if carries_registry:
        node_dir = overlay / "config" / "onex" / "overlays" / "node_lane_census_effect"
        node_dir.mkdir(parents=True)
        (node_dir / "overlay.yaml").write_bytes(_FIXTURE_REGISTRY.read_bytes())
    env_file = tmp_path / "github_env"
    env = dict(os.environ)
    env["GITHUB_ENV"] = str(env_file)
    proc = subprocess.run(
        ["bash", "-c", _place_step()],
        cwd=ws,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, (proc.returncode, proc.stderr)
    exported: dict[str, str] = {}
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line:
                key, _, value = line.partition("=")
                exported[key] = value
    return exported


def test_refresh_place_step_exports_the_registry_the_overlay_carries(
    tmp_path: Path,
) -> None:
    exported = _run_place_step(tmp_path, carries_registry=True)
    root = exported["ONEX_WORKSPACE_CONFIG_ROOT"]
    assert exported["ONEX_SKILL_OVERLAY_ROOTS"] == f"{root}/config/onex/overlays"
    assert (
        exported["LANE_CENSUS_REGISTRY"]
        == f"{root}/config/onex/overlays/node_lane_census_effect/overlay.yaml"
    )
    assert Path(exported["LANE_CENSUS_REGISTRY"]).is_file()


def test_refresh_place_step_exports_no_registry_when_the_overlay_has_none(
    tmp_path: Path,
) -> None:
    exported = _run_place_step(tmp_path, carries_registry=False)
    assert "ONEX_WORKSPACE_CONFIG_ROOT" in exported
    assert "CENSUS_DIR" in exported
    assert "LANE_CENSUS_REGISTRY" not in exported
    assert "ONEX_SKILL_OVERLAY_ROOTS" not in exported


def test_refresh_overlay_checkout_fetches_the_node_overlay() -> None:
    step = _checkout_step()
    entries = step["with"]["sparse-checkout"].split()
    assert "config/onex/overlays/node_lane_census_effect/overlay.yaml" in entries
    assert "config/lab/lane-census" in entries


def test_refresh_registry_resolves_for_the_collector(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "model_lane_census_registry",
        _REPO
        / "src"
        / "omnibase_infra"
        / "models"
        / "lane_census"
        / "model_lane_census_registry.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    registry_path = module.registry_path

    exported = _run_place_step(tmp_path, carries_registry=True)
    environ = {
        "ONEX_SKILL_OVERLAY_ROOTS": exported["ONEX_SKILL_OVERLAY_ROOTS"],
        "LANE_CENSUS_REGISTRY": exported["LANE_CENSUS_REGISTRY"],
    }
    path = registry_path(environ)
    assert Path(path).is_file()

    with pytest.raises(ValueError):
        registry_path({})
