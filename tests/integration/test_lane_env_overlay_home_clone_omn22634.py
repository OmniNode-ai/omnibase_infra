# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The overlay home reader against a real git clone and the real node contract (OMN-22634)."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env

REPO_ROOT = Path(__file__).resolve().parents[2]
# Loaded by path under its own name, as the standalone agent loads it, so the agent's unit
# tests in the same session keep the module identity they raise and catch.
_SPEC = importlib.util.spec_from_file_location(
    "lane_env_overlay_integration",
    REPO_ROOT
    / "src/omnibase_infra/nodes/node_setup_local_provision_effect/runtime_lane_env_overlay.py",
)
assert _SPEC is not None and _SPEC.loader is not None
overlay = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(overlay)
LANE = "compose-lane-a"
ENV = {"COST_ESTIMATOR_WEB_SOURCE_REPOSITORY": "web-front-end"}


def _execution() -> ModuleType:
    module = ModuleType("executor_stub")
    module.__file__ = str(REPO_ROOT / "scripts/deploy-agent/deploy_agent/executor.py")
    return module


def _parse_fragment(text: str, *, lane_id: str, source: str) -> dict[str, str]:
    env: dict[str, str] = json.loads(text)["env"]
    return env


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        env=scrub_git_location_env(os.environ),
        check=True,
        capture_output=True,
        text=True,
    )


def _clone(root: Path, declared: dict[str, object]) -> Path:
    repo = root / "clone"
    target = repo / str(declared["overlay_path"])
    target.parent.mkdir(parents=True)
    document = {
        "schema_version": "runtime_lane_env.v1",
        "lanes": {LANE: {"description": "synthetic", "env": ENV}},
    }
    target.write_text(yaml.safe_dump(document), encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "test")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "overlay")
    _git(repo, "update-ref", f"refs/remotes/{declared['ref']}", "HEAD")
    return repo


@pytest.mark.integration
def test_lane_env_reads_overlay_from_a_real_clone(tmp_path: Path) -> None:
    declared = overlay.declaration(_execution())
    clone = _clone(tmp_path, dict(declared))
    environ = {str(declared["clone_env"]): str(clone)}

    merged = overlay.lane_env_with_overlay_home(
        _execution(),
        {},
        environ,
        lane_id=LANE,
        fragment_clone=tmp_path,
        parse_fragment=_parse_fragment,
    )
    assert merged == ENV

    unset = overlay.lane_env_with_overlay_home(
        _execution(),
        {},
        {},
        lane_id=LANE,
        fragment_clone=tmp_path,
        parse_fragment=_parse_fragment,
    )
    assert unset == {}
