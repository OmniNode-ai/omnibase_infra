# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The sibling-pin preflight and the pin node agree on a real git-derived clone (OMN-23947).

The handler is resolved through the node's contract, then the staging check runs as
the build runs it (a subprocess of the running interpreter) against real git
clones: a clone at or ahead of the locked commit passes and a clone behind it is
refused.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env
from omnibase_infra.models.sibling_pin import ModelSiblingClonePinRequest

pytestmark = pytest.mark.integration

_REPO = Path(__file__).resolve().parents[3]
_CONTRACT = (
    _REPO / "src/omnibase_infra/nodes/node_sibling_clone_pin_effect/contract.yaml"
)
_CHECK = _REPO / "scripts/runtime_build/check_sibling_lock_pins.py"
_PYPROJECT = (
    '[project]\nname = "omnibase_core"\ndynamic = ["version"]\n\n'
    "[tool.hatch.version.raw-options]\n"
    'git_describe_command = ["git", "describe", "--dirty", "--tags", "--long",'
    ' "--match", "v[0-9]*"]\n'
)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.email=t@example.test",
            "-c",
            "user.name=t",
            *args,
        ],
        check=True,
        capture_output=True,
        text=True,
        env=scrub_git_location_env(os.environ),
    ).stdout.strip()


def _derived_clone(tmp_path: Path) -> tuple[Path, str, str]:
    """A clone with a git-derived version: tagged first commit, one commit after."""
    root = tmp_path / "omnibase_core"
    root.mkdir()
    _git(root, "init", "-q")
    (root / "pyproject.toml").write_text(_PYPROJECT, encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "first")
    _git(root, "tag", "v1.2.3")
    first = _git(root, "rev-parse", "HEAD")
    _git(root, "commit", "-q", "--allow-empty", "-m", "second")
    return root, first, _git(root, "rev-parse", "HEAD")


def _lock(tmp_path: Path, commit: str) -> Path:
    lock = tmp_path / "uv.lock"
    lock.write_text(
        '[[package]]\nname = "omnibase-core"\nversion = "0.47.76"\n'
        f'source = {{ git = "https://example.test/core.git?tag=dev-x#{commit}" }}\n',
        encoding="utf-8",
    )
    return lock


def _run_check(
    tmp_path: Path, lock: Path, clone: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(_CHECK),
            "--lock",
            str(lock),
            "--repo",
            f"omnibase-core={clone}",
            "--output",
            str(tmp_path / "pins.json"),
            "--build-source",
            "workspace",
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_contract_routed_handler_answers_the_derived_pin(tmp_path: Path) -> None:
    clone, _first, head = _derived_clone(tmp_path)
    route = yaml.safe_load(_CONTRACT.read_text(encoding="utf-8"))["handler_routing"][
        "handlers"
    ][0]["handler"]
    handler = getattr(importlib.import_module(route["module"]), route["name"])()

    pin = handler.handle(
        ModelSiblingClonePinRequest(package="omnibase-core", repo_root=clone)
    )

    assert pin.version_derived is True
    assert pin.version.startswith("1.2.4.dev1+g")
    assert pin.git_sha == head


def test_preflight_accepts_a_derived_clone_ahead_of_the_locked_commit(
    tmp_path: Path,
) -> None:
    clone, first, _head = _derived_clone(tmp_path)

    result = _run_check(tmp_path, _lock(tmp_path, first), clone)

    assert result.returncode == 0, result.stdout + result.stderr
    entry = json.loads((tmp_path / "pins.json").read_text(encoding="utf-8"))[
        "comparisons"
    ][0]
    assert entry["version_derived"] is True
    assert entry["clone_ahead"] is True


def test_preflight_refuses_a_derived_clone_behind_the_locked_commit(
    tmp_path: Path,
) -> None:
    clone, _first, head = _derived_clone(tmp_path)
    _git(clone, "checkout", "-q", "--detach", "HEAD~1")

    result = _run_check(tmp_path, _lock(tmp_path, head), clone)

    assert result.returncode == 1, result.stdout + result.stderr
    entry = json.loads((tmp_path / "pins.json").read_text(encoding="utf-8"))[
        "comparisons"
    ][0]
    assert entry["matches"] is False
    assert "git SHA drift" in entry["mismatch_reason"]
