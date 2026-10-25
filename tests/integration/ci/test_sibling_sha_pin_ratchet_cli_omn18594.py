# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Exercise the sibling pin ratchet's CLI against real Git trees (OMN-18594)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env
from omnibase_infra.nodes.node_sibling_sha_pin_ratchet_compute.runtime_sibling_sha_pin_ratchet import (
    sha_lines,
)

pytestmark = pytest.mark.integration

_MODULE = (
    "omnibase_infra.nodes.node_sibling_sha_pin_ratchet_compute."
    "runtime_sibling_sha_pin_ratchet"
)
_WORKFLOW = ".github/workflows/sibling.yml"
_SHA = "0123456789abcdef0123456789abcdef01234567"
_PIN = f"uses: OmniNode-ai/omniclaude/.github/workflows/check.yml@{_SHA}\n"


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
        env=scrub_git_location_env(os.environ),
    )


def _run(root: Path, base: str = "origin/dev") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", _MODULE, "--repo-root", str(root), "--base", base],
        check=False,
        capture_output=True,
        text=True,
        env=scrub_git_location_env(os.environ),
    )


@pytest.mark.parametrize(
    "change", ["unchanged", "moved", "removed", "added", "new-file", "relocated"]
)
def test_cli_compares_tracked_working_tree_with_git_base(
    tmp_path: Path, change: str
) -> None:
    _git(tmp_path, "init", "-q", "-b", "dev")
    _git(tmp_path, "config", "user.name", "Test")
    _git(tmp_path, "config", "user.email", "test@example.invalid")
    workflow = tmp_path / _WORKFLOW
    workflow.parent.mkdir(parents=True)
    workflow.write_text(_PIN, encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "base")
    _git(tmp_path, "update-ref", "refs/remotes/origin/dev", "HEAD")

    if change == "moved":
        workflow.write_text(_PIN.replace(_SHA, "f" * 40), encoding="utf-8")
    elif change == "removed":
        workflow.write_text("# no sibling pin\n", encoding="utf-8")
    elif change == "added":
        workflow.write_text(_PIN * 2, encoding="utf-8")
    elif change in {"new-file", "relocated"}:
        (workflow.parent / "new.yml").write_text(_PIN, encoding="utf-8")
        if change == "relocated":
            workflow.unlink()
    _git(tmp_path, "add", "-A")

    result = _run(tmp_path)
    if change in {"added", "new-file", "relocated"}:
        assert result.returncode == 1, result.stdout + result.stderr
        assert "FAIL: internal-sibling commit-sha pins may only fall" in result.stderr
        gained = _WORKFLOW if change == "added" else ".github/workflows/new.yml"
        assert gained in result.stderr
        assert "sibling" in result.stderr
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert "PASS: no file gained" in result.stdout
    assert _run(tmp_path, "unresolvable-base").returncode == 2


@pytest.mark.parametrize(
    "path", [".github/workflows/a:b.yml", ".github/workflows/café.yml"]
)
def test_cli_detects_new_pin_in_unusual_path(tmp_path: Path, path: str) -> None:
    _git(tmp_path, "init", "-q", "-b", "dev")
    _git(tmp_path, "config", "user.name", "Test")
    _git(tmp_path, "config", "user.email", "test@example.invalid")
    _git(tmp_path, "config", "core.quotePath", "true")
    workflow = tmp_path / _WORKFLOW
    workflow.parent.mkdir(parents=True)
    workflow.write_text("# no sibling pin\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "base")
    _git(tmp_path, "update-ref", "refs/remotes/origin/dev", "HEAD")

    pin = f"ref: {_SHA}"
    (tmp_path / path).write_text(pin + "\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")

    result = _run(tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert path in result.stderr
    assert sha_lines(tmp_path, None) == ((path, pin),)

    _git(tmp_path, "commit", "-q", "-m", "add pin")
    _git(tmp_path, "update-ref", "refs/remotes/origin/dev", "HEAD")
    assert sha_lines(tmp_path, "origin/dev") == ((path, pin),)
