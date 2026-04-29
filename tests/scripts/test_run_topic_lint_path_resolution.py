# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for run_topic_lint.sh path resolution (OMN-9573).

Verifies that the hook resolves the omnibase_infra source tree without
requiring OMNIBASE_INFRA_PATH to be set, using OMNI_HOME and sibling-walk
fallbacks.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "validation" / "run_topic_lint.sh"
LINT = REPO_ROOT / "scripts" / "validation" / "lint_topic_names.py"


def copy_validation_scripts(root: Path) -> Path:
    """Copy the lint wrapper into a fake repo so script-dir resolution is not canonical."""
    scripts_dir = root / "scripts" / "validation"
    scripts_dir.mkdir(parents=True)
    shutil.copy2(SCRIPT, scripts_dir / "run_topic_lint.sh")
    shutil.copy2(LINT, scripts_dir / "lint_topic_names.py")
    return scripts_dir / "run_topic_lint.sh"


@pytest.mark.unit
def test_script_resolves_via_script_dir_canonical(tmp_path: Path) -> None:
    """Invoked from unrelated cwd, OMNI_HOME unset — resolves via SCRIPT_DIR."""
    env = {k: v for k, v in os.environ.items() if k != "OMNIBASE_INFRA_PATH"}
    env.pop("OMNI_HOME", None)
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        env=env,
        check=False,
        timeout=120,
    )
    # Must not exit 2 (runtime error from uv/missing path).
    # Exit 0 = clean; exit 1 = topic violations found (still a valid resolution).
    assert result.returncode in (0, 1), (
        f"Script failed with unexpected exit code {result.returncode}.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    # Must not emit the "could not locate" warning — path was found.
    assert "could not locate omnibase_infra source tree" not in result.stderr


@pytest.mark.unit
def test_script_resolves_via_omnibase_infra_path(tmp_path: Path) -> None:
    """OMNIBASE_INFRA_PATH remains the explicit compatibility override."""
    script = copy_validation_scripts(tmp_path / "tooling")
    fake_infra = tmp_path / "explicit_infra"
    source_root = fake_infra / "src" / "omnibase_infra"
    source_root.mkdir(parents=True)
    (source_root / "nodes").mkdir()

    env = {k: v for k, v in os.environ.items() if k != "OMNI_HOME"}
    env["OMNIBASE_INFRA_PATH"] = str(fake_infra)

    result = subprocess.run(
        ["bash", str(script)],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        env=env,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"Exit {result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "could not locate omnibase_infra source tree" not in result.stderr


@pytest.mark.unit
def test_script_resolves_via_omni_home(tmp_path: Path) -> None:
    """OMNI_HOME set to a fake root containing omnibase_infra/src/omnibase_infra."""
    script = copy_validation_scripts(tmp_path / "tooling")
    fake_infra = tmp_path / "omnibase_infra" / "src" / "omnibase_infra"
    fake_infra.mkdir(parents=True)
    (fake_infra / "nodes").mkdir()

    env = {k: v for k, v in os.environ.items() if k != "OMNIBASE_INFRA_PATH"}
    env["OMNI_HOME"] = str(tmp_path)

    # Run from a directory that is NOT the canonical infra root so SCRIPT_DIR
    # candidate (two levels up from scripts/validation) won't match.
    result = subprocess.run(
        ["bash", str(script)],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        env=env,
        check=False,
        timeout=120,
    )
    # Empty src tree → scan returns 0 (no topics to check).
    assert result.returncode == 0, (
        f"Exit {result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "could not locate omnibase_infra source tree" not in result.stderr


@pytest.mark.unit
def test_script_warns_and_exits_zero_when_no_infra_found(tmp_path: Path) -> None:
    """No canonical path, no OMNI_HOME, no sibling — must warn and exit 0."""
    script = copy_validation_scripts(tmp_path / "tooling")
    env = {k: v for k, v in os.environ.items() if k != "OMNIBASE_INFRA_PATH"}
    env.pop("OMNI_HOME", None)

    isolated = tmp_path / "isolated_repo"
    isolated.mkdir()

    result = subprocess.run(
        ["bash", str(script)],
        capture_output=True,
        text=True,
        cwd=str(isolated),
        env=env,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"Expected graceful exit 0, got {result.returncode}.\nstderr: {result.stderr}"
    )
    assert "could not locate omnibase_infra source tree" in result.stderr


@pytest.mark.unit
def test_script_resolves_via_sibling_walk(tmp_path: Path) -> None:
    """A sibling omnibase_infra/ directory found by walking up from cwd."""
    script = copy_validation_scripts(tmp_path / "tooling")
    sibling_infra = tmp_path / "omnibase_infra" / "src" / "omnibase_infra"
    sibling_infra.mkdir(parents=True)
    (sibling_infra / "nodes").mkdir()

    child_repo = tmp_path / "omniclaude" / "src"
    child_repo.mkdir(parents=True)

    env = {k: v for k, v in os.environ.items() if k != "OMNIBASE_INFRA_PATH"}
    env.pop("OMNI_HOME", None)

    result = subprocess.run(
        ["bash", str(script)],
        capture_output=True,
        text=True,
        cwd=str(child_repo),
        env=env,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"Exit {result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "could not locate omnibase_infra source tree" not in result.stderr
