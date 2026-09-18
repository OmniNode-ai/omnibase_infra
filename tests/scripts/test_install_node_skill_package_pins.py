# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests that install-node-skill-package.sh reads its pins from the ref (OMN-18675).

Before this change the script carried two shell literals —
``omnibase-compat==0.5.5`` and ``omninode-memory==0.15.0``, written 2026-07-02
and never revisited — and installed them with ``--no-deps`` and an exact ``==``.
That is an instruction, not a floor, so a repair run in September 2026 forced a
healthy shared plugin CLI venv back to July versions and broke ``onex``
host-wide.

These tests are hermetic: the "canonical clone" is a local git repo holding a
synthetic ``pyproject.toml``, and ``uv`` is a fake on PATH that records its
argv and prints a canned plan. Nothing real is installed and no network is
used.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "install-node-skill-package.sh"

_EXIT_REFUSED = 3

# The pins the ref declares — deliberately NEWER than the literals the script
# used to carry, which is exactly the situation that produced the downgrade.
_PYPROJECT = """\
[project]
name = "omnimarket"
version = "0.4.118"
dependencies = [
    "omnibase-core>=0.47.14,<0.48.0",
    "omnibase-compat==0.5.7",
    "omninode-memory==0.18.0",
    "anthropic>=0.25.0",
]
"""

_SAFE_PLAN = """\
Resolved 3 packages in 120ms
 - omnimarket==0.4.117
 + omnimarket==0.4.118
"""

_DOWNGRADE_PLAN = """\
Resolved 3 packages in 120ms
 - omnibase-compat==0.5.7
 + omnibase-compat==0.5.5
"""


def _make_clone(root: Path, pyproject: str) -> tuple[Path, str]:
    """Create $OMNI_HOME/omnimarket holding ``pyproject``; return (path, sha)."""
    omni_home = root / "omni_home"
    clone = omni_home / "omnimarket"
    clone.mkdir(parents=True)

    def run(*argv: str) -> None:
        # A git hook exports GIT_DIR/GIT_WORK_TREE, which override cwd= and
        # would retarget the REAL worktree (OMN-14891). Scrub them.
        subprocess.run(
            argv,
            cwd=clone,
            check=True,
            capture_output=True,
            env=scrub_git_location_env(),
        )

    run("git", "init", "--quiet", "-b", "dev")
    run("git", "config", "user.email", "test@example.com")
    run("git", "config", "user.name", "Test")
    (clone / "pyproject.toml").write_text(pyproject, encoding="utf-8")
    run("git", "add", "pyproject.toml")
    run("git", "commit", "--quiet", "-m", "pyproject")
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=clone,
        check=True,
        capture_output=True,
        text=True,
        env=scrub_git_location_env(),
    ).stdout.strip()
    return omni_home, sha


def _write_fake_uv(root: Path, plan: str) -> tuple[Path, Path]:
    bin_dir = root / "bin"
    bin_dir.mkdir(exist_ok=True)
    argv_log = root / "uv-argv.log"
    plan_file = root / "plan.txt"
    plan_file.write_text(plan, encoding="utf-8")
    fake = bin_dir / "uv"
    fake.write_text(
        "#!/usr/bin/env bash\n"
        f'printf "%s\\n" "$*" >> "{argv_log}"\n'
        f'cat "{plan_file}"\n'
        "exit 0\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    return bin_dir, argv_log


def _run_script(
    *args: str, omni_home: Path, ref: str, bin_dir: Path
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env["OMNI_HOME"] = str(omni_home)
    env["OMNIMARKET_REF"] = ref
    return subprocess.run(
        ["bash", str(_SCRIPT), *args, sys.executable],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_pins_are_read_from_the_ref_not_from_shell_literals(tmp_path: Path) -> None:
    """AC1: the co-installed versions come from the ref's own pyproject.toml."""
    omni_home, sha = _make_clone(tmp_path, _PYPROJECT)
    bin_dir, argv_log = _write_fake_uv(tmp_path, _SAFE_PLAN)

    result = _run_script(omni_home=omni_home, ref=sha, bin_dir=bin_dir)
    assert result.returncode == 0, result.stdout + result.stderr

    invocations = argv_log.read_text(encoding="utf-8")
    assert "omnibase-compat==0.5.7" in invocations
    assert "omninode-memory==0.18.0" in invocations
    # The stale literals must be gone from the invocation AND from the source.
    assert "omnibase-compat==0.5.5" not in invocations
    assert "omninode-memory==0.15.0" not in invocations


def test_the_stale_literals_are_not_in_the_script_source() -> None:
    """The recurrence mechanism was a baked-in version; keep it deleted."""
    source = _SCRIPT.read_text(encoding="utf-8")
    assert 'COMPAT_PIN="omnibase-compat==' not in source
    assert 'MEMORY_PIN="omninode-memory==' not in source


def test_a_repair_that_would_downgrade_a_sibling_is_refused(tmp_path: Path) -> None:
    """AC3 falsifier: drive --execute against a plan that downgrades a sibling."""
    omni_home, sha = _make_clone(tmp_path, _PYPROJECT)
    bin_dir, argv_log = _write_fake_uv(tmp_path, _DOWNGRADE_PLAN)

    result = _run_script("--execute", omni_home=omni_home, ref=sha, bin_dir=bin_dir)
    assert result.returncode == _EXIT_REFUSED, result.stdout + result.stderr
    assert "REFUSED" in result.stderr
    assert "omnibase-compat" in result.stderr

    # Refused BEFORE mutating: uv was only ever asked for a plan.
    invocations = argv_log.read_text(encoding="utf-8").splitlines()
    assert invocations, "uv was never invoked"
    assert all("--dry-run" in line for line in invocations), invocations


def test_a_plan_run_never_mutates_the_venv(tmp_path: Path) -> None:
    """Without --execute the script resolves a real plan and applies nothing."""
    omni_home, sha = _make_clone(tmp_path, _PYPROJECT)
    bin_dir, argv_log = _write_fake_uv(tmp_path, _SAFE_PLAN)

    result = _run_script(omni_home=omni_home, ref=sha, bin_dir=bin_dir)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "DRY RUN" in result.stdout
    invocations = argv_log.read_text(encoding="utf-8").splitlines()
    assert all("--dry-run" in line for line in invocations), invocations


def test_missing_omni_home_fails_fast_rather_than_defaulting(tmp_path: Path) -> None:
    """Rule 8: no silent fallback to a baked-in pin set."""
    bin_dir, _ = _write_fake_uv(tmp_path, _SAFE_PLAN)
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env.pop("OMNI_HOME", None)
    env["OMNIMARKET_REF"] = "0" * 40
    result = subprocess.run(
        ["bash", str(_SCRIPT), sys.executable],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 1
    assert "OMNI_HOME" in result.stderr


def test_a_ref_declaring_none_of_the_pins_is_refused(tmp_path: Path) -> None:
    """Never guess a version for a package the ref does not declare."""
    omni_home, sha = _make_clone(
        tmp_path, '[project]\nname = "omnimarket"\nversion = "0.1"\ndependencies = []\n'
    )
    bin_dir, _ = _write_fake_uv(tmp_path, _SAFE_PLAN)
    result = _run_script(omni_home=omni_home, ref=sha, bin_dir=bin_dir)
    assert result.returncode == 1
    assert "Refusing to guess" in result.stderr
