# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Root resolution in `scripts/git-maintenance.sh` (OMN-19396).

The script used to default OMNI_HOME and WORKTREE_ROOT to machine paths on an
old external disk, the second of them a sibling `omni_worktrees` beside the
registry root. That sibling is a stray root (operator ruling 2026-09-24): a
script that silently scans or populates it is how work keeps landing there.

Now OMNI_HOME is required (rule 8) and WORKTREE_ROOT is derived from it. Every
test drives the real script against a throwaway registry.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "git-maintenance.sh"


def _base_env() -> dict[str, str]:
    env = dict(os.environ)
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["GIT_CONFIG_SYSTEM"] = os.devnull
    env["GIT_AUTHOR_NAME"] = "OMN-19396 Test"
    env["GIT_AUTHOR_EMAIL"] = "omn19396@example.invalid"
    env["GIT_COMMITTER_NAME"] = env["GIT_AUTHOR_NAME"]
    env["GIT_COMMITTER_EMAIL"] = env["GIT_AUTHOR_EMAIL"]
    env.pop("OMNI_HOME", None)
    env.pop("WORKTREE_ROOT", None)
    env.pop("GIT_PREFIX", None)
    return scrub_git_location_env(env)


def _git(*args: str, cwd: Path, env: dict[str, str]) -> None:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        env=scrub_git_location_env(env),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"git {args}: {result.stderr}"


def _run(env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        env=scrub_git_location_env(env),
        capture_output=True,
        text=True,
        check=False,
    )


def test_refuses_to_run_without_omni_home() -> None:
    """Fail fast: no OMNI_HOME means no guessed registry and no guessed root."""
    result = _run(_base_env(), "--dry-run")
    assert result.returncode != 0, result.stdout
    assert "OMNI_HOME must be set" in result.stderr, result.stderr
    assert "=== Worktree Cleanup ===" not in result.stdout, result.stdout


def test_worktree_root_defaults_to_omni_worktrees_inside_omni_home(
    tmp_path: Path,
) -> None:
    """With WORKTREE_ROOT unset, the root the script prunes is
    $OMNI_HOME/omni_worktrees, and a sibling omni_worktrees is never touched."""
    registry = tmp_path / "omni_home"
    registry.mkdir()
    env = _base_env()
    env["OMNI_HOME"] = str(registry)

    clone = registry / "fixture_repo"
    clone.mkdir()
    _git("init", "-q", "-b", "main", cwd=clone, env=env)
    (clone / "root.txt").write_text("root\n", encoding="utf-8")
    _git("add", "-A", cwd=clone, env=env)
    _git("commit", "-q", "-m", "seed", cwd=clone, env=env)

    inside = registry / "omni_worktrees" / "OMN-INSIDE" / "fixture_repo"
    _git("worktree", "add", "-q", str(inside), "-b", "wt/inside", cwd=clone, env=env)

    # A stray sibling root holding a ticket directory: it must survive.
    stray_ticket = tmp_path / "omni_worktrees" / "OMN-STRAY"
    stray_ticket.mkdir(parents=True)
    (stray_ticket / "marker.txt").write_text("stray\n", encoding="utf-8")

    result = _run(env, "--execute", "--prune-worktrees")
    assert result.returncode == 0, result.stderr
    assert "[removed] OMN-INSIDE" in result.stdout, result.stdout
    assert not inside.exists()
    assert "OMN-STRAY" not in result.stdout, result.stdout
    assert (stray_ticket / "marker.txt").exists()


def test_control_explicit_worktree_root_still_wins(tmp_path: Path) -> None:
    """Positive control: an explicit WORKTREE_ROOT is honoured, so the default
    above is a default and not a hardcoded path."""
    registry = tmp_path / "omni_home"
    registry.mkdir()
    elsewhere = tmp_path / "explicit_root" / "OMN-EXPLICIT"
    elsewhere.mkdir(parents=True)
    env = _base_env()
    env["OMNI_HOME"] = str(registry)
    env["WORKTREE_ROOT"] = str(tmp_path / "explicit_root")

    result = _run(env, "--dry-run", "--prune-worktrees")
    assert result.returncode == 0, result.stderr
    assert "[would remove] OMN-EXPLICIT" in result.stdout, result.stdout


def test_script_names_no_machine_path() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "/Volumes/" not in text
    assert "/Users/" not in text
