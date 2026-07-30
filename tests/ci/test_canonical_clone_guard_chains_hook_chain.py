# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Regression guard for the canonical-clone hook guard's CHAINING behaviour
(OMN-15071, on top of the OMN-7018 worktree-discipline guard).

Root cause this prevents regressing: the guard is installed by pointing
`core.hooksPath` at `scripts/git-hooks/canonical-clone/`. `core.hooksPath`
REPLACES git's hook lookup -- git never falls back to `$GIT_COMMON_DIR/hooks/`
-- so any path through the guard that returns success without invoking the real
hook silently disables the ENTIRE hook chain for that repository.

The pre-fix revision did exactly that: `exit 0` for every worktree path. On
`.200`, which root CLAUDE.md rule 11a makes the DEFAULT host for pushes and gate
runs, every `git commit` in a worktree ran zero hooks and reported success, with
no output to distinguish it from a commit that had passed every gate. Live A/B
on `.200` 2026-07-30: an identical staged violation committed clean (exit 0,
zero hook output) under the pre-fix guard and was refused (exit 1, 135 lines of
hook output, HEAD unchanged) under the fixed one; an identical `git push`
created a remote ref with zero hook output under the pre-fix guard and was
refused with zero refs created under the fixed one.

Assertion classes:

1. Behavioural, chaining -- a commit the guard PERMITS must reach the real hook.
   Proven with a sentinel hook that records its own invocation, so the test
   fails if the guard returns success on its own. This is the assertion the
   pre-fix script fails.
2. Behavioural, canonical-clone protection -- a commit in the main worktree of a
   registry clone is still refused, and the sentinel must NOT have run. Chaining
   must not be bought by trading away the OMN-7018 guarantee.
3. Behavioural, both directions -- when the real hook passes, the commit lands.
   A guard that refuses everything would satisfy (2) alone.
4. Static -- the no-runnable-hook branch fails closed rather than reporting a
   vacuous pass.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GUARD_SCRIPT = REPO_ROOT / "scripts" / "git-hooks" / "canonical_clone_guard.sh"
HOOKS_DIR = REPO_ROOT / "scripts" / "git-hooks" / "canonical-clone"
HOOK_TYPES = ("pre-commit", "pre-push", "commit-msg", "pre-merge-commit")

_SENTINEL_NAME = "sentinel-ran"


def _git(
    *args: str, cwd: Path, env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _base_env(registry: Path) -> dict[str, str]:
    env = dict(os.environ)
    # Isolate from the developer's / runner's own git configuration so the test
    # asserts the guard's behaviour, not the ambient environment's.
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["GIT_CONFIG_SYSTEM"] = os.devnull
    env["GIT_AUTHOR_NAME"] = "OMN-15071 Test"
    env["GIT_AUTHOR_EMAIL"] = "omn15071@example.invalid"
    env["GIT_COMMITTER_NAME"] = env["GIT_AUTHOR_NAME"]
    env["GIT_COMMITTER_EMAIL"] = env["GIT_AUTHOR_EMAIL"]
    env["OMNI_HOME"] = str(registry)
    env.pop("ALLOW_CANONICAL_CLONE_COMMIT", None)
    return env


def _write_sentinel_hook(clone: Path, sentinel: Path, exit_code: int) -> None:
    """Install a real hook file at the location the guard must chain to."""
    hooks_dir = clone / ".git" / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)
    hook = hooks_dir / "pre-commit"
    hook.write_text(
        f'#!/usr/bin/env bash\nprintf "chained\\n" > "{sentinel}"\nexit {exit_code}\n',
        encoding="utf-8",
    )
    hook.chmod(0o755)


@pytest.fixture
def registry(tmp_path: Path) -> Path:
    """A miniature omni_home: `<registry>/<repo>` clone + `omni_worktrees/`."""
    reg = tmp_path / "omni_home"
    (reg / "omni_worktrees").mkdir(parents=True)
    return reg


@pytest.fixture
def clone(registry: Path) -> Path:
    """A canonical clone with `core.hooksPath` pointed at the guard, plus one
    commit so a linked worktree can be created from it."""
    repo = registry / "some_repo"
    repo.mkdir(parents=True)
    env = _base_env(registry)
    _git("init", "-q", "-b", "dev", cwd=repo, env=env)
    _git("config", "core.hooksPath", str(HOOKS_DIR), cwd=repo, env=env)
    (repo / "seed.txt").write_text("seed\n", encoding="utf-8")
    _git("add", "seed.txt", cwd=repo, env=env)
    # The guard is already active, so seed via the documented emergency override
    # rather than by disabling it -- this also exercises that the override still
    # works end to end.
    seed_env = dict(env)
    seed_env["ALLOW_CANONICAL_CLONE_COMMIT"] = "1"
    seeded = _git("commit", "-m", "seed", cwd=repo, env=seed_env)
    assert seeded.returncode == 0, seeded.stderr
    return repo


def _linked_worktree(clone: Path, registry: Path, env: dict[str, str]) -> Path:
    worktree = registry / "omni_worktrees" / "OMN-15071" / "some_repo"
    result = _git(
        "worktree",
        "add",
        "-q",
        "-b",
        "jonah/omn-15071-test",
        str(worktree),
        "HEAD",
        cwd=clone,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    return worktree


def test_guard_script_and_hook_symlinks_exist() -> None:
    assert GUARD_SCRIPT.is_file(), f"expected the guard at {GUARD_SCRIPT}"
    assert os.access(GUARD_SCRIPT, os.X_OK), f"{GUARD_SCRIPT} must be executable"
    for hook_type in HOOK_TYPES:
        link = HOOKS_DIR / hook_type
        assert link.is_symlink(), f"{link} must be a symlink to the guard"
        assert link.resolve() == GUARD_SCRIPT.resolve(), (
            f"{link} must resolve to {GUARD_SCRIPT}"
        )


def test_permitted_commit_chains_to_the_real_hook(
    registry: Path, clone: Path, tmp_path: Path
) -> None:
    """Assertion class 1 -- the class the pre-fix guard fails.

    A worktree commit the guard permits must REACH `$GIT_COMMON_DIR/hooks/`.
    The sentinel exits 1, so a guard that chains produces a refused commit AND
    a sentinel file; a guard that returns success on its own produces a landed
    commit and NO sentinel file.
    """
    env = _base_env(registry)
    worktree = _linked_worktree(clone, registry, env)
    sentinel = tmp_path / _SENTINEL_NAME
    _write_sentinel_hook(clone, sentinel, exit_code=1)

    (worktree / "change.txt").write_text("change\n", encoding="utf-8")
    _git("add", "change.txt", cwd=worktree, env=env)
    head_before = _git("rev-parse", "HEAD", cwd=worktree, env=env).stdout.strip()
    result = _git(
        "commit", "-m", "should be refused by the chained hook", cwd=worktree, env=env
    )
    head_after = _git("rev-parse", "HEAD", cwd=worktree, env=env).stdout.strip()

    assert sentinel.is_file(), (
        "the guard returned without invoking $GIT_COMMON_DIR/hooks/pre-commit -- "
        "core.hooksPath replaces git's hook lookup, so this silently disables "
        "the entire hook chain (OMN-15071)"
    )
    assert result.returncode != 0, (
        "the chained hook exited 1; the commit must be refused"
    )
    assert head_after == head_before, "a refused commit must not move HEAD"


def test_permitted_commit_succeeds_when_the_real_hook_passes(
    registry: Path, clone: Path, tmp_path: Path
) -> None:
    """Assertion class 3 -- chaining must not refuse legitimate work."""
    env = _base_env(registry)
    worktree = _linked_worktree(clone, registry, env)
    sentinel = tmp_path / _SENTINEL_NAME
    _write_sentinel_hook(clone, sentinel, exit_code=0)

    (worktree / "change.txt").write_text("change\n", encoding="utf-8")
    _git("add", "change.txt", cwd=worktree, env=env)
    head_before = _git("rev-parse", "HEAD", cwd=worktree, env=env).stdout.strip()
    result = _git("commit", "-m", "legitimate worktree commit", cwd=worktree, env=env)
    head_after = _git("rev-parse", "HEAD", cwd=worktree, env=env).stdout.strip()

    assert sentinel.is_file(), "the real hook must still be reached on the passing path"
    assert result.returncode == 0, f"legitimate commit was refused: {result.stderr}"
    assert head_after != head_before, "a permitted, clean commit must land"


def test_canonical_clone_commit_is_still_blocked_before_any_chaining(
    registry: Path, clone: Path, tmp_path: Path
) -> None:
    """Assertion class 2 -- OMN-7018 is not traded away for OMN-15071."""
    env = _base_env(registry)
    sentinel = tmp_path / _SENTINEL_NAME
    _write_sentinel_hook(clone, sentinel, exit_code=0)

    (clone / "change.txt").write_text("change\n", encoding="utf-8")
    _git("add", "change.txt", cwd=clone, env=env)
    head_before = _git("rev-parse", "HEAD", cwd=clone, env=env).stdout.strip()
    result = _git(
        "commit", "-m", "must be blocked in the canonical clone", cwd=clone, env=env
    )
    head_after = _git("rev-parse", "HEAD", cwd=clone, env=env).stdout.strip()

    assert result.returncode != 0, "a canonical-clone commit must be refused"
    assert "blocked pre-commit in canonical clone" in result.stderr, result.stderr
    assert head_after == head_before, "a blocked commit must not move HEAD"
    assert not sentinel.exists(), (
        "the canonical-clone refusal must short-circuit BEFORE the chain runs"
    )


def test_canonical_clone_override_still_chains(
    registry: Path, clone: Path, tmp_path: Path
) -> None:
    """The documented emergency override suppresses the refusal only -- it is
    not a hook-chain bypass (root CLAUDE.md rule #10)."""
    env = _base_env(registry)
    env["ALLOW_CANONICAL_CLONE_COMMIT"] = "1"
    sentinel = tmp_path / _SENTINEL_NAME
    _write_sentinel_hook(clone, sentinel, exit_code=1)

    (clone / "change.txt").write_text("change\n", encoding="utf-8")
    _git("add", "change.txt", cwd=clone, env=env)
    result = _git(
        "commit", "-m", "override must still run the hooks", cwd=clone, env=env
    )

    assert sentinel.is_file(), (
        "ALLOW_CANONICAL_CLONE_COMMIT must not skip the hook chain"
    )
    assert result.returncode != 0, (
        "the chained hook exited 1; the commit must be refused"
    )


def test_missing_hook_with_a_precommit_config_fails_closed() -> None:
    """Assertion class 4 -- static.

    `pre-commit install` refuses to write hook files while `core.hooksPath` is
    set, so "no installed hook" says nothing about "no configured hooks". The
    guard must never answer that situation with a silent success.
    """
    text = GUARD_SCRIPT.read_text(encoding="utf-8")
    assert "hook-impl" in text, (
        "expected a `pre-commit hook-impl` fallback when no hook file is installed"
    )
    assert "refuses to report a vacuous pass" in text, (
        "expected an explicit fail-closed branch when the chain cannot be run"
    )


def test_guard_contains_no_hardcoded_absolute_user_paths() -> None:
    """Root CLAUDE.md rule #6. The pre-fix guard defaulted OMNI_HOME to a
    literal `/Users/...` path and matched two more literal worktree roots."""
    text = GUARD_SCRIPT.read_text(encoding="utf-8")
    for forbidden in ("/Users/", "/Volumes/"):
        assert forbidden not in text, (
            f"hardcoded absolute path {forbidden!r} in {GUARD_SCRIPT}"
        )
