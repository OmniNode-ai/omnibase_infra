# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""A second registry root is guarded exactly like the first (OMN-19388).

The registry is moving from the directory `$OMNI_HOME` names to a second root
that holds a fresh canonical clone of every repository, while `OMNI_HOME` itself
stays unchanged until the move is finished. Before this change the git-hook
family decided "is this a canonical clone?" by one test -- a strict descendant
of `$OMNI_HOME` -- so a clone in the second root, even with `core.hooksPath`
pointed at the family, was treated as an unrelated repository: every commit and
every branch switch in it was permitted, and a permitted operation prints
nothing.

`ONEX_REGISTRY_ROOTS` names the extra roots, and the installer records every
root in a `registry-roots` file beside the live hooks so that an actor with no
environment at all (launchd, `env -i`, a GUI git client) is guarded the same way.
A clone whose own config points at the family but which sits in no root the hook
can resolve fails CLOSED. These tests drive real git commands against a
throwaway pair of roots with the family installed exactly as a host installs it.
Each `test_control_*` shows a command succeeding, so each refusal is known to
come from the guard and not from a fixture that broke git.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
HOOKS_DIR = REPO_ROOT / "scripts" / "git-hooks" / "canonical-clone"
INSTALLER = REPO_ROOT / "scripts" / "install-canonical-clone-git-hooks.sh"


def _git(
    *args: str, cwd: Path, env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        env=scrub_git_location_env(env),
        capture_output=True,
        text=True,
        check=False,
    )


def _env(home: Path, registry_roots: str | None) -> dict[str, str]:
    env = dict(os.environ)
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["GIT_CONFIG_SYSTEM"] = os.devnull
    env["GIT_AUTHOR_NAME"] = "OMN-19388 Test"
    env["GIT_AUTHOR_EMAIL"] = "omn19388@example.invalid"
    env["GIT_COMMITTER_NAME"] = env["GIT_AUTHOR_NAME"]
    env["GIT_COMMITTER_EMAIL"] = env["GIT_AUTHOR_EMAIL"]
    env["OMNI_HOME"] = str(home)
    for name in (
        "ALLOW_CANONICAL_CLONE_COMMIT",
        "ONEX_CANONICAL_CONVERGE",
        "ONEX_WORKTREES_ROOT",
        "ONEX_REGISTRY_ROOTS",
    ):
        env.pop(name, None)
    if registry_roots is not None:
        env["ONEX_REGISTRY_ROOTS"] = registry_roots
    return env


def _seed_clone(repo: Path, env: dict[str, str], *, guarded: bool) -> Path:
    repo.mkdir(parents=True)
    assert _git("init", "-q", "-b", "dev", cwd=repo, env=env).returncode == 0
    (repo / "seed.txt").write_text("seed\n", encoding="utf-8")
    assert _git("add", "seed.txt", cwd=repo, env=env).returncode == 0
    # Seeding happens before the hooks are pointed at, so it proves nothing
    # about them and needs no override.
    seeded = _git("commit", "-q", "-m", "seed", cwd=repo, env=env)
    assert seeded.returncode == 0, seeded.stderr
    if guarded:
        configured = _git("config", "core.hooksPath", str(HOOKS_DIR), cwd=repo, env=env)
        assert configured.returncode == 0, configured.stderr
    return repo


@pytest.fixture
def home(tmp_path: Path) -> Path:
    root = tmp_path / "omni_home"
    (root / "omni_worktrees").mkdir(parents=True)
    return root


@pytest.fixture
def new_root(tmp_path: Path) -> Path:
    root = tmp_path / "new_registry"
    (root / "omni_worktrees").mkdir(parents=True)
    return root


@pytest.fixture
def new_clone(home: Path, new_root: Path) -> Path:
    return _seed_clone(new_root / "some_repo", _env(home, None), guarded=True)


@pytest.fixture
def home_clone(home: Path) -> Path:
    return _seed_clone(home / "home_repo", _env(home, None), guarded=True)


def _try_commit(repo: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    (repo / "change.txt").write_text("change\n", encoding="utf-8")
    assert _git("add", "change.txt", cwd=repo, env=env).returncode == 0
    return _git("commit", "-q", "-m", "change", cwd=repo, env=env)


def _head(repo: Path, env: dict[str, str]) -> str:
    return _git("rev-parse", "HEAD", cwd=repo, env=env).stdout.strip()


def test_a_second_root_clone_is_refused_when_the_setting_is_absent(
    home: Path, new_clone: Path
) -> None:
    """The fail-open the first revision shipped, now closed. A clone whose OWN
    config points `core.hooksPath` at the family was put there by the installer,
    so a guard that cannot place it in any root it knows must refuse, never
    permit: an actor whose environment lacks ONEX_REGISTRY_ROOTS (launchd, a
    GUI git client, `env -i`) is exactly the actor these hooks exist for."""
    env = _env(home, None)
    before = _head(new_clone, env)

    result = _try_commit(new_clone, env)

    assert result.returncode != 0
    assert "blocked pre-commit in canonical clone" in result.stderr
    assert "outside every registry root" in result.stderr
    assert _head(new_clone, env) == before


def test_a_commit_in_a_second_root_clone_is_refused(
    home: Path, new_root: Path, new_clone: Path
) -> None:
    env = _env(home, f"{home}:{new_root}")
    before = _head(new_clone, env)

    result = _try_commit(new_clone, env)

    assert result.returncode != 0
    assert "blocked pre-commit in canonical clone" in result.stderr
    assert _head(new_clone, env) == before


def test_a_branch_switch_in_a_second_root_clone_is_refused(
    home: Path, new_root: Path, new_clone: Path
) -> None:
    """The reference-transaction guard reads the same setting."""
    env = _env(home, str(new_root))

    result = _git("checkout", "-q", "-b", "feature", cwd=new_clone, env=env)

    assert result.returncode != 0
    assert "refused" in result.stderr
    head = _git("symbolic-ref", "--quiet", "HEAD", cwd=new_clone, env=env)
    assert head.stdout.strip() == "refs/heads/dev"


def test_the_same_branch_switch_is_refused_when_the_setting_is_absent(
    home: Path, new_clone: Path
) -> None:
    env = _env(home, None)

    result = _git("checkout", "-q", "-b", "feature", cwd=new_clone, env=env)

    assert result.returncode != 0
    assert "outside every registry root" in result.stderr
    head = _git("symbolic-ref", "--quiet", "HEAD", cwd=new_clone, env=env)
    assert head.stdout.strip() == "refs/heads/dev"


def test_control_a_repo_on_a_host_wide_hooks_path_outside_every_root_is_untouched(
    home: Path, tmp_path: Path
) -> None:
    """The position rule reads the clone's OWN config. A host-wide (global)
    `core.hooksPath` reaching an unrelated repository stays inert there, so the
    rule cannot break ordinary work elsewhere on the machine."""
    global_config = tmp_path / "global.gitconfig"
    global_config.write_text(f"[core]\n\thooksPath = {HOOKS_DIR}\n", encoding="utf-8")
    env = _env(home, None)
    outside = _seed_clone(tmp_path / "elsewhere" / "repo", env, guarded=False)
    env["GIT_CONFIG_GLOBAL"] = str(global_config)
    before = _head(outside, env)

    result = _try_commit(outside, env)

    assert result.returncode == 0, result.stderr
    assert _head(outside, env) != before


def test_a_worktree_of_a_second_root_clone_under_the_sanctioned_root_commits(
    home: Path, new_root: Path, new_clone: Path
) -> None:
    """Worktrees of second-root clones live under $OMNI_HOME/omni_worktrees
    while OMNI_HOME is unchanged; linking one and committing in it works."""
    env = _env(home, f"{home}:{new_root}")
    worktree = home / "omni_worktrees" / "OMN-19388" / "some_repo"

    linked = _git(
        "worktree", "add", "-q", "-b", "wt", str(worktree), "HEAD",
        cwd=new_clone, env=env,
    )  # fmt: skip
    assert linked.returncode == 0, linked.stderr

    result = _try_commit(worktree, env)

    assert result.returncode == 0, result.stderr


def test_the_first_root_is_still_guarded_when_the_setting_is_present(
    home: Path, new_root: Path, home_clone: Path
) -> None:
    env = _env(home, str(new_root))
    before = _head(home_clone, env)

    result = _try_commit(home_clone, env)

    assert result.returncode != 0
    assert _head(home_clone, env) == before


def test_a_root_own_tree_is_not_itself_guarded(home: Path, new_root: Path) -> None:
    """Only the clones inside a root are guarded. A root that is itself a git
    repository commits as before; guarding it is a separate decision."""
    env = _env(home, str(new_root))
    assert _git("init", "-q", "-b", "main", cwd=new_root, env=env).returncode == 0
    assert (
        _git("config", "core.hooksPath", str(HOOKS_DIR), cwd=new_root, env=env)
    ).returncode == 0

    result = _try_commit(new_root, env)

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("bad", ["relative/root", "", "{missing}"])
def test_a_malformed_setting_fails_closed(
    home: Path, new_clone: Path, tmp_path: Path, bad: str
) -> None:
    """Rule 8: a declared root the hook cannot read is a refusal naming the
    setting, never a silently smaller guarded set."""
    value = bad.format(missing=tmp_path / "does-not-exist")
    env = _env(home, value)
    before = _head(new_clone, env)

    result = _try_commit(new_clone, env)

    assert result.returncode != 0
    assert "ONEX_REGISTRY_ROOTS" in result.stderr
    assert _head(new_clone, env) == before


def _install(
    *args: str, env: dict[str, str], cwd: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(INSTALLER), *args],
        cwd=cwd,
        env=scrub_git_location_env(env),
        capture_output=True,
        text=True,
        check=False,
    )


def test_the_installer_points_second_root_clones_at_the_family(
    home: Path, new_root: Path, tmp_path: Path
) -> None:
    env = _env(home, f"{home}:{new_root}")
    repo = _seed_clone(new_root / "unguarded_repo", env, guarded=False)

    readback = _install(env=env, cwd=tmp_path)
    assert readback.returncode == 3, readback.stderr
    assert f"canonical clones in {new_root}" in readback.stdout
    assert "UNSET    unguarded_repo" in readback.stdout

    applied = _install("--apply", env=env, cwd=tmp_path)
    assert applied.returncode == 0, applied.stderr

    hooks_path = _git("config", "--get", "core.hooksPath", cwd=repo, env=env)
    assert hooks_path.stdout.strip() == str(
        home / "scripts" / "git-hooks" / "canonical-clone"
    )
    again = _install(env=env, cwd=tmp_path)
    assert again.returncode == 0, again.stdout


def test_the_installer_refuses_a_malformed_setting(home: Path, tmp_path: Path) -> None:
    env = _env(home, "relative/root")

    result = _install(env=env, cwd=tmp_path)

    assert result.returncode == 1
    assert "REFUSED" in result.stderr
    assert "ONEX_REGISTRY_ROOTS" in result.stderr


# ---------------------------------------------------------------------------
# The installed roots file: the guard holds with no environment at all
# ---------------------------------------------------------------------------


def _bare_env(env: dict[str, str]) -> dict[str, str]:
    """What launchd, `env -i` or a GUI git client hands a hook: neither
    OMNI_HOME nor ONEX_REGISTRY_ROOTS."""
    bare = dict(env)
    bare.pop("OMNI_HOME", None)
    bare.pop("ONEX_REGISTRY_ROOTS", None)
    return bare


def _installed(home: Path, new_root: Path, tmp_path: Path) -> Path:
    """Install the family into the throwaway home the way a host does, with
    both roots declared, and return the live hooks directory."""
    env = _env(home, f"{home}:{new_root}")
    applied = _install("--apply", env=env, cwd=tmp_path)
    assert applied.returncode == 0, applied.stderr + applied.stdout
    return home / "scripts" / "git-hooks" / "canonical-clone"


def _point_at(repo: Path, hooks_dir: Path, env: dict[str, str]) -> None:
    configured = _git("config", "core.hooksPath", str(hooks_dir), cwd=repo, env=env)
    assert configured.returncode == 0, configured.stderr


def test_the_installer_records_the_roots_beside_the_live_hooks(
    home: Path, new_root: Path, tmp_path: Path
) -> None:
    _installed(home, new_root, tmp_path)

    recorded = (home / "scripts" / "git-hooks" / "registry-roots").read_text(
        encoding="utf-8"
    )

    assert f"omni_home={home.resolve()}" in recorded
    assert f"root={home.resolve()}" in recorded
    assert f"root={new_root.resolve()}" in recorded


def test_with_no_environment_the_installed_hook_refuses_a_second_root_commit(
    home: Path, new_root: Path, new_clone: Path, tmp_path: Path
) -> None:
    """The verifier's reproduction: the installed pre-commit hook in a
    second-root clone, run with neither variable set. The refusal names the
    worktree root the roots file records, not one guessed from the clone."""
    hooks = _installed(home, new_root, tmp_path)
    env = _bare_env(_env(home, None))
    _point_at(new_clone, hooks, env)
    before = _head(new_clone, env)

    result = _try_commit(new_clone, env)

    assert result.returncode != 0
    assert "blocked pre-commit in canonical clone" in result.stderr
    assert f"{home.resolve()}/omni_worktrees/<ticket>/<repo>" in result.stderr
    assert "outside every registry root" not in result.stderr
    assert _head(new_clone, env) == before


def test_with_no_environment_a_first_root_commit_is_still_refused(
    home: Path, new_root: Path, home_clone: Path, tmp_path: Path
) -> None:
    hooks = _installed(home, new_root, tmp_path)
    env = _bare_env(_env(home, None))
    _point_at(home_clone, hooks, env)
    before = _head(home_clone, env)

    result = _try_commit(home_clone, env)

    assert result.returncode != 0
    assert _head(home_clone, env) == before


def test_with_no_environment_a_worktree_commit_is_still_allowed(
    home: Path, new_root: Path, new_clone: Path, tmp_path: Path
) -> None:
    """Positive control for the refusals above: the same installed hook, the
    same bare environment, and a linked worktree commits."""
    hooks = _installed(home, new_root, tmp_path)
    env = _bare_env(_env(home, None))
    _point_at(new_clone, hooks, env)
    worktree = home / "omni_worktrees" / "OMN-19388" / "some_repo"
    linked = _git(
        "worktree", "add", "-q", "-b", "wt", str(worktree), "HEAD",
        cwd=new_clone, env=env,
    )  # fmt: skip
    assert linked.returncode == 0, linked.stderr

    result = _try_commit(worktree, env)

    assert result.returncode == 0, result.stderr


def test_a_malformed_roots_file_fails_closed(
    home: Path, new_root: Path, new_clone: Path, tmp_path: Path
) -> None:
    hooks = _installed(home, new_root, tmp_path)
    roots_file = home / "scripts" / "git-hooks" / "registry-roots"
    roots_file.write_text("root=relative/root\n", encoding="utf-8")
    env = _bare_env(_env(home, None))
    _point_at(new_clone, hooks, env)
    result = _try_commit(new_clone, env)

    assert result.returncode != 0
    assert "registry-roots" in result.stderr


def test_the_installer_keeps_a_recorded_root_when_the_setting_is_later_absent(
    home: Path, new_root: Path, tmp_path: Path
) -> None:
    """An installer re-run from a shell that lacks ONEX_REGISTRY_ROOTS must not
    quietly un-declare a root it recorded earlier."""
    _installed(home, new_root, tmp_path)

    rerun = _install("--apply", env=_env(home, None), cwd=tmp_path)
    assert rerun.returncode == 0, rerun.stderr

    recorded = (home / "scripts" / "git-hooks" / "registry-roots").read_text(
        encoding="utf-8"
    )
    assert f"root={new_root.resolve()}" in recorded
    readback = _install(env=_env(home, None), cwd=tmp_path)
    assert readback.returncode == 0, readback.stdout


def test_the_installer_readback_reports_a_missing_roots_file(
    home: Path, new_root: Path, tmp_path: Path
) -> None:
    _installed(home, new_root, tmp_path)
    (home / "scripts" / "git-hooks" / "registry-roots").unlink()

    readback = _install(env=_env(home, f"{home}:{new_root}"), cwd=tmp_path)

    assert readback.returncode == 3, readback.stdout
    assert "registry-roots" in readback.stdout
