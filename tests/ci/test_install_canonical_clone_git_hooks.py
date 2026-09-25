# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Behaviour of the canonical-clone git-hook installer (OMN-16497).

Until this installer existed the git-hook family was placed on a host BY HAND.
The live copy on the development Mac happened to be byte-identical to the
tracked source, which is luck rather than a property: nothing verified it, and
nothing would have reported a host that had drifted or had never been installed
at all. The first readback of the real machine found thirteen canonical clones
with `core.hooksPath` unset -- enforcing nothing -- including the one whose
drift the 2026-09-13 friction report had already reported as a live specimen.

So the readback is the point, and these tests are mostly about it: that it
reports the truth, and that it refuses rather than guesses when it cannot.
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
INSTALLER = REPO_ROOT / "scripts" / "install-canonical-clone-git-hooks.sh"

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_PENDING = 3


def _run(
    *args: str, env: dict[str, str], cwd: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(INSTALLER), *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _clean_env(registry: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["OMNI_HOME"] = str(registry)
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["GIT_CONFIG_SYSTEM"] = os.devnull
    env.pop("ONEX_REGISTRY_ROOTS", None)
    for leaked in (
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_COMMON_DIR",
        "GIT_INDEX_FILE",
    ):
        env.pop(leaked, None)
    return env


@pytest.fixture
def registry(tmp_path: Path) -> Path:
    """A miniature registry with one canonical clone that enforces nothing."""
    reg = tmp_path / "omni_home"
    repo = reg / "some_repo"
    repo.mkdir(parents=True)
    env = _clean_env(reg)
    subprocess.run(
        ["git", "init", "-q", "-b", "dev", "."],
        cwd=repo,
        env=scrub_git_location_env(env),
        check=True,
        capture_output=True,
    )
    return reg


def test_installer_is_executable() -> None:
    assert INSTALLER.is_file()
    assert os.access(INSTALLER, os.X_OK)


def test_readback_reports_an_uninstalled_clone_and_changes_nothing(
    registry: Path, tmp_path: Path
) -> None:
    env = _clean_env(registry)

    result = _run(env=env, cwd=tmp_path)

    assert result.returncode == EXIT_PENDING, result.stderr
    assert "some_repo" in result.stdout
    assert "core.hooksPath is not set" in result.stdout
    assert "ACTION PENDING" in result.stdout
    # The readback is read-only: nothing was created and no clone was pointed.
    assert not (registry / "scripts" / "git-hooks").exists()
    current = subprocess.run(
        ["git", "-C", str(registry / "some_repo"), "config", "--get", "core.hooksPath"],
        env=scrub_git_location_env(env),
        capture_output=True,
        text=True,
        check=False,
    )
    assert current.stdout.strip() == ""


def test_apply_installs_the_family_and_the_readback_then_passes(
    registry: Path, tmp_path: Path
) -> None:
    """Applied into a THROWAWAY registry, never the real one. The fleet-wide
    install on a real host is an operator action."""
    env = _clean_env(registry)

    applied = _run("--apply", env=env, cwd=tmp_path)
    assert applied.returncode == EXIT_OK, applied.stderr

    hooks_dir = registry / "scripts" / "git-hooks" / "canonical-clone"
    ref_link = hooks_dir / "reference-transaction"
    assert ref_link.is_symlink()
    assert ref_link.resolve().name == "canonical_clone_ref_guard.sh"
    # The sibling hook types are NOT repointed: the directory composes by type.
    assert (hooks_dir / "pre-commit").resolve().name == "canonical_clone_guard.sh"

    readback = _run(env=env, cwd=tmp_path)
    assert readback.returncode == EXIT_OK, readback.stdout
    assert "every canonical clone is installed and current" in readback.stdout


def test_drifted_live_copy_is_reported_and_backed_up_on_apply(
    registry: Path, tmp_path: Path
) -> None:
    env = _clean_env(registry)
    assert _run("--apply", env=env, cwd=tmp_path).returncode == EXIT_OK

    live = registry / "scripts" / "git-hooks" / "canonical_clone_ref_guard.sh"
    live.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")

    readback = _run(env=env, cwd=tmp_path)
    assert readback.returncode == EXIT_PENDING
    assert "DRIFTED  canonical_clone_ref_guard.sh" in readback.stdout

    repaired = _run("--apply", env=env, cwd=tmp_path)
    assert repaired.returncode == EXIT_OK
    assert "previous copy kept at" in repaired.stdout
    # The drifted copy is preserved, never discarded.
    backups = list(live.parent.glob("canonical_clone_ref_guard.sh.bak.*"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == "#!/usr/bin/env bash\nexit 0\n"


@pytest.mark.parametrize(
    "leaked", ["GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE"]
)
def test_a_leaked_git_variable_is_a_refusal_not_a_guess(
    registry: Path, tmp_path: Path, leaked: str
) -> None:
    """The 2026-09-13 incident class: git exports these into hook processes and
    they override both `-C` and the cwd for every descendant git call. A leaked
    GIT_DIR pointed an installer at the shared canonical-clone hooks directory
    and every clone on the machine briefly refused commits from unregistered
    worktrees. An installer that reads a leaked variable does not install what
    it reports, so it must refuse rather than proceed."""
    env = _clean_env(registry)
    env[leaked] = str(tmp_path / "somewhere-else")

    result = _run("--apply", env=env, cwd=tmp_path)

    assert result.returncode == EXIT_ERROR
    assert "REFUSED" in result.stderr
    assert leaked in result.stderr
    assert not (registry / "scripts").exists()


def test_missing_omni_home_is_a_refusal_not_a_default(tmp_path: Path) -> None:
    """Rule 8: fail fast on missing env, never a silent fallback. A default
    registry path here would install the fleet's enforcement surface into
    whatever directory the guess happened to name."""
    env = dict(os.environ)
    env.pop("OMNI_HOME", None)
    for leaked in ("GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE"):
        env.pop(leaked, None)

    result = _run(env=env, cwd=tmp_path)

    assert result.returncode == EXIT_ERROR
    assert "OMNI_HOME" in result.stderr


def test_a_linked_worktree_is_skipped_not_installed_into(
    registry: Path, tmp_path: Path
) -> None:
    """Worktrees share their clone's config; pointing one at the hooks dir
    separately would be a second, divergent installation of the same family."""
    env = _clean_env(registry)
    repo = registry / "some_repo"
    (repo / "seed.txt").write_text("seed\n", encoding="utf-8")
    for args in (
        ["add", "seed.txt"],
        ["-c", "user.email=t@e.invalid", "-c", "user.name=T", "commit", "-qm", "seed"],
    ):
        subprocess.run(
            ["git", "-C", str(repo), *args],
            env=scrub_git_location_env(env),
            check=True,
            capture_output=True,
        )
    worktree = registry / "omni_worktrees" / "OMN-16497" / "some_repo"
    worktree.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "-C", str(repo), "worktree", "add", "-q", str(worktree), "-b", "wt"],
        env=scrub_git_location_env(env),
        check=True,
        capture_output=True,
    )

    result = _run("OMN-16497", env=env, cwd=tmp_path)

    assert "SKIP" in result.stdout
    assert "not a canonical clone" in result.stdout
