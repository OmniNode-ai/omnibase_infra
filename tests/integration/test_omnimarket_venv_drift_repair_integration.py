# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for the omnimarket venv drift repair script."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)

pytestmark = pytest.mark.integration

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "check-omnimarket-venv-drift.sh"


def _init_bare_remote(root: Path) -> Path:
    work = root / "work"
    work.mkdir()
    subprocess.run(
        ["git", "init", "--quiet", "-b", "dev"],
        cwd=work,
        check=True,
        env=_scrubbed_git_env(),
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=work,
        check=True,
        env=_scrubbed_git_env(),
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=work,
        check=True,
        env=_scrubbed_git_env(),
    )
    (work / "f.txt").write_text("x", encoding="utf-8")
    subprocess.run(
        ["git", "add", "f.txt"], cwd=work, check=True, env=_scrubbed_git_env()
    )
    subprocess.run(
        ["git", "commit", "--quiet", "-m", "init"],
        cwd=work,
        check=True,
        env=_scrubbed_git_env(),
    )

    bare = root / "bare.git"
    subprocess.run(
        ["git", "clone", "--quiet", "--bare", str(work), str(bare)],
        check=True,
        env=_scrubbed_git_env(),
    )
    return bare


def _clone_dev(bare_remote: Path, destination: Path) -> str:
    subprocess.run(
        ["git", "clone", "--quiet", str(bare_remote), str(destination)],
        check=True,
        env=_scrubbed_git_env(),
    )
    subprocess.run(
        ["git", "checkout", "--quiet", "dev"],
        cwd=destination,
        check=True,
        env=_scrubbed_git_env(),
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=destination,
        check=True,
        env=_scrubbed_git_env(),
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=destination,
        check=True,
        env=_scrubbed_git_env(),
    )
    return _head(destination)


def _head(repo: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        env=_scrubbed_git_env(),
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _scrubbed_git_env() -> dict[str, str]:
    """A git environment that cannot reach out of ``tmp_path`` (OMN-14891).

    git exports GIT_DIR / GIT_WORK_TREE / GIT_INDEX_FILE into every hook
    environment, and those OVERRIDE both ``cwd=`` and ``git -C``: this fixture
    builds throwaway repositories, so under a pre-commit or pre-push hook it
    would otherwise commit into the REAL invoking worktree. The keys are named
    literally as well as scrubbed, because the OMN-14891 guard verifies a
    module-local scrubber by reading the keys it drops and a delegated call is
    invisible to it.
    """
    env = scrub_git_location_env(os.environ)
    for key in (
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_COMMON_DIR",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    ):
        env.pop(key, None)
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    env["GIT_EDITOR"] = "true"
    return env


def _fake_python(root: Path, installed_sha: str) -> Path:
    """A target interpreter that reports `installed_sha` but is otherwise real.

    The shim used to answer EVERY invocation by echoing the sha. That stopped
    being a faithful stand-in when OMN-18663 made the script run its lock
    helper on the TARGET interpreter: the shim answered that call too, exiting
    0 without running the wrapped command, so the whole repair silently
    no-opped and the script still exited 0. The test then failed on the
    fast-forward assertion and read as a product regression, which it was not.

    The script uses the target interpreter for exactly two things, and they are
    distinguishable without guessing: the installed-distribution read feeds a
    program on STDIN (no script argument), while the lock helper is invoked
    with `heavy_lock.py` as its first argument. The first is what this test
    controls; the second is delegated to the real interpreter running the test,
    by absolute path, because the lock helper needs a modern stdlib and the
    PATH this test hands the script deliberately carries almost nothing.
    """
    shim = root / "fake_python.sh"
    shim.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ $# -gt 0 && -f ${1:-} ]]; then\n"
        f'  exec "{sys.executable}" "$@"\n'
        "fi\n"
        "cat >/dev/null\n"
        f'echo "{installed_sha}"\n',
        encoding="utf-8",
    )
    shim.chmod(0o755)
    return shim


def _script_harness(root: Path) -> tuple[Path, Path]:
    harness = root / "harness"
    harness.mkdir()

    script = harness / "check-omnimarket-venv-drift.sh"
    script.write_text(_SCRIPT.read_text(encoding="utf-8"), encoding="utf-8")
    script.chmod(0o755)

    # OMN-18627: the script resolves its helpers relative to its OWN directory
    # (`source "$SCRIPT_DIR/lib/venv_reconcile_lock.sh"`), so a harness that
    # copies the script without its `lib/` leaves it sourcing a path that does
    # not exist. OMN-18663 added that source line and this copy did not follow,
    # which made the test fail on `origin/dev` with a shell "No such file or
    # directory" -- a harness defect wearing the shape of a product failure.
    #
    # The WHOLE directory is copied rather than the one file this script
    # currently sources: the next helper added beside it is then carried
    # automatically, instead of reproducing this same failure once more.
    # `lib/venv_reconcile_lock.sh` in turn resolves `../heavy_lock.py` relative
    # to ITSELF, so the harness has to reproduce the two-level layout, not just
    # the one file. Both are copied by NAME here rather than by globbing the
    # whole of `scripts/`, which would pull an unbounded tree into every temp
    # dir; a helper added beside either is a one-line addition to this tuple
    # and, unlike the previous version of this harness, fails loudly below
    # rather than silently producing a script that cannot source itself.
    for rel in ("lib/venv_reconcile_lock.sh", "heavy_lock.py"):
        source = _SCRIPT.parent / rel
        assert source.is_file(), (
            f"{source} is missing, so this harness cannot reproduce the "
            "script's own directory layout and the failure it produces would "
            "be a harness defect wearing the shape of a product failure"
        )
        destination = harness / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    marker = harness / "install_calls.log"
    installer = harness / "install-node-skill-package.sh"
    installer.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "ref=${{OMNIMARKET_REF:-}} args=$*" >> "{marker}"\n',
        encoding="utf-8",
    )
    installer.chmod(0o755)
    return script, marker


def test_repair_fast_forwards_canonical_clone_before_install(tmp_path: Path) -> None:
    bare = _init_bare_remote(tmp_path)
    omni_home = tmp_path / "omni_home"
    omni_home.mkdir()
    omnimarket_root = omni_home / "omnimarket"
    behind_sha = _clone_dev(bare, omnimarket_root)

    advance_clone = tmp_path / "advance_clone"
    _clone_dev(bare, advance_clone)
    (advance_clone / "f2.txt").write_text("y", encoding="utf-8")
    subprocess.run(
        ["git", "add", "f2.txt"], cwd=advance_clone, check=True, env=_scrubbed_git_env()
    )
    subprocess.run(
        ["git", "commit", "--quiet", "-m", "advance"],
        cwd=advance_clone,
        check=True,
        env=_scrubbed_git_env(),
    )
    subprocess.run(
        ["git", "push", "--quiet"],
        cwd=advance_clone,
        check=True,
        env=_scrubbed_git_env(),
    )
    ahead_sha = _head(advance_clone)

    script, marker = _script_harness(tmp_path)
    result = subprocess.run(
        ["bash", str(script), "--repair", str(_fake_python(tmp_path, behind_sha))],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
        env={"PATH": "/usr/bin:/bin", "OMNI_HOME": str(omni_home)},
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert _head(omnimarket_root) == ahead_sha
    assert f"ref={ahead_sha}" in marker.read_text(encoding="utf-8")
