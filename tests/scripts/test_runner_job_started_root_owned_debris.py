# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""runner-job-started.sh must fail loud (not silently wedge) when root-owned
debris blocks the workspace-reset `rm -rf`, and must reset cleanly on the
normal (all-files-owned-by-the-hook-user) path (OMN-15134).

Live failure (run https://github.com/OmniNode-ai/omnibase_infra/actions/runs/30171892373,
2026-07-25T19:35Z): the built-in "Set up runner" job-started hook aborted
with 18 `Permission denied` lines against a `.venv` under the Actions job
workspace, before any workflow-defined step ran:

    rm: cannot remove '.../.venv/lib/python3.12/site-packages/pydantic/__pycache__/__init__.cpython-312.pyc': Permission denied
    ##[error]Process completed with exit code 1.

Root cause, confirmed directly on the omninode-deploy-runner container (not
inferred): the `.venv` was root-owned (mtime matching a same-day `uv run`
verification probe run via a bare `docker exec` -- no `-u runner` --
against this container, whose image ENTRYPOINT legitimately starts as root
for the docker-socket GID fix, so any ad hoc `docker exec` without an
explicit user silently inherits root). The hook itself runs as the
unprivileged `runner` user (the same identity every job step executes as),
so it correctly could not remove root-owned files -- but it also gave no
diagnostic and no recovery path, wedging every subsequent job on this
single-runner label until an operator hand-ran a root `docker exec rm -rf`.

These tests run the real script (not an extracted fragment -- it has no
internal functions) against a scratch HOME to prove:

1. the happy path (workspace owned by the invoking user) still resets clean.
2. a workspace containing a file the invoking user cannot delete (simulated
   via a read-only parent directory, since the test does not run as root and
   cannot fabricate genuine root ownership) makes the plain `rm -rf` fail,
   and the hook responds by (a) logging the failure loudly with the
   offending path, (b) attempting the scoped sudo self-heal fallback, and
   (c) exiting non-zero with an explicit manual-remediation instruction when
   that fallback is also unavailable -- never silently succeeding and never
   hanging.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HOOK_SCRIPT = REPO_ROOT / "docker" / "runners" / "runner-job-started.sh"


def _has_gnu_realpath_m() -> bool:
    """The hook uses `realpath -m` (GNU coreutils; resolve without requiring
    the path to exist). The production runner image is Ubuntu 22.04 (GNU
    coreutils) -- this always holds there. BSD/macOS `realpath` has no `-m`
    flag, so these subprocess-level tests skip on a bare macOS dev box
    rather than fail on an environment divergence unrelated to OMN-15134;
    they run for real on Linux CI and via the `.200` gate host."""
    try:
        result = subprocess.run(
            ["realpath", "-m", "."],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except OSError:
        return False
    return result.returncode == 0


pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(
        not _has_gnu_realpath_m(),
        reason="runner-job-started.sh requires GNU `realpath -m` (present on "
        "the Ubuntu 22.04 runner image and Linux CI; absent on BSD/macOS "
        "realpath) -- OMN-15134",
    ),
]


def _run_hook(
    runner_home: Path, workspace: Path, *, github_workspace: str | None = None
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["RUNNER_HOME"] = str(runner_home)
    env["GITHUB_WORKSPACE"] = (
        github_workspace if github_workspace is not None else str(workspace)
    )
    return subprocess.run(
        ["bash", str(HOOK_SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_script_exists_and_is_executable() -> None:
    assert HOOK_SCRIPT.is_file(), f"hook script missing: {HOOK_SCRIPT}"
    mode = HOOK_SCRIPT.stat().st_mode
    assert mode & stat.S_IXUSR, "hook script must be executable"


def test_happy_path_resets_workspace_owned_by_invoking_user(tmp_path: Path) -> None:
    """A workspace the hook's own user owns resets cleanly (regression guard --
    the OMN-15134 hardening must not break the common case)."""
    runner_home = tmp_path / "actions-runner"
    workspace = runner_home / "_work" / "omnibase_infra" / "omnibase_infra"
    workspace.mkdir(parents=True)
    (workspace / "stale_from_prior_job.txt").write_text("leftover")

    result = _run_hook(runner_home, workspace)

    assert result.returncode == 0, result.stderr
    assert workspace.is_dir()
    assert not (workspace / "stale_from_prior_job.txt").exists()


def test_undeletable_file_fails_loud_with_manual_remediation(tmp_path: Path) -> None:
    """Simulate the OMN-15134 condition: the hook's own user cannot delete a
    file under the workspace (here: via a read-only parent directory, the
    closest same-uid analog to genuine root-owned debris a non-root test
    process can fabricate). The hook must NOT silently succeed or hang -- it
    must fail loud, name the offending path, attempt the sudo fallback, and
    exit non-zero with a manual-remediation instruction when that fallback
    is unavailable (no sudoers rule exists in the test environment)."""
    runner_home = tmp_path / "actions-runner"
    workspace = runner_home / "_work" / "omnibase_infra" / "omnibase_infra"
    locked_dir = workspace / "locked"
    locked_dir.mkdir(parents=True)
    (locked_dir / "debris.txt").write_text("undeletable")
    # A read-only parent directory blocks unlink() of its children for
    # everyone including the owner (EACCES on the directory, not the file) --
    # the same externally-visible symptom as root-owned debris under a
    # writable parent: `rm -rf` fails partway through with Permission denied.
    locked_dir.chmod(0o555)

    try:
        result = _run_hook(runner_home, workspace)

        assert result.returncode != 0, (
            "hook must fail loud on undeletable debris, not silently succeed: "
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )
        assert "ERROR: rm -rf failed" in result.stderr
        assert "Attempting scoped root cleanup fallback" in result.stderr
        assert "Refusing to run the job against an unreset workspace" in result.stderr
        assert (
            "Manual remediation: docker exec -u root omninode-deploy-runner"
            in result.stderr
        )
    finally:
        # Restore permissions so pytest's own tmp_path cleanup can remove it.
        locked_dir.chmod(0o755)


def test_refuses_workspace_outside_work_root(tmp_path: Path) -> None:
    """Regression guard: the pre-existing path-confinement check must survive
    the OMN-15134 edit untouched."""
    runner_home = tmp_path / "actions-runner"
    (runner_home / "_work").mkdir(parents=True)
    outside = tmp_path / "not_under_work_root"
    outside.mkdir()

    result = _run_hook(runner_home, outside, github_workspace=str(outside))

    assert result.returncode == 1
    assert "Refusing to clean workspace outside" in result.stderr
    assert outside.is_dir()
