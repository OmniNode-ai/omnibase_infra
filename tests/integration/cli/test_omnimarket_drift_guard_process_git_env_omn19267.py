# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-19267: the drift guard, run in a real child process whose environment
exports ``GIT_DIR`` / ``GIT_WORK_TREE`` for another repository, reads the NAMED
omnimarket clone.

The unit test of the same behaviour patches ``os.environ`` in-process. This one
crosses the process boundary the way a lane does: a launcher exports the git
repository variables for its own detached worktree, and the guard, started as a
fresh interpreter against real ``git`` repositories on disk, must still report
the clone it was pointed at.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

# The child imports the package from the source tree: hosted CI runs from a
# checkout that does not install omnibase_infra into the interpreter.
_SRC = Path(__file__).resolve().parents[3] / "src"
_PATH = "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"

_GIT_ENV = {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.invalid",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.invalid",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_SYSTEM": "/dev/null",
}

_PROBE = """
import json, sys
from omnibase_infra.cli import omnimarket_drift_guard as guard

home = sys.argv[1]
print(json.dumps({
    "attachment": guard.canonical_clone_attachment(home).name,
    "commit": guard.canonical_local_omnimarket_commit(home),
}))
"""


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True,
        text=True,
        check=True,
        env={"PATH": _PATH, "HOME": str(cwd), **_GIT_ENV},
        timeout=30,
    ).stdout.strip()


def _repo(path: Path, *, detach: bool) -> str:
    path.mkdir()
    _git(path, "init", "-q", "-b", "dev")
    (path / "f.txt").write_text(f"{path.name}\n", encoding="utf-8")
    _git(path, "add", "f.txt")
    _git(path, "commit", "-q", "-m", f"commit in {path.name}")
    sha = _git(path, "rev-parse", "HEAD")
    if detach:
        _git(path, "checkout", "-q", "--detach")
    return sha


def test_a_child_process_with_exported_git_variables_reads_the_named_clone(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    clone_sha = _repo(home / "omnimarket", detach=False)
    lane = tmp_path / "lane"
    _repo(lane, detach=True)

    child_env = {
        "PATH": _PATH,
        "HOME": str(tmp_path),
        "PYTHONPATH": str(_SRC),
        **_GIT_ENV,
        "GIT_DIR": str(lane / ".git"),
        "GIT_WORK_TREE": str(lane),
        "GIT_INDEX_FILE": str(lane / ".git" / "index"),
    }
    result = subprocess.run(
        [sys.executable, "-c", _PROBE, str(home)],
        capture_output=True,
        text=True,
        check=False,
        env=child_env,
        timeout=120,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report == {"attachment": "ATTACHED", "commit": clone_sha}
