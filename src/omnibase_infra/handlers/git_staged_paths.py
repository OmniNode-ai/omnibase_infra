# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Read the staged paths of a repository through git (OMN-22867).

The migration-sequence check node is a COMPUTE node, so the git read it needs
lives here, in the effect-side handlers package, and its runtime module calls it.

Outside a repository ``git diff --cached`` reads as ``git diff --no-index`` and
answers ``unknown option`` and a usage block (exit 129), which says nothing about
discovery. ``staged_paths`` asks ``git rev-parse --git-dir`` after a failed diff
and raises ``GitDiscoveryError`` with git's own ``fatal: not a git repository``
text when that fails too, so a caller can tell the environment's fault (git
cannot find the repository) from the change's.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env

__all__ = ["GitDiscoveryError", "staged_paths"]


class GitDiscoveryError(RuntimeError):
    """git cannot find a repository at the path that was asked about.

    ``diff_exit`` is the exit status of the failed ``git diff --cached`` and
    ``detail`` the stderr of ``git rev-parse --git-dir`` on the same path.
    """

    def __init__(self, diff_exit: int, detail: str) -> None:
        super().__init__(f"git diff --cached failed (exit {diff_exit}): {detail}")
        self.diff_exit = diff_exit
        self.detail = detail


def _git(
    repo_path: Path, timeout: float, label: str, *args: str
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=repo_path,
            env=scrub_git_location_env(),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("git executable not found") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{label} timed out") from exc


def staged_paths(repo_path: Path, timeout: float) -> tuple[str, ...]:
    """The paths staged in the repository at ``repo_path``.

    Raises:
        GitDiscoveryError: git cannot find a repository at ``repo_path``.
        RuntimeError: git is missing, timed out or failed for another reason.
    """
    completed = _git(
        repo_path, timeout, "git diff --cached", "diff", "--cached", "--name-only"
    )
    if completed.returncode != 0:
        probe = _git(
            repo_path, timeout, "git rev-parse --git-dir", "rev-parse", "--git-dir"
        )
        if probe.returncode != 0:
            raise GitDiscoveryError(completed.returncode, probe.stderr.strip())
        raise RuntimeError(
            f"git diff --cached failed (exit {completed.returncode}): {completed.stderr.strip()}"
        )
    return tuple(path.strip() for path in completed.stdout.splitlines() if path.strip())
