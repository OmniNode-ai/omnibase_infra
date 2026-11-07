# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The hostile-review gate against a really rebased PR (OMN-17427).

A rebase rewrites every sha of a PR, so a reviewer thread whose ``fixed-in``
reply names the pre-rebase sha is no longer an ancestor of the head. These tests
build that history with real ``git`` (commit, move the base, ``git rebase``) and
send review threads through the gate's own ``classify_threads`` with the real
``GitFixVerifier``, so the verdict the merge sees is the one asserted. Nothing
is mocked: the patch-id match in ``_rewritten_as`` is what clears the thread,
and a fix that is not in the PR, or one that never touched the cited lines, must
still block it.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env

pytestmark = pytest.mark.integration

_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "ci"
    / "check_hostile_review_threads.py"
)
_spec = importlib.util.spec_from_file_location("check_hostile_review_threads", _SCRIPT)
assert _spec is not None and _spec.loader is not None
gate = importlib.util.module_from_spec(_spec)
sys.modules["check_hostile_review_threads"] = gate
_spec.loader.exec_module(gate)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        env=scrub_git_location_env(os.environ),
    ).stdout.strip()


def _is_ancestor(repo: Path, older: str, newer: str) -> bool:
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", older, newer],
            cwd=repo,
            env=scrub_git_location_env(os.environ),
            check=False,
        ).returncode
        == 0
    )


class _Pr:
    def __init__(self, repo: Path, base: str, old_fix: str, head: str, reviewed: str):
        self.repo = repo
        self.base = base
        self.old_fix = old_fix
        self.head = head
        self.reviewed = reviewed


def _rebased_pr(tmp_path: Path) -> _Pr:
    """A PR with a reviewed commit and a fix of line 20; dev moves; the PR rebases."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "dev")
    _git(repo, "config", "user.email", "ci@example.invalid")
    _git(repo, "config", "user.name", "ci")
    (repo / "other.py").write_text("base\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "base")
    _git(repo, "checkout", "-q", "-b", "pr")
    target = repo / "src.py"
    target.write_text("".join(f"line {i}\n" for i in range(1, 31)))
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "reviewed")
    reviewed = _git(repo, "rev-parse", "HEAD")
    lines = target.read_text().splitlines(keepends=True)
    lines[19] = "line 20 fixed\n"
    target.write_text("".join(lines))
    _git(repo, "commit", "-q", "-am", "fix line 20")
    old_fix = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "dev")
    (repo / "other.py").write_text("base\nmoved on\n")
    _git(repo, "commit", "-q", "-am", "dev moves")
    new_base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "pr")
    _git(repo, "rebase", "-q", "dev")
    return _Pr(repo, new_base, old_fix, _git(repo, "rev-parse", "HEAD"), reviewed)


def _thread(pr: _Pr, fixed_in: str, line: int) -> dict[str, Any]:
    """A hostile-reviewer thread on ``src.py:line`` with a ``fixed-in`` reply."""
    return {
        "id": "T1",
        "isResolved": True,
        "isOutdated": True,
        "path": "src.py",
        "originalLine": line,
        "originalStartLine": line,
        "subjectType": "LINE",
        "comments": {
            "nodes": [
                {
                    "body": (
                        "**[HIGH] hostile-reviewer (m)** finding\n"
                        f"<!-- {gate.MARKER} fp=0123456789ab -->"
                    ),
                    "url": "https://example.invalid/c1",
                    "originalCommit": {"oid": pr.reviewed},
                    "author": {"login": "bot"},
                },
                {
                    "body": f"fixed-in {fixed_in}",
                    "url": "https://example.invalid/c2",
                    "databaseId": 2,
                    "author": {"login": "jake-b-omni"},
                    "authorAssociation": "OWNER",
                },
            ]
        },
    }


def _classify(pr: _Pr, thread: dict[str, Any]) -> tuple[list[Any], list[Any]]:
    verifier = gate.GitFixVerifier(pr.repo, pr.head, pr.base)
    return gate.classify_threads([thread], verifier)


def test_rebase_really_rewrote_the_fix_sha(tmp_path: Path) -> None:
    pr = _rebased_pr(tmp_path)
    assert pr.old_fix != pr.head
    assert not _is_ancestor(pr.repo, pr.old_fix, pr.head)


def test_gate_clears_a_thread_whose_fixed_in_sha_a_rebase_rewrote(
    tmp_path: Path,
) -> None:
    pr = _rebased_pr(tmp_path)
    blocking, cleared = _classify(pr, _thread(pr, pr.old_fix, 20))
    assert not blocking, [v.reason for v in blocking]
    assert len(cleared) == 1
    assert cleared[0].fix_verified is True
    assert "changes src.py:20-20" in cleared[0].reason
    # The verdict names the rewritten commit, not the dead pre-rebase sha.
    assert pr.old_fix[:12] not in cleared[0].reason
    assert pr.head[:12] in cleared[0].reason


def test_gate_still_blocks_a_fixed_in_sha_that_is_not_in_the_pr(
    tmp_path: Path,
) -> None:
    pr = _rebased_pr(tmp_path)
    _git(pr.repo, "checkout", "-q", "-b", "elsewhere", pr.base)
    (pr.repo / "src.py").write_text("".join(f"stranger {i}\n" for i in range(1, 31)))
    _git(pr.repo, "add", ".")
    _git(pr.repo, "commit", "-q", "-m", "a change the PR never made")
    stranger = _git(pr.repo, "rev-parse", "HEAD")
    blocking, cleared = _classify(pr, _thread(pr, stranger, 20))
    assert not cleared
    assert len(blocking) == 1
    assert "is not in this PR (not an ancestor of the head)" in blocking[0].reason


def test_gate_blocks_a_rewritten_fix_that_never_touched_the_cited_lines(
    tmp_path: Path,
) -> None:
    pr = _rebased_pr(tmp_path)
    # The rebased fix changes line 20 only; a finding on line 3 is not addressed.
    blocking, cleared = _classify(pr, _thread(pr, pr.old_fix, 3))
    assert not cleared
    assert len(blocking) == 1
    assert "does not touch the cited lines" in blocking[0].reason
