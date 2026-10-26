# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regression tests for whole-file hostile-review findings cited at line 1."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)

pytestmark = pytest.mark.unit

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "ci"
    / "check_hostile_review_threads.py"
)
_spec = importlib.util.spec_from_file_location("check_hostile_review_threads", _SCRIPT)
assert _spec is not None and _spec.loader is not None
gate = importlib.util.module_from_spec(_spec)
sys.modules["check_hostile_review_threads"] = gate
_spec.loader.exec_module(gate)

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "ci"
    / "post_hostile_review_threads.py"
)
_spec = importlib.util.spec_from_file_location("post_hostile_review_threads", _SCRIPT)
assert _spec is not None and _spec.loader is not None
poster = importlib.util.module_from_spec(_spec)
sys.modules["post_hostile_review_threads"] = poster
_spec.loader.exec_module(poster)


def _thread(
    *,
    path: str = "src.py",
    line: int = 1,
    start_line: int | None = None,
    original_commit: str = "",
) -> dict[str, Any]:
    first: dict[str, Any] = {
        "body": (
            f"<!-- {gate.MARKER} fp=abc123def456 -->\n"
            "**[MAJOR] hostile-reviewer (sonnet)**\n\nthe finding"
        ),
        "url": "https://x/thread",
        "author": {"login": "github-actions"},
        "originalCommit": {"oid": original_commit} if original_commit else None,
    }
    return {
        "id": "T1",
        "isResolved": False,
        "isOutdated": False,
        "path": path,
        "originalLine": line,
        "originalStartLine": start_line,
        "subjectType": "LINE",
        "resolvedBy": None,
        "comments": {"nodes": [first]},
    }


@pytest.mark.parametrize(
    ("start_line", "line", "expected"),
    [
        (None, 1, (None, None)),
        (1, 1, (None, None)),
        (1, 5, (1, 5)),
        (None, 20, (20, 20)),
    ],
)
def test_cited_lines(
    start_line: int | None, line: int, expected: tuple[int | None, int | None]
) -> None:
    reviewed = "a" * 40
    cited = gate.cited_lines(
        _thread(line=line, start_line=start_line, original_commit=reviewed)
    )
    assert (cited.start, cited.end) == expected
    assert cited.path == "src.py"
    assert cited.original_commit == reviewed


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        env=scrub_git_location_env(os.environ),
    ).stdout.strip()


def test_line_one_thread_accepts_a_fix_elsewhere_in_the_file(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "dev")
    _git(repo, "config", "user.email", "ci@example.invalid")
    _git(repo, "config", "user.name", "ci")
    other = repo / "other.py"
    other.write_text("base\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "base")
    base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "-b", "pr")
    target = repo / "src.py"
    target.write_text(
        "# SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.\n"
        + "".join(f"line {i}\n" for i in range(2, 32))
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "reviewed")
    reviewed = _git(repo, "rev-parse", "HEAD")
    lines = target.read_text().splitlines(keepends=True)
    lines[19] = "line 20 fixed\n"
    target.write_text("".join(lines))
    _git(repo, "commit", "-q", "-am", "fix")
    fix = _git(repo, "rev-parse", "HEAD")

    cited = gate.cited_lines(_thread(original_commit=reviewed))
    verifier = gate.GitFixVerifier(repo, fix, base)
    ok, why = verifier(fix, cited)
    assert ok, why

    # A literal line-1 citation still refuses the honest fix: the old behavior.
    ok, why = verifier(fix, gate.CitedLines("src.py", reviewed, 1, 1))
    assert not ok
    assert "does not touch the cited lines src.py:1-1" in why

    other.write_text("unrelated fix\n")
    _git(repo, "commit", "-q", "-am", "fix different file")
    unrelated = _git(repo, "rev-parse", "HEAD")
    ok, why = gate.GitFixVerifier(repo, unrelated, base)(unrelated, cited)
    assert not ok
    assert "does not change src.py" in why


@pytest.mark.parametrize(
    ("line", "expected"),
    [(1, ("file", "src.py", None)), (12, ("line", "src.py", 12))],
)
def test_resolve_anchor(line: int, expected: tuple[str, str, int | None]) -> None:
    assert (
        poster.resolve_anchor({"file_path": f"src.py:{line}"}, {"src.py": {1, 12}})
        == expected
    )
