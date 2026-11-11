# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Git-history verification of a hostile-review ``fixed-in <sha>`` disposition.

Moved out of scripts/ci/check_hostile_review_threads.py (OMN-17427) so the
baselined script only shrinks. Standard library only: the gate runs bare
``python3`` on a runner with no package installed and loads this file by path.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
    from omnibase_infra.models.hostile_review.model_fix_verify_request import (
        ModelFixVerifyRequest,
    )
    from omnibase_infra.models.hostile_review.model_fix_verify_result import (
        ModelFixVerifyResult,
    )


@dataclass(frozen=True)
class CitedThreadLines:
    """The lines a review thread cites, at the commit it was raised against.

    The one definition of the shape: ``scripts/ci/check_hostile_review_threads.py``
    binds its own ``CitedLines`` name to this class, so the gate and the handler
    cannot drift apart. ``start``/``end`` are None for a file-level thread.
    """

    path: str
    original_commit: str
    start: int | None
    end: int | None


_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


@dataclass(frozen=True)
class Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int


def parse_hunks(diff_text: str) -> list[Hunk]:
    """Hunk headers of a ``git diff -U0`` for one file."""
    hunks: list[Hunk] = []
    for line in diff_text.splitlines():
        match = _HUNK_RE.match(line)
        if not match:
            continue
        old_start, old_count, new_start, new_count = match.groups()
        hunks.append(
            Hunk(
                old_start=int(old_start),
                old_count=1 if old_count is None else int(old_count),
                new_start=int(new_start),
                new_count=1 if new_count is None else int(new_count),
            )
        )
    return hunks


def _map_line(hunks: list[Hunk], line: int) -> tuple[int, int]:
    """Where an old-side line lands on the new side, as a span.

    A line inside a changed hunk maps to the whole span that replaced it.
    """
    offset = 0
    for hunk in sorted(hunks, key=lambda h: h.old_start):
        if hunk.old_count == 0:
            # Pure insertion after old line ``old_start``.
            if hunk.old_start < line:
                offset += hunk.new_count
            continue
        old_end = hunk.old_start + hunk.old_count - 1
        if line < hunk.old_start:
            break
        if line > old_end:
            offset += hunk.new_count - hunk.old_count
            continue
        span_end = hunk.new_start + max(hunk.new_count, 1) - 1
        return hunk.new_start, span_end
    return line + offset, line + offset


def map_range(hunks: list[Hunk], start: int, end: int) -> tuple[int, int]:
    """Carry an old-side line range across a diff to the new side."""
    low = _map_line(hunks, start)[0]
    high = _map_line(hunks, end)[1]
    return min(low, high), max(low, high)


def range_touched(hunks: list[Hunk], start: int, end: int) -> bool:
    """True when any hunk changes, deletes or inserts inside ``[start, end]``."""
    for hunk in hunks:
        if hunk.old_count == 0:
            # Insertion after old line ``old_start``: inside or adjacent.
            if start - 1 <= hunk.old_start <= end:
                return True
            continue
        if hunk.old_start <= end and hunk.old_start + hunk.old_count - 1 >= start:
            return True
    return False


_GIT_LOCATION_VARS = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_COMMON_DIR",
    "GIT_OBJECT_DIRECTORY",
)


def _git_argv(*args: str) -> list[str]:
    """``git <args>`` with git's location variables unset in the child process.

    A git hook exports GIT_DIR and friends, and those override ``cwd``, so a
    verifier run from a hook would read the invoking worktree instead of the
    checkout it was handed (OMN-18434). ``env -u`` drops them on the way into
    git rather than in this process, so this module reads no environment of its
    own: it needs none for configuration, and it cannot call
    ``omnibase_core.validators.no_unguarded_git_subprocess.scrub_git_location_env``
    because scripts/ci/check_hostile_review_threads.py loads it by path under
    bare ``python3`` with nothing installed.
    """
    unset: list[str] = []
    for name in _GIT_LOCATION_VARS:
        unset += ["-u", name]
    return ["/usr/bin/env", *unset, "git", *args]


def _patch_ids_of(repo_dir: Path, patch_text: str) -> dict[str, str]:
    """``{commit: patch-id}`` for a stream of ``git log -p`` / ``git show``."""
    out = subprocess.run(
        _git_argv("patch-id", "--stable"),
        cwd=repo_dir,
        input=patch_text,
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        return {}
    ids: dict[str, str] = {}
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2:
            ids.setdefault(parts[1], parts[0])
    return ids


class GitFixVerifier:
    """Checks ``fixed-in <sha>`` against the PR's own git history.

    ``head`` is the PR head commit and ``base`` the base branch commit; both
    must be present in the checkout (fetch-depth 0).
    """

    def __init__(self, repo_dir: Path, head: str, base: str) -> None:
        self._repo_dir = repo_dir
        self._head = head
        self._base = base

    def _git(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            _git_argv(*args),
            cwd=self._repo_dir,
            capture_output=True,
            text=True,
            check=False,
        )

    def _resolve(self, sha: str) -> str | None:
        out = self._git("rev-parse", "--verify", "--quiet", f"{sha}^{{commit}}")
        if out.returncode == 0:
            return out.stdout.strip()
        if len(sha) == 40:
            self._git("fetch", "--no-tags", "--quiet", "origin", sha)
            out = self._git("rev-parse", "--verify", "--quiet", f"{sha}^{{commit}}")
            if out.returncode == 0:
                return out.stdout.strip()
        return None

    def _is_ancestor(self, older: str, newer: str) -> bool:
        return self._git("merge-base", "--is-ancestor", older, newer).returncode == 0

    def _patch_id(self, commit: str) -> str | None:
        """The content identity (``git patch-id --stable``) of one commit."""
        shown = self._git("show", "--no-color", "--no-ext-diff", commit)
        if shown.returncode != 0 or not shown.stdout.strip():
            return None
        return _patch_ids_of(self._repo_dir, shown.stdout).get(commit)

    def _rewritten_as(self, fix: str) -> str | None:
        """The commit of this PR whose patch is ``fix``'s patch, or None.

        A rebase rewrites every sha of a PR, so a ``fixed-in`` naming the old sha
        is no longer an ancestor of the head although its change is still there.
        The change is matched by patch-id against the PR's own commits
        (``base..head``); a change that is not in the PR matches nothing.
        """
        wanted = self._patch_id(fix)
        if wanted is None:
            return None
        log = self._git(
            "log",
            "-p",
            "--no-color",
            "--no-ext-diff",
            "--no-merges",
            "--format=commit %H",
            f"{self._base}..{self._head}",
        )
        if log.returncode != 0:
            return None
        for commit, patch_id in _patch_ids_of(self._repo_dir, log.stdout).items():
            if patch_id == wanted:
                return commit
        return None

    def _diff(self, old: str, new: str, path: str) -> str:
        out = self._git(
            "diff", "-U0", "--no-color", "--no-ext-diff", old, new, "--", path
        )
        if out.returncode != 0:
            raise RuntimeError(
                f"git diff {old[:12]} {new[:12]} -- {path}: {out.stderr.strip()}"
            )
        return out.stdout

    def file_at(self, commit: str, path: str) -> str | None:
        """The file's text at ``commit``, or None when it does not exist there."""
        out = self._git("show", f"{commit}:{path}")
        return out.stdout if out.returncode == 0 else None

    def carry_forward(
        self, cited: CitedThreadLines, commit: str
    ) -> tuple[int, int] | None:
        """The cited range as it lies at ``commit`` (OMN-17427, the judge's excerpt).

        None for a file-level thread. A range whose reviewed commit is unknown
        is returned as cited.
        """
        if cited.start is None or cited.end is None:
            return None
        original = (
            self._resolve(cited.original_commit) if cited.original_commit else None
        )
        target = self._resolve(commit)
        if original is None or target is None or original == target:
            return cited.start, cited.end
        return map_range(
            parse_hunks(self._diff(original, target, cited.path)),
            cited.start,
            cited.end,
        )

    def __call__(self, sha: str, cited: CitedThreadLines) -> tuple[bool, str]:
        fix = self._resolve(sha)
        if fix is None:
            return False, f"commit {sha} is not in this repository's history"
        if not self._is_ancestor(fix, self._head):
            landed = self._rewritten_as(fix)
            if landed is None:
                return (
                    False,
                    f"commit {fix[:12]} is not in this PR (not an ancestor of the head)",
                )
            fix = landed
        if self._is_ancestor(fix, self._base):
            return (
                False,
                f"commit {fix[:12]} is already on the base branch, not a commit of this PR",
            )
        parent_out = self._git("rev-parse", "--verify", "--quiet", f"{fix}^")
        if parent_out.returncode != 0:
            return False, f"commit {fix[:12]} has no parent to diff against"
        parent = parent_out.stdout.strip()
        try:
            fix_hunks = parse_hunks(self._diff(parent, fix, cited.path))
        except RuntimeError as exc:
            return False, str(exc)
        if cited.start is None or cited.end is None:
            if fix_hunks:
                return True, f"commit {fix[:12]} changes {cited.path}"
            return False, f"commit {fix[:12]} does not change {cited.path}"
        start, end = cited.start, cited.end
        original = (
            self._resolve(cited.original_commit) if cited.original_commit else None
        )
        if original is not None and original != parent:
            try:
                start, end = map_range(
                    parse_hunks(self._diff(original, parent, cited.path)), start, end
                )
            except RuntimeError as exc:
                return False, str(exc)
        if range_touched(fix_hunks, start, end):
            return True, f"commit {fix[:12]} changes {cited.path}:{start}-{end}"
        return False, (
            f"commit {fix[:12]} does not touch the cited lines {cited.path}:{start}-{end} "
            f"(lines {cited.start}-{cited.end} at the reviewed commit "
            f"{(cited.original_commit or 'unknown')[:12]})"
        )


class HandlerGitFixVerifier:
    """Contract-bound entrypoint: verify one fixed-in disposition (OMN-17427).

    The node contract binds this class; the CI gate uses ``GitFixVerifier``
    directly. The result model and the handler enums are imported inside their
    members so this module stays standard-library only at load time.
    """

    @property
    def handler_type(self) -> EnumHandlerType:
        """Return the architectural role of this handler.

        Returns:
            EnumHandlerType.INFRA_HANDLER - infrastructure handler running git
            subprocesses against the checkout in the working directory.
        """
        from omnibase_infra.enums import EnumHandlerType

        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Return the behavioral classification of this handler.

        Returns:
            EnumHandlerTypeCategory.EFFECT - side-effecting I/O (git reads, and
            an origin fetch for a commit the checkout does not yet hold).
        """
        from omnibase_infra.enums import EnumHandlerTypeCategory

        return EnumHandlerTypeCategory.EFFECT

    def handle(self, request: ModelFixVerifyRequest) -> ModelFixVerifyResult:
        """Verify ``request.sha`` against the checkout in the working directory."""
        from omnibase_infra.models.hostile_review.model_fix_verify_result import (
            ModelFixVerifyResult,
        )

        cited = CitedThreadLines(
            request.path, request.original_commit, request.start, request.end
        )
        verifier = GitFixVerifier(Path.cwd(), request.head, request.base)
        accepted, reason = verifier(request.sha, cited)
        return ModelFixVerifyResult(accepted=accepted, reason=reason)


__all__ = [
    "CitedThreadLines",
    "GitFixVerifier",
    "HandlerGitFixVerifier",
    "Hunk",
    "map_range",
    "parse_hunks",
    "range_touched",
]
