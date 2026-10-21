# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Deterministic thread-resolution gate for hostile-reviewer findings (OMN-17492).

The LLM is the FINDER, never the gate: ``post_hostile_review_threads.py``
publishes findings as resolvable PR review threads, and THIS check is the
deterministic merge surface. No model opinion is consulted here; the inputs
are GitHub thread state, the replies on each thread and the PR's own git
history.

Blocking rule (OMN-17427, operator RULING 2026-10-21T13:00:04Z)
---------------------------------------------------------------
Only threads whose first comment carries the ``hostile-reviewer-thread``
marker are this gate's business. Each one must end in a DISPOSITION: a reply
on the thread whose line starts with one of

    fixed-in <sha>          the commit (7-40 hex) that fixed the finding
    rejected: <reason>      why the finding is wrong (at least 10 characters)

The latest valid disposition on a thread wins, so an edited or a later reply
replaces an earlier one. A marker thread blocks when:

  - it has no disposition, resolved or not. A resolved thread with no reply
    is the case the ruling closes: "resolved" alone recorded nothing about
    what was done. An OUTDATED unresolved thread blocks too; before OMN-17427
    it was only reported, so every finding whose line moved under a later
    push escaped without a record;
  - its disposition is ``fixed-in <sha>`` and that commit is not in this PR
    (an ancestor of the PR head and not of its base), or it did not touch the
    cited lines. The cited lines are the thread's original range at the
    commit the review was posted on, carried forward through the PR's own
    history to the fix commit's parent; a file-level thread needs the fix
    commit to touch the file.

An accepted disposition clears its thread whether or not the thread has been
resolved in the review UI, so replying is the one action an author takes: the
reply fires the gate (``hostile-review-thread-gate.yml``) and the gate checks
it. The disposition is the record; "resolved" alone is not. Two GitHub facts
force this shape: resolving a thread fires no workflow event, so a gate that
waited for the click would never re-run; and the workflow token cannot resolve
a thread itself (resolveReviewThread answers FORBIDDEN, "Resource not
accessible by integration", measured on omnibase_infra#5234 run 39626556756).

The disposition rule holds PRs into ``dev`` (CI Summary requires this check
there only). On any other base the gate keeps its earlier rule: an open,
current marker thread without an accepted disposition blocks; resolved and
outdated ones do not.

Override: there is none, on purpose. The operator ruling asks for an override
only a human can apply, and every agent and lane acts through the operator's
own GitHub account and tokens, so a label, an approval, a reply or a check-run
from that account proves nothing about who acted. No override is accepted
until a credential no agent holds exists (a separate human-only account whose
login this gate would accept, or a touch-required hardware signing key whose
signature it would verify); the options are recorded on OMN-17427.

No threads means nothing to check: when the reviewer was skipped, degraded or
down, there is no marker thread and the gate passes. A reviewer outage never
stops a merge here; only a finding that was posted and not answered does.

Accuracy record
---------------
Every marker thread is printed as one ``hostile-review-disposition: {json}``
line (and written to ``DISPOSITION_RECORD_PATH`` when set): the voter model,
severity, cited path and lines, who resolved the thread, the disposition, its
author and whether a ``fixed-in`` commit was verified. ``--report FILE...``
reads those records back and prints, per voter model, how many findings were
fixed, rejected and left without a disposition.

Failure posture: fail closed. An unreachable or malformed GraphQL response is
a nonzero exit, never a vacuous pass. A git error while checking a
``fixed-in`` commit blocks that thread with the error named.

Reference: OMN-17492, OMN-17427.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MARKER = "hostile-reviewer-thread"
DISPOSITION_LINE_PREFIX = "hostile-review-disposition: "
MIN_REJECT_REASON_CHARS = 10

_REJECTED_RE = re.compile(r"^\s*rejected:\s*(\S.*?)\s*$", re.MULTILINE | re.IGNORECASE)
_FIXED_ANY_CASE_RE = re.compile(
    r"^\s*fixed-in\s+([0-9a-fA-F]{7,40})\b", re.MULTILINE | re.IGNORECASE
)
_FP_RE = re.compile(rf"{MARKER} fp=([0-9a-f]{{12}})")
_LABEL_RE = re.compile(
    r"\*\*\[(?P<severity>[A-Z]+)\] hostile-reviewer \((?P<model>[^)]+)\)\*\*"
)
_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

_QUERY = """
query($owner: String!, $name: String!, $pr: Int!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $pr) {
      reviewThreads(first: 50, after: $cursor) {
        pageInfo { hasNextPage endCursor }
        nodes {
          id
          isResolved
          isOutdated
          path
          line
          originalLine
          startLine
          originalStartLine
          subjectType
          resolvedBy { login }
          comments(first: 100) {
            totalCount
            nodes {
              body
              url
              createdAt
              author { login }
              originalCommit { oid }
            }
          }
        }
      }
    }
  }
}
"""


def _graphql(query: str, variables: dict[str, Any], token: str) -> dict[str, Any]:
    payload = json.dumps({"query": query, "variables": variables}).encode()
    req = urllib.request.Request(
        "https://api.github.com/graphql",  # url-authority-ok: GitHub GraphQL endpoint (reviewThreads has no REST surface); CI-only, GITHUB_TOKEN-authenticated, same fixed origin as gh api graphql
        data=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
        data: dict[str, Any] = json.loads(resp.read())
    if data.get("errors"):
        raise RuntimeError(f"GraphQL errors: {data['errors']}")
    return data


def fetch_review_threads(repo: str, pr_number: int, token: str) -> list[dict[str, Any]]:
    """Fetch every review thread on the PR via GraphQL (paged)."""
    owner, name = repo.split("/", 1)
    threads: list[dict[str, Any]] = []
    cursor: str | None = None
    while True:
        data = _graphql(
            _QUERY,
            {"owner": owner, "name": name, "pr": pr_number, "cursor": cursor},
            token,
        )
        conn = data["data"]["repository"]["pullRequest"]["reviewThreads"]
        threads.extend(conn["nodes"])
        if not conn["pageInfo"]["hasNextPage"]:
            return threads
        cursor = conn["pageInfo"]["endCursor"]


# ---------------------------------------------------------------------------
# Disposition parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Disposition:
    """One valid disposition reply on a hostile-reviewer thread."""

    kind: str  # "fixed" | "rejected"
    author: str
    url: str
    sha: str = ""
    reason: str = ""


def _comments(thread: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = (thread.get("comments") or {}).get("nodes") or []
    return [c for c in nodes if isinstance(c, dict)]


def _login(raw: object) -> str:
    return str(raw.get("login", "")) if isinstance(raw, dict) else ""


def is_marker_thread(thread: dict[str, Any]) -> bool:
    comments = _comments(thread)
    return bool(comments) and MARKER in str(comments[0].get("body", ""))


def parse_disposition(body: str) -> tuple[str, str] | None:
    """Return ("fixed", sha) or ("rejected", reason) for a reply body, else None.

    The LAST disposition line in the body wins. A ``rejected:`` reason shorter
    than :data:`MIN_REJECT_REASON_CHARS` is not a reason and does not count.
    """
    found: list[tuple[int, str, str]] = []
    for match in _FIXED_ANY_CASE_RE.finditer(body):
        found.append((match.start(), "fixed", match.group(1).lower()))
    for match in _REJECTED_RE.finditer(body):
        reason = match.group(1).strip()
        if len(reason) >= MIN_REJECT_REASON_CHARS:
            found.append((match.start(), "rejected", reason))
    if not found:
        return None
    _, kind, value = max(found)
    return kind, value


def latest_disposition(thread: dict[str, Any]) -> Disposition | None:
    """The last valid disposition among the replies (the first comment excluded)."""
    result: Disposition | None = None
    for comment in _comments(thread)[1:]:
        parsed = parse_disposition(str(comment.get("body", "")))
        if parsed is None:
            continue
        kind, value = parsed
        result = Disposition(
            kind=kind,
            author=_login(comment.get("author")),
            url=str(comment.get("url", "")),
            sha=value if kind == "fixed" else "",
            reason=value if kind == "rejected" else "",
        )
    return result


# ---------------------------------------------------------------------------
# fixed-in verification: did the commit touch the cited lines?
# ---------------------------------------------------------------------------


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


@dataclass(frozen=True)
class CitedLines:
    path: str
    original_commit: str
    start: int | None  # None: a file-level thread
    end: int | None


def cited_lines(thread: dict[str, Any]) -> CitedLines:
    comments = _comments(thread)
    original_commit = ""
    if comments:
        original_commit = str(
            (comments[0].get("originalCommit") or {}).get("oid") or ""
        )
    path = str(thread.get("path", ""))
    end = thread.get("originalLine")
    if str(thread.get("subjectType", "")).upper() == "FILE" or end is None:
        return CitedLines(path, original_commit, None, None)
    start = thread.get("originalStartLine") or end
    return CitedLines(path, original_commit, int(start), int(end))


Verifier = Callable[[str, CitedLines], tuple[bool, str]]

_GIT_LOCATION_VARS = frozenset(
    {
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_COMMON_DIR",
        "GIT_OBJECT_DIRECTORY",
    }
)


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
        # A git hook exports GIT_DIR and friends, which override cwd; drop them
        # so the verifier always reads the checkout it was given (OMN-18434).
        env = {k: v for k, v in os.environ.items() if k not in _GIT_LOCATION_VARS}
        return subprocess.run(
            ["git", *args],
            cwd=self._repo_dir,
            capture_output=True,
            text=True,
            check=False,
            env=env,
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

    def _diff(self, old: str, new: str, path: str) -> str:
        out = self._git(
            "diff", "-U0", "--no-color", "--no-ext-diff", old, new, "--", path
        )
        if out.returncode != 0:
            raise RuntimeError(
                f"git diff {old[:12]} {new[:12]} -- {path}: {out.stderr.strip()}"
            )
        return out.stdout

    def __call__(self, sha: str, cited: CitedLines) -> tuple[bool, str]:
        fix = self._resolve(sha)
        if fix is None:
            return False, f"commit {sha} is not in this repository's history"
        if not self._is_ancestor(fix, self._head):
            return (
                False,
                f"commit {fix[:12]} is not in this PR (not an ancestor of the head)",
            )
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


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@dataclass
class ThreadVerdict:
    """The gate's verdict on one hostile-reviewer thread, and its accuracy record."""

    thread_id: str
    url: str
    path: str
    line_start: int | None
    line_end: int | None
    original_commit: str
    fingerprint: str
    model: str
    severity: str
    resolved: bool
    outdated: bool
    resolved_by: str
    disposition: str  # "fixed" | "rejected" | "none"
    disposition_author: str = ""
    disposition_url: str = ""
    fixed_in: str = ""
    rejected_reason: str = ""
    fix_verified: bool | None = None
    blocking: bool = True
    reason: str = ""
    finding: str = ""


FINDING_CHARS = 300


def finding_text(first_body: str) -> str:
    """The finding's own words from the thread's first comment, bounded.

    Drops the marker comment, the severity/model label and the italic footer,
    so a job-log reader can tell which finding a verdict is about without a
    GitHub read.
    """
    lines = []
    for line in first_body.splitlines():
        text = line.strip()
        if not text or text.startswith("<!--") or _LABEL_RE.search(text):
            continue
        if text.startswith("_") and text.endswith("_"):
            continue
        lines.append(text)
    joined = " ".join(lines)
    return joined if len(joined) <= FINDING_CHARS else joined[:FINDING_CHARS] + " …"


def evaluate_thread(
    thread: dict[str, Any], verifier: Verifier, *, require_disposition: bool = True
) -> ThreadVerdict:
    """One thread's verdict.

    ``require_disposition`` is the dev rule (OMN-17427). Off it, on a PR into
    any other base, the gate keeps its earlier rule: only an open thread that
    is not outdated and has no accepted disposition blocks.
    """
    comments = _comments(thread)
    first_body = str(comments[0].get("body", "")) if comments else ""
    label = _LABEL_RE.search(first_body)
    fp = _FP_RE.search(first_body)
    cited = cited_lines(thread)
    resolved = bool(thread.get("isResolved"))
    verdict = ThreadVerdict(
        thread_id=str(thread.get("id", "")),
        url=str(comments[0].get("url", "")) if comments else "",
        path=cited.path,
        line_start=cited.start,
        line_end=cited.end,
        original_commit=cited.original_commit,
        fingerprint=fp.group(1) if fp else "",
        finding=finding_text(first_body),
        model=label.group("model") if label else "unknown",
        severity=label.group("severity") if label else "UNKNOWN",
        resolved=resolved,
        outdated=bool(thread.get("isOutdated")),
        resolved_by=_login(thread.get("resolvedBy")),
        disposition="none",
    )
    disposition = latest_disposition(thread)
    if disposition is None:
        verdict.reason = (
            "resolved with no disposition reply"
            if resolved
            else "unresolved, no disposition reply"
        )
        if not require_disposition:
            verdict.blocking = not resolved and not verdict.outdated
        return verdict
    verdict.disposition = disposition.kind
    verdict.disposition_author = disposition.author
    verdict.disposition_url = disposition.url
    if disposition.kind == "rejected":
        verdict.rejected_reason = disposition.reason
        verdict.blocking = False
        verdict.reason = "rejected with a reason"
    else:
        verdict.fixed_in = disposition.sha
        ok, why = verifier(disposition.sha, cited)
        verdict.fix_verified = ok
        verdict.blocking = not ok and (
            require_disposition or (not resolved and not verdict.outdated)
        )
        verdict.reason = why
    return verdict


def classify_threads(
    threads: Iterable[dict[str, Any]],
    verifier: Verifier,
    *,
    require_disposition: bool = True,
) -> tuple[list[ThreadVerdict], list[ThreadVerdict]]:
    """Split marker threads into (blocking, cleared). Other threads are ignored."""
    blocking: list[ThreadVerdict] = []
    cleared: list[ThreadVerdict] = []
    for thread in threads:
        if not is_marker_thread(thread):
            continue
        verdict = evaluate_thread(
            thread, verifier, require_disposition=require_disposition
        )
        (blocking if verdict.blocking else cleared).append(verdict)
    return blocking, cleared


def record(
    verdict: ThreadVerdict, *, repo: str, pr_number: int, head_sha: str
) -> dict[str, Any]:
    row = asdict(verdict)
    row.update({"repo": repo, "pr": pr_number, "head_sha": head_sha})
    return row


# ---------------------------------------------------------------------------
# Accuracy report
# ---------------------------------------------------------------------------


def _read_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if text.startswith(DISPOSITION_LINE_PREFIX):
                text = text[len(DISPOSITION_LINE_PREFIX) :]
            elif DISPOSITION_LINE_PREFIX in text:
                # A raw job log line carries a timestamp before the prefix.
                text = text.split(DISPOSITION_LINE_PREFIX, 1)[1]
            if not text.startswith("{"):
                continue
            rows.append(json.loads(text))
    return rows


def accuracy(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Per voter model: findings fixed, rejected and left without a disposition.

    A finding is counted once per (repo, pr, fingerprint); the latest record
    read wins, so re-runs of the gate on one PR do not double count.
    """
    latest: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row.get("repo")),
            int(row.get("pr", 0)),
            str(row.get("fingerprint") or row.get("thread_id")),
        )
        latest[key] = row
    table: dict[str, dict[str, int]] = {}
    for row in latest.values():
        model = str(row.get("model", "unknown"))
        bucket = table.setdefault(
            model, {"findings": 0, "fixed": 0, "rejected": 0, "none": 0}
        )
        bucket["findings"] += 1
        disposition = str(row.get("disposition", "none"))
        if disposition == "fixed" and row.get("fix_verified") is not True:
            disposition = "none"
        bucket[disposition if disposition in bucket else "none"] += 1
    return table


def print_report(table: dict[str, dict[str, int]]) -> None:
    print(
        "model | findings | fixed | rejected | no disposition | fixed share of decided"
    )
    for model in sorted(table):
        row = table[model]
        decided = row["fixed"] + row["rejected"]
        share = f"{row['fixed'] / decided:.0%}" if decided else "n/a"
        print(
            f"{model} | {row['findings']} | {row['fixed']} | {row['rejected']} | "
            f"{row['none']} | {share}"
        )
    print(json.dumps(table, sort_keys=True))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _write_summary(blocking: list[ThreadVerdict], cleared: list[ThreadVerdict]) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    lines = [
        "## Hostile Review Thread Gate",
        "",
        f"Blocking: {len(blocking)}, cleared: {len(cleared)}",
        "",
        "| Finding | Model | Disposition | By | Verdict |",
        "|---|---|---|---|---|",
    ]
    for verdict in blocking + cleared:
        where = verdict.path + (
            f":{verdict.line_start}-{verdict.line_end}" if verdict.line_end else ""
        )
        disposition = verdict.disposition + (
            f" {verdict.fixed_in[:12]}" if verdict.fixed_in else ""
        )
        state = "BLOCKS" if verdict.blocking else "ok"
        lines.append(
            f"| [{where}]({verdict.url}) | {verdict.model} | {disposition} | "
            f"{verdict.disposition_author or verdict.resolved_by or '-'} | {state}: {verdict.reason} |"
        )
    with open(summary_path, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def run_gate() -> int:
    token = os.environ["GITHUB_TOKEN"]
    repo = os.environ["REPO"]
    pr_number = int(os.environ["PR_NUMBER"])
    head_sha = os.environ.get("HEAD_SHA", "")
    base_ref = os.environ.get("BASE_REF", "")

    try:
        threads = fetch_review_threads(repo, pr_number, token)
    except Exception as exc:  # noqa: BLE001 - fail closed on ANY fetch problem
        print(f"::error::could not fetch review threads (fail-closed): {exc}")
        return 1

    verifier = GitFixVerifier(
        Path.cwd(),
        head_sha or "HEAD",
        f"refs/remotes/origin/{base_ref}" if base_ref else "HEAD^1",
    )
    # OMN-17427: the disposition rule holds PRs into dev (an unknown base is
    # treated as dev, so a missing variable enforces rather than relaxes).
    blocking, cleared = classify_threads(
        threads, verifier, require_disposition=base_ref in ("dev", "")
    )

    record_path = os.environ.get("DISPOSITION_RECORD_PATH")
    rows = [
        record(v, repo=repo, pr_number=pr_number, head_sha=head_sha)
        for v in blocking + cleared
    ]
    for row in rows:
        print(DISPOSITION_LINE_PREFIX + json.dumps(row, sort_keys=True))
    if record_path:
        Path(record_path).write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
    _write_summary(blocking, cleared)

    if blocking:
        print(
            f"::error::{len(blocking)} hostile-reviewer finding(s) have no accepted "
            "disposition. Reply on each thread with a line `fixed-in <sha>` (a commit "
            "of this PR that changes the cited lines) or `rejected: <reason>`; the "
            "reply re-runs this gate (OMN-17427). A thread resolved without that "
            "reply blocks."
        )
        for verdict in blocking:
            print(
                f"  BLOCKING: {verdict.path} ({verdict.model}) — {verdict.reason} — {verdict.url}"
            )
        return 1

    print(
        f"Hostile Review Thread Gate PASSED — {len(cleared)} finding(s) with an "
        "accepted disposition, 0 blocking."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--report",
        nargs="+",
        type=Path,
        metavar="FILE",
        help="print per-model accuracy from disposition records or job logs",
    )
    args = parser.parse_args(argv)
    if args.report:
        print_report(accuracy(_read_records(args.report)))
        return 0
    return run_gate()


if __name__ == "__main__":
    sys.exit(main())
