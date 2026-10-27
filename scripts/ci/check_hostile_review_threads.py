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
    commit to touch the file. A thread cited at line 1 alone is read as
    file-level because reviewers cite ``:1`` for whole-file findings.

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

The judge modes (``--write-request``, ``--read-request``, ``--pending``,
``--post-verdicts``) serve the base-branch ``hostile-review-judge.yml``
``workflow_run`` workflow, which posts onexbot verdict replies bound to the
rejection's id, body digest and the PR head whose code the judge saw. A push
invalidates a prior judgement; the head recorded by ``--pending`` is retained
through posting even if the PR moves during the model call. This gate does
not yet require a verdict: a rejection with a reason still clears a finding
on dev. The rule requiring a verdict lands with OMN-17427 (omnibase_infra#5237).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MARKER = "hostile-reviewer-thread"
DISPOSITION_LINE_PREFIX = "hostile-review-disposition: "
MIN_REJECT_REASON_CHARS = 10
_WHOLE_FILE_LINE = 1

# The review bot's verdict on a rejection (OMN-17427). GraphQL spells the
# App's bot ``onexbot``, REST ``onexbot[bot]``. Only the privileged base-branch
# workflow hostile-review-judge.yml mints that token for verdict replies.
# The workflow identity github-actions no longer counts: any workflow of the
# repository can post as it.
JUDGE_MARKER = "hostile-review-judge"
JUDGE_LOGINS = frozenset({"onexbot", "onexbot[bot]"})
# Only a rejection from an account GitHub associates with the repository as
# OWNER, MEMBER or COLLABORATOR is judged. That is not a write-access check:
# MEMBER is organization membership, and COLLABORATOR covers read and triage
# roles too. It keeps an unassociated account's reply, which anyone can post on
# a public repository, from costing a paid model call; such a rejection blocks
# with this reason instead.
JUDGED_ASSOCIATIONS = frozenset({"OWNER", "MEMBER", "COLLABORATOR"})
# The full SHA-256 hex digest of the rejection body a verdict binds to.
JUDGE_DIGEST_CHARS = 64
_JUDGE_MARKER_RE = re.compile(
    rf"\A<!-- {JUDGE_MARKER} rejection=(?P<rejection>\d+) "
    rf"digest=(?P<digest>[0-9a-f]{{{JUDGE_DIGEST_CHARS}}}) "
    r"(?:head=(?P<head>[0-9a-f]{40}) )?"
    r"verdict=(?P<verdict>accepted|refused) "
    r"judge=(?P<judge>[a-z][a-z0-9-]{0,62}) "
    r"identity=(?P<identity>[^\s>]+) -->[ \t]*(?:\n|\Z)"
)
_JUDGE_SENTENCE_RE = re.compile(r"^> (?P<sentence>\S.*)$", re.MULTILINE)

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
              databaseId
              body
              url
              createdAt
              authorAssociation
              author { login __typename }
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
    comment_id: int = 0
    digest: str = ""
    author_association: str = ""


def _comments(thread: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = (thread.get("comments") or {}).get("nodes") or []
    return [c for c in nodes if isinstance(c, dict)]


def _login(raw: object) -> str:
    return str(raw.get("login", "")) if isinstance(raw, dict) else ""


def is_marker_thread(thread: dict[str, Any]) -> bool:
    comments = _comments(thread)
    return bool(comments) and MARKER in str(comments[0].get("body", ""))


def body_digest(body: str) -> str:
    """The digest a judge verdict binds to: the rejection's body as it reads now."""
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:JUDGE_DIGEST_CHARS]


def is_judge_reply(body: str) -> bool:
    """A reply that begins with the judge marker, whoever posted it.

    Such a reply is never a disposition, so a model's sentence quoted in it
    can never be read as a ``fixed-in`` or a ``rejected:`` line.
    """
    return _JUDGE_MARKER_RE.match(body) is not None


def not_judgeable(disposition: Disposition) -> str:
    """Why the review bot does not judge this rejection, or "" when it does."""
    if disposition.kind != "rejected":
        return "not a rejection"
    if not disposition.comment_id:
        return "the rejection has no comment id to bind a verdict to"
    if disposition.author_association not in JUDGED_ASSOCIATIONS:
        return (
            f"the rejection's author {disposition.author or '?'} is not an owner, "
            f"member or collaborator of this repository (association "
            f"{disposition.author_association or 'unknown'}), so the review bot does "
            "not judge it"
        )
    return ""


@dataclass(frozen=True)
class JudgeVerdict:
    """The review bot's accepted or refused verdict on one rejection."""

    verdict: str  # "accepted" | "refused"
    judge: str
    identity: str
    sentence: str
    url: str


def parse_judge_reply(body: str) -> dict[str, str] | None:
    """The marker fields and sentence of a judge reply, else None."""
    match = _JUDGE_MARKER_RE.match(body)
    if match is None:
        return None
    sentence = _JUDGE_SENTENCE_RE.search(body[match.end() :])
    fields = {k: str(v) for k, v in match.groupdict().items()}
    fields["sentence"] = sentence.group("sentence").strip() if sentence else ""
    return fields


def judge_verdict_for(
    thread: dict[str, Any], disposition: Disposition, *, head_sha: str | None = None
) -> JudgeVerdict | None:
    """The latest verdict the onexbot App posted on exactly this rejection.

    A reply from any other login is ignored, whatever it claims. The verdict
    must name the rejection's comment id, its current body digest and the
    current PR head. An unbound legacy marker is recognised as a judge reply
    but cannot suppress a new judgement of the current code.
    """
    if disposition.kind != "rejected" or not disposition.comment_id:
        return None
    current_head = os.environ.get("HEAD_SHA", "") if head_sha is None else head_sha
    if re.fullmatch(r"[0-9a-f]{40}", current_head) is None:
        return None
    found: JudgeVerdict | None = None
    for comment in _comments(thread)[1:]:
        author = comment.get("author")
        if _login(author) not in JUDGE_LOGINS:
            continue
        if (
            isinstance(author, dict)
            and "__typename" in author
            and author["__typename"] != "Bot"
        ):
            continue
        fields = parse_judge_reply(str(comment.get("body", "")))
        if fields is None:
            continue
        if int(fields["rejection"]) != disposition.comment_id:
            continue
        if fields["digest"] != disposition.digest:
            continue
        if fields.get("head") != current_head:
            continue
        found = JudgeVerdict(
            verdict=fields["verdict"],
            judge=fields["judge"],
            identity=fields["identity"],
            sentence=fields["sentence"],
            url=str(comment.get("url", "")),
        )
    return found


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
    """The last valid disposition among the replies (the first comment excluded).

    A judge reply is never a disposition (:func:`is_judge_reply`).
    """
    result: Disposition | None = None
    for comment in _comments(thread)[1:]:
        body = str(comment.get("body", ""))
        if is_judge_reply(body):
            continue
        parsed = parse_disposition(body)
        if parsed is None:
            continue
        kind, value = parsed
        raw_id = comment.get("databaseId")
        result = Disposition(
            kind=kind,
            author=_login(comment.get("author")),
            url=str(comment.get("url", "")),
            sha=value if kind == "fixed" else "",
            reason=value if kind == "rejected" else "",
            comment_id=raw_id if isinstance(raw_id, int) else 0,
            digest=body_digest(body),
            author_association=str(comment.get("authorAssociation") or ""),
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
    if start == end == _WHOLE_FILE_LINE:
        return CitedLines(path, original_commit, None, None)
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

    def file_at(self, commit: str, path: str) -> str | None:
        """The file's text at ``commit``, or None when it does not exist there."""
        out = self._git("show", f"{commit}:{path}")
        return out.stdout if out.returncode == 0 else None

    def carry_forward(self, cited: CitedLines, commit: str) -> tuple[int, int] | None:
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
    # OMN-17427: the review bot's verdict on a rejection: "accepted",
    # "refused", "unjudged" (no verdict on the latest rejection), or "" when
    # the disposition is not a rejection.
    judge_verdict: str = ""
    judge_voter: str = ""
    judge_identity: str = ""
    judge_sentence: str = ""
    judge_url: str = ""
    blocking: bool = True
    reason: str = ""
    finding: str = ""


FINDING_CHARS = 300


def finding_text(first_body: str, limit: int = FINDING_CHARS) -> str:
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
    return joined if len(joined) <= limit else joined[:limit] + " …"


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
# The review bot's inputs and replies (OMN-17427)
# ---------------------------------------------------------------------------
#
# The judgement itself is omniintelligence's ``cli_review judge-rejections``
# (node_review_rejection_judge_compute), run in the reviewer's own model
# environment by hostile-review-judge.yml on the base branch. This script keeps what it has
# always owned, GitHub thread state and the PR's git history: ``--pending``
# writes the rejections that await a verdict, with the cited lines at the
# head, and ``--post-verdicts`` turns the verdicts into the judge replies this
# gate reads, under the onexbot App token. A judgement model never touches
# GitHub, and nothing here calls a model.

JUDGE_FINDING_CHARS = 4_000
JUDGE_EXCERPT_CONTEXT_LINES = 20
JUDGE_FILE_EXCERPT_LINES = 400
JUDGE_EXCERPT_CHARS = 16_000
JUDGE_SENTENCE_CHARS = 400
JUDGE_VERDICTS = frozenset({"accepted", "refused"})
_JUDGE_ID_RE = re.compile(r"\A[a-z][a-z0-9-]{0,62}\Z")

FORK_UNAVAILABLE = (
    "this is a fork pull request: its workflow token is read-only and no model "
    "key reaches it, so the review bot can neither judge a rejection nor reply; "
    "fix the finding and reply `fixed-in <sha>`"
)
NO_JUDGE_RESULT = (
    "the review bot's judgement step wrote no result: it failed before judging "
    "(an install or overlay step, or an omniintelligence without `cli_review "
    "judge-rejections`); its log names why"
)


@dataclass(frozen=True)
class PendingRejection:
    """A rejection the review bot has to judge, and where its verdict goes."""

    key: str
    top_comment_id: int
    digest: str
    rejection_url: str
    raised_by: str
    finding: str
    rejection: str
    cited: CitedLines


def pending_rejections(
    threads: Iterable[dict[str, Any]],
    *,
    head_sha: str | None = None,
) -> tuple[list[PendingRejection], dict[str, str]]:
    """Rejections still to judge, and notes for those the bot will not judge.

    Only a thread's latest disposition counts, and a rejection that already
    carries the onexbot App's verdict on its id, current body and reviewed
    head is done. Legacy or prior-head verdicts remain pending.
    """
    pending: list[PendingRejection] = []
    notes: dict[str, str] = {}
    for thread in threads:
        if not is_marker_thread(thread):
            continue
        disposition = latest_disposition(thread)
        if disposition is None or disposition.kind != "rejected":
            continue
        if judge_verdict_for(thread, disposition, head_sha=head_sha) is not None:
            continue
        key = str(disposition.comment_id or thread.get("id", ""))
        why = not_judgeable(disposition)
        top = _comments(thread)[0]
        top_id = top.get("databaseId")
        if why or not isinstance(top_id, int):
            notes[key] = why or "the finding's first comment has no id to reply to"
            continue
        first_body = str(top.get("body", ""))
        label = _LABEL_RE.search(first_body)
        pending.append(
            PendingRejection(
                key=key,
                top_comment_id=int(top_id),
                digest=disposition.digest,
                rejection_url=disposition.url,
                raised_by=label.group("model") if label else "unknown",
                finding=finding_text(first_body, JUDGE_FINDING_CHARS),
                rejection=disposition.reason,
                cited=cited_lines(thread),
            )
        )
    return pending, notes


def code_excerpt(
    history: GitFixVerifier, cited: CitedLines, head: str
) -> tuple[str, str]:
    """(label, numbered excerpt) of the cited lines as they lie at the head."""
    text = history.file_at(head, cited.path)
    if text is None:
        return (
            f"{cited.path} (absent at the head)",
            "(the file does not exist at the pull request head)",
        )
    lines = text.splitlines()
    note = ""
    try:
        span = history.carry_forward(cited, head)
    except RuntimeError as exc:
        span = (cited.start, cited.end) if cited.start and cited.end else None
        note = f"; the range could not be carried to the head ({exc})"
    if span is None:
        low, high = 1, min(len(lines), JUDGE_FILE_EXCERPT_LINES)
        label = (
            f"{cited.path} (a file-level finding; lines {low}-{high} of "
            f"{len(lines)}){note}"
        )
    else:
        start, end = span
        low = max(1, start - JUDGE_EXCERPT_CONTEXT_LINES)
        high = min(len(lines), end + JUDGE_EXCERPT_CONTEXT_LINES)
        label = (
            f"{cited.path}:{start}-{end} (shown with "
            f"{JUDGE_EXCERPT_CONTEXT_LINES} lines of context){note}"
        )
    numbered = "\n".join(f"{n:>6}  {lines[n - 1]}" for n in range(low, high + 1))
    if len(numbered) > JUDGE_EXCERPT_CHARS:
        numbered = numbered[:JUDGE_EXCERPT_CHARS] + "\n…[excerpt truncated]"
    return label, numbered or "(the cited range is empty at the head)"


def _write_json(path: str | Path, data: object) -> None:
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", "utf-8")


def _read_status(path: str | None) -> dict[str, Any]:
    status: dict[str, Any] = {"unavailable": "", "notes": {}, "replies": {}}
    if path and Path(path).is_file():
        try:
            loaded = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            status["unavailable"] = f"the judge status file is unreadable: {exc}"
            return status
        if isinstance(loaded, dict):
            status.update(loaded)
    return status


def _set_output(name: str, value: str) -> None:
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(f"{name}={value}\n")


def _request_fields(data: object) -> tuple[int, str, int]:
    """Validate the untrusted request's exact schema before using any field."""
    if not isinstance(data, dict) or set(data) != {"pr", "head_sha", "count"}:
        raise ValueError(
            "the judge request must contain exactly pr, head_sha and count"
        )
    pr, head_sha, count = data["pr"], data["head_sha"], data["count"]
    if type(pr) is not int or pr <= 0:
        raise ValueError("the judge request pr must be a positive integer")
    if not isinstance(head_sha, str) or re.fullmatch(r"[0-9a-f]{40}", head_sha) is None:
        raise ValueError(
            "the judge request head_sha must be 40 lowercase hex characters"
        )
    if type(count) is not int or count < 0:
        raise ValueError("the judge request count must be a non-negative integer")
    return pr, head_sha, count


def run_write_request(path: Path) -> int:
    """Write only the PR number, head SHA and pending count for the base judge."""
    try:
        data = {
            "pr": int(os.environ["PR_NUMBER"]),
            "head_sha": os.environ["HEAD_SHA"],
            "count": int(os.environ["PENDING_COUNT"]),
        }
        _request_fields(data)
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(path, data)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        print(f"::error::could not write the judge request (fail-closed): {exc}")
        return 1
    return 0


def run_read_request(
    path: Path,
    *,
    api: Callable[..., Any] | None = None,
    base: str | None = None,
) -> int:
    """Read the gate artifact as data and verify its head against GitHub.

    No output is emitted until the exact schema, triggering run's head and
    current same-repository PR head all agree. A stale or forged request
    fails closed before any PR objects are fetched or credentials minted.
    """
    try:
        pr, head_sha, count = _request_fields(json.loads(path.read_text("utf-8")))
        if head_sha != os.environ["WORKFLOW_RUN_HEAD_SHA"]:
            raise ValueError(
                "the judge request head_sha differs from WORKFLOW_RUN_HEAD_SHA"
            )
        repo = os.environ["REPO"]
        if api is None or base is None:
            import post_hostile_review_threads as poster

            api = api or poster._api
            base = base or poster.repo_api_base(repo)
        pull = api("GET", f"{base}/pulls/{pr}", os.environ["GITHUB_TOKEN"])
        if pull["head"]["sha"] != head_sha:
            raise ValueError("the PR head.sha differs from the judge request")
        if pull["head"]["repo"]["full_name"] != repo:
            raise ValueError("the PR head repository differs from REPO (fork refused)")
        _set_output("pr", str(pr))
        _set_output("head_sha", head_sha)
        _set_output("count", str(count))
    except (OSError, RuntimeError, KeyError, TypeError, ValueError) as exc:
        print(f"::error::could not read the judge request (fail-closed): {exc}")
        return 1
    return 0


def run_pending(
    pending_path: Path,
    *,
    fetch: Callable[[str, int, str], list[dict[str, Any]]] | None = None,
    repo_dir: Path | None = None,
) -> int:
    """Write the rejections awaiting the review bot, and ``count=<n>``."""
    repo = os.environ["REPO"]
    pr_number = int(os.environ["PR_NUMBER"])
    head_sha = os.environ["HEAD_SHA"]
    status_path = os.environ.get("JUDGE_STATUS_PATH")
    try:
        threads = (fetch or fetch_review_threads)(
            repo, pr_number, os.environ["GITHUB_TOKEN"]
        )
    except (OSError, RuntimeError, KeyError, TypeError, ValueError) as exc:
        print(f"::error::could not fetch review threads (fail-closed): {exc}")
        return 1
    pending, notes = pending_rejections(threads, head_sha=head_sha)
    unavailable = ""
    if pending and os.environ.get("HOSTILE_REVIEW_IS_FORK", "false") == "true":
        unavailable = FORK_UNAVAILABLE
        pending = []
    history = GitFixVerifier(repo_dir or Path.cwd(), head_sha, head_sha)
    items = []
    for item in pending:
        lines_label, excerpt = code_excerpt(history, item.cited, head_sha)
        items.append(
            {
                "key": item.key,
                "raised_by": item.raised_by,
                "finding": item.finding or "(the finding has no text)",
                "lines_label": lines_label,
                "excerpt": excerpt,
                "rejection": item.rejection,
            }
        )
    _write_json(
        pending_path,
        {"repo": repo, "pr": pr_number, "head_sha": head_sha, "items": items},
    )
    if status_path:
        _write_json(
            status_path,
            {
                "unavailable": unavailable,
                "notes": notes,
                "replies": {
                    item.key: {
                        "top_comment_id": item.top_comment_id,
                        "digest": item.digest,
                        "head_sha": head_sha,
                        "rejection_url": item.rejection_url,
                        "raised_by": item.raised_by,
                    }
                    for item in pending
                },
            },
        )
    for key, why in notes.items():
        print(f"rejection {key} is not judged: {why}")
    if unavailable:
        print(f"::warning::the review bot cannot judge here: {unavailable}")
    print(f"{len(items)} rejection(s) await the review bot's judgement.")
    _set_output("count", str(len(items)))
    return 0


def _flat(text: str, limit: int) -> str:
    """One bounded line that cannot open markup."""
    flat = " ".join(str(text).replace("<", " ").replace(">", " ").split())
    return flat if len(flat) <= limit else flat[:limit].rsplit(" ", 1)[0] + " …"


def verdict_body(
    *,
    rejection_id: int,
    digest: str,
    head_sha: str,
    verdict: str,
    judge: str,
    identity: str,
    sentence: str,
) -> str:
    """The judge reply this gate reads: the marker line, the verdict, its sentence."""
    identity = re.sub(r"[\s>]", "_", identity) or "unknown"
    marker = (
        f"<!-- {JUDGE_MARKER} rejection={rejection_id} digest={digest} "
        f"head={head_sha} "
        f"verdict={verdict} judge={judge} identity={identity} -->"
    )
    if verdict == "accepted":
        tail = (
            "This finding no longer blocks the `Hostile Review Thread Gate`. The "
            "judgement is bound to the rejection as written and this PR head: "
            "a push, an edit of the rejection or a later disposition replaces it."
        )
    else:
        tail = (
            "This finding still blocks the `Hostile Review Thread Gate`. Reply "
            "with a line `fixed-in <sha>` (a commit of this PR that changes the "
            "cited lines), or a new `rejected: <reason>` that answers the "
            "sentence above; a new or edited rejection is judged again."
        )
    return (
        f"{marker}\n"
        f"**Review bot: rejection {verdict}** (judge `{judge}`, model "
        f"`{identity}`, OMN-17427)\n\n"
        f"> {_flat(sentence, JUDGE_SENTENCE_CHARS)}\n\n"
        f"{tail}"
    )


def run_post_verdicts(
    result_path: Path,
    *,
    api: Callable[..., Any] | None = None,
    base: str | None = None,
) -> int:
    """Post the judge's verdicts as replies under the onexbot App token.

    Fails closed: a missing or unreadable result, a malformed verdict or a
    reply GitHub refuses posts nothing for that rejection, and the reason is
    written to the status the gate step reads.
    """
    repo = os.environ["REPO"]
    pr_number = int(os.environ["PR_NUMBER"])
    token = os.environ["GITHUB_TOKEN"]
    status_path = os.environ.get("JUDGE_STATUS_PATH")
    status = _read_status(status_path)
    notes: dict[str, str] = dict(status.get("notes") or {})
    replies: dict[str, Any] = dict(status.get("replies") or {})
    judged: list[dict[str, Any]] = []
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        result = {"unavailable": NO_JUDGE_RESULT}
    if not isinstance(result, dict):
        result = {"unavailable": NO_JUDGE_RESULT}
    unavailable = str(status.get("unavailable") or result.get("unavailable") or "")
    for key, why in (result.get("notes") or {}).items():
        notes[str(key)] = str(why)
    if api is None or base is None:
        # The thread poster's REST call and base: the same origin and token.
        import post_hostile_review_threads as poster

        api = api or poster._api
        base = base or poster.repo_api_base(repo)
    for record in result.get("judgements") or []:
        key = str(record.get("key", "")) if isinstance(record, dict) else ""
        target = replies.get(key)
        if not isinstance(target, dict):
            notes[key or "?"] = "the judge returned a verdict for no pending rejection"
            continue
        verdict = str(record.get("verdict", ""))
        judge = str(record.get("judge_id", ""))
        identity = str(record.get("identity", ""))
        sentence = _flat(str(record.get("sentence", "")), JUDGE_SENTENCE_CHARS)
        if verdict not in JUDGE_VERDICTS or not _JUDGE_ID_RE.match(judge):
            notes[key] = f"the judge's verdict is malformed: {record!r}"[:400]
            continue
        if not sentence or not identity:
            notes[key] = "the judge's verdict carries no sentence or identity"
            continue
        judged_head = target.get("head_sha")
        if (
            not isinstance(judged_head, str)
            or re.fullmatch(r"[0-9a-f]{40}", judged_head) is None
        ):
            notes[key] = "the pending rejection has no valid judged head SHA"
            continue
        body = verdict_body(
            rejection_id=int(key),
            digest=str(target["digest"]),
            head_sha=judged_head,
            verdict=verdict,
            judge=judge,
            identity=identity,
            sentence=sentence,
        )
        url = (
            f"{base}/pulls/{pr_number}/comments/{int(target['top_comment_id'])}/replies"
        )
        try:
            api("POST", url, token, {"body": body})
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            notes[key] = (
                f"the judge {verdict} it, but GitHub refused the verdict reply "
                f"({exc.code}): {detail}"
            )
            continue
        except urllib.error.URLError as exc:
            notes[key] = f"the judge {verdict} it, but the reply was not sent: {exc}"
            continue
        row = {
            "head_sha": judged_head,
            "rejection_comment_id": int(key),
            "rejection_url": target.get("rejection_url", ""),
            "voter_model": target.get("raised_by", ""),
            "judge": judge,
            "judge_identity": identity,
            "verdict": verdict,
            "sentence": sentence,
        }
        judged.append(row)
        print("hostile-review-judgement: " + json.dumps(row, sort_keys=True))
    unanswered = [
        key
        for key in replies
        if key not in {str(r["rejection_comment_id"]) for r in judged}
    ]
    for key in unanswered:
        if key not in notes and not unavailable:
            notes[key] = "the judge returned no verdict for it"
    if status_path:
        _write_json(
            status_path,
            {
                "unavailable": unavailable,
                "notes": notes,
                "replies": replies,
                "judged": judged,
            },
        )
    for key, why in notes.items():
        print(f"rejection {key} is not judged: {why}")
    if unavailable or unanswered:
        print(
            f"::error::the review bot could not judge {len(unanswered)} "
            "rejection(s); their findings stay blocked (fail-closed): "
            f"{unavailable or 'reasons above'}"
        )
        return 1
    print(f"The review bot judged {len(judged)} rejection(s).")
    return 0


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
        if verdict.judge_verdict:
            disposition += f" ({verdict.judge_verdict} by the review bot)"
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
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--write-request",
        type=Path,
        metavar="FILE",
        help="write the minimal request for the base-branch judge",
    )
    mode.add_argument(
        "--read-request",
        type=Path,
        metavar="FILE",
        help="validate the gate request against the triggering run and GitHub",
    )
    mode.add_argument(
        "--pending",
        type=Path,
        metavar="FILE",
        help="write the rejections awaiting the review bot (OMN-17427)",
    )
    mode.add_argument(
        "--post-verdicts",
        type=Path,
        metavar="FILE",
        help="post the review bot's verdicts from cli_review judge-rejections",
    )
    args = parser.parse_args(argv)
    if args.report:
        print_report(accuracy(_read_records(args.report)))
        return 0
    if args.pending:
        return run_pending(args.pending)
    if args.write_request:
        return run_write_request(args.write_request)
    if args.read_request:
        return run_read_request(args.read_request)
    if args.post_verdicts:
        return run_post_verdicts(args.post_verdicts)
    return run_gate()


if __name__ == "__main__":
    sys.exit(main())
