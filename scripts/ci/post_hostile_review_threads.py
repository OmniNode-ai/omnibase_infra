# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Post hostile-reviewer findings as resolvable PR review threads (OMN-17492).

Why this exists
---------------
The ``hostile-reviewer.yml`` workflow runs
``omniintelligence.review_pairing.cli_review`` on every PR, but until
OMN-17492 the findings CONTENT went nowhere: the verdict step parsed
``merged_findings`` / ``total_input_findings`` — fields
``ModelMultiReviewResult`` has never had — so the summary comment always
reported zero findings and the "blocked" verdict was dead code. The review
ran into the void.

Doctrine (finder, never the gate)
---------------------------------
The LLM is the FINDER. This script publishes its findings as ONE pull-request
review per run, with per-finding comments that become resolvable review
threads. The DETERMINISTIC merge surface is thread resolution: the companion
``Hostile Review Thread Gate`` job fails while unresolved hostile-reviewer
threads exist. No job ever exits nonzero because of a model's opinion.

Behavior
--------
- Reads the ``ModelMultiReviewResult`` JSON written by ``cli_review --output``.
- Drops ``hint``-severity findings (nit-class noise; counted, not posted).
- Dedupes against threads posted by earlier runs of this PR via a stable
  fingerprint embedded in each comment (``fp=<sha12>``), so a ``synchronize``
  re-run does not re-post the same finding (resolved threads stay resolved).
- Anchors each finding to the PR diff when it can:
  line-anchored (``path:line`` parsed from the finding location and present
  in the patch's RIGHT side) > file-anchored (``subject_type: file`` for a
  changed file) > review body (everything else).
- Caps thread comments per run (:data:`MAX_THREAD_COMMENTS`); the overflow is
  listed in the review body under a LOUD truncation banner.
- Review event: ``REQUEST_CHANGES`` when any critical/major
  (``error``/``warning``) finding is being posted, else ``COMMENT``.
  Zero postable findings -> no review is posted at all.

Failure posture
---------------
Posting failures exit nonzero (the job must not report a review that was
never published) EXCEPT when ``HOSTILE_REVIEW_IS_FORK=true``: fork PRs get a
read-only ``GITHUB_TOKEN`` (OMN-16235 precedent), so posting is structurally
impossible and the script exits 0 after logging.

Reference: OMN-17492 (GLM-backed hostile review in CI, thread-gated).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request
from typing import Any

MARKER = "hostile-reviewer-thread"
MAX_THREAD_COMMENTS = 25
MAX_BODY_CHARS = 60_000
MAX_FINDING_CHARS = 1_500

# Severity values as emitted by review_pairing (EnumFindingSeverity.value),
# already mapped from the prompt's critical/major/minor/nit scale.
_SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2, "hint": 3}
_SEVERITY_LABEL = {
    "error": "CRITICAL",
    "warning": "MAJOR",
    "info": "MINOR",
    "hint": "NIT",
}
_REQUEST_CHANGES_SEVERITIES = frozenset({"error", "warning"})

_LOCATION_LINE_RE = re.compile(r"^(?P<path>[^:\s][^:]*?):(?P<line>\d{1,6})\b")


def _api(
    method: str,
    url: str,
    token: str,
    payload: dict[str, Any] | None = None,
) -> Any:
    """Minimal GitHub REST call. Raises urllib.error.HTTPError on failure."""
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(  # noqa: S310 - fixed https://api.github.com base
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
        body = resp.read()
    return json.loads(body) if body else None


def _paginate(url: str, token: str) -> list[dict[str, Any]]:
    """Fetch every page of a list endpoint (per_page=100, page=N)."""
    results: list[dict[str, Any]] = []
    page = 1
    while True:
        sep = "&" if "?" in url else "?"
        chunk = _api("GET", f"{url}{sep}per_page=100&page={page}", token)
        if not chunk:
            break
        results.extend(chunk)
        if len(chunk) < 100:
            break
        page += 1
    return results


def collect_findings(review_result: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten per-model findings from a ModelMultiReviewResult dump.

    Attaches ``model`` to each finding and drops nothing here — severity
    filtering is a separate, testable step.
    """
    findings: list[dict[str, Any]] = []
    for model_result in review_result.get("results", []):
        if not model_result.get("success"):
            continue
        for finding in model_result.get("findings", []):
            enriched = dict(finding)
            enriched["model"] = model_result.get("model", "unknown")
            findings.append(enriched)
    return findings


def split_by_noise_policy(
    findings: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Return (postable findings sorted most-severe-first, suppressed count).

    ``hint`` (nit-class) findings are suppressed: style nits are noise in an
    adversarial review surface (OMN-17492 design constraint).
    """
    postable = [f for f in findings if str(f.get("severity", "info")) != "hint"]
    postable.sort(key=lambda f: _SEVERITY_ORDER.get(str(f.get("severity", "info")), 2))
    return postable, len(findings) - len(postable)


def finding_fingerprint(finding: dict[str, Any]) -> str:
    """Stable dedupe fingerprint for one finding across workflow runs.

    Uses model + rule_id + file_path + the normalized message head — NOT the
    line number, so a rebase that shifts lines does not re-post the same
    finding as a new thread.
    """
    basis = "|".join(
        [
            str(finding.get("model", "")),
            str(finding.get("rule_id", "")),
            str(finding.get("file_path", "")),
            str(finding.get("normalized_message", ""))[:200],
        ]
    )
    return hashlib.sha256(basis.encode()).hexdigest()[:12]


def parse_right_side_lines(patch: str | None) -> set[int]:
    """RIGHT-side (new file) line numbers present in a unified diff patch.

    GitHub line-anchored review comments require the line to appear in the
    diff; anchoring anywhere else is rejected with a 422.
    """
    if not patch:
        return set()
    lines: set[int] = set()
    new_line = 0
    for raw in patch.splitlines():
        if raw.startswith("@@"):
            match = re.search(r"\+(\d+)(?:,(\d+))?", raw)
            if not match:
                continue
            new_line = int(match.group(1))
            continue
        if raw.startswith("+"):
            lines.add(new_line)
            new_line += 1
        elif raw.startswith("-"):
            continue
        elif raw.startswith("\\"):
            # "\ No newline at end of file" — no line consumed either side.
            continue
        else:
            lines.add(new_line)
            new_line += 1
    return lines


def resolve_anchor(
    finding: dict[str, Any],
    changed_files: dict[str, set[int]],
) -> tuple[str, str | None, int | None]:
    """Classify a finding's diff anchor.

    Returns:
        ("line", path, line) — line-anchored comment;
        ("file", path, None) — file-level comment on a changed file;
        ("body", None, None) — render in the review body.
    """
    raw_path = str(finding.get("file_path", "") or "")
    match = _LOCATION_LINE_RE.match(raw_path)
    if match:
        path = match.group("path")
        line = int(match.group("line"))
        if path in changed_files and line in changed_files[path]:
            return ("line", path, line)
        if path in changed_files:
            return ("file", path, None)
    if raw_path in changed_files:
        return ("file", raw_path, None)
    return ("body", None, None)


def format_finding_body(finding: dict[str, Any]) -> str:
    """Render one finding as a review-comment body (bounded size)."""
    severity = str(finding.get("severity", "info"))
    label = _SEVERITY_LABEL.get(severity, severity.upper())
    model = str(finding.get("model", "unknown"))
    message = str(finding.get("raw_message", "")) or str(
        finding.get("normalized_message", "")
    )
    if len(message) > MAX_FINDING_CHARS:
        message = message[:MAX_FINDING_CHARS] + " …[truncated]"
    fp = finding_fingerprint(finding)
    return (
        f"<!-- {MARKER} fp={fp} -->\n"
        f"**[{label}] hostile-reviewer ({model})**\n\n"
        f"{message}\n\n"
        f"_Resolve this thread when addressed — the `Hostile Review Thread "
        f"Gate` blocks while hostile-reviewer threads are unresolved "
        f"(OMN-17492)._"
    )


def build_review(
    findings: list[dict[str, Any]],
    changed_files: dict[str, set[int]],
    existing_fps: set[str],
    *,
    suppressed_hints: int,
    models_succeeded: list[str],
    models_failed: list[str],
    max_thread_comments: int = MAX_THREAD_COMMENTS,
) -> tuple[dict[str, Any] | None, dict[str, int]]:
    """Assemble the single review payload for this run.

    Returns (payload_or_None, stats). payload is None when there is nothing
    new to post (all findings deduped, or no postable findings at all).
    """
    stats = {
        "posted_threads": 0,
        "body_findings": 0,
        "deduped": 0,
        "truncated": 0,
        "suppressed_hints": suppressed_hints,
    }
    comments: list[dict[str, Any]] = []
    body_sections: list[str] = []
    truncated: list[dict[str, Any]] = []
    request_changes = False

    for finding in findings:
        fp = finding_fingerprint(finding)
        if fp in existing_fps:
            stats["deduped"] += 1
            continue
        existing_fps.add(fp)
        severity = str(finding.get("severity", "info"))
        if severity in _REQUEST_CHANGES_SEVERITIES:
            request_changes = True
        anchor_kind, path, line = resolve_anchor(finding, changed_files)
        if anchor_kind != "body" and len(comments) < max_thread_comments:
            comment: dict[str, Any] = {
                "path": path,
                "body": format_finding_body(finding),
            }
            if anchor_kind == "line":
                comment["line"] = line
                comment["side"] = "RIGHT"
            else:
                comment["subject_type"] = "file"
            comments.append(comment)
        elif anchor_kind != "body":
            truncated.append(finding)
        else:
            body_sections.append(
                f"- {format_finding_body(finding)}".replace("\n", "\n  ")
            )
            stats["body_findings"] += 1

    stats["posted_threads"] = len(comments)
    stats["truncated"] = len(truncated)

    if not comments and not body_sections and not truncated:
        return None, stats

    body_parts = [
        f"<!-- {MARKER} summary -->",
        "## Hostile Reviewer — adversarial findings (OMN-17492)",
        "",
        f"Models succeeded: {', '.join(models_succeeded) or 'none'}  ",
        f"Models failed: {', '.join(models_failed) or 'none'}  ",
        f"New finding threads: {len(comments)}  ",
        f"Deduped (already posted on this PR): {stats['deduped']}  ",
        f"Nit-level findings suppressed: {suppressed_hints}",
        "",
        "The model is the FINDER, never the gate: merge is gated only by the",
        "deterministic `Hostile Review Thread Gate`, which blocks while",
        "hostile-reviewer threads are unresolved. Resolve each thread after",
        "addressing (or rejecting, with a reply) its finding.",
    ]
    if truncated:
        body_parts += [
            "",
            f"### ⚠️ TRUNCATED: {len(truncated)} finding(s) over the "
            f"{MAX_THREAD_COMMENTS}-thread cap",
            "",
            "These did NOT get threads this run (they will on a later run "
            "once threads resolve):",
            "",
        ]
        body_parts += [
            f"- **[{_SEVERITY_LABEL.get(str(f.get('severity', 'info')), '?')}]** "
            f"`{f.get('file_path', '?')}` — "
            f"{str(f.get('normalized_message', ''))[:200]}"
            for f in truncated
        ]
    if body_sections:
        body_parts += ["", "### Findings not anchored to a changed file", ""]
        body_parts += body_sections

    body = "\n".join(body_parts)
    if len(body) > MAX_BODY_CHARS:
        body = body[:MAX_BODY_CHARS] + "\n\n…[review body truncated]"

    payload: dict[str, Any] = {
        "event": "REQUEST_CHANGES" if request_changes else "COMMENT",
        "body": body,
        "comments": comments,
    }
    return payload, stats


def main() -> int:
    token = os.environ["GITHUB_TOKEN"]
    repo = os.environ["REPO"]
    pr_number = int(os.environ["PR_NUMBER"])
    review_json_path = os.environ["REVIEW_JSON_PATH"]
    is_fork = os.environ.get("HOSTILE_REVIEW_IS_FORK", "false") == "true"
    head_sha = os.environ.get("HEAD_SHA", "")

    if is_fork:
        print(
            "Fork PR: GITHUB_TOKEN is read-only (OMN-16235); skipping thread "
            "posting entirely."
        )
        return 0

    try:
        with open(review_json_path, encoding="utf-8") as fh:
            review_result = json.load(fh)
    except FileNotFoundError:
        print(
            "No hostile-review result JSON exists; review step already degraded "
            "and reported zero postable findings."
        )
        return 0
    except (OSError, json.JSONDecodeError) as exc:
        print(f"::error::cannot read review JSON at {review_json_path}: {exc}")
        return 1

    findings = collect_findings(review_result)
    postable, suppressed_hints = split_by_noise_policy(findings)

    base = f"https://api.github.com/repos/{repo}"  # url-authority-ok: GitHub REST API base for the CI thread poster; runs only inside GitHub Actions with GITHUB_TOKEN, same fixed origin every gh CLI call in this repo resolves

    # Existing hostile-reviewer fingerprints on this PR: review comments
    # (threads) AND review bodies (summary/body-only findings live there).
    existing_fps: set[str] = set()
    for comment in _paginate(f"{base}/pulls/{pr_number}/comments", token):
        for fp in re.findall(
            rf"{MARKER} fp=([0-9a-f]{{12}})", comment.get("body") or ""
        ):
            existing_fps.add(fp)
    for review in _paginate(f"{base}/pulls/{pr_number}/reviews", token):
        for fp in re.findall(
            rf"{MARKER} fp=([0-9a-f]{{12}})", review.get("body") or ""
        ):
            existing_fps.add(fp)

    changed_files: dict[str, set[int]] = {}
    for file_info in _paginate(f"{base}/pulls/{pr_number}/files", token):
        changed_files[str(file_info.get("filename"))] = parse_right_side_lines(
            file_info.get("patch")
        )

    payload, stats = build_review(
        postable,
        changed_files,
        existing_fps,
        suppressed_hints=suppressed_hints,
        models_succeeded=list(review_result.get("models_succeeded", [])),
        models_failed=list(review_result.get("models_failed", [])),
    )
    print(f"thread-poster stats: {json.dumps(stats)}")

    if payload is None:
        print("No new findings to post (all deduped or none postable).")
        return 0

    if head_sha:
        payload["commit_id"] = head_sha
    try:
        _api("POST", f"{base}/pulls/{pr_number}/reviews", token, payload)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:1000]
        if payload["comments"]:
            # A single bad anchor 422s the WHOLE review. Retry once with every
            # finding demoted to the review body so nothing is silently lost.
            print(
                f"::warning::review POST failed ({exc.code}): {detail} — "
                "retrying with all findings demoted to the review body"
            )
            demoted = [c["body"].replace("\n", "\n  ") for c in payload["comments"]]
            payload_retry = {
                "event": payload["event"],
                "body": (
                    payload["body"]
                    + "\n\n### Findings demoted from threads (anchor rejected)\n\n"
                    + "\n".join(f"- {body}" for body in demoted)
                )[:MAX_BODY_CHARS],
                "comments": [],
            }
            if head_sha:
                payload_retry["commit_id"] = head_sha
            try:
                _api("POST", f"{base}/pulls/{pr_number}/reviews", token, payload_retry)
            except urllib.error.HTTPError as retry_exc:
                retry_detail = retry_exc.read().decode(errors="replace")[:1000]
                print(
                    f"::error::review POST retry failed "
                    f"({retry_exc.code}): {retry_detail}"
                )
                return 1
        else:
            print(f"::error::review POST failed ({exc.code}): {detail}")
            return 1

    print(
        f"Posted hostile review ({payload['event']}) with "
        f"{len(payload['comments'])} thread comment(s)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
