# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Resolve the PR whose DoD ``check_values`` ``contract-compliance`` must run.

OMN-16508. ``ci.yml``'s ``contract-compliance`` job invokes
``run_contract_compliance_check.py --pr <n>``. ``<n>`` came straight from
``github.event.pull_request.number``, which only the ``pull_request`` event
populates -- but the job's ``if:`` also admits ``push`` and ``merge_group``.
The pre-fix step met the empty case with ``exit 0``.

That ``exit 0`` was a **vacuous pass**, and a load-bearing one:
``"Contract Compliance Check"`` is a ``GATE_JOBS`` entry in
:mod:`scripts.ci.ci_summary_gate`, and ``CI Summary`` is the sole required
branch-protection context on this repo. The umbrella poller therefore counted a
run that evaluated zero ``check_values`` as a *provable* pass. Unlike a red
gate, a green one that checked nothing never prompts investigation.

Live fail-open surface, stated precisely: ``ci.yml``'s ``on.push.branches`` is
``[main]``, so this never fired on a ``dev`` commit. The two real events are
``push`` to ``main`` (release-synced fast-forwards) and ``merge_group`` --
the latter unexercised today (no repo in the registry has a queue on ``dev`` as
of 2026-08-24) but a latent bypass of the repo's only required context the
moment a queue is re-enabled.

Resolution, not narrowing
-------------------------
Excluding ``push``/``merge_group`` with a job-level ``if:`` would publish a
``skipped`` check run, which branch protection counts as passing -- the same
skip-vector class ``reject-required-check-skip-vector`` (OMN-14863) rejects,
and which ``ci_summary_gate``'s external-context assertion treats as a hard
failure. So the job keeps running on all three events and this module answers
*which* PR's ``check_values`` apply:

``pull_request``
    ``github.event.pull_request.number``, verbatim. No API call -- this path
    stays behaviourally identical to pre-fix.
``merge_group``
    The queue ref carries it. GitHub names the branch
    ``refs/heads/gh-readonly-queue/<base>/pr-<n>-<sha>``, so no API call is
    needed here either.
``push``
    ``gh api repos/{repo}/commits/{sha}/pulls`` -- GitHub associates a
    squash-merge commit with its source PR. Only a MERGED association counts,
    and a PR whose ``merge_commit_sha`` is exactly this SHA wins over one that
    merely contains the commit.

Anything unresolved -- including a transient ``gh`` failure -- exits non-zero.
A gate that cannot determine what to check must not report that it checked.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess  # fixed argv, no shell, trusted gh binary
import sys
from typing import Any, Protocol

EXIT_PASS = 0
EXIT_FAIL = 1

# GitHub's merge-queue branch shape: refs/heads/gh-readonly-queue/<base>/pr-<n>-<sha>
_MERGE_QUEUE_REF = re.compile(r"gh-readonly-queue/[^/]+/pr-(\d+)-")


class CommitPullsFetcher(Protocol):
    """Fetches the PRs GitHub associates with a commit."""

    def __call__(self, repo: str, sha: str) -> list[dict[str, Any]]: ...


def parse_merge_queue_ref(ref: str) -> int | None:
    """Return the source PR number carried by a merge-queue branch ref."""
    match = _MERGE_QUEUE_REF.search(ref)
    return int(match.group(1)) if match else None


def fetch_commit_pulls_via_gh(repo: str, sha: str) -> list[dict[str, Any]]:
    """Ask the GitHub API which PRs are associated with ``sha``."""
    completed = subprocess.run(
        [
            "gh",
            "api",
            f"repos/{repo}/commits/{sha}/pulls",
            "-H",
            "Accept: application/vnd.github+json",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    payload = json.loads(completed.stdout)
    if not isinstance(payload, list):
        raise RuntimeError(f"unexpected /pulls payload shape: {type(payload).__name__}")
    return [entry for entry in payload if isinstance(entry, dict)]


def resolve_pr_number(
    *,
    repo: str,
    pr_number: str,
    merge_group_ref: str,
    sha: str,
    fetch_commit_pulls: CommitPullsFetcher,
) -> int | None:
    """Resolve the governing PR number, or ``None`` to fail closed."""
    if pr_number.strip():
        return int(pr_number.strip())

    from_queue = parse_merge_queue_ref(merge_group_ref)
    if from_queue is not None:
        return from_queue

    if not sha.strip():
        return None

    try:
        associated = fetch_commit_pulls(repo, sha.strip())
    except Exception as exc:  # noqa: BLE001 - any failure must fail closed
        print(
            f"::warning::could not list PRs for commit {sha}: {exc}",
            file=sys.stderr,
        )
        return None

    merged = [entry for entry in associated if entry.get("merged_at")]
    if not merged:
        return None

    # A commit can be associated with several PRs; prefer the one that
    # actually produced it.
    for entry in merged:
        if entry.get("merge_commit_sha") == sha.strip():
            return int(entry["number"])
    return int(merged[0]["number"])


def main(
    argv: list[str] | None = None,
    *,
    fetch_commit_pulls: CommitPullsFetcher = fetch_commit_pulls_via_gh,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="owner/name of this repository")
    parser.add_argument(
        "--pr-number", default="", help="github.event.pull_request.number"
    )
    parser.add_argument(
        "--merge-group-ref", default="", help="github.event.merge_group.head_ref"
    )
    parser.add_argument("--sha", default="", help="the commit to resolve a PR for")
    parser.add_argument("--event-name", default="", help="github.event_name, for logs")
    args = parser.parse_args(argv)

    resolved = resolve_pr_number(
        repo=args.repo,
        pr_number=args.pr_number,
        merge_group_ref=args.merge_group_ref,
        sha=args.sha,
        fetch_commit_pulls=fetch_commit_pulls,
    )
    if resolved is None:
        print(
            f"::error::Could not resolve a merged PR for event="
            f"{args.event_name or '?'} sha={args.sha or '?'}. DoD check_values "
            f"are PR-scoped, so there is nothing to evaluate -- failing closed "
            f"rather than reporting a green gate that ran zero checks "
            f"(OMN-16508).",
            file=sys.stderr,
        )
        return EXIT_FAIL

    print(resolved)
    return EXIT_PASS


if __name__ == "__main__":
    raise SystemExit(main())
