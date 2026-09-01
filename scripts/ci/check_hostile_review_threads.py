# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Deterministic thread-resolution gate for hostile-reviewer findings (OMN-17492).

The LLM is the FINDER, never the gate: ``post_hostile_review_threads.py``
publishes findings as resolvable PR review threads, and THIS check is the
deterministic merge surface — it fails while unresolved hostile-reviewer
threads exist on the PR, and passes the moment humans/sessions resolve them.
No model opinion is consulted here; the only inputs are GitHub thread states.

Blocking rule
-------------
A thread blocks iff ALL of:
  - its first comment carries the ``hostile-reviewer-thread`` marker
    (other bots'/humans' threads are none of this gate's business), and
  - ``isResolved`` is false, and
  - ``isOutdated`` is false — an outdated thread's anchor content changed
    under it (force-push/fix), a deterministic signal the code moved; it is
    REPORTED but does not block, so a rebase cannot wedge the PR on a stale
    anchor. Resolve outdated threads for hygiene.

Failure posture: fail closed. An unreachable/malformed GraphQL response is a
nonzero exit, never a vacuous pass.

Reference: OMN-17492.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from typing import Any

MARKER = "hostile-reviewer-thread"

_QUERY = """
query($owner: String!, $name: String!, $pr: Int!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $pr) {
      reviewThreads(first: 100, after: $cursor) {
        pageInfo { hasNextPage endCursor }
        nodes {
          isResolved
          isOutdated
          path
          comments(first: 1) { nodes { body url } }
        }
      }
    }
  }
}
"""


def fetch_review_threads(repo: str, pr_number: int, token: str) -> list[dict[str, Any]]:
    """Fetch every review thread on the PR via GraphQL (paged)."""
    owner, name = repo.split("/", 1)
    threads: list[dict[str, Any]] = []
    cursor: str | None = None
    while True:
        payload = json.dumps(
            {
                "query": _QUERY,
                "variables": {
                    "owner": owner,
                    "name": name,
                    "pr": pr_number,
                    "cursor": cursor,
                },
            }
        ).encode()
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
            data = json.loads(resp.read())
        if data.get("errors"):
            raise RuntimeError(f"GraphQL errors: {data['errors']}")
        conn = data["data"]["repository"]["pullRequest"]["reviewThreads"]
        threads.extend(conn["nodes"])
        if not conn["pageInfo"]["hasNextPage"]:
            return threads
        cursor = conn["pageInfo"]["endCursor"]


def classify_threads(
    threads: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split marker threads into (blocking, outdated_unresolved).

    Resolved marker threads and non-marker threads are ignored entirely.
    """
    blocking: list[dict[str, Any]] = []
    outdated: list[dict[str, Any]] = []
    for thread in threads:
        comments = thread.get("comments", {}).get("nodes") or []
        first_body = str(comments[0].get("body", "")) if comments else ""
        if MARKER not in first_body:
            continue
        if thread.get("isResolved"):
            continue
        if thread.get("isOutdated"):
            outdated.append(thread)
        else:
            blocking.append(thread)
    return blocking, outdated


def main() -> int:
    token = os.environ["GITHUB_TOKEN"]
    repo = os.environ["REPO"]
    pr_number = int(os.environ["PR_NUMBER"])

    try:
        threads = fetch_review_threads(repo, pr_number, token)
    except Exception as exc:  # noqa: BLE001 - fail closed on ANY fetch problem
        print(f"::error::could not fetch review threads (fail-closed): {exc}")
        return 1

    blocking, outdated = classify_threads(threads)

    for thread in outdated:
        comments = thread.get("comments", {}).get("nodes") or []
        url = comments[0].get("url", "?") if comments else "?"
        print(
            f"::warning::outdated unresolved hostile-reviewer thread at "
            f"{thread.get('path', '?')} ({url}) — the anchored code changed; "
            "not blocking, resolve for hygiene"
        )

    if blocking:
        print(
            f"::error::{len(blocking)} unresolved hostile-reviewer thread(s) "
            "block this gate. Address or reject each finding, then resolve "
            "its thread in the PR review UI (OMN-17492: the model finds, "
            "thread resolution gates)."
        )
        for thread in blocking:
            comments = thread.get("comments", {}).get("nodes") or []
            url = comments[0].get("url", "?") if comments else "?"
            print(f"  UNRESOLVED: {thread.get('path', '?')} — {url}")
        return 1

    print(
        f"Hostile Review Thread Gate PASSED — "
        f"{len(outdated)} outdated-unresolved (non-blocking), 0 blocking."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
