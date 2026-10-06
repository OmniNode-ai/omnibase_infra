# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Decide the one declared case in which Contract Compliance evaluates nothing.

OMN-20135. The Contract Compliance Check now fails closed when the PR's ticket
has no contract at its evidence commit, or when no ticket resolves at all. One
kind of pull request carries no ticket by design: a dependency-bot bump, which
doctrine's PR-title rule exempts from carrying a ticket token. Such a PR has no
contract and no evidence reference, so evaluating it could only fail.

This module is that exemption, stated once and narrowly. It is the same
predicate ``ci_summary_gate`` already applies to the change-control caller jobs
(``_declared_ticketless_dependency_bot_skip``, OMN-19167), composed from the
same public helpers so the two cannot drift:

1. the event is ``pull_request`` or ``merge_group`` (push carries no PR and admits
   nothing). A merge_group event carries no author or title either, so they are
   read from the source PR the queue ref names, through the GitHub API: the
   queue re-evaluates the same PR the pull_request run already exempted, and
   judging it on empty facts ejected dependency-bot bumps from the queue;
2. the author is one of the dependency bots (``DEPENDENCY_BOT_AUTHORS``);
3. the mirrored PR-title rule exempts the PR from carrying a ticket (its
   bot-author arm already does for both bots);
4. neither the title nor the head ref carries a ticket token.

Anything else is evaluated. A bot PR that a lane retitled with a ticket is
judged against that ticket's contract; a human PR with a bump-shaped title must
carry a ticket like any other.

Writes ``exempt=true`` or ``exempt=false`` to ``--github-output``.
"""

from __future__ import annotations

import argparse
import json
import subprocess  # fixed argv, no shell, trusted gh binary
import sys
from pathlib import Path

from scripts.ci.ci_summary_gate import (
    DEPENDENCY_BOT_AUTHORS,
    PullRequestContext,
    title_rule_exempts_ticket,
)
from scripts.ci.resolve_contract_compliance_pr import parse_merge_queue_ref

_EXEMPTABLE_EVENTS = frozenset({"pull_request", "merge_group"})


def fetch_pull_request_context(repo: str, pr_number: str) -> PullRequestContext:
    """Read the PR's real author, title and head ref; empty on any failure.

    An unresolved context admits nothing, so a failed read fails closed into
    full evaluation.
    """

    try:
        completed = subprocess.run(
            [
                "gh",
                "api",
                f"repos/{repo}/pulls/{int(pr_number)}",
                "-H",
                "Accept: application/vnd.github+json",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        payload = json.loads(completed.stdout)
        return PullRequestContext(
            author=str(payload["user"]["login"]),
            title=str(payload["title"]),
            head_ref=str(payload["head"]["ref"]),
        )
    except (
        subprocess.SubprocessError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
    ) as exc:
        sys.stderr.write(f"::warning::could not read PR {pr_number}: {exc}\n")
        return PullRequestContext()


def is_declared_exemption(*, event_name: str, ctx: PullRequestContext) -> bool:
    """True only for a ticketless dependency-bot bump on a pull_request event."""

    if event_name not in _EXEMPTABLE_EVENTS or not ctx.is_resolved:
        return False
    if ctx.author not in DEPENDENCY_BOT_AUTHORS:
        return False
    if not title_rule_exempts_ticket(author=ctx.author, title=ctx.title):
        return False
    return not ctx.carries_ticket_token


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-name", required=True)
    parser.add_argument("--pr-author", default="")
    parser.add_argument("--pr-title", default="")
    parser.add_argument("--pr-head-ref", default="")
    parser.add_argument("--repo", default="")
    parser.add_argument("--merge-group-ref", default="")
    parser.add_argument("--github-output", required=True, type=Path)
    args = parser.parse_args(argv)

    ctx = PullRequestContext(
        author=args.pr_author, title=args.pr_title, head_ref=args.pr_head_ref
    )
    queue_pr = parse_merge_queue_ref(args.merge_group_ref)
    if args.event_name == "merge_group" and args.repo and queue_pr is not None:
        ctx = fetch_pull_request_context(args.repo, str(queue_pr))
    exempt = is_declared_exemption(event_name=args.event_name, ctx=ctx)
    with args.github_output.open("a", encoding="utf-8") as out:
        out.write(f"exempt={'true' if exempt else 'false'}\n")
    if exempt:
        sys.stdout.write(
            f"Declared exemption: ticketless dependency-bot bump by {ctx.author} "
            "(PR-title rule); no ticket contract applies, so no DoD item is run.\n"
        )
    else:
        sys.stdout.write(
            "Not exempt: the PR's ticket contract is resolved and evaluated.\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
