#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Sanctioned deletion of an orphan local branch in a canonical clone (OMN-18370).

## The defect this closes

A worktree pruner removes a worktree under the worktrees root and then has to
delete the branch that worktree held. The branch lives in the CLONE, so the
delete lands on the canonical-clone reference-transaction guard
(`scripts/git-hooks/canonical_clone_ref_guard.sh`) and is refused. Each half is
correct alone; where they meet, the worktree goes and the branch stays.

Measured 2026-09-16: a prune pass removed 171 rescue-only worktrees under the
operator consent row at `docs/tracking/ROLLING_WORK_LEDGER.md:3624`, and every
one of the 171 `git branch -D` calls that followed exited 128 against the
guard. Third occurrence of the same defect.

## What this is NOT

It is not a bypass. The guard still refuses a bare `git branch -D`, and the
Claude PreToolUse guard still refuses one typed as a tool call. This tool is the
declared path: it does the policy work the guard cannot do, then opens the
guard's narrow, per-ref door for exactly the refs it has cleared.

## The bar a branch must clear

1. The branch exists in the clone and is not checked out by any live worktree
   (a branch a worktree still holds is not an orphan).
2. It has NO OPEN pull request. A `gh` failure refuses the branch; an
   unanswerable question is not a pass.
3. It is EITHER fully merged into the clone's upstream default branch, OR its
   tip oid is recorded verbatim in the cited ledger, so the commit survives the
   deletion as a resolvable reference.
4. The consent citation resolves to an OPERATOR-CONSENT row.

Anything that cannot be established refuses the branch. Dry run is the default;
`--execute` is required to delete.

## Honest limit

This enforces evidence and blast radius, not authorisation. No file proves a
human said the words in the cited row. What it removes is the SILENT deletion:
one that leaves no artifact naming a ref, an oid and a consent row.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

CONSENT_MARKER = "| OPERATOR-CONSENT |"
CITATION_RE = re.compile(r"^(?P<path>.+):(?P<line>[1-9][0-9]*)$")
OID_RE = re.compile(r"^[0-9a-f]{40}$")


class RefusedError(Exception):
    """A precondition this tool cannot establish. Always fails closed."""


@dataclass(frozen=True)
class BranchVerdict:
    """One branch's outcome. `deleted` is only ever true under --execute."""

    repo: str
    branch: str
    ref: str
    tip: str | None
    eligible: bool
    reason: str
    deleted: bool


def _run(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run a command with stderr CAPTURED, never discarded (CLAUDE.md rule 16)."""
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False)


def resolve_consent(citation: str, omni_home: Path) -> str:
    """Return the cited row, or raise. A relative path resolves against OMNI_HOME."""
    match = CITATION_RE.match(citation)
    if match is None:
        raise RefusedError(
            f"consent citation {citation!r} is not of the form <path>:<line>"
        )
    raw = Path(match.group("path"))
    path = raw if raw.is_absolute() else omni_home / raw
    if not path.is_file():
        raise RefusedError(f"consent citation path does not exist: {path}")
    number = int(match.group("line"))
    lines = path.read_text(encoding="utf-8").splitlines()
    if number > len(lines):
        raise RefusedError(
            f"consent citation line {number} is past the end of {path} "
            f"({len(lines)} lines)"
        )
    row = lines[number - 1]
    if CONSENT_MARKER not in row:
        raise RefusedError(
            f"{path}:{number} is not an OPERATOR-CONSENT row; it reads: {row[:120]!r}"
        )
    return row


def repo_slug(repo: Path) -> str | None:
    """`OmniNode-ai/<name>` from the clone's origin, or None when there is none."""
    result = _run(["git", "-C", str(repo), "remote", "get-url", "origin"])
    if result.returncode != 0:
        return None
    url = result.stdout.strip()
    match = re.search(r"[:/]([^/:]+/[^/]+?)(?:\.git)?$", url)
    return match.group(1) if match else None


def default_branch(repo: Path) -> str | None:
    """The clone's upstream default ref, resolved from origin/HEAD."""
    result = _run(
        ["git", "-C", str(repo), "symbolic-ref", "--quiet", "refs/remotes/origin/HEAD"]
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def checked_out_branches(repo: Path) -> set[str]:
    """Every branch a live worktree of this clone currently holds."""
    result = _run(["git", "-C", str(repo), "worktree", "list", "--porcelain"])
    if result.returncode != 0:
        raise RefusedError(
            f"cannot enumerate worktrees of {repo}: {result.stderr.strip()}"
        )
    return {
        line.split(" ", 1)[1].strip()
        for line in result.stdout.splitlines()
        if line.startswith("branch ")
    }


def has_open_pr(slug: str, branch: str) -> bool:
    """True when an OPEN pull request exists for the branch. Raises on failure."""
    result = _run(
        [
            "gh",
            "pr",
            "list",
            "--repo",
            slug,
            "--head",
            branch,
            "--state",
            "open",
            "--limit",
            "50",
            "--json",
            "number",
        ]
    )
    if result.returncode != 0:
        raise RefusedError(
            f"cannot determine open pull requests for {slug} {branch}: "
            f"{result.stderr.strip()}"
        )
    return bool(json.loads(result.stdout or "[]"))


def tip_is_recorded(tip: str, ledger: Path) -> bool:
    """True when the full 40-hex tip appears verbatim in the ledger."""
    if not ledger.is_file():
        raise RefusedError(f"ledger not readable for tip lookup: {ledger}")
    return tip in ledger.read_text(encoding="utf-8")


def classify(
    repo: Path,
    branch: str,
    *,
    slug: str | None,
    upstream: str | None,
    held: set[str],
    ledger: Path,
) -> BranchVerdict:
    """Decide one branch. Every refusal names the clause it failed."""
    ref = f"refs/heads/{branch}"

    def verdict(eligible: bool, reason: str, tip: str | None = None) -> BranchVerdict:
        return BranchVerdict(
            repo=repo.name,
            branch=branch,
            ref=ref,
            tip=tip,
            eligible=eligible,
            reason=reason,
            deleted=False,
        )

    resolved = _run(["git", "-C", str(repo), "rev-parse", "--verify", "--quiet", ref])
    tip = resolved.stdout.strip()
    if resolved.returncode != 0 or not OID_RE.match(tip):
        return verdict(False, "branch_absent")

    if ref in held:
        return verdict(False, "worktree_still_holds_branch", tip)

    if slug is None:
        return verdict(False, "origin_slug_unresolvable", tip)
    if has_open_pr(slug, branch):
        return verdict(False, "open_pull_request", tip)

    merged = False
    if upstream is not None:
        ancestor = _run(
            ["git", "-C", str(repo), "merge-base", "--is-ancestor", tip, upstream]
        )
        merged = ancestor.returncode == 0
    if merged:
        return verdict(True, "merged_into_upstream_default", tip)

    if tip_is_recorded(tip, ledger):
        return verdict(True, "tip_recorded_in_ledger", tip)

    return verdict(False, "unmerged_and_tip_not_recorded", tip)


def delete(repo: Path, verdict: BranchVerdict, citation: str) -> BranchVerdict:
    """Open the guard's per-ref door for exactly this ref, then delete."""
    env = dict(os.environ)
    env["ONEX_BRANCH_DELETE_CONSENT"] = citation
    env["ONEX_BRANCH_DELETE_REFS"] = verdict.ref
    result = subprocess.run(
        ["git", "-C", str(repo), "branch", "-D", verdict.branch],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return BranchVerdict(
            **{
                **asdict(verdict),
                "eligible": False,
                "reason": f"delete_failed: {result.stderr.strip()[:200]}",
            }
        )
    return BranchVerdict(**{**asdict(verdict), "deleted": True})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Delete orphan local branches in a canonical clone, under a "
        "cited operator consent row (OMN-18370).",
    )
    parser.add_argument(
        "--consent",
        required=True,
        help="`<path>:<line>` citing an OPERATOR-CONSENT ledger row. A relative "
        "path resolves against OMNI_HOME.",
    )
    parser.add_argument("--repo", required=True, help="Path to the canonical clone.")
    parser.add_argument(
        "--branch",
        action="append",
        default=[],
        help="Branch to delete. Repeatable.",
    )
    parser.add_argument(
        "--branches-file",
        help="File of branch names, one per line. Blank lines and `#` comments "
        "are ignored.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete. Without it the run is a dry classification.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the report as JSON.")
    args = parser.parse_args(argv)

    omni_home_raw = os.environ.get("OMNI_HOME")
    if not omni_home_raw:
        print("ERROR: OMNI_HOME is not set", file=sys.stderr)
        return 2
    omni_home = Path(omni_home_raw)

    branches = list(args.branch)
    if args.branches_file:
        for line in Path(args.branches_file).read_text(encoding="utf-8").splitlines():
            name = line.strip()
            if name and not name.startswith("#"):
                branches.append(name)
    if not branches:
        print("ERROR: no branches given", file=sys.stderr)
        return 2

    repo = Path(args.repo).resolve()
    try:
        consent_row = resolve_consent(args.consent, omni_home)
        held = checked_out_branches(repo)
    except RefusedError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 3

    # The ledger the tip must be recorded in is the file the consent cites: the
    # authorisation and the durable record of what was destroyed are the same
    # artifact, so a tip can never be "recorded" somewhere nobody is reading.
    ledger_raw = Path(args.consent.rsplit(":", 1)[0])
    ledger = ledger_raw if ledger_raw.is_absolute() else omni_home / ledger_raw

    slug = repo_slug(repo)
    upstream = default_branch(repo)

    verdicts: list[BranchVerdict] = []
    for branch in branches:
        try:
            verdict = classify(
                repo,
                branch,
                slug=slug,
                upstream=upstream,
                held=held,
                ledger=ledger,
            )
        except RefusedError as exc:
            verdict = BranchVerdict(
                repo=repo.name,
                branch=branch,
                ref=f"refs/heads/{branch}",
                tip=None,
                eligible=False,
                reason=f"refused: {exc}",
                deleted=False,
            )
        if verdict.eligible and args.execute:
            verdict = delete(repo, verdict, args.consent)
        verdicts.append(verdict)

    report = {
        "repo": str(repo),
        "consent": args.consent,
        "consent_row": consent_row[:240],
        "executed": args.execute,
        "eligible": sum(1 for v in verdicts if v.eligible),
        "deleted": sum(1 for v in verdicts if v.deleted),
        "refused": sum(1 for v in verdicts if not v.eligible),
        "branches": [asdict(v) for v in verdicts],
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for v in verdicts:
            state = (
                "DELETED" if v.deleted else ("ELIGIBLE" if v.eligible else "REFUSED")
            )
            print(f"{state:9} {v.branch} {(v.tip or '-')[:12]} {v.reason}")
        print(
            f"-- {report['eligible']} eligible, {report['deleted']} deleted, "
            f"{report['refused']} refused"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
