#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unattended worktree prune, run by launchd rather than by a session (OMN-18832).

## The defect this closes

The worktree prune automation existed only as a Workflow script
(`.claude/workflows/morning-worktree-prune.js`), which runs when a human starts
a session and invokes it. Measured 2026-09-19: the ledger carries zero rows
from any lane named `morning-worktree-prune`, so it had never run once, and
`scripts/launchd/` carried no worktree entry. launchd DOES fire on this Mac —
the claim that it does not was corrected under OMN-17173 — so the timer was
simply never written.

## What this is NOT

It is **not a second classifier.** `omniclaude/scripts/worktree_auto_prune.py`
is 3,099 lines and already collects facts, classifies them against the pure
predicate in `omniclaude/src/omniclaude/hooks/lib/worktree_prune_policy.py`,
captures git's exit code and stderr, and retries a timeout once when host load
falls. That README states the rule directly: *do not move policy into the
workflow script*, because a predicate that lives only inside a morning agent
brief cannot be called by an event. This runner drives that script as a
subprocess and adds only the four things it does not do.

## The four things it adds

1. **A schedule.** It is the launchd entrypoint, so the prune no longer waits
   for somebody to open a session.

2. **The removal-debris path (OMN-18826).** `git worktree remove` without
   `--force` is not atomic on git 2.50.1: one unwritable subdirectory makes it
   exit 255, delete the admin directory and leave survivors behind a stale
   `.git` file. The second removal over that half-deleted tree refuses with
   `contains modified or untracked files`, which reads as a peer lane's work
   when it is debris from the first attempt. The signature that tells them
   apart is the porcelain — debris is unstaged DELETIONS and nothing else —
   and a forced removal is still taken only once the branch's pull request is
   MERGED, because the pull request is what says the content survives.

3. **A ledger row.** One NOTE row per run carrying the counts and the
   refusals by reason, so a run that nobody watched is still visible.

4. **A refusal review by a LOCAL model.** Operator ruling 2026-09-19 ~13:36Z,
   verbatim: *"if we need an LLM in the loop for triage it should go to local
   models"*, recorded at `docs/tracking/ROLLING_WORK_LEDGER.md:2914`. The
   deterministic pass stays deterministic; what it REFUSED goes to
   `onex delegate` on the deployed lane in batches under the prompt ceiling,
   and the proposals are written into the report and acted on by NOBODY.

## Authorisation

The standing consent citation is read from a config file the operator can
revoke — never hardcoded, because a source-literal citation cannot be revoked
by editing a config. An absent config, a disabled flag, a citation that does
not resolve to an OPERATOR-CONSENT row, and an unreadable ledger each refuse
the whole run before anything is removed.

## Honest limit

This enforces evidence and blast radius, not authorisation. No file proves a
human said the words in the cited row. What it removes is the prune that never
ran at all, and the silent removal that left no artifact naming a path, an
exit code and a consent row.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

CONSENT_MARKER = "| OPERATOR-CONSENT |"
CITATION_RE = re.compile(r"^(?P<path>.+):(?P<line>[1-9][0-9]*)$")

# Measured 2026-09-19 on the local heavy-reasoning backend: a 5,497-character
# prompt hit the 240s handler budget and was cancelled with a typed timeout; a
# 2,509-character prompt carrying the same facts completed in 78s. The ceiling
# is that measurement, not a guess.
DELEGATE_PROMPT_CEILING = 2_500

# The removal budget. The fixed 90s this replaces called a real 17,600-file
# removal a timeout under load 40, when it needed 280s (OMN-18826).
BUDGET_FLOOR_SECONDS = 90
BUDGET_SECONDS_PER_FILE = 0.02
BUDGET_CEILING_SECONDS = 3_600

# git status --porcelain: an unstaged deletion, and nothing else, is debris.
UNSTAGED_DELETION_RE = re.compile(r"^ D .+$")


class RefusedError(Exception):
    """A precondition this runner cannot establish. Always fails closed."""


@dataclass(frozen=True)
class ModelDebrisVerdict:
    """Whether one half-deleted tree may be removed with `--force`."""

    eligible: bool
    reason: str


@dataclass(frozen=True)
class ModelTimerConfig:
    """The operator's revocable authorisation for an unattended run."""

    enabled: bool
    consent_citation: str
    delegate_refusal_review: bool


@dataclass
class ModelRunReport:
    """One run, as it is written to disk and summarised into the ledger."""

    started_at: str
    executed: bool
    consent_citation: str
    classifier_exit: int | None = None
    worktrees_removed: int = 0
    debris_forced: int = 0
    branches_deleted: int = 0
    refusals: dict[str, int] = field(default_factory=dict)
    refusal_items: list[str] = field(default_factory=list)
    removal_failures: list[dict[str, object]] = field(default_factory=list)
    delegate_receipt: dict[str, object] = field(default_factory=dict)
    delegate_proposals: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pure predicates
# ---------------------------------------------------------------------------


def classify_removal_debris(porcelain: str, *, pr_merged: bool) -> ModelDebrisVerdict:
    """Decide whether a tree is OMN-18826 removal debris.

    Debris is unstaged deletions and NOTHING else. An untracked file is
    content that exists nowhere else; a modified file is an edit; a STAGED
    deletion is somebody having run `git rm` on purpose. Each of those is real
    work, so any one of them refuses the whole tree. An unrecognised status
    code refuses too, rather than being assumed benign.

    The porcelain says the tree is debris. The merged pull request is what
    says the deleted CONTENT survives the removal, so both are required.

    Args:
        porcelain: `git status --porcelain` output for the tree.
        pr_merged: Whether the tree's branch has a MERGED pull request,
            established live rather than from a cached list.

    Returns:
        A frozen verdict naming the clause it turned on.
    """
    lines = [line for line in porcelain.splitlines() if line.strip()]
    if not lines:
        return ModelDebrisVerdict(
            eligible=False,
            reason="clean tree: nothing to force, a plain removal suffices",
        )

    offending = [line for line in lines if not UNSTAGED_DELETION_RE.match(line)]
    if offending:
        shown = ", ".join(repr(line) for line in offending[:3])
        more = "…" if len(offending) > 3 else ""
        return ModelDebrisVerdict(
            eligible=False,
            reason=(
                f"{len(offending)} of {len(lines)} porcelain entries are not "
                f"unstaged deletions, so this is work rather than removal "
                f"debris: {shown}{more}"
            ),
        )

    if not pr_merged:
        return ModelDebrisVerdict(
            eligible=False,
            reason=(
                f"{len(lines)} unstaged deletions and nothing else, which is "
                "the debris shape, but the branch has no merged pull request "
                "so the deleted content is not proven to survive removal"
            ),
        )

    return ModelDebrisVerdict(
        eligible=True,
        reason=(
            f"{len(lines)} unstaged deletions, zero untracked and zero "
            "modified entries, and the branch's pull request is merged: "
            "OMN-18826 removal debris from a non-atomic worktree remove"
        ),
    )


def removal_budget_seconds(file_count: int) -> int:
    """A removal budget that scales with the tree rather than a constant.

    Bounded at both ends: an empty tree still gets a real budget, and a
    pathological count cannot hang a timer that nobody is watching.
    """
    scaled = BUDGET_FLOOR_SECONDS + BUDGET_SECONDS_PER_FILE * max(0, file_count)
    return int(min(BUDGET_CEILING_SECONDS, max(BUDGET_FLOOR_SECONDS, scaled)))


def batch_refusals(
    items: list[str] | tuple[str, ...], ceiling: int
) -> tuple[tuple[str, ...], ...]:
    """Split the refusal list into prompt-sized batches, dropping nothing.

    An item longer than the ceiling rides in a batch of its own rather than
    being dropped: silently shrinking the refusal list is the one thing a
    triage handoff must never do.
    """
    batches: list[tuple[str, ...]] = []
    current: list[str] = []
    size = 0
    for item in items:
        cost = len(item) + (1 if current else 0)
        if current and size + cost > ceiling:
            batches.append(tuple(current))
            current, size = [], 0
            cost = len(item)
        current.append(item)
        size += cost
        if size > ceiling:
            batches.append(tuple(current))
            current, size = [], 0
    if current:
        batches.append(tuple(current))
    return tuple(batches)


# ---------------------------------------------------------------------------
# Authorisation
# ---------------------------------------------------------------------------


def load_timer_config(path: Path) -> ModelTimerConfig:
    """Read the operator's revocable authorisation, or refuse.

    Parsed with a deliberately small reader rather than a YAML dependency:
    this file is three scalar keys and runs from launchd, where an import
    failure is an invisible non-run.
    """
    if not path.is_file():
        raise RefusedError(
            f"timer config does not exist: {path}. An unattended prune with no "
            "authorisation file does not run — deleting the file IS the revoke."
        )
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip() if not raw.strip().startswith("#") else ""
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        values[key.strip()] = value.strip().strip("\"'")

    citation = values.get("consent_citation", "")
    if not citation:
        raise RefusedError(
            f"timer config {path} declares no consent_citation. The ledger row "
            "the deletions are authorised by is never hardcoded in the source, "
            "because a source literal cannot be revoked."
        )
    return ModelTimerConfig(
        enabled=values.get("enabled", "").lower() == "true",
        consent_citation=citation,
        delegate_refusal_review=values.get("delegate_refusal_review", "true").lower()
        != "false",
    )


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
    lines = path.read_text(encoding="utf-8").splitlines()
    number = int(match.group("line"))
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


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def _run(
    args: list[str], *, cwd: Path | None = None, timeout: int | None = None
) -> subprocess.CompletedProcess[str]:
    """Run a command with stderr CAPTURED, never discarded (CLAUDE.md rule 16).

    `git-maintenance.sh:106` discarding stderr and printing a guessed cause is
    precisely how the OMN-18826 debris read as peer work for an hour.
    """
    return subprocess.run(
        args, cwd=cwd, capture_output=True, text=True, check=False, timeout=timeout
    )


def delegate_refusal_review(
    refusals: list[str], omni_home: Path, report: ModelRunReport
) -> None:
    """Hand the refusal list to a LOCAL model and record what it said.

    Never acts on a proposal. A refusal or a timeout is recorded with its
    exact text and the items stay listed unclassified, because an unanswered
    triage question is not a triaged item.
    """
    if not refusals:
        report.notes.append("no refusals to review")
        return

    wrapper = omni_home / "omnibase_infra" / "scripts" / "onex"
    batches = batch_refusals(refusals, DELEGATE_PROMPT_CEILING)
    receipts: list[dict[str, object]] = []

    for index, batch in enumerate(batches, start=1):
        prompt = (
            "For each worktree line below output exactly one line: "
            "path | proposed disposition | reason. "
            "Dispositions: remove, keep, escalate.\n" + "\n".join(batch)
        )
        argv = [
            "bash",
            str(wrapper),
            "delegate",
            prompt,
            "--bus",
            "kafka",
            "--lane",
            "dev",
            "--locus",
            "deployed-lane",
        ]
        try:
            result = _run(argv, timeout=420)
        except subprocess.TimeoutExpired:
            receipts.append(
                {"batch": index, "ok": False, "error": "delegate wall-clock timeout"}
            )
            report.notes.append(
                f"batch {index}: delegate timed out; its {len(batch)} items "
                "stay listed unclassified"
            )
            continue
        if result.returncode != 0:
            receipts.append(
                {
                    "batch": index,
                    "ok": False,
                    "exit_code": result.returncode,
                    "stderr": (result.stderr or result.stdout).strip()[:2000],
                }
            )
            report.notes.append(
                f"batch {index}: delegate refused (exit {result.returncode}); "
                f"its {len(batch)} items stay listed unclassified"
            )
            continue
        report.delegate_proposals.extend(
            line for line in result.stdout.splitlines() if "|" in line
        )
        receipts.append(
            {
                "batch": index,
                "ok": True,
                "items": len(batch),
                "raw": result.stdout.strip()[:4000],
            }
        )

    report.delegate_receipt = {"batches": len(batches), "receipts": receipts}


def _branch_pr_is_merged(canonical: Path, branch: str) -> bool:
    """LIVE, per branch, at the moment of removal — never a cached list.

    Reuses the sanctioned deletion tool's own predicate so the bar a forced
    worktree removal clears is byte-for-byte the bar a branch deletion clears
    (OMN-18825): a MERGED pull request whose head ref IS this branch and whose
    head oid CONTAINS the local tip.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from delete_orphan_branch import (
        head_contains_tip,
        merged_pull_requests,
        repo_slug,
    )

    slug = repo_slug(canonical)
    if slug is None:
        return False
    tip = _run(
        ["git", "-C", str(canonical), "rev-parse", "--verify", "--quiet", branch]
    ).stdout.strip()
    if not tip:
        return False
    try:
        pulls = merged_pull_requests(slug, branch)
    except Exception:  # noqa: BLE001 — an unanswerable question refuses
        return False
    return any(head_contains_tip(canonical, tip, p.head_ref_oid) for p in pulls)


def force_remove_debris(
    worktree: Path, canonical: Path, branch: str | None, report: ModelRunReport
) -> bool:
    """Force-remove ONE tree, only once it is proven OMN-18826 debris.

    One retry, never a loop: a removal that fails twice is a finding, and a
    loop under load is how a timer nobody is watching eats a host. Every
    failure records git's exit code and its stderr verbatim — a guessed cause
    over a discarded stderr is what made this debris read as peer work.
    """
    status = _run(["git", "-C", str(worktree), "status", "--porcelain"])
    if status.returncode != 0:
        report.removal_failures.append(
            {
                "path": str(worktree),
                "exit_code": status.returncode,
                "stderr": (status.stderr or status.stdout).strip()[:2000],
                "detail": "git status refused; the tree was not classified",
            }
        )
        return False

    merged = bool(branch) and _branch_pr_is_merged(canonical, str(branch))
    verdict = classify_removal_debris(status.stdout, pr_merged=merged)
    if not verdict.eligible:
        report.refusals["not_removal_debris"] = (
            report.refusals.get("not_removal_debris", 0) + 1
        )
        report.refusal_items.append(f"{worktree} | keep | {verdict.reason}")
        return False

    file_count = len([ln for ln in status.stdout.splitlines() if ln.strip()])
    budget = removal_budget_seconds(file_count)
    argv = ["git", "-C", str(canonical), "worktree", "remove", "--force", str(worktree)]

    for attempt in (1, 2):
        try:
            result = _run(argv, timeout=budget)
        except subprocess.TimeoutExpired:
            if attempt == 1:
                continue
            report.removal_failures.append(
                {
                    "path": str(worktree),
                    "exit_code": -1,
                    "stderr": "",
                    "detail": (
                        f"timed out twice at {budget}s for {file_count} files; "
                        "not a safety finding"
                    ),
                }
            )
            return False
        if result.returncode == 0:
            report.debris_forced += 1
            report.notes.append(f"{worktree}: forced removal — {verdict.reason}")
            return True
        if attempt == 1:
            continue
        report.removal_failures.append(
            {
                "path": str(worktree),
                "exit_code": result.returncode,
                "stderr": (result.stderr or result.stdout).strip()[:2000],
                "detail": f"forced removal refused after one retry, budget {budget}s",
            }
        )
    return False


def append_ledger_note(ledger: Path, report: ModelRunReport, omni_home: Path) -> str:
    """Append ONE STATUS row through the mutex wrapper, never by hand."""
    reasons = (
        ", ".join(f"{k}={v}" for k, v in sorted(report.refusals.items())) or "none"
    )
    failures = "; ".join(
        f"{f['path']} exit={f['exit_code']} stderr={str(f['stderr'])[:160]}"
        for f in report.removal_failures[:5]
    )
    row = (
        f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} | STATUS | "
        f"lane=worktree-prune-timer | actor=launchd | model=none | "
        f"ticket=OMN-18832 | unattended prune run, consent="
        f"{report.consent_citation} | worktrees_removed="
        f"{report.worktrees_removed} debris_forced={report.debris_forced} "
        f"branches_deleted={report.branches_deleted} | refusals by reason: "
        f"{reasons} | removal failures (exit code and stderr, never discarded): "
        f"{failures or 'none'} | refusal review: "
        f"{report.delegate_receipt.get('batches', 0)} batch(es) to a local model "
        f"via onex delegate, {len(report.delegate_proposals)} proposal line(s) "
        f"recorded and NONE acted on | est ~0 lane-hours; displaces nothing"
    )
    result = _run(
        [
            "python3",
            str(omni_home / "scripts" / "ledger_lock.py"),
            str(ledger),
            "--append",
            row,
        ]
    )
    if result.returncode != 0:
        report.notes.append(
            f"ledger append failed (exit {result.returncode}): "
            f"{(result.stderr or result.stdout).strip()[:400]}"
        )
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Unattended worktree prune, run by launchd (OMN-18832).",
    )
    parser.add_argument("--config", required=True, help="Path to the timer config.")
    parser.add_argument("--ledger", required=True, help="Path to the rolling ledger.")
    parser.add_argument(
        "--state-dir", required=True, help="Where run reports and logs are written."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually remove. Without it the run classifies and reports only.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the report as JSON.")
    args = parser.parse_args(argv)

    omni_home_raw = os.environ.get("OMNI_HOME")
    if not omni_home_raw:
        print("ERROR: OMNI_HOME is not set", file=sys.stderr)
        return 2
    omni_home = Path(omni_home_raw)
    ledger = Path(args.ledger)

    report = ModelRunReport(
        started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        executed=bool(args.execute),
        consent_citation="",
    )

    try:
        # The ledger is BOTH the authorisation and the only durable record of
        # what was destroyed. A run that could not write one must not destroy
        # anything, so this is checked before the config, not after.
        if not ledger.is_file():
            raise RefusedError(
                f"ledger is not readable: {ledger}. The run would have no "
                "durable record of what it destroyed."
            )
        config = load_timer_config(Path(args.config))
        if not config.enabled:
            raise RefusedError(
                f"timer is disabled in {args.config} (enabled: false). This is "
                "the operator's revoke and it is honoured before any other check."
            )
        consent_row = resolve_consent(config.consent_citation, omni_home)
        report.consent_citation = config.consent_citation
    except RefusedError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 3

    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1 — the EXISTING omniclaude classifier, as a subprocess. Its
    # predicate is not reimplemented here; see this module's docstring.
    classifier = omni_home / "omniclaude" / "scripts" / "worktree_auto_prune.py"
    if classifier.is_file() and shutil.which("uv"):
        stamp = report.started_at.replace(":", "").replace("-", "")
        argv_classify = [
            "uv",
            "run",
            "--project",
            str(omni_home / "omniclaude"),
            "python",
            str(classifier),
            "--ledger",
            str(ledger),
            "--report-md",
            str(state_dir / f"{stamp}-prune.md"),
            "--report-json",
            str(state_dir / f"{stamp}-prune.json"),
        ]
        if args.execute:
            argv_classify.append("--execute")
        result = _run(argv_classify, cwd=omni_home, timeout=BUDGET_CEILING_SECONDS)
        report.classifier_exit = result.returncode
        if result.returncode != 0:
            report.notes.append(
                f"classifier exited {result.returncode}: "
                f"{(result.stderr or result.stdout).strip()[-1500:]}"
            )
        decisions = state_dir / f"{stamp}-prune.json"
        if decisions.is_file():
            try:
                payload = json.loads(decisions.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                report.notes.append(f"classifier report is not JSON: {exc}")
            else:
                rows = payload.get("decisions") or payload.get("worktrees") or []
                for row in rows if isinstance(rows, list) else []:
                    if not isinstance(row, dict):
                        continue
                    disposition = str(row.get("disposition", ""))
                    if "REMOV" in disposition.upper() and args.execute:
                        report.worktrees_removed += 1
                        continue
                    for reason in row.get("block_reasons") or ["unclassified"]:
                        key = str(reason)
                        report.refusals[key] = report.refusals.get(key, 0) + 1
                    report.refusal_items.append(
                        f"{row.get('path', '?')} | {disposition or 'refused'} | "
                        f"{', '.join(str(r) for r in (row.get('block_reasons') or []))}"
                    )
    else:
        report.notes.append(
            f"classifier not invoked: {classifier} present={classifier.is_file()}, "
            f"uv on PATH={bool(shutil.which('uv'))}"
        )

    # Phase 2 — the OMN-18826 debris pass, over what phase 1 could NOT remove.
    # A half-deleted tree is refused forever by a plain `git worktree remove`,
    # so without this pass the classifier reports the same rows every morning.
    if args.execute:
        still_refused = list(report.refusal_items)
        report.refusal_items = []
        carried: dict[str, int] = {}
        for item in still_refused:
            path_text = item.split("|", 1)[0].strip()
            worktree = Path(path_text)
            if not worktree.is_dir():
                report.refusal_items.append(item)
                continue
            canonical_out = _run(
                [
                    "git",
                    "-C",
                    str(worktree),
                    "rev-parse",
                    "--path-format=absolute",
                    "--git-common-dir",
                ]
            ).stdout.strip()
            if not canonical_out:
                report.refusal_items.append(item)
                continue
            canonical = Path(canonical_out).parent
            branch = _run(
                ["git", "-C", str(worktree), "rev-parse", "--abbrev-ref", "HEAD"]
            ).stdout.strip()
            if not force_remove_debris(worktree, canonical, branch or None, report):
                report.refusal_items.append(item)
        report.refusals = {**carried, **report.refusals}

    # Phase 3 — the refusal review, by a LOCAL model, acted on by nobody.
    if config.delegate_refusal_review:
        delegate_refusal_review(report.refusal_items, omni_home, report)
    else:
        report.notes.append("refusal review disabled in the timer config")

    stamp = report.started_at.replace(":", "").replace("-", "")
    (state_dir / f"{stamp}-run.json").write_text(
        json.dumps(
            {**vars(report), "consent_row": consent_row[:240]},
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    # Phase 4 — one ledger row, only for a run that actually acted.
    if args.execute:
        append_ledger_note(ledger, report, omni_home)

    if args.json:
        print(json.dumps({**vars(report), "consent_row": consent_row[:240]}, indent=2))
    else:
        print(
            f"prune {'EXECUTED' if args.execute else 'DRY'}: "
            f"{report.worktrees_removed} worktrees removed, "
            f"{report.debris_forced} debris forced, "
            f"{len(report.refusal_items)} refused, "
            f"{len(report.delegate_proposals)} local-model proposal line(s)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
