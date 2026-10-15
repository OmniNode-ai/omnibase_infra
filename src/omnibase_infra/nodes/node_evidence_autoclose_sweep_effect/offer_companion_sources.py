# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Where an offered ticket's merged evidence companion is read from (OMN-21146).

``--offer`` used to resolve the companion with ``gh api search/issues``. The
GitHub-search guard refuses that call on every host, so every offered ticket
ended in ERROR_GITHUB_API before the verifier ran. The companion is a merged
commit on the change-control repository's default branch and a record in the
PR watcher's state, so it is read from there:

1. the PR watcher's state file (request ``pr_watcher_state_path``, else
   ``<clone_root>/.claude_scratch/pr-watcher/state.json``). A MERGED record is
   terminal and is served at any watcher age;
2. the canonical clone's default branch, ``git log --grep <ticket>`` (the clone
   root is the request's ``clone_root``), which also yields the changed-file
   list of each squash commit, so no file listing is fetched;
3. ``node_github_repo_gateway_effect`` through ``onex scm`` (``pr-heads`` then
   ``pr-view``), only for what the first two did not answer.

GitHub search is never used. This module only gathers candidates; the handler's
title/file binding filter decides which of them is the nominated ticket's.

When the change-control copy is absent (OCC is being retired), the contract the
verifier reads lives in the product repository. ``find_product_contract`` finds
that copy in the canonical clones: a ``contracts/<ticket>.yaml`` on a clone's
default branch, in a repository other than the change-control one.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import subprocess
import sys
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

logger = logging.getLogger(__name__)

WATCHER_STATE_RELATIVE: Final[str] = ".claude_scratch/pr-watcher/state.json"
GIT_TIMEOUT_SECONDS: Final[int] = 30
# `pr-heads` pages inside the gateway node up to this many merged PRs.
GATEWAY_HEADS_LIMIT: Final[int] = 300
# A ticket-named head branch rarely maps to more PRs than this; each one costs a
# `pr-view` bus call, so the probe is bounded.
GATEWAY_MAX_PR_VIEWS: Final[int] = 5

_SQUASH_SUBJECT: Final = re.compile(r"\(#([1-9][0-9]*)\)\s*$")
_TICKET: Final = re.compile(r"^OMN-[0-9]+$")
_FIELD_SEP: Final[str] = "\x1f"

# (clone, git args) -> stdout, or None when git could not answer.
TypeRunGit = Callable[[Path, tuple[str, ...]], str | None]
# (onex scm args, timeout) -> (parsed JSON, error). Same shape as the gh runner.
TypeRunGateway = Callable[[list[str], float], Awaitable[tuple[object | None, str]]]


def run_git_real(clone: Path, args: tuple[str, ...]) -> str | None:
    """``git -C <clone> <args>``; None on a non-zero exit, a timeout or an OS error."""
    try:
        done = subprocess.run(
            ["git", "-C", str(clone), *args],
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("git %s in %s failed: %s", args[0], clone, exc)
        return None
    if done.returncode != 0:
        logger.warning(
            "git %s in %s exited %d: %s",
            args[0],
            clone,
            done.returncode,
            done.stderr.strip()[-300:],
        )
        return None
    return done.stdout


def watcher_state_path(clone_root: Path | None, declared: str) -> Path | None:
    """The PR watcher's state file: the declared path, else under the clone root."""
    if declared:
        return Path(declared)
    if clone_root is None:
        return None
    return clone_root / WATCHER_STATE_RELATIVE


def _repo_parts(repo: str) -> tuple[str, str]:
    owner, _, name = repo.partition("/")
    return owner, name


def _utc(stamp: str) -> str:
    """An ISO-8601 stamp with any offset, as ``YYYY-MM-DDTHH:MM:SSZ``; '' if unreadable."""
    try:
        parsed = datetime.fromisoformat(stamp.strip().replace("Z", "+00:00"))
    except ValueError:
        return ""
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _candidate(
    repo: str, number: int, title: str, merged_at: str, files: list[str] | None
) -> dict[str, object]:
    candidate: dict[str, object] = {
        "number": number,
        "html_url": f"https://github.com/{repo}/pull/{number}",
        "title": title,
        "merged_at": merged_at,
        "updated_at": merged_at,
    }
    if files is not None:
        candidate["files"] = files
    return candidate


def read_watcher_candidates(
    state_path: Path | None, repo: str, ticket_id: str
) -> tuple[list[dict[str, object]], bool]:
    """MERGED watcher records of ``repo`` whose title names ``ticket_id``.

    Returns ``(candidates, answered)``; ``answered`` is False when the state
    file is absent or unreadable, so the caller can tell "no record" from "no
    state to read".
    """
    if state_path is None or not state_path.is_file():
        return [], False
    try:
        state = json.loads(state_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("PR watcher state %s unreadable: %s", state_path, exc)
        return [], False
    records = state.get("prs") if isinstance(state, dict) else None
    if not isinstance(records, dict):
        return [], False
    _owner, name = _repo_parts(repo)
    wanted = ticket_id.upper()
    found: list[dict[str, object]] = []
    for key, record in records.items():
        if not isinstance(key, str) or not key.startswith(f"{name}#"):
            continue
        facts = record.get("facts") if isinstance(record, dict) else None
        if not isinstance(facts, dict) or facts.get("state") != "MERGED":
            continue
        title = str(facts.get("title") or "")
        merged_at = _utc(str(facts.get("merged_at") or ""))
        number = facts.get("number")
        if wanted not in title.upper() or not merged_at or not isinstance(number, int):
            continue
        found.append(_candidate(repo, number, title, merged_at, None))
    return found, True


def _default_ref(clone: Path, run_git: TypeRunGit) -> str:
    head = run_git(clone, ("symbolic-ref", "--short", "refs/remotes/origin/HEAD"))
    return head.strip() if head and head.strip() else ""


def read_clone_candidates(
    clone: Path | None, repo: str, ticket_id: str, run_git: TypeRunGit
) -> tuple[list[dict[str, object]], bool]:
    """Squash commits on the clone's default branch whose message names ``ticket_id``.

    ``answered`` is False when there is no clone or no ``origin/HEAD`` to read.
    """
    if clone is None or not (clone / ".git").exists():
        return [], False
    ref = _default_ref(clone, run_git)
    if not ref:
        return [], False
    listing = run_git(
        clone,
        (
            "log",
            ref,
            "--fixed-strings",
            f"--grep={ticket_id}",
            f"--format=%H{_FIELD_SEP}%cI{_FIELD_SEP}%s",
        ),
    )
    if listing is None:
        return [], False
    found: list[dict[str, object]] = []
    for line in listing.splitlines():
        sha, _, rest = line.partition(_FIELD_SEP)
        stamp, _, subject = rest.partition(_FIELD_SEP)
        match = _SQUASH_SUBJECT.search(subject)
        merged_at = _utc(stamp)
        if not sha or match is None or not merged_at:
            continue
        names = run_git(clone, ("show", "--name-only", "--format=", sha))
        files = (
            [path for path in names.splitlines() if path.strip()]
            if names is not None
            else None
        )
        found.append(_candidate(repo, int(match.group(1)), subject, merged_at, files))
    return found, True


def merge_candidates(
    watcher: list[dict[str, object]], clone: list[dict[str, object]]
) -> list[dict[str, object]]:
    """One candidate per PR number: the clone's changed files, the watcher's merge time.

    The watcher's ``merged_at`` is GitHub's own; the clone supplies the file
    list the watcher does not carry.
    """
    by_number: dict[object, dict[str, object]] = {c["number"]: dict(c) for c in clone}
    for record in watcher:
        existing = by_number.get(record["number"])
        if existing is None:
            by_number[record["number"]] = dict(record)
        else:
            existing["merged_at"] = record["merged_at"]
            existing["updated_at"] = record["merged_at"]
            existing["title"] = record["title"]
    return list(by_number.values())


def _gateway_args(
    verb: str, repo: str, extra: list[str], lane: str, workspace_root: str
) -> list[str]:
    onex = Path(sys.executable).parent / "onex"
    binary = str(onex) if onex.exists() else (shutil.which("onex") or "onex")
    return [
        binary,
        "scm",
        verb,
        repo,
        *extra,
        "--source",
        "bus",
        "--bus-lane",
        lane,
        # The lane declaration lives in the workspace root's omnimarket checkout.
        *(["--omnibase-path", workspace_root] if workspace_root else []),
    ]


async def run_gateway_real(
    args: list[str], timeout: float
) -> tuple[object | None, str]:
    """Run one ``onex scm`` read and parse its JSON answer."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
    except OSError as exc:
        return None, f"OS error launching {' '.join(args[:3])}: {exc}"
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return None, f"Timeout running: {' '.join(args[:3])}"
    if proc.returncode != 0:
        return None, stderr.decode(errors="replace").strip()[-500:] or (
            f"{' '.join(args[:3])} exited {proc.returncode}"
        )
    try:
        return json.loads(stdout.decode()), ""
    except json.JSONDecodeError as exc:
        return None, f"Invalid JSON from {' '.join(args[:3])}: {exc}"


def _gateway_result(answer: object, operation: str) -> dict[str, object] | None:
    if not isinstance(answer, dict):
        return None
    if answer.get("outcome") != "completed" or answer.get("operation") != operation:
        return None
    result = answer.get("result")
    return result if isinstance(result, dict) else None


async def read_gateway_candidates(
    repo: str,
    ticket_id: str,
    run_gateway: TypeRunGateway,
    timeout: float,
    bus_lane: str,
    workspace_root: str,
) -> tuple[list[dict[str, object]], str]:
    """Merged PRs on a ticket-named head branch, read through the gateway node.

    ``pr-heads`` lists merged PRs with their head branch; the ones whose branch
    names the ticket are read with ``pr-view`` for their title. A companion on a
    branch that does not name the ticket (a batch-window ``auto/window-*``
    companion) is not reachable this way; the local sources are the ones that
    reach it. ``(candidates, error)`` with a non-empty error when the gateway
    could not answer.
    """
    heads, error = await run_gateway(
        _gateway_args(
            "pr-heads",
            repo,
            ["--limit", str(GATEWAY_HEADS_LIMIT)],
            bus_lane,
            workspace_root,
        ),
        timeout,
    )
    if heads is None:
        return [], error or "gateway pr-heads returned no data"
    result = _gateway_result(heads, "pr_heads")
    prs = result.get("prs") if result is not None else None
    if not isinstance(prs, list):
        return [], "gateway pr-heads answer carried no 'prs' list"
    wanted = ticket_id.lower()
    named = [
        pr
        for pr in prs
        if isinstance(pr, dict)
        and pr.get("state") == "MERGED"
        and wanted in str(pr.get("head_ref") or "").lower()
        and isinstance(pr.get("number"), int)
    ][:GATEWAY_MAX_PR_VIEWS]
    found: list[dict[str, object]] = []
    for pr in named:
        number = int(str(pr["number"]))
        view, error = await run_gateway(
            _gateway_args("pr-view", f"{repo}#{number}", [], bus_lane, workspace_root),
            timeout,
        )
        if view is None:
            return [], error or f"gateway pr-view returned no data for #{number}"
        detail = _gateway_result(view, "pr_text")
        if detail is None:
            return [], f"gateway pr-view answer for #{number} was not a completed read"
        merged_at = _utc(str(detail.get("merged_at") or pr.get("merged_at") or ""))
        if str(detail.get("state")) != "MERGED" or not merged_at:
            continue
        found.append(
            _candidate(repo, number, str(detail.get("title") or ""), merged_at, None)
        )
    return found, ""


def find_product_contract(
    root: Path | None,
    owner: str,
    change_control_name: str,
    ticket_id: str,
    run_git: TypeRunGit,
) -> tuple[dict[str, object] | None, str]:
    """The merged ``contracts/<ticket>.yaml`` in a product repository's clone.

    Looks at every clone under ``root`` except the change-control repository's
    own, on that clone's default branch. Returns ``(contract, error)``: a
    contract dict (repo, commit, merged_at, url) when exactly one repository
    holds one, ``(None, "")`` when none does, and ``(None, reason)`` when more
    than one does, which this run will not choose between.
    """
    if root is None or not root.is_dir() or _TICKET.match(ticket_id) is None:
        return None, ""
    path = f"contracts/{ticket_id}.yaml"
    holders: list[dict[str, object]] = []
    for clone in sorted(root.iterdir()):
        if clone.name == change_control_name or not (clone / ".git").exists():
            continue
        ref = _default_ref(clone, run_git)
        if not ref:
            continue
        present = run_git(clone, ("ls-tree", "--name-only", ref, "--", path))
        if present is None or present.strip() != path:
            continue
        last = run_git(
            clone, ("log", ref, "-1", f"--format=%H{_FIELD_SEP}%cI", "--", path)
        )
        sha, _, stamp = (last or "").strip().partition(_FIELD_SEP)
        merged_at = _utc(stamp)
        if not sha or not merged_at:
            continue
        repo = f"{owner}/{clone.name}"
        holders.append(
            {
                "repo": repo,
                "commit": sha,
                "merged_at": merged_at,
                "url": f"https://github.com/{repo}/blob/{sha}/{path}",
            }
        )
    if len(holders) > 1:
        names = ", ".join(sorted(str(h["repo"]) for h in holders))
        return None, (
            f"{path} is on the default branch of more than one product "
            f"repository ({names}); this run will not choose between them"
        )
    return (holders[0] if holders else None), ""


__all__ = [
    "TypeRunGateway",
    "TypeRunGit",
    "find_product_contract",
    "merge_candidates",
    "read_clone_candidates",
    "read_gateway_candidates",
    "read_watcher_candidates",
    "run_gateway_real",
    "run_git_real",
    "watcher_state_path",
]
