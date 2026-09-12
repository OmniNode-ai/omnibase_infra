# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18233 — the four-clause verified-supersession predicate.

The closer's cited-PR merge conjunct reads ``merged_at`` and refuses anything
without one. That is correct for an OPEN pull request, which can still merge,
and permanently wrong for a CLOSED one, which cannot. Cascade bump pull requests
carry the RELEASING ticket's id, so a bump that was closed in favour of a better
one blocks its ticket forever: OMN-18201 had all four criteria evidenced and was
refused because ``omnibase_infra#3446`` closed unmerged, superseded by ``#3448``.

Ignoring closed bumps outright is what the plan review refused, and it was right
to: a closed pull request with no replacement is ABANDONED work, and a predicate
that waved it through would convert a real signal into none. So the bump is
ignorable only when supersession is PROVEN, by these four clauses:

  1. the closed pull request carries cascade provenance declaring a source
     package and a required version;
  2. a MERGED pull request exists in the same repository whose diff changes that
     package's pin or lockfile entry;
  3. the version that merged pull request delivers is >= the required version,
     compared as a VERSION, not as a string;
  4. the delivered version is readable from the repository's own default branch
     after the merge, not only from a pull request body.

Two properties are load-bearing and are pinned by tests rather than asserted
here. **Nothing reads a title.** Titles are prose a human or a generator wrote,
they are what the original OMN-17292 mis-flip keyed on, and a predicate that
read one could be satisfied by renaming a branch. **Ordering is not a clause.**
On the real pair the replacement merged roughly three hours BEFORE the
superseded bump was closed, so a `merged_at > closed_at` requirement would
refuse the single case this predicate was built for.

Every clause fails CLOSED. "I could not read the default branch" resolves to
"this bump still blocks", never to "so I will ignore it" — the same direction
every other fence in this node fails in.

This module sits at the node ROOT and not under ``handlers/`` deliberately: it
is not a handler, it is routed by no contract, and it is invoked as a function
by the sweep handler. Everything under ``handlers/`` is audited as a handler and
must appear in the contract's ``handler_routing`` — a rule this module would
have to be exempted from rather than satisfy. ``node_chain_canary_effect``'s
``lane_transport`` is the same shape. It is split out of the handler module only
because that file is already around six thousand lines.
"""

from __future__ import annotations

import base64
import re
from collections.abc import Awaitable, Callable

from packaging.version import InvalidVersion, Version

from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.model_cascade_provenance import (
    ModelCascadeProvenance,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.model_supersession_verdict import (
    ModelSupersessionVerdict,
)

#: The provenance block's heading, as `dependency-cascade.yml` emits it. Matched
#: on the heading TEXT so the `(OMN-16286)` suffix may change without breaking
#: the parser; OMN-18235 makes the generator's own test assert this shape.
_PROVENANCE_HEADING_RE = re.compile(r"^#{2,6}\s+cascade\s+provenance\b", re.IGNORECASE)
_ANY_HEADING_RE = re.compile(r"^#{1,6}\s+")
_SOURCE_REPO_RE = re.compile(
    r"^\s*[-*]\s*Source repo:\s*`?([\w.-]+/[\w.-]+)`?\s*$", re.IGNORECASE
)
_RELEASED_VERSION_RE = re.compile(
    r"^\s*[-*]\s*Released version:\s*`?([^`\s]+)`?\s*$", re.IGNORECASE
)

#: PEP 503 normalisation: runs of `-`, `_` and `.` collapse to a single `-`.
_NORMALISE_RE = re.compile(r"[-_.]+")

#: Where a pin lives, in the order the predicate reads them. `pyproject.toml`
#: first because an exact pin there is the declaration; the lockfile is the
#: resolution, and a repo that constrains rather than pins only has the latter.
_PIN_PATHS: tuple[str, ...] = ("pyproject.toml", "uv.lock")

#: Bounds on the clause-2 walk. A pin the predicate cannot find within these is
#: UNRESOLVED, which holds — never "absent", which would ignore.
_MAX_PIN_COMMITS = 20
_MAX_REPLACEMENT_CANDIDATES = 10

#: `(args, timeout) -> (payload | None, error)`, the node's injected gh runner.
GhRunner = Callable[[list[str], float], Awaitable[tuple[object | None, str]]]


def normalise_distribution(name: str) -> str:
    """PEP 503 normalised distribution name (`omnibase_core` -> `omnibase-core`)."""
    return _NORMALISE_RE.sub("-", name).lower()


def parse_cascade_provenance(body: str) -> ModelCascadeProvenance | None:
    """Clause 1. The provenance block a cascade bump declares, or ``None``.

    The two fields bind to the provenance HEADING, not to the body at large.
    A pull request whose summary happens to spell "Source repo:" is not a
    cascade bump, and reading the fields loose would make it one.
    """
    lines = body.splitlines()
    in_section = False
    source_repo = ""
    released_version = ""
    for line in lines:
        if _PROVENANCE_HEADING_RE.match(line):
            in_section = True
            continue
        if in_section and _ANY_HEADING_RE.match(line):
            break
        if not in_section:
            continue
        source_match = _SOURCE_REPO_RE.match(line)
        if source_match and not source_repo:
            source_repo = source_match.group(1)
            continue
        version_match = _RELEASED_VERSION_RE.match(line)
        if version_match and not released_version:
            released_version = version_match.group(1)
    if not source_repo or not released_version:
        return None
    try:
        Version(released_version)
    except InvalidVersion:
        # A version clause 3 cannot compare is a clause 1 that did not resolve.
        return None
    return ModelCascadeProvenance(
        source_repo=source_repo,
        required_version=released_version,
        distribution=normalise_distribution(source_repo.rsplit("/", 1)[-1]),
    )


def pinned_version_from_pyproject(text: str, distribution: str) -> str | None:
    """The exact version `pyproject.toml` pins for ``distribution``, if any.

    Only an exact `==` pin is a delivered version. A `>=` constraint states what
    the repo will ACCEPT, which is not the same fact and cannot discharge
    clause 4 — the lockfile carries the resolution in that case.
    """
    wanted = normalise_distribution(distribution)
    pattern = re.compile(
        r"""["']\s*([\w.-]+)(?:\[[^\]]+\])?\s*==\s*([^"',\s\[\]]+)""",
    )
    for name, version in pattern.findall(text):
        if normalise_distribution(name) == wanted:
            return version
    return None


def pinned_version_from_lockfile(text: str, distribution: str) -> str | None:
    """The version ``uv.lock`` resolves ``distribution`` to, if any."""
    wanted = normalise_distribution(distribution)
    current: dict[str, str] = {}

    def resolved(block: dict[str, str]) -> str | None:
        if normalise_distribution(block.get("name", "")) == wanted:
            return block.get("version") or None
        return None

    for raw in [*text.splitlines(), "[[package]]"]:
        line = raw.strip()
        if line == "[[package]]":
            version = resolved(current)
            if version:
                return version
            current = {}
            continue
        name_match = re.match(r'^name\s*=\s*"([^"]+)"$', line)
        if name_match:
            current["name"] = name_match.group(1)
            continue
        version_match = re.match(r'^version\s*=\s*"([^"]+)"$', line)
        if version_match:
            current["version"] = version_match.group(1)
    return None


async def _read_pinned_versions(
    *,
    repo: str,
    ref: str,
    distribution: str,
    run_gh_command: GhRunner,
    gh_timeout_seconds: int,
) -> dict[str, str]:
    """Every readable pinned version for ``distribution`` at ``ref``, by path."""
    versions: dict[str, str] = {}
    for path in _PIN_PATHS:
        payload, _error = await run_gh_command(
            ["gh", "api", f"repos/{repo}/contents/{path}?ref={ref}"],
            gh_timeout_seconds,
        )
        if not isinstance(payload, dict):
            continue
        encoded = str(payload.get("content") or "")
        if not encoded:
            continue
        try:
            text = base64.b64decode(encoded).decode(errors="replace")
        except ValueError:
            continue
        reader = (
            pinned_version_from_pyproject
            if path == "pyproject.toml"
            else pinned_version_from_lockfile
        )
        version = reader(text, distribution)
        if version:
            versions[path] = version
    return versions


async def _read_pinned_version(
    *,
    repo: str,
    ref: str,
    distribution: str,
    run_gh_command: GhRunner,
    gh_timeout_seconds: int,
) -> tuple[str, str]:
    """``(version, path)`` for ``distribution`` at ``ref``; ``("", "")`` if none.

    Reads the declaration first and the resolution second. An unreadable path is
    indistinguishable from an absent one HERE on purpose: both mean this ref
    does not prove a version, and the caller holds on either.
    """
    versions = await _read_pinned_versions(
        repo=repo,
        ref=ref,
        distribution=distribution,
        run_gh_command=run_gh_command,
        gh_timeout_seconds=gh_timeout_seconds,
    )
    for path in _PIN_PATHS:
        version = versions.get(path, "")
        if version:
            return version, path
    return "", ""


async def _merged_pr_that_moved_the_pin(
    *,
    repo: str,
    default_branch: str,
    distribution: str,
    closed_pr_number: int,
    run_gh_command: GhRunner,
    gh_timeout_seconds: int,
) -> tuple[int, str]:
    """Clause 2. The merged pull request whose diff changed the pin, or ``0``.

    Walks the default branch's own commit history for the pin-carrying paths,
    newest first, and for each commit asks which pull request delivered it. A
    candidate qualifies only when the pinned version DIFFERS between its base
    commit and its merge commit — "touched pyproject.toml" is not "moved the
    pin", and the routine unrelated edit is the common case.
    """
    seen: set[int] = set()
    examined = 0
    for path in _PIN_PATHS:
        commits, _error = await run_gh_command(
            [
                "gh",
                "api",
                f"repos/{repo}/commits?path={path}&sha={default_branch}"
                f"&per_page={_MAX_PIN_COMMITS}",
            ],
            gh_timeout_seconds,
        )
        if not isinstance(commits, list):
            continue
        for commit in commits[:_MAX_PIN_COMMITS]:
            if not isinstance(commit, dict):
                continue
            sha = str(commit.get("sha") or "")
            if not sha:
                continue
            associated, _pr_error = await run_gh_command(
                ["gh", "api", f"repos/{repo}/commits/{sha}/pulls"],
                gh_timeout_seconds,
            )
            if not isinstance(associated, list):
                continue
            for entry in associated:
                if not isinstance(entry, dict):
                    continue
                number = entry.get("number")
                if not isinstance(number, int) or number in seen:
                    continue
                if number == closed_pr_number:
                    continue
                if not str(entry.get("merged_at") or ""):
                    continue
                seen.add(number)
                if examined >= _MAX_REPLACEMENT_CANDIDATES:
                    return 0, (
                        f"clause 2 unresolved: examined the "
                        f"{_MAX_REPLACEMENT_CANDIDATES} most recent merged pull "
                        f"requests touching {'/'.join(_PIN_PATHS)} on "
                        f"{default_branch} and none moved the {distribution} pin"
                    )
                examined += 1
                detail = await _pull_request_moved_the_pin(
                    repo=repo,
                    pr_number=number,
                    distribution=distribution,
                    run_gh_command=run_gh_command,
                    gh_timeout_seconds=gh_timeout_seconds,
                )
                if detail:
                    return number, detail
    return 0, (
        f"clause 2 unresolved: no merged pull request in {repo} was found whose "
        f"diff moves the {distribution} pin on {default_branch}"
    )


async def _pull_request_moved_the_pin(
    *,
    repo: str,
    pr_number: int,
    distribution: str,
    run_gh_command: GhRunner,
    gh_timeout_seconds: int,
) -> str:
    """Non-empty when ``pr_number``'s diff changed ``distribution``'s pin."""
    payload, _error = await run_gh_command(
        ["gh", "api", f"repos/{repo}/pulls/{pr_number}"], gh_timeout_seconds
    )
    if not isinstance(payload, dict):
        return ""
    if not str(payload.get("merged_at") or ""):
        return ""
    merge_sha = str(payload.get("merge_commit_sha") or "")
    base = payload.get("base")
    base_sha = str(base.get("sha") or "") if isinstance(base, dict) else ""
    if not merge_sha or not base_sha:
        return ""
    after_versions = await _read_pinned_versions(
        repo=repo,
        ref=merge_sha,
        distribution=distribution,
        run_gh_command=run_gh_command,
        gh_timeout_seconds=gh_timeout_seconds,
    )
    if not after_versions:
        return ""
    before_versions = await _read_pinned_versions(
        repo=repo,
        ref=base_sha,
        distribution=distribution,
        run_gh_command=run_gh_command,
        gh_timeout_seconds=gh_timeout_seconds,
    )
    for path in _PIN_PATHS:
        after = after_versions.get(path, "")
        if not after:
            continue
        before = before_versions.get(path, "")
        if before != after:
            return (
                f"{repo}#{pr_number} merged and moved the {distribution} entry "
                f"in {path} from {before or 'absent'} to {after}"
            )
    return ""


async def resolve_verified_supersession(
    *,
    repo: str,
    closed_pr_number: int,
    closed_pr_body: str,
    run_gh_command: GhRunner,
    gh_timeout_seconds: int,
) -> ModelSupersessionVerdict:
    """The four-clause predicate. ``superseded`` only when all four resolve."""
    provenance = parse_cascade_provenance(closed_pr_body)
    if provenance is None:
        return ModelSupersessionVerdict(
            superseded=False,
            detail=(
                f"clause 1 unresolved: {repo}#{closed_pr_number} carries no "
                "machine-readable cascade provenance block declaring a source "
                "repo and a released version, so it is not a cascade bump this "
                "predicate can reason about"
            ),
        )

    repo_payload, repo_error = await run_gh_command(
        ["gh", "api", f"repos/{repo}"], gh_timeout_seconds
    )
    if not isinstance(repo_payload, dict):
        return ModelSupersessionVerdict(
            superseded=False,
            detail=(
                f"clause 4 unresolved: could not read {repo}'s default branch "
                f"({repo_error or 'no payload'})"
            ),
            required_version=provenance.required_version,
        )
    default_branch = str(repo_payload.get("default_branch") or "")
    if not default_branch:
        return ModelSupersessionVerdict(
            superseded=False,
            detail=f"clause 4 unresolved: {repo} reports no default branch",
            required_version=provenance.required_version,
        )

    delivered, delivered_path = await _read_pinned_version(
        repo=repo,
        ref=default_branch,
        distribution=provenance.distribution,
        run_gh_command=run_gh_command,
        gh_timeout_seconds=gh_timeout_seconds,
    )
    if not delivered:
        return ModelSupersessionVerdict(
            superseded=False,
            detail=(
                f"clause 4 unresolved: no {provenance.distribution} version is "
                f"readable from {repo}@{default_branch} in "
                f"{' or '.join(_PIN_PATHS)}"
            ),
            required_version=provenance.required_version,
        )

    try:
        delivered_version = Version(delivered)
    except InvalidVersion:
        return ModelSupersessionVerdict(
            superseded=False,
            detail=(
                f"clause 3 unresolved: {repo}@{default_branch} carries "
                f"{provenance.distribution} {delivered!r}, which is not a "
                "comparable version"
            ),
            required_version=provenance.required_version,
        )
    if delivered_version < Version(provenance.required_version):
        return ModelSupersessionVerdict(
            superseded=False,
            detail=(
                f"clause 3 refuted: {repo}@{default_branch} delivers "
                f"{provenance.distribution} {delivered} in {delivered_path}, "
                f"below the {provenance.required_version} "
                f"{repo}#{closed_pr_number} was opened to deliver"
            ),
            required_version=provenance.required_version,
            delivered_version=delivered,
        )

    replacement_pr, clause_two_detail = await _merged_pr_that_moved_the_pin(
        repo=repo,
        default_branch=default_branch,
        distribution=provenance.distribution,
        closed_pr_number=closed_pr_number,
        run_gh_command=run_gh_command,
        gh_timeout_seconds=gh_timeout_seconds,
    )
    if not replacement_pr:
        return ModelSupersessionVerdict(
            superseded=False,
            detail=clause_two_detail,
            required_version=provenance.required_version,
            delivered_version=delivered,
        )

    return ModelSupersessionVerdict(
        superseded=True,
        detail=(
            f"{repo}#{closed_pr_number} closed unmerged and is PROVEN "
            f"superseded: it declared {provenance.source_repo} "
            f"{provenance.required_version}; {clause_two_detail}; and "
            f"{repo}@{default_branch} reads {provenance.distribution} "
            f"{delivered} in {delivered_path}, which is at or above what the "
            "closed bump was opened to deliver"
        ),
        replacement_pr=replacement_pr,
        required_version=provenance.required_version,
        delivered_version=delivered,
    )


__all__ = [
    "GhRunner",
    "normalise_distribution",
    "parse_cascade_provenance",
    "pinned_version_from_lockfile",
    "pinned_version_from_pyproject",
    "resolve_verified_supersession",
]
