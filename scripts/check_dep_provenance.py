#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Dep-provenance gate — forbid first-party git-source overrides on dev/main.

Root cause this closes (OMN-13873): omnibase_infra PR #2184 merged to dev
carrying ``[tool.uv.sources]`` git-rev overrides pinning ``omnibase-core`` /
``omnibase-spi`` to UNRELEASED commits. Every CI check passed because CI
resolved those exact commits and ran green against them — the breakage is
dependency *provenance*, not runtime behavior, so no test catches it. This is a
pure static provenance gate: it FAILS closed if any PyPI-published first-party
dependency is sourced from git instead of PyPI.

Forbidden first-party deps (both hyphen and underscore spellings):

    omnibase-core   omnibase-spi   omnibase-compat

A ``[tool.uv.sources]`` entry for any of the above with a ``git`` / ``rev`` /
``branch`` / ``tag`` key is a forbidden override and fails the gate.

``onex-change-control`` is deliberately NOT checked — it follows an
immutable-main pin release model (different from the three PyPI-released deps),
so its git pin is intentional and must remain allowed.

Escape hatch (Rule-10 style): a forbidden source line may carry an inline
comment ``# raw-override-ok: <ticket>`` with a NON-EMPTY token. This exempts the
single line. An empty token (``# raw-override-ok:`` with nothing after) does NOT
exempt — the gate still fails. Because the TOML parser drops comments, the token
is detected by reading the raw source line for each flagged package.

Exit codes:
    0  — no forbidden first-party git-source override present
    1  — a forbidden override was found (or a hard error, e.g. missing file)

The override-forbid check above (``find_violations`` / the default CLI
invocation with no flags) is deterministic and offline — it makes no network
calls. It is unchanged and remains what pre-commit and
``dep-provenance-gate.yml`` invoke by default.

Content-lineage check (OMN-15604, opt-in via ``--check-lineage``)
-------------------------------------------------------------------
The gate above is SHAPE-only: it forbids a git source unconditionally, but a
line carrying a well-formed ``# raw-override-ok: <ticket>`` token is exempt
from it *unconditionally and forever* — the token is validated only for
non-emptiness, never against whether the pinned rev's content actually matches
the PyPI version declared alongside it. Live incident this reproduces:
``omnibase_infra@dev`` declared ``omnibase-core==0.46.8`` while
``[tool.uv.sources]`` pinned git rev ``3d51b047`` (escaped via
``# raw-override-ok: OMN-15414``) whose ``src/`` tree measurably DIFFERED from
released tag ``v0.46.8`` (9 files, +381/-27 lines) — a version string that was a
label on a tree it did not describe, invisible to every existing gate.

``find_lineage_violations`` closes that gap: for every forbidden package that
has BOTH a ``==X.Y.Z`` version constraint (in ``project.dependencies`` or
``[tool.uv] override-dependencies``) AND a ``[tool.uv.sources]`` git override,
it resolves the ``src/`` git tree object of the pinned ref and of released tag
``vX.Y.Z`` via the GitHub REST API (no local clone required — same technique as
``scripts/ci/check_pin_reachability.py``) and fails if the two tree SHAs
differ. This runs **regardless of a ``# raw-override-ok:`` token** — the token
exempts a line from the "forbid git source" rule only; it was never designed to
(and must not) exempt a pin from matching the content it claims to build.

Because tree resolution requires network access, this is opt-in
(``--check-lineage``) and never runs from pre-commit (offline hook) or the
default no-args CLI invocation. It fails closed on network failure — an
unresolved rev/tag is reported as a lineage violation, not silently skipped —
unless ``--allow-undetermined-lineage`` is passed, which is refused when ``CI``
is set (same posture as ``check_pin_reachability.py``).

A non-exact declared constraint (a range like ``>=0.46.8,<0.47.0``, or no
declared constraint at all) sitting next to a git override is ALSO a lineage
violation, not a free pass: an earlier, single-operator-only version lookup
recognized only ``pkg==X.Y.Z``, so loosening the constraint from ``==`` to a
range was enough to make ``find_lineage_violations`` skip the package entirely
while ``find_violations`` still exempted the same line via its escape token —
a complete bypass of both checks. ``_declared_version_specs`` closes that gap
by capturing every requirement shape (see below), so a range/unversioned
constraint is now itself flagged rather than silently skipped. There is no
released tree that a range constraint unambiguously names, so it cannot be
lineage-verified; the constraint must be tightened to an exact pin (or the
override deleted) before this check can prove anything about it.

Escape-token reconciliation (OMN-15604 AC3, opt-in via ``--check-token-expiry``)
---------------------------------------------------------------------------------
The ``# raw-override-ok: <ticket>`` escape token in ``find_violations`` never
expires and never checks whether ``<ticket>`` is still open. ``
find_escape_token_violations`` closes that gap two ways, either of which fails
the line:

1. an explicit ``until=YYYY-MM-DD`` suffix on the token
   (``# raw-override-ok: OMN-15414 until=2026-09-01``) that has passed, or
2. the cited ticket resolves (via the Linear API, ``LINEAR_API_KEY``) to a
   Done/Cancelled/Duplicate status — a closed ticket is no longer a live
   justification for an unconditional override.

Like the escape-token check itself, LINEAR_API_KEY is optional infrastructure
(same graceful-degradation posture as ``scripts/validation/check_stale_todos.py``
/ ``.github/workflows/stale-todo-gate.yml``): when it is unset, ticket-status
resolution is skipped (not failed) so a repo that has never provisioned the
secret is not permanently red. The ``until=`` date check requires no network
and always runs.

**Mandatory-``until=`` residual (closed here).** ``LINEAR_API_KEY`` is not
provisioned as a repo *or* org secret anywhere in OmniNode-ai as of this
writing (verified via ``gh secret list`` / ``gh api orgs/.../actions/secrets``)
-- so condition 2 above never actually fires in any live enforcing
environment; it is dead code in production, not merely optional. Left as
originally shipped, a token with no ``until=`` suffix (e.g. the literal
``# raw-override-ok: OMN-15414`` from the live incident this ticket names)
would fall straight through both conditions and pass unconditionally and
forever -- the exact defect AC3 exists to close, reproduced under the exact
live incident token. A token that supplies neither a live ``until=`` date nor
a resolvable ticket status is therefore *itself* a violation: graceful
degradation on a missing ``LINEAR_API_KEY`` only applies when the ``until=``
date is present and did the enforcing (see ``find_escape_token_violations``).

Cascade-movability check (OMN-15604 AC4, ``--check-movable <PACKAGE>``)
--------------------------------------------------------------------------
``uv lock --upgrade-package <pkg>==<version>`` cannot move a
``[tool.uv.sources]`` git-source override — uv always prefers an explicit
source override over registry resolution, so re-locking against the SAME
override typically re-resolves to a byte-identical ``uv.lock``. The automated
cascade in ``.github/workflows/dependency-cascade.yml`` reads that as
"no lockfile changes — already on latest", which is a false-positive SKIP:
the repo is not on latest, it is still stuck on the git pin.
``find_unmovable_cascade_targets`` gives that workflow (or any caller) an
explicit, actionable failure instead of a silent no-op: it fails when
``PACKAGE`` currently has an active ``[tool.uv.sources]`` git override,
regardless of a ``raw-override-ok`` token (the token was never designed to
exempt a cascade's ability to move the pin either).

Usage::

    uv run python scripts/check_dep_provenance.py
    uv run python scripts/check_dep_provenance.py --pyproject pyproject.toml
    uv run python scripts/check_dep_provenance.py --check-lineage
    uv run python scripts/check_dep_provenance.py --check-token-expiry
    uv run python scripts/check_dep_provenance.py --check-movable omnibase-core
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path

_ResolveFn = Callable[[str, str], "tuple[str | None, str]"]
_TicketResolveFn = Callable[[str], "tuple[str | None, str]"]

# ---------------------------------------------------------------------------
# First-party PyPI-published deps that must be resolved from PyPI, never git.
# Names are stored in canonical hyphen form; underscore spellings are
# normalized on lookup so both `omnibase-core` and `omnibase_core` are caught.
# ---------------------------------------------------------------------------

_FORBIDDEN_PACKAGES: frozenset[str] = frozenset(
    {
        "omnibase-core",
        "omnibase-spi",
        "omnibase-compat",
    }
)

# Source-override keys that indicate a git provenance (any one is forbidden for
# a first-party dep). A PyPI source has none of these.
_GIT_SOURCE_KEYS: frozenset[str] = frozenset({"git", "rev", "branch", "tag"})

# Inline escape token: `# raw-override-ok: <ticket>` with a non-empty token.
_ESCAPE_TOKEN_RE = re.compile(r"#\s*raw-override-ok:\s*(\S+)")

# Optional `until=YYYY-MM-DD` suffix on the escape token, e.g.
# `# raw-override-ok: OMN-15414 until=2026-09-01`.
_ESCAPE_TOKEN_UNTIL_RE = re.compile(
    r"#\s*raw-override-ok:\s*\S+\s+until=(\d{4}-\d{2}-\d{2})"
)

# A ticket identifier at the start of a token, e.g. `OMN-15414` out of a token
# that might carry trailing punctuation.
_TICKET_ID_RE = re.compile(r"^([A-Za-z]+-\d+)")

# `pkg==X.Y.Z` inside a project.dependencies / override-dependencies string.
_DECLARED_VERSION_RE = re.compile(r"^([A-Za-z0-9_.-]+)==([A-Za-z0-9_.+-]+)$")

# Any requirement string naming a package, regardless of operator/shape:
# captures the package name and the raw remainder (may be an exact `==` pin,
# a range, or empty for a bare unversioned name).
_REQUIREMENT_SPEC_RE = re.compile(r"^([A-Za-z0-9_.-]+)\s*(.*)$")

# An exact-pin spec, e.g. `==0.46.8` (the remainder captured above).
_EXACT_PIN_SUFFIX_RE = re.compile(r"^==\s*([A-Za-z0-9_.+-]+)$")

# GitHub org all first-party OmniNode repos live under, and the REST root.
_ORG = "OmniNode-ai"
_GITHUB_API = "https://api.github.com"  # url-authority-ok: fixed public REST API, no ONEX routing authority
_LINEAR_API = "https://api.linear.app/graphql"  # url-authority-ok: fixed public GraphQL API, no ONEX routing authority
_REQUEST_TIMEOUT_SECONDS = 10.0

# Linear issue-state names/types that mean "closed" -- same set as
# scripts/validation/check_stale_todos.py's done_statuses.
_TICKET_DONE_STATUSES = frozenset(
    {"done", "completed", "canceled", "cancelled", "duplicate"}
)

# `git = "https://github.com/OmniNode-ai/<repo>.git"` — extracts <repo>.
_SOURCE_URL_RE = re.compile(
    r"\Ahttps://github\.com/" + _ORG + r"/(?P<repo>[\w.-]+?)(?:\.git)?/?\Z"
)

# ---------------------------------------------------------------------------
# [tool.uv.sources] parsing. TOML is authoritative for source classification so
# single quotes, indentation, and package subtables cannot hide a git override.
# Raw text is still used only to find the inline escape token, because TOML drops
# comments.
# ---------------------------------------------------------------------------

_UVS_BLOCK_RE = re.compile(
    r"^\[tool\.uv\.sources\](.*?)(?=^\[(?!tool\.uv\.sources\.)|\Z)",
    re.MULTILINE | re.DOTALL,
)
_UVS_SUBTABLE_RE = re.compile(r"^\[tool\.uv\.sources\.([^\]]+)\]")


def _normalize(pkg: str) -> str:
    """Canonicalize a package name to hyphen form for comparison."""
    return pkg.strip().strip('"').strip("'").replace("_", "-").lower()


def _uv_sources_block(text: str) -> str | None:
    """Return the raw text of the [tool.uv.sources] block, or None if absent."""
    block_m = _UVS_BLOCK_RE.search(text)
    return block_m.group(1) if block_m else None


def _parse_uv_source_entries(text: str) -> dict[str, dict[str, object]]:
    """Return {normalized_pkg: source_mapping} from pyproject TOML."""
    try:
        parsed = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"invalid TOML: {exc}") from exc

    tool = parsed.get("tool", {})
    if not isinstance(tool, dict):
        return {}
    uv = tool.get("uv", {})
    if not isinstance(uv, dict):
        return {}
    raw_sources = uv.get("sources", {})
    if not isinstance(raw_sources, dict):
        return {}

    sources: dict[str, dict[str, object]] = {}
    for pkg, attrs in raw_sources.items():
        if isinstance(attrs, dict):
            sources[_normalize(str(pkg))] = attrs
    return sources


def _line_for_package(block: str, pkg: str) -> str | None:
    """Return the raw source line (with any trailing comment) declaring `pkg`."""
    for raw_line in block.splitlines():
        stripped = raw_line.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        subtable_m = _UVS_SUBTABLE_RE.match(stripped)
        if subtable_m and _normalize(subtable_m.group(1)) == pkg:
            return raw_line
        # Entry key is the text before the first '=' on the line.
        key = stripped.split("=", 1)[0] if "=" in stripped else ""
        if key and _normalize(key) == pkg:
            return raw_line
    return None


# ---------------------------------------------------------------------------
# Core check
# ---------------------------------------------------------------------------


def find_violations(text: str) -> list[str]:
    """Return diagnostic messages for each forbidden git-source override.

    An empty list means the file is clean (exit 0). A non-empty list means the
    gate fails (exit 1). Lines carrying a valid `# raw-override-ok: <token>`
    escape are excluded.
    """
    block = _uv_sources_block(text) or text

    try:
        entries = _parse_uv_source_entries(text)
    except ValueError as exc:
        return [str(exc)]
    violations: list[str] = []

    for pkg, attrs in entries.items():
        if pkg not in _FORBIDDEN_PACKAGES:
            continue
        git_keys = sorted(_GIT_SOURCE_KEYS & set(attrs))
        if not git_keys:
            # A non-git source (unusual, but not a provenance violation).
            continue

        raw_line = _line_for_package(block, pkg)
        if raw_line is not None:
            escape_m = _ESCAPE_TOKEN_RE.search(raw_line)
            if escape_m and escape_m.group(1).strip():
                # Valid non-empty escape token — this line is exempt.
                continue

        keys_desc = ", ".join(f"{k}={attrs[k]!r}" for k in git_keys)
        violations.append(
            f"{pkg}: forbidden git-source override ({keys_desc}). "
            f"First-party deps must resolve from PyPI, not git. "
            f"line: {raw_line.strip() if raw_line else '<unresolved>'}"
        )

    return violations


# ---------------------------------------------------------------------------
# Content-lineage check (OMN-15604) — network, opt-in via --check-lineage.
# ---------------------------------------------------------------------------


def _declared_version_specs(parsed: dict[str, object]) -> dict[str, str]:
    """Return {normalized_pkg: raw_version_spec} for every requirement string
    naming a package, regardless of operator/shape.

    Scans `project.dependencies` and `[tool.uv] override-dependencies` — the
    two loci the live incident used (pyproject.toml:36 and :189). Later
    entries win over earlier ones so override-dependencies (which is what uv
    actually resolves against) takes precedence over a project.dependencies
    entry for the same package, if they ever disagree.

    Captures EVERY shape referencing the package -- an exact pin
    (`==0.46.8`), a range (`>=0.46.8,<0.47.0`), or a bare unversioned name
    (empty spec) -- so a loosened constraint cannot silently evade
    comparison the way an exact-only, single-operator lookup would (OMN-15604:
    this was the exact bypass a ranged constraint produced against
    `find_lineage_violations` before this function replaced that lookup).
    Callers that need only the exact `==` pins filter this dict's values
    through `_EXACT_PIN_SUFFIX_RE` themselves (see `find_lineage_violations`).
    """
    specs: dict[str, str] = {}

    project = parsed.get("project", {})
    dependencies = project.get("dependencies", []) if isinstance(project, dict) else []

    tool = parsed.get("tool", {})
    uv = tool.get("uv", {}) if isinstance(tool, dict) else {}
    overrides = uv.get("override-dependencies", []) if isinstance(uv, dict) else []

    for requirement_list in (dependencies, overrides):
        if not isinstance(requirement_list, list):
            continue
        for requirement in requirement_list:
            if not isinstance(requirement, str):
                continue
            match = _REQUIREMENT_SPEC_RE.match(requirement.strip())
            if match:
                specs[_normalize(match.group(1))] = match.group(2).strip()

    return specs


def _repo_from_git_url(git_url: str) -> str | None:
    """Extract the bare `<repo>` name from an OmniNode-ai github.com URL."""
    match = _SOURCE_URL_RE.match(git_url.strip())
    return match.group("repo") if match else None


def _api_get(url: str) -> tuple[int | None, dict[str, object] | None, str]:
    """GET a GitHub REST endpoint. Returns ``(status, body, detail)``.

    ``status is None`` means the request could not be performed at all. Same
    shape/posture as ``scripts/ci/check_pin_reachability.py``'s helper of the
    same name (not imported directly — that module's helper is private, and
    duplicating ~15 lines of stdlib HTTP plumbing is cheaper than coupling two
    independently-evolving CI gates through a private symbol).
    """
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "omnibase-infra-dep-provenance-lineage-gate (OMN-15604)",
    }
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)  # noqa: S310 - fixed https host
    try:
        with urllib.request.urlopen(  # noqa: S310 - fixed https host
            request, timeout=_REQUEST_TIMEOUT_SECONDS
        ) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
            body = payload if isinstance(payload, dict) else None
            return response.status, body, f"HTTP {response.status}"
    except urllib.error.HTTPError as exc:
        detail = f"HTTP {exc.code}"
        try:
            body = json.loads(exc.read().decode("utf-8", errors="replace"))
            message = body.get("message", "") if isinstance(body, dict) else ""
            if message:
                detail = f"HTTP {exc.code}: {message}"
        except (ValueError, OSError):
            pass
        return exc.code, None, detail
    except (urllib.error.URLError, OSError, TimeoutError, ValueError) as exc:
        return None, None, f"transport error: {exc}"


def resolve_src_tree_sha(repo: str, ref: str) -> tuple[str | None, str]:
    """Resolve the git tree SHA of ``<ref>:src`` in ``OmniNode-ai/<repo>``.

    Two REST calls, no local clone: (1) resolve ``ref`` to its commit and read
    the commit's root tree SHA, (2) list that root tree's top-level entries and
    return the ``sha`` of the entry named ``src`` (type ``tree``). That `sha`
    IS the tree object GitHub/git would produce for ``git rev-parse <ref>:src``
    — trees are addressed by content, so identical trees hash identically
    regardless of which commit/ref reached them.

    Returns ``(tree_sha, "ok")`` on success, or ``(None, detail)`` describing
    why resolution failed (network failure, missing ref, no top-level `src/`).
    """
    status, body, detail = _api_get(
        f"{_GITHUB_API}/repos/{_ORG}/{repo}/commits/{urllib.parse.quote(ref, safe='')}"
    )
    if status != 200 or body is None:
        return None, f"could not resolve commit for {ref!r}: {detail}"
    commit = body.get("commit")
    root_tree = commit.get("tree") if isinstance(commit, dict) else None
    root_tree_sha = root_tree.get("sha") if isinstance(root_tree, dict) else None
    if not isinstance(root_tree_sha, str):
        return None, f"commit response for {ref!r} missing commit.tree.sha"

    status, body, detail = _api_get(
        f"{_GITHUB_API}/repos/{_ORG}/{repo}/git/trees/{root_tree_sha}"
    )
    if status != 200 or body is None:
        return None, f"could not list root tree for {ref!r}: {detail}"
    entries = body.get("tree")
    if not isinstance(entries, list):
        return None, f"tree response for {ref!r} missing 'tree' entries"
    for entry in entries:
        if (
            isinstance(entry, dict)
            and entry.get("path") == "src"
            and entry.get("type") == "tree"
            and isinstance(entry.get("sha"), str)
        ):
            return entry["sha"], "ok"
    return None, f"no top-level 'src' tree entry found at {ref!r}"


def find_lineage_violations(
    text: str,
    *,
    resolve: _ResolveFn = resolve_src_tree_sha,
) -> list[str]:
    """RED when a git-pinned override's `src/` tree differs from the released
    tree of the version declared alongside it (OMN-15604).

    Applies to every forbidden package (omnibase-core / omnibase-spi /
    omnibase-compat) that has a `[tool.uv.sources]` git override — regardless
    of a `# raw-override-ok:` escape token, which exempts a line from
    `find_violations` only. A package with a git override but NO requirement
    string referencing it anywhere is skipped (nothing to compare against —
    that shape is a `find_violations` failure, not a lineage failure). A
    package that IS referenced but not with an exact `pkg==X.Y.Z` pin (a
    range, or a bare unversioned name) is a VIOLATION, not a skip: a range
    constraint does not unambiguously name one released tree to compare
    against, and treating it as "nothing to compare" is precisely the bypass
    that let a `>=0.46.8,<0.47.0`-style loosening evade this check entirely
    while `find_violations` still exempted the same line via its token.

    `resolve` is injectable for hermetic unit tests; it defaults to the live
    `resolve_src_tree_sha`, which calls the GitHub REST API.
    """
    resolver = resolve

    try:
        parsed = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        return [f"invalid TOML: {exc}"]

    declared_specs = _declared_version_specs(parsed)
    sources = _parse_uv_source_entries(text)

    violations: list[str] = []
    for pkg, attrs in sources.items():
        if pkg not in _FORBIDDEN_PACKAGES:
            continue
        git_url = attrs.get("git")
        if not isinstance(git_url, str):
            continue
        ref = attrs.get("rev") or attrs.get("tag") or attrs.get("branch")
        if not isinstance(ref, str) or not ref:
            continue

        spec = declared_specs.get(pkg)
        if spec is None:
            # Never referenced by any dependency/override-dependency entry --
            # nothing to compare against, and out of this check's scope.
            continue

        exact_m = _EXACT_PIN_SUFFIX_RE.match(spec)
        if exact_m is None:
            violations.append(
                f"{pkg}: git-pinned override (rev={ref!r}) sits alongside a "
                f"non-exact declared constraint ({spec!r}) instead of a "
                f"single `{pkg}==X.Y.Z` pin. A range/loosened/unversioned "
                "constraint cannot be lineage-verified against one released "
                "tree -- this is the exact shape that bypasses this check "
                "by construction (loosen the pin, keep the escape token). "
                "Pin an exact `pkg==X.Y.Z` version, or delete the "
                "[tool.uv.sources] override, so lineage can be proven."
            )
            continue
        version = exact_m.group(1)

        repo = _repo_from_git_url(git_url)
        if repo is None:
            violations.append(
                f"{pkg}: cannot resolve a repo name from git url {git_url!r} to "
                "verify lineage against declared version "
                f"{version!r}"
            )
            continue

        pinned_sha, pinned_detail = resolver(repo, ref)
        released_sha, released_detail = resolver(repo, f"v{version}")

        if pinned_sha is None or released_sha is None:
            violations.append(
                f"{pkg}: UNDETERMINED lineage for rev={ref!r} vs declared "
                f"version {version!r} (tag v{version}) — pinned: {pinned_detail}; "
                f"released: {released_detail}"
            )
            continue

        if pinned_sha != released_sha:
            violations.append(
                f"{pkg}: pinned rev {ref!r} src/ tree ({pinned_sha}) differs from "
                f"released v{version} src/ tree ({released_sha}) — the declared "
                f"version {version!r} does not describe what this override "
                "actually builds. Either delete the [tool.uv.sources] override "
                "and re-lock from the declared version, or correct the declared "
                "version to match what the pin actually builds."
            )

    return violations


# ---------------------------------------------------------------------------
# Escape-token reconciliation (OMN-15604 AC3) — network (Linear), opt-in via
# --check-token-expiry. The `until=` date half never needs network.
# ---------------------------------------------------------------------------


def _parse_escape_token(raw_line: str) -> tuple[str, str | None] | None:
    """Return `(ticket, until_date)` from a raw source line's
    `# raw-override-ok: <ticket> [until=YYYY-MM-DD]` comment, or `None` if no
    valid non-empty token is present on the line.
    """
    token_m = _ESCAPE_TOKEN_RE.search(raw_line)
    if not token_m or not token_m.group(1).strip():
        return None
    ticket = token_m.group(1).strip()
    until_m = _ESCAPE_TOKEN_UNTIL_RE.search(raw_line)
    until_date = until_m.group(1) if until_m else None
    return ticket, until_date


def resolve_ticket_status(ticket_id: str) -> tuple[str | None, str]:
    """Resolve a Linear ticket's status name via the Linear GraphQL API.

    Returns `(status_name, "ok")` on success, or `(None, detail)` if the
    ticket could not be resolved: `LINEAR_API_KEY` unset, transport failure,
    or the ticket not found. `detail == "LINEAR_API_KEY not set"` is the
    specific sentinel `find_escape_token_violations` checks to apply the same
    graceful-degradation posture as this org's other already-shipped,
    LINEAR_API_KEY-gated Linear check (the stale ticket-tag scanner under
    `scripts/validation/`, wired as its own required CI gate): without a
    credential the check cannot run at all, so it is skipped rather than
    failing every PR in a repo that never provisioned the secret.

    Uses `issue(id: "<identifier>")`, not the `issueSearch` filter shape that
    scanner uses — that filter shape (`identifier: { eq: ... }`) is rejected
    by the live Linear schema (`GRAPHQL_VALIDATION_FAILED`); `issue(id:)`
    accepts a human-readable identifier directly and was verified live
    against the real OMN-15414 ticket during this ticket's own build.
    """
    api_key = os.environ.get("LINEAR_API_KEY", "")
    if not api_key:
        return None, "LINEAR_API_KEY not set"

    query = {
        "query": (
            'query { issue(id: "'
            + ticket_id.replace('"', "")
            + '") { identifier state { name type } } }'
        )
    }
    request = urllib.request.Request(  # noqa: S310 - fixed https host
        _LINEAR_API,
        data=json.dumps(query).encode("utf-8"),
        headers={"Authorization": api_key, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(  # noqa: S310 - fixed https host
            request, timeout=_REQUEST_TIMEOUT_SECONDS
        ) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        return None, f"HTTP {exc.code}"
    except (urllib.error.URLError, OSError, TimeoutError, ValueError) as exc:
        return None, f"transport error: {exc}"

    if not isinstance(payload, dict):
        return None, "malformed Linear API response"
    if payload.get("errors"):
        return None, f"Linear API error: {payload['errors']}"
    issue = (
        payload.get("data", {}).get("issue")
        if isinstance(payload.get("data"), dict)
        else None
    )
    if not isinstance(issue, dict):
        return None, f"ticket {ticket_id!r} not found in Linear"
    state = issue.get("state") if isinstance(issue.get("state"), dict) else {}
    name = state.get("name") if isinstance(state, dict) else None
    if not isinstance(name, str) or not name:
        return None, f"ticket {ticket_id!r} has no resolvable state"
    return name, "ok"


def find_escape_token_violations(
    text: str,
    *,
    resolve_ticket: _TicketResolveFn = resolve_ticket_status,
    today: date | None = None,
) -> list[str]:
    """RED when a `# raw-override-ok: <ticket>` escape token has expired
    (OMN-15604 AC3).

    The token exempted a forbidden git-source override unconditionally and
    forever in the original OMN-13873 design. This closes that gap with
    either of the two conditions the ticket accepts as sufficient:

    1. an explicit `until=YYYY-MM-DD` suffix has passed `today`
       (`# raw-override-ok: OMN-15414 until=2026-09-01`), or
    2. the cited ticket resolves (via Linear) to a Done/Cancelled/Duplicate
       status — the override's justification is closed, so the override
       itself is stale.

    `LINEAR_API_KEY` is not provisioned anywhere in this org today, so
    condition 2 never actually resolves in the live enforcing environment --
    a token with no `until=` suffix would otherwise fall through both
    conditions and pass forever, exactly reproducing the AC3 defect against
    the live incident token. A token supplying neither a live `until=` date
    nor a resolvable ticket status is therefore a violation in its own right:
    graceful degradation on a missing `LINEAR_API_KEY` only applies once an
    `until=` date has already provided a live, network-free enforcement path
    for this line.

    A line with NO escape token is out of scope here — `find_violations`
    already fails it unconditionally; this function only reconciles tokens
    that `find_violations` currently treats as a permanent pass.

    `resolve_ticket` is injectable for hermetic unit tests; it defaults to
    the live `resolve_ticket_status`, which calls the Linear API.
    """
    if today is None:
        today = datetime.now(tz=UTC).date()

    block = _uv_sources_block(text) or text
    try:
        entries = _parse_uv_source_entries(text)
    except ValueError as exc:
        return [str(exc)]

    violations: list[str] = []
    for pkg, attrs in entries.items():
        if pkg not in _FORBIDDEN_PACKAGES:
            continue
        if not (_GIT_SOURCE_KEYS & set(attrs)):
            continue

        raw_line = _line_for_package(block, pkg)
        if raw_line is None:
            continue
        parsed_token = _parse_escape_token(raw_line)
        if parsed_token is None:
            continue
        raw_ticket, until_date = parsed_token

        if until_date is not None:
            try:
                expiry = date.fromisoformat(until_date)
            except ValueError:
                violations.append(
                    f"{pkg}: raw-override-ok token for {raw_ticket} has an "
                    f"unparseable until= date ({until_date!r}); expected "
                    "YYYY-MM-DD"
                )
                continue
            if today > expiry:
                violations.append(
                    f"{pkg}: raw-override-ok token for {raw_ticket} EXPIRED "
                    f"on {until_date} (today: {today.isoformat()}) — the "
                    "override is no longer exempt from the forbid-git-source "
                    "gate. Renew with a new until= date, or resolve the "
                    "override."
                )
                continue

        ticket_m = _TICKET_ID_RE.match(raw_ticket)
        ticket_id = ticket_m.group(1) if ticket_m else raw_ticket
        status_name, detail = resolve_ticket(ticket_id)
        if status_name is None:
            if detail == "LINEAR_API_KEY not set":
                if until_date is None:
                    # Neither reconciliation condition is live: there is no
                    # until= date to fail closed on, and ticket-status
                    # resolution cannot run without LINEAR_API_KEY (unset
                    # everywhere in this org today). Falling through here
                    # would reproduce the exact OMN-13873 unconditional-
                    # forever pass this function exists to close, so the
                    # token itself is the violation.
                    violations.append(
                        f"{pkg}: raw-override-ok token for {raw_ticket} has "
                        "no until= expiry date, and Linear ticket-status "
                        "resolution is unavailable (LINEAR_API_KEY not set) "
                        "-- neither reconciliation condition is live, so the "
                        "override would otherwise be exempt unconditionally "
                        "and forever. Add an explicit `until=YYYY-MM-DD` "
                        "suffix to the token."
                    )
                    continue
                # An until= date IS present (and, since we reached this line,
                # already checked above as not yet expired) -- that date is
                # itself a live, network-free fail-closed condition, so a
                # missing credential for this secondary ticket-status check
                # is not fatal. Graceful degradation, same posture as
                # check_stale_todos.py.
                continue
            violations.append(
                f"{pkg}: could not resolve Linear status for cited ticket "
                f"{ticket_id} to verify the raw-override-ok token is still "
                f"valid: {detail}"
            )
            continue
        if status_name.strip().lower() in _TICKET_DONE_STATUSES:
            violations.append(
                f"{pkg}: raw-override-ok token cites {ticket_id}, whose "
                f"Linear status is {status_name!r} — a closed ticket is no "
                "longer a valid justification for an unconditional "
                "git-source override. File a reconciliation ticket, or "
                "resolve the override."
            )

    return violations


# ---------------------------------------------------------------------------
# Cascade-movability check (OMN-15604 AC4) — offline, `--check-movable`.
# ---------------------------------------------------------------------------


def find_unmovable_cascade_targets(text: str, package: str) -> list[str]:
    """Return a violation message if `package` cannot be moved by a
    dependency cascade (OMN-15604 AC4).

    `uv lock --upgrade-package <pkg>==<version>` re-resolves the dependency
    graph, but a `[tool.uv.sources]` entry for that package takes precedence
    over registry resolution regardless — uv re-resolves against the SAME
    pinned git ref and typically produces a byte-identical `uv.lock`, which
    `.github/workflows/dependency-cascade.yml` currently reads as "no
    lockfile changes — already on latest" (a false-positive SKIP: the repo is
    not on latest, it is still stuck on the git pin).

    Returns a non-empty list (one message) when `package` has an active
    `[tool.uv.sources]` git override — regardless of a `raw-override-ok`
    escape token, since the token only ever exempted the forbid-git-source
    rule, never a cascade's ability to move the pin. Returns `[]` when
    `package` has no such override (movable), including when `package` isn't
    tracked at all (`onex-change-control`-style git pins are legitimate and
    out of scope for this check, same as `find_violations`/
    `find_lineage_violations`).
    """
    pkg = _normalize(package)
    try:
        sources = _parse_uv_source_entries(text)
    except ValueError as exc:
        return [str(exc)]

    attrs = sources.get(pkg)
    if attrs is None:
        return []
    git_keys = sorted(_GIT_SOURCE_KEYS & set(attrs))
    if not git_keys:
        return []

    keys_desc = ", ".join(f"{k}={attrs[k]!r}" for k in git_keys)
    return [
        f"{pkg}: pinned via [tool.uv.sources] git override ({keys_desc}). "
        "'uv lock --upgrade-package' cannot move a [tool.uv.sources] git "
        "pin -- it re-resolves against the SAME override and will silently "
        "report no lockfile change, masking a no-op cascade. Remove the "
        "[tool.uv.sources] override for this package before running a "
        "dependency cascade against it."
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pyproject",
        default="pyproject.toml",
        help="Path to pyproject.toml (default: pyproject.toml)",
    )
    parser.add_argument(
        "--check-lineage",
        action="store_true",
        help=(
            "Additionally run the content-lineage check (OMN-15604): fail if a "
            "[tool.uv.sources] git-pinned rev's src/ tree differs from the "
            "released tree of the version declared alongside it. Applies even "
            "to lines carrying a valid raw-override-ok token. Makes GitHub "
            "REST API calls — off by default; never invoked from pre-commit."
        ),
    )
    parser.add_argument(
        "--allow-undetermined-lineage",
        action="store_true",
        help=(
            "Downgrade an UNDETERMINED lineage resolution (network failure) "
            "from failure to a warning. Local/offline use only — REFUSED when "
            "CI is set, matching check_pin_reachability.py's posture."
        ),
    )
    parser.add_argument(
        "--check-token-expiry",
        action="store_true",
        help=(
            "Additionally run the escape-token reconciliation check "
            "(OMN-15604 AC3): fail if a raw-override-ok token has no "
            "until= date, if its until= date has passed, or if its cited "
            "ticket resolves (via the Linear API) to a closed status. "
            "Ticket-status resolution requires LINEAR_API_KEY and is "
            "gracefully skipped (not failed) when unset -- but ONLY once "
            "a live until= date is present; a token with neither is a "
            "violation, matching scripts/validation/check_stale_todos.py's "
            "graceful-degradation posture for the secondary check only."
        ),
    )
    parser.add_argument(
        "--check-movable",
        metavar="PACKAGE",
        default=None,
        help=(
            "Standalone check (OMN-15604 AC4): fail if PACKAGE has an active "
            "[tool.uv.sources] git override, which a `uv lock "
            "--upgrade-package` dependency cascade cannot move and will "
            "silently no-op against. Runs instead of the default "
            "find_violations check; does not combine with --check-lineage "
            "or --check-token-expiry."
        ),
    )
    args = parser.parse_args(argv)

    in_ci = bool(os.environ.get("CI"))
    if args.allow_undetermined_lineage and in_ci:
        print(
            "error: --allow-undetermined-lineage is refused under CI. The "
            "enforcing surface must fail closed on an unresolvable pin.",
            file=sys.stderr,
        )
        return 2

    pyproject_path = Path(args.pyproject)
    if not pyproject_path.exists():
        print(
            f"ERROR: pyproject.toml not found: {pyproject_path}",
            file=sys.stderr,
        )
        return 1

    if args.check_movable:
        text = pyproject_path.read_text()
        movable_violations = find_unmovable_cascade_targets(text, args.check_movable)
        if movable_violations:
            print(
                f"FAIL: {args.check_movable} cannot be moved by a dependency cascade:",
                file=sys.stderr,
            )
            for msg in movable_violations:
                print(f"  - {msg}", file=sys.stderr)
            return 1
        print(
            f"OK: {args.check_movable} has no [tool.uv.sources] git override "
            "-- movable by a dependency cascade."
        )
        return 0

    text = pyproject_path.read_text()
    violations = find_violations(text)

    if violations:
        print(
            "FAIL: forbidden first-party git-source override(s) in "
            f"{pyproject_path} [tool.uv.sources]:",
            file=sys.stderr,
        )
        for msg in violations:
            print(f"  - {msg}", file=sys.stderr)
        print(
            "\nomnibase-core / omnibase-spi / omnibase-compat are PyPI-published "
            "first-party deps and must NOT be pinned to git commits/branches/tags "
            "on dev/main. Resolve them from PyPI (release the dep first if the "
            "needed version is unpublished). If a temporary override is genuinely "
            "unavoidable, annotate the exact line with "
            "'# raw-override-ok: <ticket>' (non-empty token).",
            file=sys.stderr,
        )
        return 1

    print(
        f"OK: no forbidden first-party git-source override in "
        f"{pyproject_path} [tool.uv.sources]."
    )

    if not args.check_lineage and not args.check_token_expiry:
        return 0

    failed = False

    if args.check_lineage:
        lineage_violations = find_lineage_violations(text)
        undetermined = [v for v in lineage_violations if "UNDETERMINED lineage" in v]
        diverged = [v for v in lineage_violations if v not in undetermined]

        if diverged:
            print(
                "\nFAIL: git-pinned override content diverges from the released "
                f"tree of its declared version in {pyproject_path}:",
                file=sys.stderr,
            )
            for msg in diverged:
                print(f"  - {msg}", file=sys.stderr)
            failed = True
        elif undetermined and not args.allow_undetermined_lineage:
            print(
                f"\nFAIL: {len(undetermined)} pin(s) could not be "
                "lineage-resolved. An unresolvable pin is not a passing pin.",
                file=sys.stderr,
            )
            for msg in undetermined:
                print(f"  - {msg}", file=sys.stderr)
            failed = True
        elif undetermined:
            print(
                f"\nWARNING: {len(undetermined)} pin(s) UNDETERMINED and "
                "--allow-undetermined-lineage was passed. This run proved "
                "nothing about lineage for those pins; CI is the enforcing "
                "surface."
            )
        else:
            print(
                "OK: no git-pinned override content diverges from its declared version."
            )

    if args.check_token_expiry:
        token_violations = find_escape_token_violations(text)
        if token_violations:
            print(
                "\nFAIL: raw-override-ok escape token(s) need reconciliation "
                f"in {pyproject_path}:",
                file=sys.stderr,
            )
            for msg in token_violations:
                print(f"  - {msg}", file=sys.stderr)
            failed = True
        else:
            print(
                "OK: no raw-override-ok escape token is expired or cites a "
                "closed ticket."
            )

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
