#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Dependency provenance checks for first-party uv sources on dev/main.

The default offline mode (``find_violations``) refuses first-party git overrides
unless their line has a nonempty ``# raw-override-ok: <ticket>`` token or their
only ref is a dev-merge tag, ``dev-YYYYMMDDHHMMSS-<sha12>`` (OMN-17427).
It exits 0 for admitted sources and 1 for violations or missing/invalid input.

``--check-lineage`` additionally compares a git pin's ``src/`` tree with its
exact declared release, regardless of escape tokens; ranges fail. Dev-merge
tags are skipped because core's ``dev_tag_lineage`` validator owns their tag
identity, dev ancestry, version and adjacent lock verification. CLI resolution
tries the runner's git mirror, then GitHub REST; unresolved facts fail closed.
It exits 0 on success and 1 on violations. ``--allow-undetermined-lineage``
warns locally on unresolved facts but is refused under CI with exit 2.

``--check-token-expiry`` additionally refuses expired ``until=YYYY-MM-DD``
escape tokens and tokens citing closed Linear tickets. Without LINEAR_API_KEY,
ticket lookup is skipped only when an until date already enforces expiry;
a token with neither is a violation. It exits 0 on success and 1 on violations.

``--check-movable PACKAGE`` runs alone: active git overrides prevent a release
cascade except for dev-merge tags, which the cascade verifies with core before
removal. With ``--target-version VERSION``, declared constraints must also
admit that version. It exits 0 if movable and 1 on violations or invalid input.

``--drop-dev-tag-override PACKAGE`` runs alone after the caller verifies the
package with core. It removes only a single-line dev-merge tag source, preserving
all other bytes and requiring the parsed result to equal the original TOML
minus that source. It exits 0 for removal or absence, and 1 for any other shape
or invalid input, without changing the file on refusal. No verification or
network access occurs in this mode. Argument errors exit 2 in every mode.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

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


def _is_dev_merge_tag(attrs: dict[str, object]) -> bool:
    """Recognize the sanctioned ref shape; core verifies its identity."""
    tag = attrs.get("tag")
    return (
        isinstance(attrs.get("git"), str)
        and isinstance(tag, str)
        and re.fullmatch(r"dev-\d{14}-[0-9a-f]{12}", tag) is not None
        and not ({"rev", "branch"} & attrs.keys())
    )


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
    escape or a dev-merge tag ref are excluded; core verifies tag identity.
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
        if not git_keys or _is_dev_merge_tag(attrs):
            # Registry sources and core-verified dev-merge tag shapes are admitted.
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


# ---------------------------------------------------------------------------
# Hermetic mirror-based resolution (OMN-16096) -- the merge-required path must
# not depend on live GitHub API reachability. See
# docker/runners/git-mirror-refresh.sh (OMN-14027 C2) and the
# `_C2_MIRROR_HOST`/`_C2_MIRROR_PORT` wiring in
# docker/runners/runner-job-started.sh (OMN-16053) for the mirror this reuses.
# ---------------------------------------------------------------------------

# Host-level bare git mirror served over the self-hosted runner's own docker
# bridge network -- reachable without ever leaving the runner host, so it is
# immune to the github.com egress degradation that produced the
# `transport error: The read operation timed out` failures this closes.
# ubuntu-latest (public-fork PRs) has no route to this address at all;
# `resolve_src_tree_sha_hermetic` below falls back to the live resolver there,
# unchanged from pre-OMN-16096 behavior for that lane.
_MIRROR_HOST = os.environ.get("OMNI_GIT_MIRROR_HOST", "172.18.0.1")
_MIRROR_PORT = os.environ.get("OMNI_GIT_MIRROR_PORT", "9418")
_MIRROR_TIMEOUT_SECONDS = 15.0
# git subprocess calls talk only to the mirror (git:// protocol, no auth) --
# disabling a terminal credential prompt is defense in depth, not a fix for
# an observed hang.
_MIRROR_GIT_ENV = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}


def resolve_src_tree_sha_via_mirror(
    repo: str, ref: str, *, timeout: float = _MIRROR_TIMEOUT_SECONDS
) -> tuple[str | None, str]:
    """Resolve the git tree SHA of ``<ref>:src`` via the host-level git mirror.

    Fetches exactly the requested ref from the runner-host bare mirror
    (``git://<host>:<port>/<repo>.git``, OMN-16053) into a throwaway local
    repo and resolves the tree SHA with local git plumbing
    (``git rev-parse FETCH_HEAD:src``) -- no REST call, no github.com
    reachability required at all. Tree objects are content-addressed, so this
    returns the byte-identical SHA :func:`resolve_src_tree_sha` would compute
    over the REST API for the same ref (the same invariant that function's own
    docstring documents).

    Returns ``(tree_sha, "ok (mirror)")`` on success, or ``(None, detail)``
    describing why resolution failed (mirror unreachable, repo/ref not served
    by this mirror, no top-level ``src/`` tree). Every failure mode returns
    cleanly -- a missing ``git`` binary, a dead mirror, or an unmirrored repo
    all yield ``(None, detail)`` rather than raising, so a caller can always
    fall back to another resolver.
    """
    mirror_url = f"git://{_MIRROR_HOST}:{_MIRROR_PORT}/{repo}.git"
    try:
        with tempfile.TemporaryDirectory(prefix="dep-provenance-mirror-") as tmp:
            init = subprocess.run(
                ["git", "init", "--quiet", "--bare", tmp],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=_MIRROR_GIT_ENV,
                check=False,
            )
            if init.returncode != 0:
                return (
                    None,
                    f"mirror resolution failed: git init error: {init.stderr.strip()}",
                )

            fetch = subprocess.run(
                ["git", "-C", tmp, "fetch", "--quiet", "--depth=1", mirror_url, ref],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=_MIRROR_GIT_ENV,
                check=False,
            )
            if fetch.returncode != 0:
                return (
                    None,
                    f"could not fetch {ref!r} from mirror {mirror_url}: "
                    f"{fetch.stderr.strip() or 'unknown git error'}",
                )

            rev = subprocess.run(
                [
                    "git",
                    "-C",
                    tmp,
                    "rev-parse",
                    "--verify",
                    "--quiet",
                    "FETCH_HEAD:src",
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=_MIRROR_GIT_ENV,
                check=False,
            )
            tree_sha = rev.stdout.strip()
            if rev.returncode != 0 or not tree_sha:
                return (
                    None,
                    f"ref {ref!r} resolved via mirror {mirror_url} but has no "
                    "top-level 'src' tree",
                )
            return tree_sha, "ok (mirror)"
    except subprocess.TimeoutExpired:
        return None, f"mirror resolution timed out after {timeout:.0f}s for {ref!r}"
    except OSError as exc:
        return None, f"mirror resolution failed: {exc}"


def resolve_src_tree_sha_hermetic(repo: str, ref: str) -> tuple[str | None, str]:
    """Merge-path resolver (OMN-16096): the host-level git mirror first,
    falling back to the live GitHub REST API only when the mirror itself
    cannot serve the ref (e.g. ubuntu-latest public-fork runners, which have
    no route to the runner host's docker network at all, or a repo the mirror
    does not yet carry).

    This is the resolver ``--check-lineage`` uses by default. It removes the
    merge-required gate's dependency on github.com reachability on the fleet
    where the OMN-16053 mirror is actually served (self-hosted omnibase-ci);
    it changes nothing on lanes that never had the mirror to begin with.
    """
    mirror_sha, mirror_detail = resolve_src_tree_sha_via_mirror(repo, ref)
    if mirror_sha is not None:
        return mirror_sha, mirror_detail
    live_sha, live_detail = resolve_src_tree_sha(repo, ref)
    if live_sha is not None:
        return live_sha, live_detail
    return None, f"mirror: {mirror_detail}; live API: {live_detail}"


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

    Dev-merge tag refs are skipped; core's dev_tag_lineage owns their identity.
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
        if _is_dev_merge_tag(attrs):
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


def _range_cap_violation(pkg: str, spec_str: str, target_version: str) -> str | None:
    """Return a violation message if the declared requirement `spec_str`
    (e.g. `>=0.46.13,<0.47.0`, `==0.46.8`, or `` for an unversioned/absent
    requirement) does NOT admit `target_version` (OMN-16926).

    Uses `packaging.specifiers.SpecifierSet`, so every PEP 440 operator is
    covered generically (`<`, `<=`, `==`, `~=`, `!=`, and combinations) rather
    than a hand-rolled per-operator parser -- an empty/unversioned spec
    admits every version (nothing to cap), matching uv's own resolution
    semantics.

    Returns `None` (no violation / fail-open on this specific comparison)
    when either side fails to parse — an unparseable spec or target is a
    caller/input error, not evidence of a cap; `find_unmovable_cascade_targets`
    still fails closed overall on a hard TOML parse error via its own
    try/except.
    """
    if not spec_str:
        return None
    try:
        target = Version(target_version)
    except InvalidVersion:
        return None
    try:
        specifier = SpecifierSet(spec_str)
    except InvalidSpecifier:
        return None
    if target in specifier:
        return None
    return (
        f"{pkg}: pinned via a range constraint ({spec_str!r}) in "
        "[project.dependencies] that does not admit target version "
        f"{target_version!r}. 'uv lock --upgrade-package' cannot select a "
        "version outside its own pyproject.toml constraint -- it will "
        "silently re-resolve within the SAME capped range and report no "
        "lockfile change, masking a no-op cascade (this repo is NOT already "
        f"on the latest version; it is capped below {target_version!r} and "
        "cannot reach it). Widen the pyproject.toml constraint for "
        f"{pkg} to admit {target_version!r}, then re-run this dependency "
        "cascade."
    )


def find_unmovable_cascade_targets(
    text: str, package: str, target_version: str | None = None
) -> list[str]:
    """Return a violation message if `package` cannot be moved by a
    dependency cascade (OMN-15604 AC4, extended OMN-16926).

    Dev-merge tag refs yield to core verification and removal in the cascade.
    Two independent immovability channels, checked in this order:

    1. **Git-source override** (OMN-15604 AC4): `uv lock --upgrade-package
       <pkg>==<version>` re-resolves the dependency graph, but a
       `[tool.uv.sources]` entry for that package takes precedence over
       registry resolution regardless — uv re-resolves against the SAME
       pinned git ref and typically produces a byte-identical `uv.lock`.
       Checked unconditionally (does not require `target_version`),
       regardless of a `raw-override-ok` escape token, since the token only
       ever exempted the forbid-git-source rule (`find_violations`), never a
       cascade's ability to move the pin.
    2. **Range cap** (OMN-16926): when `target_version` is supplied and no
       git override fired, the package's declared requirement in
       `[project.dependencies]` (any PEP 440 shape) is checked against
       `target_version` via `_range_cap_violation`. A constraint that
       excludes the target is a violation with the SAME "cannot be moved"
       consequence as a git override — `uv lock --upgrade-package` cannot
       cross a `pyproject.toml` constraint any more than it can cross a git
       source pin, and the pre-OMN-16926 checker silently missed this
       channel entirely (5 of 6 downstream legs on the v0.47.0 cascade were
       misreported as "already on latest" through exactly this gap).

    Returns `[]` when `package` has neither channel active (genuinely
    movable), including when `package` isn't tracked at all
    (a package that is not one of the three forbidden first-party deps is out
    of scope for this check, same as `find_violations`/`find_lineage_violations`),
    and when `target_version` is omitted (pre-OMN-16926 behavior: range caps
    are not evaluated without a target to compare against).
    """
    pkg = _normalize(package)
    try:
        sources = _parse_uv_source_entries(text)
    except ValueError as exc:
        return [str(exc)]

    attrs = sources.get(pkg)
    if attrs is not None:
        git_keys = sorted(_GIT_SOURCE_KEYS & set(attrs))
        if git_keys and not _is_dev_merge_tag(attrs):
            keys_desc = ", ".join(f"{k}={attrs[k]!r}" for k in git_keys)
            return [
                f"{pkg}: pinned via [tool.uv.sources] git override "
                f"({keys_desc}). 'uv lock --upgrade-package' cannot move a "
                "[tool.uv.sources] git pin -- it re-resolves against the "
                "SAME override and will silently report no lockfile "
                "change, masking a no-op cascade. Remove the "
                "[tool.uv.sources] override for this package before "
                "running a dependency cascade against it."
            ]

    if target_version is None:
        return []

    try:
        parsed = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        return [str(exc)]
    declared_specs = _declared_version_specs(parsed)
    spec_str = declared_specs.get(pkg)
    if spec_str is None:
        return []

    violation = _range_cap_violation(pkg, spec_str, target_version)
    return [violation] if violation else []


def drop_dev_tag_override(text: str, package: str) -> str:
    """Remove one single-line source without touching any neighbouring bytes."""
    pkg = _normalize(package)
    entries = _parse_uv_source_entries(text)
    parsed = tomllib.loads(text)
    sources = parsed
    for part in ("tool", "uv", "sources"):
        sources = sources.get(part, {}) if isinstance(sources, dict) else {}
    if not isinstance(sources, dict):
        raise ValueError("tool.uv.sources must be a table")
    keys = [key for key in sources if _normalize(key) == pkg]
    if not keys:
        return text
    attrs = entries.get(pkg, {})
    block = _uv_sources_block(text)
    raw_line = _line_for_package(block, pkg) if block is not None else None
    if len(keys) != 1 or not _is_dev_merge_tag(attrs) or raw_line is None:
        raise ValueError(f"{pkg}: expected one single-line dev-merge tag source")
    if _parse_uv_source_entries("[tool.uv.sources]\n" + raw_line) != {pkg: attrs}:
        raise ValueError(f"{pkg}: source is not a self-contained single-line entry")
    match = _UVS_BLOCK_RE.search(text)
    assert match is not None and block is not None
    line = next(
        line
        for line in block.splitlines(keepends=True)
        if line.rstrip("\r\n") == raw_line
    )
    result = text[: match.start(1)] + block.replace(line, "", 1) + text[match.end(1) :]
    del sources[keys[0]]
    if tomllib.loads(result) != parsed:
        raise ValueError(f"{pkg}: removing source would change other TOML values")
    return result


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
            "to lines carrying a valid raw-override-ok token. Resolves via the "
            "OMN-16053 host-level git mirror first, falling back to the "
            "GitHub REST API only when the mirror cannot serve the ref "
            "(OMN-16096) — off by default; never invoked from pre-commit."
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
            "Standalone check (OMN-15604 AC4, extended OMN-16926): fail if "
            "PACKAGE has an active [tool.uv.sources] git override, OR (when "
            "--target-version is also given) a [project.dependencies] range "
            "constraint that excludes --target-version -- either of which a "
            "`uv lock --upgrade-package` dependency cascade cannot move and "
            "will silently no-op against. Runs instead of the default "
            "find_violations check; does not combine with --check-lineage "
            "or --check-token-expiry."
        ),
    )
    parser.add_argument(
        "--target-version",
        metavar="VERSION",
        default=None,
        help=(
            "OMN-16926: the version a dependency cascade is trying to move "
            "--check-movable's PACKAGE to. Enables the range-cap channel of "
            "--check-movable (a pyproject.toml constraint that excludes "
            "this version fails the check); omitted, only the "
            "[tool.uv.sources] git-override channel is checked (pre-"
            "OMN-16926 behavior). Has no effect without --check-movable."
        ),
    )
    parser.add_argument(
        "--drop-dev-tag-override",
        metavar="PACKAGE",
        help="Remove a single-line dev-merge tag source after caller verification.",
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

    if args.drop_dev_tag_override:
        try:
            original = pyproject_path.read_bytes()
            result = drop_dev_tag_override(
                original.decode("utf-8"), args.drop_dev_tag_override
            )
            if result.encode("utf-8") != original:
                pyproject_path.write_bytes(result.encode("utf-8"))
        except (ValueError, OSError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.check_movable:
        text = pyproject_path.read_text()
        movable_violations = find_unmovable_cascade_targets(
            text, args.check_movable, target_version=args.target_version
        )
        if movable_violations:
            print(
                f"FAIL: {args.check_movable} cannot be moved by a dependency cascade:",
                file=sys.stderr,
            )
            for msg in movable_violations:
                print(f"  - {msg}", file=sys.stderr)
            return 1
        target_desc = (
            f" -- admits target version {args.target_version}"
            if args.target_version
            else ""
        )
        print(
            f"OK: {args.check_movable} has no blocking [tool.uv.sources] git override "
            f"and no excluding pyproject.toml range cap{target_desc} -- "
            "movable after core verifies and the cascade drops any dev-merge tag source."
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
        lineage_violations = find_lineage_violations(
            text, resolve=resolve_src_tree_sha_hermetic
        )
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
