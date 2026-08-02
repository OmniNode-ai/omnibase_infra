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

Usage::

    uv run python scripts/check_dep_provenance.py
    uv run python scripts/check_dep_provenance.py --pyproject pyproject.toml
    uv run python scripts/check_dep_provenance.py --check-lineage
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
from pathlib import Path

_ResolveFn = Callable[[str, str], "tuple[str | None, str]"]

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

# `pkg==X.Y.Z` inside a project.dependencies / override-dependencies string.
_DECLARED_VERSION_RE = re.compile(r"^([A-Za-z0-9_.-]+)==([A-Za-z0-9_.+-]+)$")

# GitHub org all first-party OmniNode repos live under, and the REST root.
_ORG = "OmniNode-ai"
_GITHUB_API = "https://api.github.com"  # url-authority-ok: fixed public REST API, no ONEX routing authority
_REQUEST_TIMEOUT_SECONDS = 10.0

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


def _declared_versions(parsed: dict[str, object]) -> dict[str, str]:
    """Return {normalized_pkg: version} for every `pkg==X.Y.Z` pin.

    Scans `project.dependencies` and `[tool.uv] override-dependencies` — the
    two loci the live incident used (pyproject.toml:36 and :189). Later
    entries win over earlier ones so override-dependencies (which is what uv
    actually resolves against) takes precedence over a project.dependencies
    entry for the same package, if they ever disagree.
    """
    versions: dict[str, str] = {}

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
            match = _DECLARED_VERSION_RE.match(requirement.strip())
            if match:
                versions[_normalize(match.group(1))] = match.group(2)

    return versions


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
    omnibase-compat) that has BOTH a `[tool.uv.sources]` git override AND a
    `pkg==X.Y.Z` version constraint — **regardless of a `# raw-override-ok:`
    escape token**, which exempts a line from `find_violations` only. A
    package with a git override but no parseable declared version is skipped
    (nothing to compare against — that shape is a `find_violations` failure,
    not a lineage failure).

    `resolve` is injectable for hermetic unit tests; it defaults to the live
    `resolve_src_tree_sha`, which calls the GitHub REST API.
    """
    resolver = resolve

    try:
        parsed = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        return [f"invalid TOML: {exc}"]

    declared = _declared_versions(parsed)
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
        version = declared.get(pkg)
        if version is None:
            continue

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

    if not args.check_lineage:
        return 0

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
        return 1

    if undetermined and not args.allow_undetermined_lineage:
        print(
            f"\nFAIL: {len(undetermined)} pin(s) could not be lineage-resolved. "
            "An unresolvable pin is not a passing pin.",
            file=sys.stderr,
        )
        for msg in undetermined:
            print(f"  - {msg}", file=sys.stderr)
        return 1

    if undetermined:
        print(
            f"\nWARNING: {len(undetermined)} pin(s) UNDETERMINED and "
            "--allow-undetermined-lineage was passed. This run proved nothing "
            "about lineage for those pins; CI is the enforcing surface."
        )

    print("OK: no git-pinned override content diverges from its declared version.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
