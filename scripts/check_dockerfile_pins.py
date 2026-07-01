#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Validate Dockerfile plugin pin ranges against published PyPI releases.

This script parses Dockerfile.runtime, extracts ``uv pip install`` lines that
use ``--no-deps`` (the runtime-plugin pattern), and verifies that:

1. The installed version range is satisfiable against the latest published
   release on PyPI (i.e. the latest release falls within the declared range).
2. No package is pinned to an exact version (``==``) without an accompanying
   ``--no-deps`` flag; exact pins without ``--no-deps`` risk dep-resolver
   conflicts on coordinated releases.

Usage::

    # From the repo root
    uv run python scripts/check_dockerfile_pins.py
    uv run python scripts/check_dockerfile_pins.py --dockerfile docker/Dockerfile.runtime

Exit codes:
    0 — all checks passed
    1 — one or more checks failed (stderr has details)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class PinEntry(NamedTuple):
    package: str
    specifier: str  # e.g. ">=0.3.0,<0.5.0" or "==0.3.0"
    line_number: int
    no_deps: bool


class NpmPinEntry(NamedTuple):
    """A single ``npm install -g`` global-package spec parsed from a Dockerfile.

    npm CLI globals (codex / claude — OMN-13248) are pinned to an EXACT
    ``@version`` (the opposite of the PyPI ``--no-deps`` range pins) so the
    runtime image is fully reproducible: a moving npm tag would silently change
    the shipped agent CLI on every rebuild.
    """

    package: str  # e.g. "@openai/codex" or "@anthropic-ai/claude-code"
    version: str  # exact version after the trailing "@", or "" if unpinned
    line_number: int


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_INSTALL_LINE_RE = re.compile(r"(?:^|\s)(?:uv|uv-with-retry)\s+pip\s+install\b(.+)$")
# Matches a package specifier such as "omninode-claude>=0.3.0,<0.5.0" (with or
# without surrounding quotes).
_PACKAGE_SPEC_RE = re.compile(
    r'"?([A-Za-z0-9_\-]+)([>=<!,\d\.\*]+)"?',
)

# Matches an `npm install -g` / `npm i -g` line and captures the trailing args.
_NPM_GLOBAL_INSTALL_RE = re.compile(
    r"(?:^|\s)npm\s+(?:install|i)\s+(?:-g|--global)\b(.+)$"
)
# Matches one npm package spec token: an optional @scope/, the package name, and
# an optional trailing @version. Examples:
#   @openai/codex@0.141.0  -> ("@openai/codex", "0.141.0")
#   @anthropic-ai/claude-code  -> ("@anthropic-ai/claude-code", "")  (unpinned)
_NPM_PACKAGE_SPEC_RE = re.compile(
    r"(?P<package>@?[A-Za-z0-9_][A-Za-z0-9_.\-]*(?:/[A-Za-z0-9_][A-Za-z0-9_.\-]*)?)"
    r"(?:@(?P<version>[0-9][A-Za-z0-9_.\-]*))?"
)
_NPM_ALIAS_MARKER = "@npm:"


def _iter_logical_lines(text: str) -> list[tuple[int, str]]:
    """Join backslash-continued physical lines into logical lines.

    Returns a list of ``(start_line_number, joined_line)`` tuples.
    """
    logical_lines: list[tuple[int, str]] = []
    pending = ""
    start_line = 1
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.rstrip()
        if pending == "":
            start_line = line_no
        if stripped.endswith("\\"):
            pending += stripped[:-1] + " "
            continue
        logical_lines.append((start_line, pending + stripped))
        pending = ""
    if pending:
        logical_lines.append((start_line, pending))
    return logical_lines


def _parse_dockerfile(path: Path) -> list[PinEntry]:
    """Return all uv pip install entries from *path*."""
    text = path.read_text()
    entries: list[PinEntry] = []

    for line_no, logical_line in _iter_logical_lines(text):
        m = _INSTALL_LINE_RE.search(logical_line)
        if not m:
            continue
        args = m.group(1)
        no_deps = "--no-deps" in args

        for pkg_m in _PACKAGE_SPEC_RE.finditer(args):
            package = pkg_m.group(1)
            specifier = pkg_m.group(2)
            # Skip flag-looking tokens
            if package.startswith("-"):
                continue
            entries.append(
                PinEntry(
                    package=package,
                    specifier=specifier,
                    line_number=line_no,
                    no_deps=no_deps,
                )
            )

    return entries


def _parse_npm_global_installs(path: Path) -> list[NpmPinEntry]:
    """Return all ``npm install -g`` global-package specs from *path*."""
    text = path.read_text()
    entries: list[NpmPinEntry] = []

    for line_no, logical_line in _iter_logical_lines(text):
        m = _NPM_GLOBAL_INSTALL_RE.search(logical_line)
        if not m:
            continue
        # Only the package list up to the first shell separator belongs to THIS
        # npm install (e.g. "...; npm cache clean --force" is a separate command
        # whose tokens must not be mistaken for install targets).
        args = re.split(r"[;&|]", m.group(1), maxsplit=1)[0]
        for token in args.split():
            # Skip flags (-g already consumed; --foo, etc.) and shell glue.
            if token.startswith("-") or (not token[0].isalnum() and token[0] != "@"):
                continue
            if _NPM_ALIAS_MARKER in token:
                alias_name, target_spec = token.split(_NPM_ALIAS_MARKER, maxsplit=1)
                _target_package, sep, target_version = target_spec.rpartition("@")
                if not alias_name or not sep or not target_version:
                    continue
                entries.append(
                    NpmPinEntry(
                        package=alias_name,
                        version=target_version,
                        line_number=line_no,
                    )
                )
                continue
            spec_m = _NPM_PACKAGE_SPEC_RE.fullmatch(token)
            if not spec_m:
                continue
            entries.append(
                NpmPinEntry(
                    package=spec_m.group("package"),
                    version=spec_m.group("version") or "",
                    line_number=line_no,
                )
            )

    return entries


# ---------------------------------------------------------------------------
# PyPI queries
# ---------------------------------------------------------------------------


def _latest_pypi_version(package: str) -> str | None:
    """Return the latest stable version of *package* from PyPI, or None on error."""
    url = f"https://pypi.org/pypi/{package}/json"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310
            data = json.load(resp)
        return data["info"]["version"]
    except Exception as exc:  # noqa: BLE001 — boundary: prints error and degrades
        print(
            f"  WARNING: could not fetch PyPI data for {package!r}: {exc}",
            file=sys.stderr,
        )
        return None


# ---------------------------------------------------------------------------
# Version range check (lightweight — no packaging dependency)
# ---------------------------------------------------------------------------


def _parse_version(v: str) -> tuple[int, ...]:
    """Parse a dotted version string into a tuple of ints."""
    return tuple(int(x) for x in v.split(".")[:3])


def _satisfies_specifier(version_str: str, specifier: str) -> bool:
    """Return True if *version_str* satisfies *specifier*.

    Supports simple ``>=``, ``<=``, ``>``, ``<``, ``==``, ``!=`` clauses
    joined by commas.  Not a full PEP 440 implementation — sufficient for the
    range pins used in this Dockerfile.
    """
    version = _parse_version(version_str)
    for clause in specifier.split(","):
        clause = clause.strip()
        for op in (">=", "<=", "!=", ">", "<", "=="):
            if clause.startswith(op):
                rhs = _parse_version(clause[len(op) :])
                if (
                    (op == ">=" and version < rhs)
                    or (op == "<=" and version > rhs)
                    or (op == ">" and version <= rhs)
                    or (op == "<" and version >= rhs)
                    or (op == "==" and version != rhs)
                    or (op == "!=" and version == rhs)
                ):
                    return False
                break
    return True


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def _check_no_exact_pin_without_no_deps(entry: PinEntry) -> str | None:
    """Return an error message if an exact pin is used without --no-deps."""
    if "==" in entry.specifier and not entry.no_deps:
        return (
            f"  Line {entry.line_number}: {entry.package}{entry.specifier} uses an exact pin "
            f"without --no-deps. Exact pins without --no-deps risk dep-resolver conflicts on "
            f"coordinated releases. Either switch to a range pin or add --no-deps."
        )
    return None


def _check_no_exact_pin_for_plugins(entry: PinEntry) -> str | None:
    """Return an error if a --no-deps plugin uses an exact pin (OMN-5474).

    Exact pins (==) on --no-deps plugins cause CI failures when new versions
    are published before the Dockerfile is updated. Range pins (>=x,<y) stay
    valid across multiple releases and only need updating when the upper bound
    is reached.
    """
    if "==" in entry.specifier and entry.no_deps:
        return (
            f"  Line {entry.line_number}: {entry.package}"
            f"{entry.specifier} uses an exact pin with "
            f"--no-deps. Use a range pin (e.g., "
            f'">=x.y.z,<2.0.0") to avoid CI failures '
            f"when new versions are published (OMN-5474)."
        )
    return None


def _check_range_covers_latest(entry: PinEntry) -> str | None:
    """Return an error message if the latest PyPI release is outside the declared range."""
    latest = _latest_pypi_version(entry.package)
    if latest is None:
        return None  # Cannot check; warning already printed
    if not _satisfies_specifier(latest, entry.specifier):
        return (
            f"  Line {entry.line_number}: {entry.package}{entry.specifier} does not cover "
            f"the latest PyPI release ({latest}). Update the range pin in Dockerfile.runtime."
        )
    return None


def _check_npm_exact_pin(entry: NpmPinEntry) -> str | None:
    """Return an error if an ``npm install -g`` package lacks an exact version pin.

    npm CLI globals (codex / claude — OMN-13248) MUST carry an exact ``@version``
    so the runtime image is reproducible. A bare package name (or a dist-tag like
    ``@latest``) would silently change the shipped agent CLI on every rebuild —
    the OMN-13246 Phase 0 auth proof pinned specific CLI versions, so the image
    must ship exactly those.
    """
    if not entry.version:
        return (
            f"  Line {entry.line_number}: npm global package {entry.package!r} is not "
            f"pinned to an exact version. Use {entry.package}@<exact-version> (e.g. "
            f"{entry.package}@1.2.3) so the runtime image is reproducible (OMN-13248)."
        )
    if not entry.version[0].isdigit():
        return (
            f"  Line {entry.line_number}: npm global package "
            f"{entry.package}@{entry.version} uses a non-exact pin (dist-tag or range). "
            f"Use an exact semver version so the runtime image is reproducible (OMN-13248)."
        )
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dockerfile",
        default="docker/Dockerfile.runtime",
        help="Path to Dockerfile.runtime (default: docker/Dockerfile.runtime)",
    )
    parser.add_argument(
        "--no-pypi",
        action="store_true",
        help="Skip PyPI version checks (offline / fast mode)",
    )
    args = parser.parse_args(argv)

    dockerfile_path = Path(args.dockerfile)
    if not dockerfile_path.exists():
        print(f"ERROR: Dockerfile not found: {dockerfile_path}", file=sys.stderr)
        return 1

    entries = _parse_dockerfile(dockerfile_path)
    npm_entries = _parse_npm_global_installs(dockerfile_path)
    if not entries and not npm_entries:
        print(
            "No uv pip install or npm install -g entries found in Dockerfile — nothing to check."
        )
        return 0

    errors: list[str] = []

    for entry in entries:
        err = _check_no_exact_pin_without_no_deps(entry)
        if err:
            errors.append(err)

        # OMN-5474: exact pins on --no-deps plugins cause forward-looking failures
        err = _check_no_exact_pin_for_plugins(entry)
        if err:
            errors.append(err)

        if not args.no_pypi and entry.no_deps:
            # Only range-check the --no-deps plugin pins (the ones this script guards)
            err = _check_range_covers_latest(entry)
            if err:
                errors.append(err)

    # OMN-13248: npm global CLI installs (codex / claude) must be exactly pinned.
    for npm_entry in npm_entries:
        npm_err = _check_npm_exact_pin(npm_entry)
        if npm_err:
            errors.append(npm_err)

    if errors:
        print("Dockerfile pin check FAILED:\n", file=sys.stderr)
        for e in errors:
            print(e, file=sys.stderr)
        print(
            "\nSee docs/conventions/dockerfile-plugin-pins.md for the --no-deps pattern.",
            file=sys.stderr,
        )
        return 1

    print(
        f"Dockerfile pin check passed ({len(entries)} pip + {len(npm_entries)} npm entries validated)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
