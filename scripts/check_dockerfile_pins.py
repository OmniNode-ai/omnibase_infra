#!/usr/bin/env python3
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


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_INSTALL_LINE_RE = re.compile(
    r"^\s*uv pip install\b(.+)$",
    re.MULTILINE,
)
# Matches a package specifier such as "omninode-claude>=0.3.0,<0.5.0" (with or
# without surrounding quotes).
_PACKAGE_SPEC_RE = re.compile(
    r'"?([A-Za-z0-9_\-]+)([>=<!,\d\.\*]+)"?',
)


def _parse_dockerfile(path: Path) -> list[PinEntry]:
    """Return all uv pip install entries from *path*."""
    text = path.read_text()
    entries: list[PinEntry] = []

    for m in _INSTALL_LINE_RE.finditer(text):
        args = m.group(1)
        no_deps = "--no-deps" in args
        line_no = text[: m.start()].count("\n") + 1

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
    except Exception as exc:
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
    if not entries:
        print("No uv pip install entries found in Dockerfile — nothing to check.")
        return 0

    errors: list[str] = []

    for entry in entries:
        err = _check_no_exact_pin_without_no_deps(entry)
        if err:
            errors.append(err)

        if not args.no_pypi and entry.no_deps:
            # Only range-check the --no-deps plugin pins (the ones this script guards)
            err = _check_range_covers_latest(entry)
            if err:
                errors.append(err)

    if errors:
        print("Dockerfile pin check FAILED:\n", file=sys.stderr)
        for e in errors:
            print(e, file=sys.stderr)
        print(
            "\nSee docs/conventions/dockerfile-plugin-pins.md for the --no-deps pattern.",
            file=sys.stderr,
        )
        return 1

    print(f"Dockerfile pin check passed ({len(entries)} entries validated).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
