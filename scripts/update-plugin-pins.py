#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Update omninode plugin pin lines in Dockerfile.runtime to latest PyPI versions.

Usage:
    update-plugin-pins.py [--dry-run] [--dockerfile PATH]

Behaviour:
    1. Fetch latest versions from PyPI for omninode-claude and omninode-memory.
    2. Rewrite Dockerfile.runtime lines matching:
         RUN pip install omninode-claude==X.Y.Z
         RUN pip install omninode-memory==X.Y.Z
       or range-pin patterns such as:
         uv pip install ... "omninode-claude>=A,<B"
    3. --dry-run: print diff without writing.
    4. Exit 0 on success, non-zero on fetch failure.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PLUGINS = ["omninode-claude", "omninode-memory"]

# Default Dockerfile path relative to the repo root (two directories above
# this script, which lives in <repo>/scripts/).
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
DEFAULT_DOCKERFILE = _REPO_ROOT / "docker" / "Dockerfile.runtime"


# ---------------------------------------------------------------------------
# PyPI helpers
# ---------------------------------------------------------------------------


def fetch_latest_version(package: str) -> str:
    """Return the latest stable version of *package* from PyPI.

    Raises:
        RuntimeError: if the HTTP request fails or the JSON is malformed.
    """
    url = f"https://pypi.org/pypi/{package}/json"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:  # noqa: S310
            data = json.loads(resp.read())
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
        raise RuntimeError(
            f"Failed to fetch PyPI metadata for {package!r}: {exc}"
        ) from exc

    try:
        version: str = data["info"]["version"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            f"Unexpected PyPI JSON structure for {package!r}: {exc}"
        ) from exc

    return version


# ---------------------------------------------------------------------------
# Rewrite helpers
# ---------------------------------------------------------------------------


# Matches any of the following patterns inside a Dockerfile RUN line:
#   "omninode-claude==1.2.3"
#   "omninode-claude>=1.0,<2.0"
#   omninode-claude==1.2.3   (without quotes)
#   omninode-claude>=1.0,<2.0  (without quotes)
_PIN_RE = re.compile(
    r'(?P<q>["\']?)(?P<pkg>omninode-(?:claude|memory))(?P<spec>[^\s"\'\\]+)(?P=q)'
)


def _replace_pin(match: re.Match[str], versions: dict[str, str]) -> str:
    """Replace a single regex match with an exact pin."""
    pkg = match.group("pkg")
    quote = match.group("q")
    latest = versions.get(pkg)
    if latest is None:
        return match.group(0)  # unknown package — leave untouched
    return f"{quote}{pkg}=={latest}{quote}"


def rewrite_content(content: str, versions: dict[str, str]) -> str:
    """Return *content* with all plugin pin specifiers updated to *versions*.

    Only lines that contain at least one PLUGINS package reference are
    modified.  Unrelated lines are returned verbatim.
    """
    lines_in = content.splitlines(keepends=True)
    lines_out: list[str] = []
    for line in lines_in:
        # Only rewrite non-comment lines that reference a plugin package.
        stripped = line.lstrip()
        if not stripped.startswith("#") and any(pkg in line for pkg in PLUGINS):
            line = _PIN_RE.sub(lambda m: _replace_pin(m, versions), line)
        lines_out.append(line)
    return "".join(lines_out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update omninode plugin pins in Dockerfile.runtime to latest PyPI versions.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print diff without writing the file.",
    )
    parser.add_argument(
        "--dockerfile",
        type=Path,
        default=DEFAULT_DOCKERFILE,
        metavar="PATH",
        help=f"Path to the Dockerfile to update (default: {DEFAULT_DOCKERFILE}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dockerfile: Path = args.dockerfile

    if not dockerfile.exists():
        print(f"ERROR: Dockerfile not found: {dockerfile}", file=sys.stderr)
        return 2

    # Fetch latest versions from PyPI.
    versions: dict[str, str] = {}
    for pkg in PLUGINS:
        try:
            versions[pkg] = fetch_latest_version(pkg)
            print(f"PyPI latest  {pkg}: {versions[pkg]}")
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    # Read and rewrite.
    original = dockerfile.read_text(encoding="utf-8")
    updated = rewrite_content(original, versions)

    if original == updated:
        print("No changes — Dockerfile pins already up to date.")
        return 0

    # Show diff.
    diff_lines = list(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            updated.splitlines(keepends=True),
            fromfile=f"a/{dockerfile.name}",
            tofile=f"b/{dockerfile.name}",
        )
    )
    print("".join(diff_lines), end="")

    if args.dry_run:
        print("Dry-run mode — no file written.")
        return 0

    dockerfile.write_text(updated, encoding="utf-8")
    print(f"Updated {dockerfile}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
