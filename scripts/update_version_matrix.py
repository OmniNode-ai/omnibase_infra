#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Update the fallback version matrix in version_compatibility.py.

This script is called during the release BUMP phase to keep the
``_FALLBACK_MATRIX`` in ``version_compatibility.py`` in sync with
``pyproject.toml``.  The module already derives ``VERSION_MATRIX``
dynamically at import time when a source tree is present, so the
fallback is only used for installed (non-editable) packages.  This
script ensures that the fallback is never stale after a release.

Usage::

    uv run python scripts/update_version_matrix.py [--check]

Options:
    --check     Exit with code 1 if the fallback is out of date (CI mode).
                Default: rewrite the file in-place.

Exit codes:
    0   Success (or --check with no drift detected).
    1   --check mode: fallback is out of date.
    2   pyproject.toml not found or unparseable.
    3   Could not extract bounds for one or more tracked packages.

Related:
    - OMN-3203: Automate version_compatibility.py matrix updates
    - omnibase_infra/src/omnibase_infra/runtime/version_compatibility.py
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
TARGET = REPO_ROOT / "src" / "omnibase_infra" / "runtime" / "version_compatibility.py"

# Packages whose bounds appear in _FALLBACK_MATRIX (underscore-normalised names)
TRACKED_PACKAGES: tuple[str, ...] = ("omnibase_core", "omnibase_spi")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_dep_bounds(spec: str) -> tuple[str, str] | None:
    """Return (min_version, max_version) from a PEP 508 specifier, or None."""
    min_match = re.search(r">=\s*([0-9][^\s,;\"']*)", spec)
    max_match = re.search(r"<\s*([0-9][^\s,;\"']*)", spec)
    if min_match and max_match:
        return min_match.group(1), max_match.group(1)
    return None


def _load_bounds_from_pyproject() -> dict[str, tuple[str, str]]:
    """Return {pkg: (min, max)} for all TRACKED_PACKAGES found in pyproject.toml."""
    if not PYPROJECT.is_file():
        print(f"ERROR: {PYPROJECT} not found", file=sys.stderr)
        sys.exit(2)

    with open(PYPROJECT, "rb") as fh:
        data = tomllib.load(fh)

    deps: list[str] = data.get("project", {}).get("dependencies", [])
    spec_map: dict[str, str] = {}
    for dep in deps:
        dep_clean = dep.strip().strip("\"'")
        name_part = re.split(r"[><=!~]", dep_clean, maxsplit=1)[0].strip()
        normalised = name_part.replace("-", "_")
        spec_map[normalised] = dep_clean

    result: dict[str, tuple[str, str]] = {}
    missing: list[str] = []
    for pkg in TRACKED_PACKAGES:
        spec = spec_map.get(pkg)
        if spec is None:
            missing.append(pkg)
            continue
        bounds = _parse_dep_bounds(spec)
        if bounds is None:
            missing.append(pkg)
            continue
        result[pkg] = bounds

    if missing:
        print(
            f"ERROR: could not extract >=/<  bounds for: {missing}\n"
            "       Ensure pyproject.toml uses 'pkg>=X.Y.Z,<A.B.C' format.",
            file=sys.stderr,
        )
        sys.exit(3)

    return result


def _build_fallback_block(bounds: dict[str, tuple[str, str]]) -> str:
    """Render the _FALLBACK_MATRIX list literal."""
    lines = ["_FALLBACK_MATRIX: list[VersionConstraint] = ["]
    for pkg, (min_v, max_v) in bounds.items():
        lines.append("    VersionConstraint(")
        lines.append(f'        package="{pkg}",')
        lines.append(f'        min_version="{min_v}",')
        lines.append(f'        max_version="{max_v}",')
        lines.append("    ),")
    lines.append("]")
    return "\n".join(lines)


# Regex that matches the entire _FALLBACK_MATRIX block (list literal only,
# not the surrounding comment block or the VERSION_MATRIX assignment).
_FALLBACK_BLOCK_RE = re.compile(
    r"^_FALLBACK_MATRIX: list\[VersionConstraint\] = \[$.*?^\]$",
    re.MULTILINE | re.DOTALL,
)


def _update_file(bounds: dict[str, tuple[str, str]], *, check: bool) -> bool:
    """Rewrite TARGET with updated fallback block.

    Returns True if the file was changed (or would be changed in --check mode).
    """
    if not TARGET.is_file():
        print(f"ERROR: target file not found: {TARGET}", file=sys.stderr)
        sys.exit(2)

    original = TARGET.read_text(encoding="utf-8")
    new_block = _build_fallback_block(bounds)

    updated = _FALLBACK_BLOCK_RE.sub(new_block, original)

    if updated == original:
        print("Fallback matrix is already up to date.")
        return False

    if check:
        print(
            "DRIFT DETECTED: _FALLBACK_MATRIX is out of date with pyproject.toml.\n"
            "Run  uv run python scripts/update_version_matrix.py  to fix.",
            file=sys.stderr,
        )
        return True

    TARGET.write_text(updated, encoding="utf-8")
    print(f"Updated {TARGET.relative_to(REPO_ROOT)}")
    for pkg, (min_v, max_v) in bounds.items():
        print(f"  {pkg}: >={min_v},<{max_v}")
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync _FALLBACK_MATRIX in version_compatibility.py with pyproject.toml"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check for drift only; exit 1 if out of date (CI mode)",
    )
    args = parser.parse_args()

    bounds = _load_bounds_from_pyproject()
    changed = _update_file(bounds, check=args.check)

    if args.check and changed:
        sys.exit(1)


if __name__ == "__main__":
    main()
