#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""
check_version_pins.py

Checks each repo's pyproject.toml against the shared version matrix
(standards/version-matrix.yaml) and reports compliance.

Usage:
    python scripts/check_version_pins.py [--repo <name>] [--matrix <path>] [--root <path>]

Options:
    --repo    Check a single repo (by directory name). If omitted, checks all.
    --matrix  Path to version-matrix.yaml (default: version-matrix.yaml next to this script)
    --root    Root directory containing repo clones (default: parent of this script's dir)

Exit codes:
    0  All repos are compliant
    1  One or more repos have out-of-compliance pins
    2  Configuration error (missing matrix, missing repo, etc.)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    # Fallback: parse YAML manually for the simple structure we use
    yaml = None  # type: ignore[assignment]


def parse_yaml_simple(text: str) -> dict:
    """Minimal YAML parser for our simple version-matrix.yaml structure."""
    result: dict = {}
    current_section: str | None = None
    current_list: list | None = None

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # Top-level key with nested content
        if not line.startswith(" ") and stripped.endswith(":"):
            current_section = stripped[:-1]
            result[current_section] = {}
            current_list = None
            continue

        if current_section and stripped.startswith("- "):
            if current_list is None:
                result[current_section] = []
                current_list = result[current_section]
            # Handle "- name: value" style
            item_str = stripped[2:]
            if ": " in item_str:
                # Start of a dict item in a list
                key, val = item_str.split(": ", 1)
                item = {key: val.strip().strip('"').strip("'")}
                current_list.append(item)
            continue

        if current_list and stripped.startswith("label:"):
            # Continuation of the last list item
            key, val = stripped.split(": ", 1)
            current_list[-1][key] = val.strip().strip('"').strip("'")
            continue

        # Key-value in a section
        if current_section and ": " in stripped:
            key, val = stripped.split(": ", 1)
            val = val.strip().strip('"').strip("'")
            if isinstance(result[current_section], dict):
                result[current_section][key] = val
            current_list = None

    return result


def load_matrix(matrix_path: Path) -> dict:
    """Load the version matrix YAML file."""
    text = matrix_path.read_text()
    if yaml is not None:
        return yaml.safe_load(text)
    return parse_yaml_simple(text)


def parse_version(version_str: str) -> tuple[int, ...]:
    """Parse a version string like '0.26.0' into a tuple of ints."""
    match = re.match(r"(\d+(?:\.\d+)*)", version_str)
    if not match:
        return (0,)
    return tuple(int(x) for x in match.group(1).split("."))


def parse_requirement(req_str: str) -> tuple[str, str, tuple[int, ...]]:
    """Parse a requirement like '>=0.26.0' into (operator, version_str, version_tuple)."""
    match = re.match(r"([><=!]+)\s*(\S+)", req_str)
    if not match:
        return (">=", req_str, parse_version(req_str))
    return (match.group(1), match.group(2), parse_version(match.group(2)))


def extract_dependencies(pyproject_path: Path) -> dict[str, str]:
    """Extract dependency versions from a pyproject.toml file.

    Returns a dict mapping package name to its version specifier.
    Only extracts our shared packages (omnibase-core, omnibase-spi, etc.).
    """
    deps: dict[str, str] = {}
    text = pyproject_path.read_text()

    # Match lines like: "omnibase-core>=0.26.0", or "omnibase-core==0.26.0",
    # handling both with and without quotes
    pattern = re.compile(
        r'"?\s*(omnibase-(?:core|spi|infra|compat))\s*([><=!]+\s*[\d.]+)\s*"?'
    )

    for match in pattern.finditer(text):
        pkg_name = match.group(1)
        version_spec = match.group(2)
        deps[pkg_name] = version_spec

    return deps


def check_compliance(
    actual_spec: str,
    required_spec: str,
) -> tuple[bool, str]:
    """Check if an actual version spec meets the required minimum.

    Returns (is_compliant, reason).
    """
    _req_op, _req_ver_str, req_ver = parse_requirement(required_spec)
    act_op, _act_ver_str, act_ver = parse_requirement(actual_spec)

    # For >= requirements, the actual pinned version must be >= the required minimum
    if act_op in ("==", ">="):
        if act_ver >= req_ver:
            return True, f"{actual_spec} meets {required_spec}"
        return False, f"{actual_spec} is below required {required_spec}"

    # For other operators, flag as needing review
    return False, f"Unexpected operator in {actual_spec}, expected >= or =="


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check repo dependency pins against the shared version matrix."
    )
    parser.add_argument(
        "--repo",
        help="Check a single repo by directory name",
    )
    parser.add_argument(
        "--matrix",
        default=None,
        help="Path to version-matrix.yaml",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Root directory containing repo clones",
    )
    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).resolve().parent
    root = Path(args.root) if args.root else script_dir.parent

    if args.matrix:
        matrix_path = Path(args.matrix)
    else:
        matrix_path = script_dir / "version-matrix.yaml"

    if not matrix_path.exists():
        print(f"ERROR: Version matrix not found at {matrix_path}", file=sys.stderr)
        return 2

    matrix = load_matrix(matrix_path)
    packages = matrix.get("packages", {})
    repos = matrix.get("repos", [])

    if not packages:
        print("ERROR: No packages defined in version matrix", file=sys.stderr)
        return 2

    # Filter to single repo if specified
    if args.repo:
        repos = [r for r in repos if r["name"] == args.repo]
        if not repos:
            print(f"ERROR: Repo '{args.repo}' not found in matrix", file=sys.stderr)
            return 2

    # Results tracking
    all_compliant = True
    results: list[tuple[str, str, str, bool, str]] = []

    print("=" * 72)
    print("Version Pin Compliance Check")
    print(f"Matrix: {matrix_path}")
    print(f"Root:   {root}")
    print("=" * 72)
    print()

    for repo_entry in repos:
        repo_name = repo_entry["name"]
        repo_label = repo_entry.get("label", repo_name)
        pyproject = root / repo_name / "pyproject.toml"

        if not pyproject.exists():
            print(f"  SKIP  {repo_label:<25} pyproject.toml not found")
            continue

        deps = extract_dependencies(pyproject)

        if not deps:
            print(f"  OK    {repo_label:<25} no shared package dependencies")
            continue

        for pkg_name, required_spec in packages.items():
            if pkg_name not in deps:
                continue

            actual_spec = deps[pkg_name]
            compliant, reason = check_compliance(actual_spec, required_spec)
            status = "OK" if compliant else "FAIL"

            if not compliant:
                all_compliant = False

            results.append((repo_label, pkg_name, actual_spec, compliant, reason))
            print(
                f"  {status:<4}  {repo_label:<25} {pkg_name:<20} "
                f"actual={actual_spec:<12} required={required_spec}"
            )

    print()
    print("-" * 72)

    if all_compliant:
        print("RESULT: All repos are compliant with the version matrix.")
        return 0

    print("RESULT: FAILED - one or more repos have out-of-compliance pins.")
    print()
    print("Non-compliant repos:")
    for repo_label, pkg_name, actual_spec, compliant, reason in results:
        if not compliant:
            print(f"  - {repo_label}: {pkg_name} {actual_spec} ({reason})")

    return 1


if __name__ == "__main__":
    sys.exit(main())
