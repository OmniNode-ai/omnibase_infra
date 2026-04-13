# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Entry-point generation and drift-validation script.

Usage:
    # Generate mode (default) — scan node dirs, write to pyproject.toml
    uv run python scripts/generate_entry_points.py omnimarket

    # Check mode — exit 0 if clean, exit 1 with diff if drifted
    uv run python scripts/generate_entry_points.py omnimarket --check

    # Exclude nodes already migrated to another repo (e.g. omnimemory → omnimarket)
    uv run python scripts/generate_entry_points.py omnimemory --exclude-migrated omnimarket
"""

from __future__ import annotations

import argparse
import difflib
import os
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Core helpers (imported by tests)
# ---------------------------------------------------------------------------


def collect_node_dirs(
    repo_root: Path,
    package_name: str,
    exclude: set[str] | None = None,
) -> list[Path]:
    """Return sorted list of node_* directories under <repo_root>/src/<package>/nodes/.

    Args:
        repo_root: Root directory of the repository.
        package_name: Python package name (e.g. "omnimarket").
        exclude: Optional set of node directory names to skip.

    Returns:
        Sorted list of Path objects for each node_* directory.
    """
    nodes_dir = repo_root / "src" / package_name / "nodes"
    if not nodes_dir.is_dir():
        return []

    exclude_set = exclude or set()
    result = [
        p
        for p in nodes_dir.iterdir()
        if p.is_dir() and p.name.startswith("node_") and p.name not in exclude_set
    ]
    return sorted(result, key=lambda p: p.name)


def build_entry_point_section(node_dirs: list[Path], package_name: str) -> str:
    """Build the [project.entry-points."onex.nodes"] TOML section string.

    Args:
        node_dirs: List of node directory Paths (as returned by collect_node_dirs).
        package_name: Python package name (e.g. "omnimarket").

    Returns:
        TOML string for the entry-points section, ready to splice into pyproject.toml.
    """
    lines = ['[project.entry-points."onex.nodes"]']
    for node_dir in node_dirs:
        name = node_dir.name
        lines.append(f'{name} = "{package_name}.nodes.{name}"')
    return "\n".join(lines) + "\n"


def _read_pyproject(repo_root: Path) -> str:
    """Read pyproject.toml content."""
    pyproject = repo_root / "pyproject.toml"
    if not pyproject.exists():
        raise FileNotFoundError(f"pyproject.toml not found at {pyproject}")
    return pyproject.read_text(encoding="utf-8")


def _replace_entry_point_section(current: str, new_section: str) -> str:
    """Replace or append the onex.nodes entry-points section in pyproject.toml content.

    Replaces the entire [project.entry-points."onex.nodes"] block if it exists,
    otherwise appends it before the next [section] or at end of file.
    """
    # Pattern: match the header line through the next blank line followed by [
    # or end of file. We use a non-greedy match per line.
    pattern = re.compile(
        r'(\[project\.entry-points\."onex\.nodes"\]\n(?:[^\[]*\n)*)',
        re.MULTILINE,
    )
    if pattern.search(current):
        return pattern.sub(new_section + "\n", current)

    # Section not present — append before next top-level section after [project]
    # or at the end.
    return current.rstrip() + "\n\n" + new_section + "\n"


def check_mode(repo_root: Path, package_name: str, node_dirs: list[Path]) -> int:
    """Compare generated entry-point section against current pyproject.toml.

    Returns:
        0 if the pyproject.toml is in sync with the filesystem.
        1 if it has drifted, printing a diff to stdout.
    """
    generated = build_entry_point_section(node_dirs, package_name)

    try:
        current = _read_pyproject(repo_root)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    # Extract the existing entry-points section from pyproject.toml for comparison.
    pattern = re.compile(
        r'(\[project\.entry-points\."onex\.nodes"\]\n(?:[^\[]*\n)*)',
        re.MULTILINE,
    )
    match = pattern.search(current)
    existing_section = match.group(1) if match else ""

    if existing_section.rstrip("\n") == generated.rstrip("\n"):
        return 0

    diff = list(
        difflib.unified_diff(
            existing_section.splitlines(keepends=True),
            generated.splitlines(keepends=True),
            fromfile="pyproject.toml (current)",
            tofile="pyproject.toml (generated)",
        )
    )
    print("".join(diff))
    return 1


def generate_mode(repo_root: Path, package_name: str, node_dirs: list[Path]) -> None:
    """Write the generated entry-point section into pyproject.toml (idempotent)."""
    new_section = build_entry_point_section(node_dirs, package_name)
    current = _read_pyproject(repo_root)
    updated = _replace_entry_point_section(current, new_section)

    pyproject = repo_root / "pyproject.toml"
    pyproject.write_text(updated, encoding="utf-8")
    print(f"Written {len(node_dirs)} entry points to {pyproject}")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def _resolve_repo(repo_arg: str) -> tuple[Path, str]:
    """Resolve repo name/path to (repo_root, package_name).

    Accepts either:
    - A bare repo name like "omnimarket" (resolved relative to OMNI_HOME or cwd parent)
    - An absolute path
    """
    p = Path(repo_arg)
    if p.is_absolute() and p.is_dir():
        src = p / "src"
        if src.is_dir():
            subdirs = [
                d for d in src.iterdir() if d.is_dir() and not d.name.startswith("_")
            ]
            if subdirs:
                return p, subdirs[0].name
        raise ValueError(
            f"Cannot determine package name for absolute path '{p}': "
            f"no src/ subdirectory with a package found."
        )

    # Try OMNI_HOME env var first, then cwd's parent (typical omni_home layout)
    omni_home_candidates = []
    omni_home_env = os.environ.get("OMNI_HOME")
    if omni_home_env:
        omni_home_candidates.append(Path(omni_home_env))

    # The script lives in omnibase_infra/scripts/; omni_home is two levels up
    script_dir = Path(__file__).parent
    omni_home_candidates.append(script_dir.parent.parent)

    for omni_home in omni_home_candidates:
        candidate = omni_home / repo_arg
        if candidate.is_dir():
            # Package name is the top-level dir under src/
            src = candidate / "src"
            if src.is_dir():
                subdirs = [
                    d
                    for d in src.iterdir()
                    if d.is_dir() and not d.name.startswith("_")
                ]
                if subdirs:
                    return candidate, subdirs[0].name
            return candidate, repo_arg.replace("-", "_")

    raise ValueError(
        f"Cannot resolve repo '{repo_arg}'. Set OMNI_HOME or pass an absolute path."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate or validate onex.nodes entry points in pyproject.toml"
    )
    parser.add_argument("repo", help="Repo name (e.g. omnimarket) or absolute path")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validation mode: exit 0 if clean, exit 1 with diff if drifted",
    )
    parser.add_argument(
        "--exclude-migrated",
        metavar="OTHER_REPO",
        help="Skip nodes already registered in another repo (e.g. for omnimemory migration)",
    )
    args = parser.parse_args()

    repo_root, package_name = _resolve_repo(args.repo)

    exclude: set[str] | None = None
    if args.exclude_migrated:
        other_root, other_pkg = _resolve_repo(args.exclude_migrated)
        other_dirs = collect_node_dirs(other_root, other_pkg)
        exclude = {d.name for d in other_dirs}
        print(f"Excluding {len(exclude)} nodes already in {args.exclude_migrated}")

    node_dirs = collect_node_dirs(repo_root, package_name, exclude=exclude)
    print(
        f"Found {len(node_dirs)} node directories in {repo_root / 'src' / package_name / 'nodes'}"
    )

    if args.check:
        return check_mode(repo_root, package_name, node_dirs)

    generate_mode(repo_root, package_name, node_dirs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
