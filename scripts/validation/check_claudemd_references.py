#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
#
# CLAUDE.md Reference Validator
#
# Scans CLAUDE.md files across all repos under omni_home and verifies that
# file/directory paths referenced in the docs actually exist on disk.
#
# Extracts paths from:
#   - Inline code: `path/to/file.py`
#   - Code blocks: lines that look like file paths
#   - Markdown links: [text](path/to/file)
#   - Table cells: | `path/to/file` |
#
# Ignores:
#   - URLs (http://, https://)
#   - Package names (e.g., omnibase-infra)
#   - Python import paths (e.g., omnibase_infra.models.foo)
#   - Environment variables ($VAR, ${VAR})
#   - Placeholder paths with <angle brackets>
#
# Usage:
#   python scripts/validation/check_claudemd_references.py /path/to/omni_home
#
# Exit codes:
#   0 = all references valid
#   1 = broken references found

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

_OMNI_HOME_DEFAULT = Path(os.environ.get("OMNI_HOME", "."))

# Patterns to extract file paths from CLAUDE.md content
BACKTICK_PATH = re.compile(r"`([^`]+)`")
LINK_PATH = re.compile(r"\[[^\]]*\]\(([^)]+)\)")

# Heuristics for "looks like a file path"
PATH_INDICATORS = re.compile(
    r"^[\w.~/$][\w.~/$-]*/"  # starts with word chars and contains /
    r"|\.(?:py|ts|tsx|js|jsx|sh|yaml|yml|json|md|sql|toml|txt|css|html)$"  # file extension
)

# Things that are NOT file paths
SKIP_PATTERNS = [
    re.compile(r"^https?://"),  # URLs
    re.compile(r"^git@"),  # Git SSH URLs
    re.compile(r"^\$"),  # Env vars
    re.compile(r"<[^>]+>"),  # Angle bracket placeholders
    re.compile(r"XXXX|xxxx"),  # Template placeholders
    re.compile(r"^\d+\.\d+\.\d+"),  # Version numbers
    re.compile(
        r"^[a-z_]+\.[a-z_]+\.[a-z_]+"
    ),  # Python imports (3+ segments, all lowercase)
    re.compile(r"^pip install"),  # pip commands
    re.compile(r"^uv "),  # uv commands
    re.compile(r"^npm "),  # npm commands
    re.compile(r"^git "),  # git commands
    re.compile(r"^docker "),  # docker commands
    re.compile(r"^curl "),  # curl commands
    re.compile(r"^bash\s"),  # bash commands
    re.compile(r"^[A-Z_]+="),  # ENV=value assignments
    re.compile(r"^#"),  # Comments
    re.compile(r"^\|"),  # Table separators
    re.compile(r"^-{2,}"),  # Horizontal rules
    re.compile(r"^~/.claude/"),  # Shared Claude config (not in repo)
    re.compile(r"^contract\.yaml$"),  # Generic contract reference
    re.compile(r"^plugin\.py$"),  # Generic plugin reference
    re.compile(r"^node\.py$"),  # Generic node reference
    re.compile(r"^v\d"),  # Version directory examples (e.g., v2/, v1_0_0/)
    re.compile(r"^test_\w+\.py$"),  # Generic test file references
    re.compile(r"^omni_home/"),  # References to omni_home parent
    re.compile(r"\s"),  # Paths with spaces are likely command fragments
]


def looks_like_path(s: str) -> bool:
    """Heuristic: does this string look like a file/directory path?"""
    s = s.strip()
    if len(s) < 3 or len(s) > 200:
        return False
    for pat in SKIP_PATTERNS:
        if pat.search(s):
            return False
    if PATH_INDICATORS.search(s):
        return True
    return False


def extract_paths(content: str) -> list[str]:
    """Extract candidate file paths from CLAUDE.md content."""
    paths: list[str] = []

    # From backtick code spans
    for match in BACKTICK_PATH.finditer(content):
        candidate = match.group(1).strip()
        if looks_like_path(candidate):
            paths.append(candidate)

    # From markdown links
    for match in LINK_PATH.finditer(content):
        candidate = match.group(1).strip()
        if looks_like_path(candidate) and not candidate.startswith("http"):
            paths.append(candidate)

    return list(set(paths))


def resolve_path(path_str: str, repo_root: Path, omni_home: Path) -> Path | None:
    """Try to resolve a path relative to repo root or omni_home."""
    # Expand ~ to home
    if path_str.startswith("~/"):
        return Path.home() / path_str[2:]

    # Absolute paths
    if path_str.startswith("/"):
        return Path(path_str)

    # Relative to repo root
    candidate = repo_root / path_str
    if candidate.exists():
        return candidate

    # Relative to omni_home
    candidate = omni_home / path_str
    if candidate.exists():
        return candidate

    # Try stripping leading repo name (e.g., "omnibase_core/docs/...")
    parts = Path(path_str).parts
    if len(parts) > 1:
        candidate = omni_home / path_str
        if candidate.exists():
            return candidate

    return None


def check_claudemd(
    claudemd_path: Path, repo_root: Path, omni_home: Path
) -> list[tuple[str, str]]:
    """Check a single CLAUDE.md file. Returns [(path, reason)]."""
    broken: list[tuple[str, str]] = []

    try:
        content = claudemd_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return [("(file)", "Could not read file")]

    paths = extract_paths(content)

    for path_str in paths:
        resolved = resolve_path(path_str, repo_root, omni_home)
        if resolved is None:
            # Could not find it anywhere
            broken.append((path_str, "path not found"))
        elif not resolved.exists():
            broken.append((path_str, f"resolved to {resolved} but does not exist"))

    return broken


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate file path references in CLAUDE.md files"
    )
    parser.add_argument(
        "omni_home",
        type=Path,
        nargs="?",
        default=_OMNI_HOME_DEFAULT,
        help="Path to omni_home directory",
    )
    args = parser.parse_args()

    omni_home = args.omni_home.resolve()
    if not omni_home.is_dir():
        print(f"ERROR: {omni_home} is not a directory", file=sys.stderr)
        return 1

    print(f"Scanning CLAUDE.md files under {omni_home}...")
    total_broken = 0
    files_checked = 0

    # Check top-level CLAUDE.md
    top_level = omni_home / "CLAUDE.md"
    if top_level.exists():
        broken = check_claudemd(top_level, omni_home, omni_home)
        files_checked += 1
        if broken:
            print(f"\n  {top_level.relative_to(omni_home)}:")
            for path_str, reason in broken:
                print(f"    BROKEN: {path_str} — {reason}")
                total_broken += 1

    # Check each repo's CLAUDE.md
    for repo_dir in sorted(omni_home.iterdir()):
        if not repo_dir.is_dir() or repo_dir.name.startswith("."):
            continue

        claudemd = repo_dir / "CLAUDE.md"
        if not claudemd.exists():
            continue

        broken = check_claudemd(claudemd, repo_dir, omni_home)
        files_checked += 1
        if broken:
            print(f"\n  {claudemd.relative_to(omni_home)}:")
            for path_str, reason in broken:
                print(f"    BROKEN: {path_str} — {reason}")
                total_broken += 1

    print(f"\nChecked {files_checked} CLAUDE.md files.")

    if total_broken > 0:
        print(f"FAIL: {total_broken} broken reference(s) found.")
        return 1

    print("OK: All file references in CLAUDE.md files are valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
