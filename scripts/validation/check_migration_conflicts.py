#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
#
# Cross-Repo Migration Conflict Checker
#
# Scans all migration directories across omni_home repos and flags:
#   1. Duplicate migration prefixes (same NNNN prefix in different repos)
#   2. Duplicate table names in CREATE TABLE statements across repos
#   3. Same-prefix conflicts within a single repo (e.g., 0003 and 0003b)
#
# Migration directories scanned:
#   - <repo>/migrations/
#   - <repo>/sql/migrations/
#   - <repo>/db/migrations/
#   - <repo>/docker/migrations/
#
# Usage:
#   python scripts/validation/check_migration_conflicts.py /path/to/omni_home
#
# Exit codes:
#   0 = no conflicts found
#   1 = conflicts detected

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

MIGRATION_DIRS = [
    "migrations",
    "sql/migrations",
    "db/migrations",
    "docker/migrations",
    "server/migrations",
    "k8s/migrations",
]

# Extract numeric prefix from migration filenames like 0003_description.sql
PREFIX_PATTERN = re.compile(r"^(\d{4})[a-z]?_")

# Extract table names from CREATE TABLE statements
CREATE_TABLE_PATTERN = re.compile(
    r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?\"?(\w+)\"?",
    re.IGNORECASE,
)

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "dist"}


def find_migration_dirs(omni_home: Path) -> dict[str, list[Path]]:
    """Find all migration directories. Returns {repo_name: [migration_dirs]}."""
    result: dict[str, list[Path]] = {}

    for repo_dir in sorted(omni_home.iterdir()):
        if not repo_dir.is_dir() or repo_dir.name.startswith("."):
            continue
        if repo_dir.name in SKIP_DIRS:
            continue

        dirs: list[Path] = []
        for rel_path in MIGRATION_DIRS:
            mig_dir = repo_dir / rel_path
            if mig_dir.is_dir():
                dirs.append(mig_dir)
        if dirs:
            result[repo_dir.name] = dirs

    return result


def extract_tables_from_sql(path: Path) -> list[str]:
    """Extract CREATE TABLE names from a SQL file."""
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    return [m.group(1).lower() for m in CREATE_TABLE_PATTERN.finditer(content)]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check for migration conflicts across repos"
    )
    parser.add_argument(
        "omni_home",
        type=Path,
        nargs="?",
        default=Path("/Users/jonah/Code/omni_home"),
        help="Path to omni_home directory",
    )
    args = parser.parse_args()

    omni_home = args.omni_home.resolve()
    if not omni_home.is_dir():
        print(f"ERROR: {omni_home} is not a directory", file=sys.stderr)
        return 1

    print(f"Scanning migration directories under {omni_home}...")
    migration_dirs = find_migration_dirs(omni_home)

    if not migration_dirs:
        print("No migration directories found.")
        return 0

    # Collect all migrations: {prefix: [(repo, file_path)]}
    prefix_map: dict[str, list[tuple[str, Path]]] = {}
    # Collect all CREATE TABLE names: {table: [(repo, file_path)]}
    table_map: dict[str, list[tuple[str, Path]]] = {}
    total_files = 0

    for repo_name, dirs in migration_dirs.items():
        for mig_dir in dirs:
            for sql_file in sorted(mig_dir.glob("*.sql")):
                total_files += 1
                # Check prefix
                match = PREFIX_PATTERN.match(sql_file.name)
                if match:
                    prefix = match.group(1)
                    prefix_map.setdefault(prefix, []).append((repo_name, sql_file))

                # Check table names
                for table_name in extract_tables_from_sql(sql_file):
                    table_map.setdefault(table_name, []).append((repo_name, sql_file))

    print(f"Found {total_files} migration files across {len(migration_dirs)} repos.\n")

    conflicts = 0

    # Check 1: Cross-repo prefix conflicts
    # (same numeric prefix in different repos — potential ordering confusion)
    print("=== Cross-Repo Prefix Conflicts ===")
    prefix_conflicts = 0
    for prefix, entries in sorted(prefix_map.items()):
        repos = {repo for repo, _ in entries}
        if len(repos) > 1:
            prefix_conflicts += 1
            print(f"  Prefix {prefix} used in {len(repos)} repos:")
            for repo, path in entries:
                print(f"    {repo}: {path.name}")
    if prefix_conflicts == 0:
        print("  None found.")
    else:
        conflicts += prefix_conflicts
    print()

    # Check 2: Cross-repo table name conflicts
    # (same CREATE TABLE name in different repos — real collision risk)
    print("=== Cross-Repo Table Name Conflicts ===")
    table_conflicts = 0
    for table_name, entries in sorted(table_map.items()):
        repos = {repo for repo, _ in entries}
        if len(repos) > 1:
            table_conflicts += 1
            print(f"  Table '{table_name}' created in {len(repos)} repos:")
            for repo, path in entries:
                print(f"    {repo}: {path.name}")
    if table_conflicts == 0:
        print("  None found.")
    else:
        conflicts += table_conflicts

    print()

    if conflicts > 0:
        print(f"FAIL: {conflicts} migration conflict(s) found.")
        return 1

    print("OK: No cross-repo migration conflicts found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
