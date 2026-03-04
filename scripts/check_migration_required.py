#!/usr/bin/env python3
"""
CI gate: any PR touching a writer_postgres.py or handler_*_postgres.py file
must either include a new migration file OR have a '# no-migration: <reason>'
comment in the changed file.

Usage (CI — compare against base branch):
    python scripts/check_migration_required.py --ci

Usage (pre-commit — check staged files):
    python scripts/check_migration_required.py --pre-commit
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

WRITER_PATTERNS = [
    re.compile(r"writer_postgres\.py$"),
    re.compile(r"handler_\w+_postgres\.py$"),
]

MIGRATION_DIRS = [
    "docker/migrations/forward/",
    "src/omnibase_infra/migrations/forward/",
]

BYPASS_RE = re.compile(r"#\s*no-migration\s*:", re.IGNORECASE)


def is_writer_file(path: str) -> bool:
    return any(p.search(path) for p in WRITER_PATTERNS)


def has_bypass_comment(content: str) -> bool:
    return bool(BYPASS_RE.search(content))


def get_changed_files(ci: bool) -> list[str]:
    if ci:
        base = subprocess.check_output(
            ["git", "diff", "--name-only", "origin/main...HEAD"],
            text=True,
        )
    else:
        base = subprocess.check_output(
            ["git", "diff", "--cached", "--name-only"],
            text=True,
        )
    return [f.strip() for f in base.splitlines() if f.strip()]


def find_violations(
    changed_files: list[str],
    bypass_comment: str | None,
) -> list[str]:
    """Return list of writer files that lack a corresponding migration."""
    writer_files = [f for f in changed_files if is_writer_file(f)]
    if not writer_files:
        return []

    has_migration = any(
        any(f.startswith(d) and f.endswith(".sql") for d in MIGRATION_DIRS)
        for f in changed_files
    )
    if has_migration:
        return []

    violations = []
    for wf in writer_files:
        path = Path(wf)
        if path.exists():
            content = path.read_text()
            if has_bypass_comment(content):
                continue
        if bypass_comment and has_bypass_comment(bypass_comment):
            continue
        violations.append(wf)

    return violations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ci", action="store_true")
    parser.add_argument("--pre-commit", action="store_true")
    args = parser.parse_args()

    changed = get_changed_files(ci=args.ci)
    violations = find_violations(changed, bypass_comment=None)

    if violations:
        print("ERROR: Writer file(s) changed without a migration file in this PR:")
        for v in violations:
            print(f"  {v}")
        print()
        print("Either:")
        print(
            "  1. Add a migration file to docker/migrations/forward/"
            " or src/omnibase_infra/migrations/forward/"
        )
        print(
            "  2. Add '# no-migration: <reason>' to the writer file"
            " if no schema change is needed"
        )
        sys.exit(1)

    print("OK: writer-migration coupling check passed.")


if __name__ == "__main__":
    main()
