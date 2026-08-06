#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""
CI gate (OMN-15717): every node-owned migration file vendored under
``docker/migrations/forward/nodes/**/*.sql`` must carry a checked-in
stream/domain declaration in
``docker/migrations/forward/_ledger/application-migrations.tsv``, or an
explicit block in
``docker/migrations/forward/_ledger/application-migration-blocks.tsv``,
BEFORE the PR that introduces it can merge.

WHY THIS EXISTS
    This is the pre-merge mirror of the "exactly one declaration or block per
    discovered node migration file" invariant that
    ``validate_application_migration_manifest()`` in
    ``scripts/run-forward-migrations.sh`` already enforces at deploy time
    (against the same two TSV files). That deploy-time check only runs
    against a live Postgres target inside a migration container, so an
    undeclared migration was previously only ever caught by a runtime
    failure -- or not at all, if the affected database was never re-migrated.

    OMN-15717: ``node_pr_review_bot``'s
    ``001_create_review_bot_bypass_log.sql`` was vendored into this tree
    (commit history: vendored, then removed as "stale" once omnimarket
    deleted the node) without ever gaining a row in
    ``application-migrations.tsv``. The gap surfaced only when a live
    database's pre-OMN-15413 legacy ledger row for that migration hit
    ``bootstrap.sql``'s fail-closed "unknown migration stream/domain" guard
    during a workspace-mode ``refresh_stability_lane.sh`` run -- weeks after
    the omission was introduced, and with no pre-merge signal at the time it
    was introduced. This script is the missing pre-merge signal.

Usage:
    python scripts/check_node_migration_declarations.py [--ci]

Exit codes:
    0 -- every vendored node migration file is declared (or explicitly
         blocked) exactly once.
    1 -- one or more vendored node migration files are undeclared, or
         declared AND blocked (ambiguous).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NODE_MIGRATIONS_DIR = REPO_ROOT / "docker" / "migrations" / "forward" / "nodes"
DEFAULT_MANIFEST_PATH = (
    REPO_ROOT
    / "docker"
    / "migrations"
    / "forward"
    / "_ledger"
    / "application-migrations.tsv"
)
DEFAULT_BLOCKS_PATH = (
    REPO_ROOT
    / "docker"
    / "migrations"
    / "forward"
    / "_ledger"
    / "application-migration-blocks.tsv"
)


def _load_declared_artifact_paths(tsv_path: Path) -> set[str]:
    """Return the set of ``artifact_path`` (first TSV column) values declared in tsv_path.

    Raises FileNotFoundError if tsv_path does not exist -- an absent
    declaration surface is a fatal configuration error, not "nothing
    declared".
    """
    if not tsv_path.is_file():
        raise FileNotFoundError(f"declaration surface missing: {tsv_path}")

    declared: set[str] = set()
    for line in tsv_path.read_text().splitlines():
        if not line.strip():
            continue
        artifact_path = line.split("\t", 1)[0]
        declared.add(artifact_path)
    return declared


def find_undeclared_migrations(
    node_migrations_dir: Path,
    manifest_path: Path,
    blocks_path: Path,
) -> list[str]:
    """Return artifact paths under node_migrations_dir with no declaration or with a
    conflicting declaration (both declared AND blocked).

    Mirrors the artifact-path grammar used by run-forward-migrations.sh:
    ``nodes/<node>/<file>.sql`` relative to the forward-migration root.
    """
    if not node_migrations_dir.is_dir():
        return []

    declared = _load_declared_artifact_paths(manifest_path)
    blocked = _load_declared_artifact_paths(blocks_path)

    problems: list[str] = []
    for sql_file in sorted(node_migrations_dir.glob("*/*.sql")):
        artifact_path = f"nodes/{sql_file.relative_to(node_migrations_dir)}"
        in_manifest = artifact_path in declared
        in_blocks = artifact_path in blocked
        # Exactly one of {declared, blocked} must hold. Equal booleans means
        # either both (ambiguous) or neither (undeclared) -- both are defects.
        if in_manifest == in_blocks:
            problems.append(artifact_path)
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ci",
        action="store_true",
        help="no-op flag kept for invocation symmetry with sibling migration gates",
    )
    parser.parse_args()

    node_migrations_dir = DEFAULT_NODE_MIGRATIONS_DIR
    manifest_path = DEFAULT_MANIFEST_PATH
    blocks_path = DEFAULT_BLOCKS_PATH

    if not node_migrations_dir.is_dir():
        print(
            f"[check-node-migration-declarations] no vendored node tree at "
            f"{node_migrations_dir}; nothing to check."
        )
        return 0

    try:
        problems = find_undeclared_migrations(
            node_migrations_dir, manifest_path, blocks_path
        )
    except FileNotFoundError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1

    if problems:
        for artifact_path in problems:
            print(
                f"FATAL: {artifact_path} has no unambiguous checked-in declaration -- "
                f"add exactly one row to "
                f"{manifest_path.relative_to(REPO_ROOT)} (or an explicit block in "
                f"{blocks_path.relative_to(REPO_ROOT)}) before merge.",
                file=sys.stderr,
            )
        print(
            f"[check-node-migration-declarations] {len(problems)} node migration file(s) "
            "missing an unambiguous declaration.",
            file=sys.stderr,
        )
        return 1

    print(
        f"[check-node-migration-declarations] all node migrations under "
        f"{node_migrations_dir.relative_to(REPO_ROOT)} are declared."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
