#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""CI twin for schema fingerprint validation (OMN-2149).

Computes a deterministic SHA-256 fingerprint from the contents of all
forward migration SQL files and compares it to the committed artifact at
``docker/migrations/schema_fingerprint.sha256``.

This is the CI-side mirror of the B2 runtime assertion. While the runtime
validates the live database schema against the expected fingerprint stored
in ``db_metadata``, this CI twin validates that the committed artifact
stays in sync with the migration source files.

Usage::

    # Verify: compare migration-derived fingerprint to committed artifact
    python scripts/check_schema_fingerprint.py verify

    # Stamp: regenerate the artifact from current migration files
    python scripts/check_schema_fingerprint.py stamp

    # Dry-run: compute fingerprint without writing
    python scripts/check_schema_fingerprint.py stamp --dry-run

Exit codes:
    0 -- Artifact is current (verify) or stamp succeeded
    1 -- Usage error or unexpected failure
    2 -- Fingerprint mismatch (artifact is stale)
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

# Repository root is two levels up from this script
_REPO_ROOT = Path(__file__).resolve().parent.parent
_MIGRATIONS_DIR = _REPO_ROOT / "docker" / "migrations" / "forward"
_ARTIFACT_PATH = _REPO_ROOT / "docker" / "migrations" / "schema_fingerprint.sha256"


def compute_migration_fingerprint(migrations_dir: Path) -> tuple[str, int]:
    """Compute SHA-256 fingerprint from sorted forward migration files.

    Algorithm:
        1. Glob all ``*.sql`` files in the migrations directory.
        2. Sort by filename (lexicographic) for determinism.
        3. For each file, compute SHA-256 of its contents.
        4. Concatenate all ``(filename, file_hash)`` pairs as a canonical
           JSON-like string and SHA-256 the result.

    Args:
        migrations_dir: Path to the forward migrations directory.

    Returns:
        Tuple of (fingerprint_hex, file_count).
    """
    sql_files = sorted(migrations_dir.glob("*.sql"))

    if not sql_files:
        # Empty migrations dir -- produce a deterministic empty hash
        empty_hash = hashlib.sha256(b"[]").hexdigest()
        return empty_hash, 0

    # Build canonical representation: sorted list of (filename, content_hash)
    entries: list[str] = []
    for sql_file in sql_files:
        content = sql_file.read_bytes()
        file_hash = hashlib.sha256(content).hexdigest()
        entries.append(f"{sql_file.name}:{file_hash}")

    canonical = "\n".join(entries)
    overall_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return overall_hash, len(sql_files)


def read_artifact(artifact_path: Path) -> str | None:
    """Read the committed fingerprint from the artifact file.

    Parses the ``sha256:<hex>`` line from the artifact. Returns None if
    the file does not exist, is unreadable, or contains ``PENDING``.

    Args:
        artifact_path: Path to the schema_fingerprint.sha256 file.

    Returns:
        The hex fingerprint string, or None if unavailable.
    """
    if not artifact_path.exists():
        return None

    for line in artifact_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("sha256:"):
            value = stripped[len("sha256:") :]
            if value == "PENDING":
                return None
            return value

    return None


def write_artifact(
    artifact_path: Path,
    fingerprint: str,
    file_count: int,
) -> None:
    """Write the fingerprint artifact file.

    Args:
        artifact_path: Destination path.
        fingerprint: SHA-256 hex digest.
        file_count: Number of migration files included.
    """
    from datetime import UTC, datetime

    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    content = (
        "# Schema migration fingerprint for omnibase_infra (auto-generated)\n"
        "# Regenerate: python scripts/check_schema_fingerprint.py stamp\n"
        "# Verify:     python scripts/check_schema_fingerprint.py verify\n"
        f"sha256:{fingerprint}\n"
        f"generated_at: {now}\n"
        f"migration_file_count: {file_count}\n"
    )
    artifact_path.write_text(content, encoding="utf-8")


def cmd_verify(
    migrations_dir: Path,
    artifact_path: Path,
) -> int:
    """Verify that the committed artifact matches current migration files.

    Returns:
        Exit code: 0 if matching, 2 if stale.
    """
    fingerprint, file_count = compute_migration_fingerprint(migrations_dir)
    committed = read_artifact(artifact_path)

    print(f"Migration files:    {file_count}")
    print(f"Computed:           {fingerprint}")
    print(f"Committed artifact: {committed or '(missing/PENDING)'}")

    if committed is None:
        print(
            "\nFAILED: Schema fingerprint artifact is missing or PENDING.",
            file=sys.stderr,
        )
        print(
            "Run: python scripts/check_schema_fingerprint.py stamp",
            file=sys.stderr,
        )
        return 2

    if fingerprint != committed:
        print(
            "\nFAILED: Schema fingerprint artifact is stale.",
            file=sys.stderr,
        )
        print(
            f"  Expected (from migrations): {fingerprint}",
            file=sys.stderr,
        )
        print(
            f"  Committed (in artifact):    {committed}",
            file=sys.stderr,
        )
        print(
            "\nMigration files have changed but the artifact was not regenerated.",
            file=sys.stderr,
        )
        print(
            "Run: python scripts/check_schema_fingerprint.py stamp",
            file=sys.stderr,
        )
        return 2

    print("\nSchema fingerprint OK")
    return 0


def cmd_stamp(
    migrations_dir: Path,
    artifact_path: Path,
    *,
    dry_run: bool = False,
) -> int:
    """Regenerate the fingerprint artifact.

    Returns:
        Exit code: 0 on success.
    """
    fingerprint, file_count = compute_migration_fingerprint(migrations_dir)

    print(f"Migration files: {file_count}")
    print(f"Fingerprint:     {fingerprint}")

    if dry_run:
        print("\n--dry-run: skipping artifact write")
        return 0

    write_artifact(artifact_path, fingerprint, file_count)
    try:
        display = artifact_path.relative_to(_REPO_ROOT)
    except ValueError:
        display = artifact_path
    print(f"\nArtifact written to {display}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="check_schema_fingerprint",
        description="CI twin: schema fingerprint drift detection (OMN-2149).",
    )
    sub = parser.add_subparsers(dest="command")

    stamp_parser = sub.add_parser(
        "stamp",
        help="Regenerate the fingerprint artifact from migration files.",
    )
    stamp_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute fingerprint without writing the artifact.",
    )
    stamp_parser.add_argument(
        "--migrations-dir",
        type=Path,
        default=_MIGRATIONS_DIR,
        help=f"Path to forward migrations directory (default: {_MIGRATIONS_DIR}).",
    )
    stamp_parser.add_argument(
        "--artifact",
        type=Path,
        default=_ARTIFACT_PATH,
        help=f"Path to fingerprint artifact (default: {_ARTIFACT_PATH}).",
    )

    verify_parser = sub.add_parser(
        "verify",
        help="Verify committed artifact matches current migration files.",
    )
    verify_parser.add_argument(
        "--migrations-dir",
        type=Path,
        default=_MIGRATIONS_DIR,
        help=f"Path to forward migrations directory (default: {_MIGRATIONS_DIR}).",
    )
    verify_parser.add_argument(
        "--artifact",
        type=Path,
        default=_ARTIFACT_PATH,
        help=f"Path to fingerprint artifact (default: {_ARTIFACT_PATH}).",
    )

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "stamp":
        return cmd_stamp(
            args.migrations_dir,
            args.artifact,
            dry_run=args.dry_run,
        )
    elif args.command == "verify":
        return cmd_verify(args.migrations_dir, args.artifact)

    return 1


if __name__ == "__main__":
    sys.exit(main())
