#!/usr/bin/env python3
"""Pre-commit validator: block duplicate migration sequence numbers.

Checks each migration directory independently for duplicate leading-integer
sequence numbers. docker/migrations/forward/ and src/omnibase_infra/migrations/
forward/ use separate sequence namespaces because run-migrations.py tracks
them with distinct migration_id keys (source_set/filename). Cross-directory
sequence collisions are intentionally allowed.

Exit Codes:
    0 - All sequence numbers are unique within each directory (or no files staged)
    1 - Duplicate sequence number detected within the same directory
    2 - Script error (git unavailable, not a repo, unexpected exception)

Ticket: OMN-3570
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Directories to validate independently for intra-directory duplicate sequences.
# Each directory forms its own sequence namespace — run-migrations.py tracks
# migrations as "{source_set}/{filename}" so cross-directory collisions are safe.
MIGRATION_DIRS: tuple[str, ...] = (
    "docker/migrations/forward",
    "src/omnibase_infra/migrations/forward",
)


@dataclass
class DuplicateConflict:
    """A pair of migration files that share a sequence number."""

    sequence: int
    file_a: str
    file_b: str

    def __str__(self) -> str:
        return f"  seq {self.sequence:03d}: {self.file_a!r}  <-->  {self.file_b!r}"


@dataclass
class SequenceValidationResult:
    """Result of migration sequence uniqueness validation."""

    conflicts: list[DuplicateConflict] = field(default_factory=list)
    files_scanned: int = 0
    has_staged_migrations: bool = False

    @property
    def is_valid(self) -> bool:
        return len(self.conflicts) == 0

    def __bool__(self) -> bool:
        return self.is_valid


def extract_sequence_number(filename: str) -> int | None:
    """Extract leading integer sequence number from a migration filename.

    Identical to run-migrations.py's extract_sequence_number() — one definition
    of sequence number across the codebase.

    Only .sql files are considered migrations (matches run-migrations.py).

    Returns:
        Integer sequence number, or None if file is not a .sql migration.
    """
    path = Path(filename)
    if path.suffix.lower() != ".sql":
        return None
    stem = path.stem
    prefix = ""
    for ch in stem:
        if ch.isdigit():
            prefix += ch
        else:
            break
    if not prefix:
        return None
    return int(prefix)


def _get_staged_paths(repo_path: Path) -> list[str]:
    """Return paths staged in the git index (git diff --cached --name-only).

    Raises:
        RuntimeError: If git is unavailable or repo_path is not a git repo.
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            capture_output=True,
            text=True,
            cwd=str(repo_path),
            timeout=10,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("git executable not found") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("git diff --cached timed out") from exc

    if result.returncode != 0:
        raise RuntimeError(
            f"git diff --cached failed (exit {result.returncode}): {result.stderr.strip()}"
        )

    return [p.strip() for p in result.stdout.strip().splitlines() if p.strip()]


def _scan_migration_dir(repo_path: Path, rel_dir: str) -> list[Path]:
    """Return all .sql files in rel_dir (relative to repo_path), if it exists."""
    abs_dir = repo_path / rel_dir
    if not abs_dir.is_dir():
        return []
    return sorted(abs_dir.glob("*.sql"))


def validate_migration_sequence(
    repo_path: Path,
) -> SequenceValidationResult:
    """Validate that all migration .sql files have unique sequence numbers.

    Each migration directory is checked independently — docker/ and src/ sets
    have separate sequence namespaces because run-migrations.py tracks them
    with distinct migration_id keys ({source_set}/{filename}). Only intra-
    directory duplicates are reported.

    Staged paths are included in the scan so newly added files are caught
    before they are committed.

    Args:
        repo_path: Absolute path to the repository root.

    Returns:
        SequenceValidationResult with any conflicts found.

    Raises:
        RuntimeError: If git is unavailable or not a git repository.
    """
    result = SequenceValidationResult()

    # Collect staged paths to determine whether any migration was staged
    staged_paths = _get_staged_paths(repo_path)

    # Check whether any staged file is a migration file
    staged_migration_rel: set[str] = set()
    for sp in staged_paths:
        for mdir in MIGRATION_DIRS:
            if sp.startswith((mdir + "/", mdir.replace("/", "\\") + "\\")):
                seq = extract_sequence_number(Path(sp).name)
                if seq is not None:
                    staged_migration_rel.add(sp)
                    break

    result.has_staged_migrations = bool(staged_migration_rel)

    # If no migration files are staged, exit early (hook does not fire)
    if not result.has_staged_migrations:
        return result

    # Check each migration directory independently for intra-directory duplicates
    total_scanned = 0
    for mdir in MIGRATION_DIRS:
        seen: dict[int, str] = {}  # seq -> relative path string

        # Gather all .sql files from this directory (filesystem scan)
        dir_files: list[str] = []
        for f in _scan_migration_dir(repo_path, mdir):
            rel = str(f.relative_to(repo_path))
            dir_files.append(rel)

        # Also include staged paths in this directory that may not be on disk yet
        for sp in staged_migration_rel:
            if sp.startswith((mdir + "/", mdir.replace("/", "\\") + "\\")):
                if sp not in dir_files:
                    dir_files.append(sp)

        total_scanned += len(dir_files)

        for rel in sorted(dir_files):
            seq = extract_sequence_number(Path(rel).name)
            if seq is None:
                continue
            if seq in seen:
                result.conflicts.append(
                    DuplicateConflict(
                        sequence=seq,
                        file_a=seen[seq],
                        file_b=rel,
                    )
                )
            else:
                seen[seq] = rel

    result.files_scanned = total_scanned

    return result


def generate_report(result: SequenceValidationResult) -> str:
    """Generate a human-readable validation report."""
    if not result.has_staged_migrations:
        return "Migration Sequence: no migration files staged — skipped"

    if result.is_valid:
        return (
            f"Migration Sequence: PASS"
            f" ({result.files_scanned} file(s) scanned, no duplicates)"
        )

    lines: list[str] = [
        "ERROR: DUPLICATE MIGRATION SEQUENCE NUMBER",
        "=" * 60,
        "",
        f"Found {len(result.conflicts)} duplicate sequence number(s):",
        "",
    ]
    for conflict in result.conflicts:
        lines.append(str(conflict))
    lines.extend(
        [
            "",
            "=" * 60,
            "Each migration file must have a unique leading sequence number",
            "within its directory (docker/ and src/ are checked independently).",
            "",
            "To fix: renumber the new migration to use the next available sequence",
            "within its directory.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    """Main entry point for CLI and pre-commit hook invocation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate migration sequence number uniqueness"
    )
    parser.add_argument(
        "repo_path",
        nargs="?",
        default=None,
        help="Path to repository root (default: auto-detect from script location)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.repo_path:
        repo_path = Path(args.repo_path).resolve()
    else:
        # Auto-detect: script lives at scripts/validation/validate_migration_sequence.py
        repo_path = Path(__file__).resolve().parent.parent.parent

    try:
        result = validate_migration_sequence(repo_path)
        report = generate_report(result)
        if args.verbose or not result.is_valid or result.has_staged_migrations:
            print(report)
        return 0 if result.is_valid else 1

    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
