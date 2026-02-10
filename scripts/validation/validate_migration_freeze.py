#!/usr/bin/env python3
"""Migration freeze enforcement for ONEX repositories.

When a `.migration_freeze` file exists in the repository root, this script
prevents new migration files from being committed. This enforces the schema
freeze during the DB-per-repo refactor (OMN-2055).

The freeze allows:
  - Modifying existing migration files (bug fixes, comments)
  - Deleting migration files (cleanup)

The freeze blocks:
  - Adding NEW migration files (forward or rollback)
  - Renaming files into migration directories (treated as new migrations)

Usage:
    python scripts/validation/validate_migration_freeze.py
    python scripts/validation/validate_migration_freeze.py --verbose
    python scripts/validation/validate_migration_freeze.py /path/to/repo

Exit Codes:
    0 - No freeze active, or no new migrations detected
    1 - Freeze active and new migration files detected
    2 - Script error
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Directories containing migration files.
# A single prefix is sufficient — startswith() matches all subdirectories
# (docker/migrations/forward/, docker/migrations/rollback/, etc.).
# This covers the complete migration layout for this repository.
MIGRATION_DIRS: tuple[str, ...] = ("docker/migrations/",)

# No extension filter — any new file in a migration directory during a
# freeze is suspicious. Matches the bash CI script (check_migration_freeze.sh)
# which checks all Added/Renamed files regardless of extension.


@dataclass
class FreezeViolation:
    """A migration file added while freeze is active."""

    file_path: str
    migration_dir: str

    def __str__(self) -> str:
        return f"  NEW: {self.file_path}"


@dataclass
class FreezeValidationResult:
    """Result of migration freeze validation."""

    freeze_active: bool = False
    violations: list[FreezeViolation] = field(default_factory=list)
    new_files_checked: int = 0

    @property
    def is_valid(self) -> bool:
        return len(self.violations) == 0

    def __bool__(self) -> bool:
        """Allow using result in boolean context.

        Warning:
            **Non-standard __bool__ behavior**: Returns ``True`` only when
            ``is_valid`` is True (no violations). Differs from typical
            dataclass behavior where any instance is truthy.
        """
        return self.is_valid


def _get_new_staged_files(repo_path: Path) -> list[str]:
    """Get files that are newly added or renamed in the staging area.

    Uses ``--diff-filter=AR`` to match the bash CI script behavior:
    Added (A) files are new migrations, Renamed (R) files are treated as
    new migrations in their destination directory.
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=AR"],
            capture_output=True,
            text=True,
            cwd=str(repo_path),
            timeout=10,
            check=False,
        )
        if result.returncode != 0:
            return []
        return [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []


def _get_merge_base(repo_path: Path) -> str:
    """Auto-detect the merge base against origin/main (or origin/master)."""
    for branch in ("origin/main", "origin/master"):
        try:
            result = subprocess.run(
                ["git", "merge-base", "HEAD", branch],
                capture_output=True,
                text=True,
                cwd=str(repo_path),
                timeout=10,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue
    # Fallback: HEAD~1 may not exist in shallow clones or single-commit repos,
    # causing git diff to return an empty list (no violations detected). This is
    # acceptable because CI uses check_migration_freeze.sh with GITHUB_BASE_REF
    # instead, and pre-commit mode uses --check-staged (not merge-base).
    return "HEAD~1"


def _get_new_committed_files(repo_path: Path, base_ref: str | None = None) -> list[str]:
    """Get files that are newly added or renamed in committed changes.

    Uses ``--diff-filter=AR`` to match the bash CI script behavior:
    Added (A) files are new migrations, Renamed (R) files are treated as
    new migrations in their destination directory.

    Args:
        repo_path: Repository root path.
        base_ref: Base ref for comparison. If None, auto-detects merge base
                  against origin/main to cover all commits on the branch.
    """
    if base_ref is None:
        base_ref = _get_merge_base(repo_path)
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "--diff-filter=AR", f"{base_ref}..HEAD"],
            capture_output=True,
            text=True,
            cwd=str(repo_path),
            timeout=10,
            check=False,
        )
        if result.returncode != 0:
            return []
        return [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []


def _is_migration_file(file_path: str) -> tuple[bool, str]:
    """Check if a file is in a watched migration directory.

    No extension filter — any new file in a migration directory during a
    freeze is treated as a violation. This matches the bash CI script.

    Returns:
        Tuple of (is_migration, migration_dir).
    """
    for migration_dir in MIGRATION_DIRS:
        if file_path.startswith(migration_dir):
            return True, migration_dir
    return False, ""


def validate_migration_freeze(
    repo_path: Path,
    verbose: bool = False,
    check_staged: bool = True,
) -> FreezeValidationResult:
    """Validate that no new migrations are added while freeze is active.

    Args:
        repo_path: Path to the repository root.
        verbose: Enable verbose output.
        check_staged: If True, check staged files (pre-commit mode).
                      If False, check committed files against merge base.

    Returns:
        FreezeValidationResult with any violations found.
    """
    result = FreezeValidationResult()

    freeze_file = repo_path / ".migration_freeze"
    if not freeze_file.exists():
        if verbose:
            print("Migration Freeze: inactive (no .migration_freeze file)")
        return result

    result.freeze_active = True
    if verbose:
        print("Migration Freeze: ACTIVE")

    # Get new files from git
    if check_staged:
        new_files = _get_new_staged_files(repo_path)
    else:
        new_files = _get_new_committed_files(repo_path)

    # Check each new file against migration directories
    for file_path in new_files:
        result.new_files_checked += 1
        is_migration, migration_dir = _is_migration_file(file_path)
        if is_migration:
            result.violations.append(
                FreezeViolation(file_path=file_path, migration_dir=migration_dir)
            )

    if verbose:
        print(f"  New files checked: {result.new_files_checked}")
        print(f"  Migration violations: {len(result.violations)}")

    return result


def generate_report(result: FreezeValidationResult, repo_path: Path) -> str:
    """Generate a validation report."""
    if not result.freeze_active:
        return "Migration Freeze: inactive (no .migration_freeze file)"

    if result.is_valid:
        return f"Migration Freeze: PASS (freeze active, {result.new_files_checked} new files checked, no violations)"

    lines = [
        "❌ MIGRATION FREEZE VIOLATION",
        "=" * 60,
        "",
        f"Schema migrations are FROZEN (see {repo_path / '.migration_freeze'}).",
        f"Found {len(result.violations)} new migration file(s):",
        "",
    ]

    for v in result.violations:
        lines.append(str(v))

    lines.extend(
        [
            "",
            "=" * 60,
            "ALLOWED during freeze:",
            "  - Modifying existing migration files (bug fixes)",
            "  - Moving migration files between repos",
            "  - Deleting migration files",
            "",
            "NOT ALLOWED during freeze:",
            "  - Adding NEW migration files",
            "",
            "To lift the freeze, remove .migration_freeze and reference OMN-2055.",
            "",
        ]
    )

    return "\n".join(lines)


def main() -> int:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate migration freeze enforcement"
    )
    parser.add_argument(
        "repo_path",
        nargs="?",
        default=None,
        help="Path to repository root (default: auto-detect)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--check-committed",
        action="store_true",
        help="Check committed files instead of staged (for CI)",
    )

    args = parser.parse_args()

    if args.repo_path:
        repo_path = Path(args.repo_path).resolve()
    else:
        script_path = Path(__file__).resolve()
        repo_path = script_path.parent.parent.parent

    try:
        result = validate_migration_freeze(
            repo_path,
            verbose=args.verbose,
            check_staged=not args.check_committed,
        )
        report = generate_report(result, repo_path)
        print(report)
        return 0 if result.is_valid else 1

    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
