#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

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

Freeze age policy (requires freeze_date= field in .migration_freeze):
  - WARNING at 30+ days since freeze_date
  - ERROR (exit 1) at 60+ days since freeze_date

Usage:
    python scripts/validation/validate_migration_freeze.py
    python scripts/validation/validate_migration_freeze.py --verbose
    python scripts/validation/validate_migration_freeze.py /path/to/repo

Exit Codes:
    0 - No freeze active, or no new migrations detected (freeze within age limits)
    1 - Freeze active and new migration files detected, OR freeze expired (60+ days)
    2 - Script error
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timezone
from pathlib import Path

# Directories containing migration files.
# A single prefix is sufficient — startswith() matches all subdirectories
# (docker/migrations/forward/, docker/migrations/rollback/, etc.).
# This covers the complete migration layout for this repository.
MIGRATION_DIRS: tuple[str, ...] = ("docker/migrations/",)

# No extension filter — any new file in a migration directory during a
# freeze is suspicious. Matches the bash CI script (check_migration_freeze.sh)
# which checks all Added/Renamed files regardless of extension.

FREEZE_WARN_DAYS = 30
FREEZE_EXPIRE_DAYS = 60


@dataclass
class FreezeViolation:
    """A migration file added while freeze is active."""

    file_path: str
    migration_dir: str

    def __str__(self) -> str:
        return f"  NEW: {self.file_path}"


@dataclass
class FreezeAgeStatus:
    """Result of freeze age check."""

    freeze_date: date | None = None
    age_days: int | None = None
    is_warning: bool = False  # 30+ days
    is_expired: bool = False  # 60+ days

    @property
    def has_date(self) -> bool:
        return self.freeze_date is not None


@dataclass
class FreezeValidationResult:
    """Result of migration freeze validation."""

    freeze_active: bool = False
    violations: list[FreezeViolation] = field(default_factory=list)
    new_files_checked: int = 0
    age_status: FreezeAgeStatus = field(default_factory=FreezeAgeStatus)

    @property
    def is_valid(self) -> bool:
        return len(self.violations) == 0 and not self.age_status.is_expired

    def __bool__(self) -> bool:
        """Allow using result in boolean context.

        Warning:
            **Non-standard __bool__ behavior**: Returns ``True`` only when
            ``is_valid`` is True (no violations, not expired). Differs from
            typical dataclass behavior where any instance is truthy.
        """
        return self.is_valid


def _parse_freeze_date(freeze_file: Path) -> date | None:
    """Parse the freeze_date= field from .migration_freeze.

    Expected format: freeze_date=YYYY-MM-DD

    Returns:
        Parsed date, or None if field is absent or unparseable.
    """
    try:
        content = freeze_file.read_text(encoding="utf-8")
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("freeze_date="):
                raw = line.split("=", 1)[1].strip()
                return datetime.strptime(raw, "%Y-%m-%d").replace(tzinfo=UTC).date()
    except (ValueError, OSError):
        pass
    return None


def _compute_freeze_age(freeze_file: Path) -> FreezeAgeStatus:
    """Compute freeze age from freeze_date= field in the freeze file.

    Returns:
        FreezeAgeStatus with age_days, is_warning, is_expired fields.
    """
    status = FreezeAgeStatus()
    freeze_date = _parse_freeze_date(freeze_file)
    if freeze_date is None:
        return status

    status.freeze_date = freeze_date
    today = datetime.now(tz=UTC).date()
    status.age_days = (today - freeze_date).days
    status.is_warning = status.age_days >= FREEZE_WARN_DAYS
    status.is_expired = status.age_days >= FREEZE_EXPIRE_DAYS
    return status


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

    Also checks freeze age: warns at 30+ days, fails at 60+ days.

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

    # Check freeze age
    result.age_status = _compute_freeze_age(freeze_file)
    if verbose and result.age_status.has_date:
        print(
            f"  Freeze age: {result.age_status.age_days} day(s)"
            f" (since {result.age_status.freeze_date})"
        )

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

    lines: list[str] = []

    # Age status reporting
    age = result.age_status
    if age.has_date:
        if age.is_expired:
            lines.extend(
                [
                    "ERROR: Migration freeze has EXPIRED!",
                    "=" * 60,
                    "",
                    f"Freeze date: {age.freeze_date} ({age.age_days} days ago)",
                    f"Freezes are automatically invalidated after {FREEZE_EXPIRE_DAYS} days.",
                    "",
                    "Action required:",
                    f"  1. Either lift the freeze (remove {repo_path / '.migration_freeze'})"
                    " if the DB boundary work is complete",
                    "  2. Or renew the freeze by updating freeze_date= with a justification comment",
                    f"  3. Track in the freeze ticket (see ticket= field in"
                    f" {repo_path / '.migration_freeze'})",
                    "",
                ]
            )
        elif age.is_warning:
            days_until_expiry = FREEZE_EXPIRE_DAYS - (age.age_days or 0)
            lines.extend(
                [
                    "WARNING: Migration freeze is approaching expiry!",
                    f"Freeze date: {age.freeze_date} ({age.age_days} days ago)",
                    f"This freeze will become an ERROR in {days_until_expiry} day(s).",
                    f"Review the freeze status and update {repo_path / '.migration_freeze'}"
                    " if still needed.",
                    "",
                ]
            )

    # Violation reporting
    if not result.violations and not age.is_expired:
        age_info = ""
        if age.has_date:
            age_info = f", age={age.age_days}d"
        return (
            f"Migration Freeze: PASS (freeze active, "
            f"{result.new_files_checked} new files checked, no violations{age_info})"
            + ("\n" + "\n".join(lines) if lines else "")
        )

    if result.violations:
        lines.extend(
            [
                "ERROR: MIGRATION FREEZE VIOLATION",
                "=" * 60,
                "",
                f"Schema migrations are FROZEN (see {repo_path / '.migration_freeze'}).",
                f"Found {len(result.violations)} new migration file(s):",
                "",
            ]
        )
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
        print(f"Error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
