#!/usr/bin/env python3
"""CI validator to detect direct os.getenv/os.environ usage.

This script enforces the SecretResolver pattern by blocking direct environment
variable access in production code paths. Use the allowlist for migration.

Exit codes:
    0: No violations found
    1: Violations found (blocks CI)
    2: Configuration error (malformed allowlist)

Usage:
    python scripts/validate_no_direct_env.py
    python scripts/validate_no_direct_env.py --verbose
    python scripts/validate_no_direct_env.py --fix-allowlist  # Generate allowlist entries
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Patterns to detect
FORBIDDEN_PATTERNS = [
    re.compile(r"os\.getenv\s*\("),
    re.compile(r"os\.environ\s*\["),
    re.compile(r"os\.environ\.get\s*\("),
    re.compile(r"os\.environ\.setdefault\s*\("),
]

# Production paths to scan (relative to repo root)
SCAN_PATHS = [
    "src/omnibase_infra/handlers/",
    "src/omnibase_infra/runtime/",
    "src/omnibase_infra/nodes/",
    "src/omnibase_infra/adapters/",
    "src/omnibase_infra/event_bus/",
]

# Paths to always exclude
EXCLUDE_PATHS = [
    "__pycache__",
    ".pyc",
    "test_",
    "_test.py",
    "conftest.py",
]

# Bootstrap exception - these files may use os.getenv for Vault bootstrap only
# Uses exact relative paths to prevent overly broad exemptions
BOOTSTRAP_EXCEPTION_PATHS = [
    "src/omnibase_infra/runtime/secret_resolver.py",  # The resolver itself needs bootstrap access
]

ALLOWLIST_FILE = ".secretresolver_allowlist"


class Violation:
    """Represents a single violation."""

    def __init__(
        self, filepath: str, line_number: int, line_content: str, pattern: str
    ) -> None:
        self.filepath = filepath
        self.line_number = line_number
        self.line_content = line_content.strip()
        self.pattern = pattern

    def __str__(self) -> str:
        return (
            f"{self.filepath}:{self.line_number}: {self.pattern} - {self.line_content}"
        )

    def allowlist_key(self) -> str:
        """Generate allowlist key for this violation."""
        return f"{self.filepath}:{self.line_number}"


class AllowlistValidationError(Exception):
    """Raised when allowlist contains malformed entries."""

    def __init__(self, malformed_entries: list[tuple[int, str]]) -> None:
        self.malformed_entries = malformed_entries
        super().__init__(f"Malformed allowlist entries: {len(malformed_entries)}")


# Pattern for valid allowlist entries: filepath:line_number
ALLOWLIST_ENTRY_PATTERN = re.compile(r"^[^:]+:\d+$")


def load_allowlist(repo_root: Path, verbose: bool = False) -> set[str]:
    """Load allowlist entries from file.

    Format: filepath:line_number # optional comment
    Lines starting with # are comments.

    Args:
        repo_root: Repository root path.
        verbose: If True, log warnings for file read errors.

    Raises:
        AllowlistValidationError: If any entries are malformed.
    """
    allowlist_path = repo_root / ALLOWLIST_FILE
    if not allowlist_path.exists():
        return set()

    allowlist: set[str] = set()
    malformed_entries: list[tuple[int, str]] = []

    try:
        lines = allowlist_path.read_text().splitlines()
    except (OSError, UnicodeDecodeError) as e:
        if verbose:
            print(f"WARNING: Could not read allowlist file: {e}")
        return set()

    for line_number, line in enumerate(lines, start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Strip comments
        if " #" in line:
            line = line.split(" #")[0].strip()

        # Validate format: filepath:line_number
        if not ALLOWLIST_ENTRY_PATTERN.match(line):
            malformed_entries.append((line_number, line))
            continue

        allowlist.add(line)

    if malformed_entries:
        raise AllowlistValidationError(malformed_entries)

    return allowlist


def find_repo_root() -> Path:
    """Find repository root by looking for .git directory."""
    current = Path.cwd()
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent
    return Path.cwd()


def should_scan_file(filepath: Path, repo_root: Path) -> bool:
    """Check if file should be scanned."""
    relative = str(filepath.relative_to(repo_root))

    # Check exclusions
    for exclude in EXCLUDE_PATHS:
        if exclude in relative:
            return False

    # Must be a Python file
    if filepath.suffix != ".py":
        return False

    return True


def is_bootstrap_exception(filepath: Path, repo_root: Path) -> bool:
    """Check if file is allowed bootstrap exception.

    Uses exact relative path matching to prevent overly broad exemptions.
    For example, only src/omnibase_infra/runtime/secret_resolver.py is exempt,
    not any file named secret_resolver.py in other locations.

    Args:
        filepath: Absolute path to the file being checked.
        repo_root: Repository root path for computing relative path.

    Returns:
        True if the file is in the bootstrap exception list.
    """
    relative = str(filepath.relative_to(repo_root))
    return relative in BOOTSTRAP_EXCEPTION_PATHS


def scan_file(filepath: Path, repo_root: Path) -> list[Violation]:
    """Scan a single file for violations."""
    violations: list[Violation] = []
    relative_path = str(filepath.relative_to(repo_root))

    try:
        content = filepath.read_text()
    except (OSError, UnicodeDecodeError):
        # File unreadable (permissions, encoding) - skip without failing
        return violations

    for line_number, line in enumerate(content.splitlines(), start=1):
        # Skip comments
        stripped = line.strip()
        if stripped.startswith("#"):
            continue

        # Skip if has inline allowlist marker
        if "# ONEX_EXCLUDE: secret_resolver" in line:
            continue

        for pattern in FORBIDDEN_PATTERNS:
            if pattern.search(line):
                violations.append(
                    Violation(
                        filepath=relative_path,
                        line_number=line_number,
                        line_content=line,
                        pattern=pattern.pattern,
                    )
                )
                break  # Only report first pattern match per line

    return violations


def scan_directory(scan_path: Path, repo_root: Path) -> list[Violation]:
    """Scan directory recursively for violations."""
    violations: list[Violation] = []

    if not scan_path.exists():
        return violations

    for filepath in scan_path.rglob("*.py"):
        if not should_scan_file(filepath, repo_root):
            continue

        if is_bootstrap_exception(filepath, repo_root):
            continue

        violations.extend(scan_file(filepath, repo_root))

    return violations


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate no direct os.getenv usage in production code"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed output"
    )
    parser.add_argument(
        "--fix-allowlist",
        action="store_true",
        help="Generate allowlist entries for current violations",
    )
    args = parser.parse_args()

    repo_root = find_repo_root()

    try:
        allowlist = load_allowlist(repo_root, verbose=args.verbose)
    except AllowlistValidationError as e:
        print(f"ERROR: Malformed allowlist entries in {ALLOWLIST_FILE}")
        print()
        print("Each entry must match format: filepath:line_number")
        print()
        print("Malformed entries:")
        for line_number, entry in e.malformed_entries:
            print(f"  Line {line_number}: {entry!r}")
        print()
        print("Example valid entry:")
        print("  src/omnibase_infra/handlers/handler_foo.py:42 # OMN-764 migration")
        return 2

    if args.verbose:
        print(f"Repository root: {repo_root}")
        print(f"Allowlist entries: {len(allowlist)}")
        print(f"Scanning paths: {SCAN_PATHS}")
        print()

    all_violations: list[Violation] = []

    for scan_path_str in SCAN_PATHS:
        scan_path = repo_root / scan_path_str
        violations = scan_directory(scan_path, repo_root)
        all_violations.extend(violations)

    # Filter out allowlisted violations
    new_violations = [v for v in all_violations if v.allowlist_key() not in allowlist]

    if args.fix_allowlist:
        # Generate allowlist entries
        print("# SecretResolver migration allowlist")
        print("# Generated entries - remove as you migrate")
        print()
        for v in sorted(all_violations, key=lambda x: (x.filepath, x.line_number)):
            print(f"{v.allowlist_key()} # OMN-764 migration pending")
        return 0

    if new_violations:
        print(
            f"ERROR: Found {len(new_violations)} direct os.getenv/os.environ usage(s)"
        )
        print()
        print("Violations:")
        for v in sorted(new_violations, key=lambda x: (x.filepath, x.line_number)):
            print(f"  {v}")
        print()
        print("To fix:")
        print("  1. Use SecretResolver.get_secret() instead of os.getenv()")
        print("  2. Or add to .secretresolver_allowlist with OMN-764 ticket reference")
        print()
        print("Example allowlist entry:")
        if new_violations:
            print(f"  {new_violations[0].allowlist_key()} # OMN-764 migration pending")
        return 1

    if args.verbose:
        print("OK: No new violations found")
        print(f"    Total scanned: {len(all_violations)} occurrences")
        print(f"    Allowlisted: {len(all_violations) - len(new_violations)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
