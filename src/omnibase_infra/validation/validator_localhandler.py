# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""LocalHandler Import Validator for ONEX Production Code Policy.

This module provides validation to detect forbidden LocalHandler imports
in production code. LocalHandler is a test-only handler that must NEVER
be imported in src/omnibase_infra/.

Policy:
    - BLOCKED: Any import of LocalHandler in src/omnibase_infra/
    - ALLOWED: Imports in tests/ directory for comparison tests

The validator uses regex-based line scanning to detect import patterns.

Import Patterns Detected:
    - ``from omnibase_core.handlers import LocalHandler``
    - ``from omnibase_core.handlers.handler_local import LocalHandler``
    - ``from omnibase_core.handlers.local_handler import LocalHandler``
    - ``import omnibase_core.handlers.LocalHandler``
    - Any other pattern containing 'LocalHandler' in an import statement

Usage:
    >>> from omnibase_infra.validation.validator_localhandler import (
    ...     validate_localhandler,
    ...     validate_localhandler_ci,
    ... )
    >>> violations = validate_localhandler(Path("src/omnibase_infra"))
    >>> result = validate_localhandler_ci(Path("src/omnibase_infra"))
    >>> if not result.passed:
    ...     print(result.format_summary())
    ...     sys.exit(1)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from omnibase_infra.models.validation.model_localhandler_validation_result import (
    ModelLocalHandlerValidationResult,
)
from omnibase_infra.models.validation.model_localhandler_violation import (
    ModelLocalHandlerViolation,
)

logger = logging.getLogger(__name__)

# Regex patterns to detect LocalHandler imports
# Pattern 1: from ... import LocalHandler (with optional alias)
_PATTERN_FROM_IMPORT = re.compile(
    r"^\s*from\s+[\w.]+\s+import\s+.*\bLocalHandler\b",
    re.MULTILINE,
)

# Pattern 2: import statement with LocalHandler in path
_PATTERN_IMPORT = re.compile(
    r"^\s*import\s+[\w.]*LocalHandler",
    re.MULTILINE,
)

# Combined pattern for line-by-line checking (more precise)
_PATTERN_LOCALHANDLER_IMPORT = re.compile(
    r"^\s*(?:from\s+[\w.]+\s+import\s+.*\bLocalHandler\b|import\s+[\w.]*LocalHandler)"
)

# Maximum file size to process (in bytes).
# Files larger than this are skipped to prevent hangs on auto-generated code.
_MAX_FILE_SIZE_BYTES: int = 1_000_000  # 1MB

# Directories to skip (exact name matching)
_SKIP_DIRECTORIES: frozenset[str] = frozenset(
    {
        "tests",  # Test files are allowed to use LocalHandler
        "__pycache__",  # Python bytecode cache
        ".git",  # Git directory
        ".venv",  # Virtual environment
        "venv",  # Virtual environment
        "node_modules",  # Node modules (unlikely but safe)
    }
)


def _should_skip_file(filepath: Path) -> bool:
    """Check if a file should be skipped based on directory matching.

    Uses exact parent directory matching to prevent false positives.

    Args:
        filepath: Path to check.

    Returns:
        True if the file should be skipped.
    """
    parts = filepath.parts

    # Check parent directories for exact matches (exclude filename)
    for part in parts[:-1]:
        if part in _SKIP_DIRECTORIES:
            return True

    # Skip files starting with underscore (private modules, test fixtures)
    if filepath.name.startswith("_"):
        return True

    return False


def validate_localhandler_in_file(filepath: Path) -> list[ModelLocalHandlerViolation]:
    """Validate a single Python file for LocalHandler import violations.

    Args:
        filepath: Path to the Python file to validate.

    Returns:
        List of detected violations. Empty list if no violations found.
    """
    try:
        content = filepath.read_text(encoding="utf-8")
    except OSError as e:
        logger.warning(
            "Failed to read file",
            extra={"file": str(filepath), "error": str(e)},
        )
        return []

    violations: list[ModelLocalHandlerViolation] = []
    lines = content.split("\n")

    for line_number, line in enumerate(lines, start=1):
        # Skip comment lines
        stripped = line.strip()
        if stripped.startswith("#"):
            continue

        # Check if line contains LocalHandler import
        if _PATTERN_LOCALHANDLER_IMPORT.match(stripped):
            violations.append(
                ModelLocalHandlerViolation(
                    file_path=filepath,
                    line_number=line_number,
                    import_line=stripped,
                )
            )

    return violations


def validate_localhandler(
    directory: Path,
    recursive: bool = True,
) -> list[ModelLocalHandlerViolation]:
    """Validate all Python files in a directory for LocalHandler import violations.

    This is the main entry point for batch validation.

    Args:
        directory: Path to the directory to validate.
        recursive: If True, recursively validate subdirectories.

    Returns:
        List of all detected violations across all files.

    Example:
        >>> violations = validate_localhandler(Path("src/omnibase_infra"))
        >>> for v in violations:
        ...     print(v.format_human_readable())
    """
    violations: list[ModelLocalHandlerViolation] = []
    pattern = "**/*.py" if recursive else "*.py"

    for filepath in directory.glob(pattern):
        if filepath.is_file() and not _should_skip_file(filepath):
            # Skip very large files
            try:
                file_size = filepath.stat().st_size
                if file_size > _MAX_FILE_SIZE_BYTES:
                    logger.debug(
                        "Skipping large file",
                        extra={"file": str(filepath), "size_bytes": file_size},
                    )
                    continue
            except OSError as e:
                logger.warning(
                    "Failed to stat file",
                    extra={"file": str(filepath), "error": str(e)},
                )
                continue

            try:
                file_violations = validate_localhandler_in_file(filepath)
                violations.extend(file_violations)
            except Exception as e:  # catch-all-ok: validation continues on file errors
                logger.warning(
                    "Failed to validate file",
                    extra={
                        "file": str(filepath),
                        "error_type": type(e).__name__,
                        "error": str(e),
                    },
                )
                continue

    return violations


def validate_localhandler_ci(
    directory: Path,
    recursive: bool = True,
) -> ModelLocalHandlerValidationResult:
    """CI gate for LocalHandler validation.

    This function is designed for CI pipeline integration. It returns a
    structured result model containing the pass/fail status and all violations
    for reporting.

    Args:
        directory: Path to the directory to validate.
        recursive: If True, recursively validate subdirectories.

    Returns:
        ModelLocalHandlerValidationResult containing pass/fail status and violations.

    Example:
        >>> result = validate_localhandler_ci(Path("src/omnibase_infra"))
        >>> if not result.passed:
        ...     print(result.format_summary())
        ...     for line in result.format_for_ci():
        ...         print(line)
        ...     sys.exit(1)
    """
    # Count files (excluding skipped patterns and large files)
    pattern = "**/*.py" if recursive else "*.py"
    files_checked = 0

    for f in directory.glob(pattern):
        if f.is_file() and not _should_skip_file(f):
            try:
                if f.stat().st_size <= _MAX_FILE_SIZE_BYTES:
                    files_checked += 1
            except OSError:
                continue

    violations = validate_localhandler(directory, recursive=recursive)
    return ModelLocalHandlerValidationResult.from_violations(violations, files_checked)


def _print_fix_instructions() -> None:
    """Print instructions for fixing LocalHandler violations."""
    print("\nHow to fix LocalHandler violations:")
    print("   1. Remove the LocalHandler import from production code")
    print("   2. LocalHandler is ONLY for test code (tests/ directory)")
    print("   3. For production handlers, use appropriate protocol implementations")
    print("")
    print("   Why LocalHandler is test-only:")
    print("   - LocalHandler is a simple echo handler for testing")
    print("   - It bypasses production handler resolution")
    print("   - Production code must use proper handler types")


__all__ = [
    "ModelLocalHandlerValidationResult",
    "ModelLocalHandlerViolation",
    "validate_localhandler",
    "validate_localhandler_ci",
    "validate_localhandler_in_file",
]
