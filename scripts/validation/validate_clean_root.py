#!/usr/bin/env python3
"""Root directory cleanliness validation for ONEX repositories.

This script validates that the project root directory contains ONLY allowed files
and directories. This enforces a clean, organized repository structure suitable
for public release.

PHILOSOPHY:
    The root directory of a repository is its "front door". It should contain
    ONLY essential files:
    - Configuration files (.gitignore, pyproject.toml, etc.)
    - Standard documentation (README.md, LICENSE, CONTRIBUTING.md, etc.)
    - Required directories (src/, tests/, docs/, scripts/, etc.)

    Working documents, development notes, and implementation details should live
    in the docs/ directory, NOT in the project root.

Usage:
    python scripts/validation/validate_clean_root.py
    python scripts/validation/validate_clean_root.py --fix
    python scripts/validation/validate_clean_root.py --verbose
    python scripts/validation/validate_clean_root.py /path/to/repo

Exit Codes:
    0 - Root directory is clean
    1 - Violations found (unexpected files in root)
    2 - Script error

Portability:
    This script auto-detects the project root from its location.
    It works with any ONEX repository - no configuration needed.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# =============================================================================
# Root Directory Allowlist
# =============================================================================
# These are the ONLY files and directories allowed in the project root.
# Everything else is a violation.

ALLOWED_ROOT_FILES: frozenset[str] = frozenset(
    {
        # Version control
        ".gitignore",
        # NOTE: `.env.*` variants (e.g. .env.local, .env.staging) are intentionally
        # NOT in this allowlist. The Infisical provisioning plan requires all non-example
        # env files to live in ~/.omnibase/ (shared) or be managed by Infisical at
        # runtime. Any such file in the repo root is a misconfiguration — it should
        # either be a committed .env.example template or sourced from ~/.omnibase/.env.
        ".gitattributes",
        ".gitmodules",
        # Python packaging (required)
        "pyproject.toml",
        "uv.lock",
        "setup.py",
        "setup.cfg",
        "MANIFEST.in",
        # Type checking
        "mypy.ini",
        "pyrightconfig.json",
        ".mypy.ini",
        "py.typed",
        # Linting/formatting
        ".pre-commit-config.yaml",
        ".yamlfmt",
        ".yamllint.yaml",
        ".ruff.toml",
        ".editorconfig",
        ".flake8",
        ".pylintrc",
        ".isort.cfg",
        # Markdown link validation
        ".markdown-link-check.json",
        # Standard documentation
        "README.md",
        "README.rst",
        "LICENSE",
        "LICENSE.md",
        "LICENSE.txt",
        "CHANGELOG.md",
        "CHANGELOG.rst",
        "CONTRIBUTING.md",
        "CONTRIBUTING.rst",
        "CODE_OF_CONDUCT.md",
        "SECURITY.md",
        # ONEX-specific documentation
        "CLAUDE.md",
        # Environment
        ".env.example",
        # Docker (if needed at root)
        "Dockerfile",
        "docker-compose.yml",
        "docker-compose.yaml",
        ".dockerignore",
        # Makefile
        "Makefile",
        # CI/CD configuration (some systems require root placement)
        ".travis.yml",
        "tox.ini",
        "noxfile.py",
        "Taskfile.yml",
        "justfile",
        # Security
        ".secretresolver_allowlist",
        ".secrets.baseline",
        # Special files
        ".migration_freeze",
        ".mcp.json",
        ".tool-versions",
        ".python-version",
        ".nvmrc",
        ".node-version",
    }
)

ALLOWED_ROOT_DIRECTORIES: frozenset[str] = frozenset(
    {
        # Required directories
        "src",
        "tests",
        "docs",
        "scripts",
        # Common optional directories
        "config",
        "contracts",
        "docker",
        "examples",
        "benchmarks",
        "tools",
        "bin",
        # Hidden directories (generally allowed)
        ".git",
        ".github",
        ".gitlab",
        ".circleci",
        ".vscode",
        ".idea",
        ".claude",
        # Cache directories (should be gitignored)
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".hypothesis",
        ".tox",
        ".nox",
        "__pycache__",
        # Virtual environments (should be gitignored)
        ".venv",
        "venv",
        "env",
        # Build directories (should be gitignored)
        "build",
        "dist",
        "tmp",
        "htmlcov",
        "coverage",
        ".coverage",
        # Test results (should be gitignored)
        "test_split_results",
    }
)

# Pattern-based allowlist for files that match patterns
ALLOWED_ROOT_PATTERNS: tuple[str, ...] = (
    "*.egg-info",  # Build artifacts (should be gitignored)
)


@dataclass
class RootViolation:
    """Represents a root directory violation."""

    path: Path
    suggestion: str
    is_directory: bool = False

    def __str__(self) -> str:
        item_type = "Directory" if self.is_directory else "File"
        return f"  {item_type}: {self.path.name}\n    → {self.suggestion}"


@dataclass
class ValidationResult:
    """Result of root directory validation."""

    violations: list[RootViolation] = field(default_factory=list)
    checked_items: int = 0

    @property
    def is_valid(self) -> bool:
        return len(self.violations) == 0

    def __bool__(self) -> bool:
        """Returns True if validation passed (no violations)."""
        return self.is_valid


def _matches_pattern(name: str, patterns: tuple[str, ...]) -> bool:
    """Check if a name matches any of the given glob patterns."""
    import fnmatch

    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def _get_gitignored_set(items: list[Path]) -> set[Path]:
    """Return the set of paths from *items* that are gitignored.

    Uses a single ``git check-ignore`` invocation for all paths instead of
    one subprocess per path, which keeps validation fast even in large repos.

    Gitignored files (e.g. .env, __pycache__, .venv) are expected to exist on
    the developer's machine but must not be committed. The validator should not
    flag them as violations — only files that ARE committed (not ignored) matter.
    """
    if not items:
        return set()
    try:
        result = subprocess.run(
            ["git", "check-ignore", "--", *[str(p) for p in items]],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            cwd=items[0].parent.resolve(),
        )
        # git check-ignore prints one ignored path per line.  Because we pass
        # absolute paths, the output lines are also absolute paths — so we
        # match against str(p) directly.
        ignored_paths: set[str] = {
            line.strip()
            for line in result.stdout.decode("utf-8", errors="replace").splitlines()
            if line.strip()
        }
        return {p for p in items if str(p) in ignored_paths}
    except FileNotFoundError:
        # git not available — treat nothing as ignored
        return set()


def _suggest_action(item: Path) -> str:
    """Generate a suggestion for what to do with a misplaced item."""
    name = item.name.lower()

    # Markdown files that look like documentation
    if item.suffix.lower() == ".md":
        if any(
            keyword in name
            for keyword in [
                "plan",
                "execution",
                "summary",
                "enhancement",
                "improvement",
                "fix",
                "audit",
                "wiring",
                "error",
                "handling",
                "log",
                "test",
                "hook",
                "migration",
                "design",
                "architecture",
                "mvp",
                "todo",
                "notes",
            ]
        ):
            return "Move to docs/ or delete if no longer relevant"
        return "Move to docs/ or rename to follow standard conventions"

    # Build/test artifacts
    if any(
        keyword in name
        for keyword in ["coverage", "report", "audit", "log", ".tmp", ".bak"]
    ):
        return "Delete (build/test artifact) or add to .gitignore"

    # Unknown files
    if item.is_file():
        return "Move to appropriate directory (src/, docs/, scripts/) or delete"

    # Unknown directories
    return "Move contents to appropriate location or add to ALLOWED_ROOT_DIRECTORIES"


def validate_root_directory(
    repo_path: Path,
    verbose: bool = False,
) -> ValidationResult:
    """
    Validate that the project root contains only allowed files and directories.

    Args:
        repo_path: Path to the repository root
        verbose: Enable verbose output

    Returns:
        ValidationResult with any violations found
    """
    result = ValidationResult()

    if not repo_path.exists():
        raise FileNotFoundError(f"Repository path does not exist: {repo_path}")

    if not repo_path.is_dir():
        raise ValueError(f"Repository path is not a directory: {repo_path}")

    # Collect all root items first so we can batch the gitignore check.
    all_items = sorted(repo_path.iterdir())

    # Items that pass the allowlist checks immediately — no git call needed.
    unresolved: list[Path] = []
    for item in all_items:
        result.checked_items += 1
        name = item.name

        # Skip hidden files/directories that are common
        if name.startswith(".") and (
            name in ALLOWED_ROOT_FILES or name in ALLOWED_ROOT_DIRECTORIES
        ):
            if verbose:
                print(f"  ✓ {name} (allowed)")
            continue

        # Check if it's an allowed file
        if item.is_file():
            if name in ALLOWED_ROOT_FILES:
                if verbose:
                    print(f"  ✓ {name} (allowed file)")
                continue
            if _matches_pattern(name, ALLOWED_ROOT_PATTERNS):
                if verbose:
                    print(f"  ✓ {name} (matches allowed pattern)")
                continue

        # Check if it's an allowed directory
        if item.is_dir():
            if name in ALLOWED_ROOT_DIRECTORIES:
                if verbose:
                    print(f"  ✓ {name}/ (allowed directory)")
                continue

        # Not on the explicit allowlist — defer to a single batched git call.
        unresolved.append(item)

    # Batch all gitignore checks into ONE subprocess call (N paths → 1 spawn).
    # Gitignored files are expected to exist locally (e.g. .env, build dirs)
    # but must not be committed. Skip them — they are not a repo violation.
    gitignored = _get_gitignored_set(unresolved)

    for item in unresolved:
        if item in gitignored:
            if verbose:
                print(f"  ~ {item.name} (gitignored, skipped)")
            continue

        # This is a violation
        result.violations.append(
            RootViolation(
                path=item,
                suggestion=_suggest_action(item),
                is_directory=item.is_dir(),
            )
        )

    return result


def generate_report(result: ValidationResult, repo_path: Path) -> str:
    """Generate a validation report."""
    if result.is_valid:
        return f"✅ Root directory is clean ({result.checked_items} items checked)"

    report_lines = [
        "❌ ROOT DIRECTORY VALIDATION FAILED",
        "=" * 60,
        "",
        f"Found {len(result.violations)} item(s) that should not be in the project root:",
        "",
    ]

    for violation in result.violations:
        report_lines.append(str(violation))
        report_lines.append("")

    report_lines.extend(
        [
            "=" * 60,
            "WHY THIS MATTERS:",
            "  • The root directory is the repository's 'front door'",
            "  • It should contain ONLY essential configuration and documentation",
            "  • Working documents and notes belong in docs/",
            "  • Build artifacts should be in .gitignore",
            "",
            "HOW TO FIX:",
            "  1. Review each file/directory above",
            "  2. Move documentation to docs/",
            "  3. Delete obsolete files",
            "  4. Add build artifacts to .gitignore",
            "",
            "To add a new allowed file/directory, edit:",
            f"  {repo_path / 'scripts/validation/validate_clean_root.py'}",
            "",
        ]
    )

    return "\n".join(report_lines)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate that project root contains only allowed files/directories"
    )
    parser.add_argument(
        "repo_path",
        nargs="?",
        default=None,
        help="Path to repository root (default: auto-detect from script location)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show all checked items, not just violations",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Show commands to fix violations (does not execute them)",
    )

    args = parser.parse_args()

    # Auto-detect repo path if not provided
    if args.repo_path:
        repo_path = Path(args.repo_path).resolve()
    else:
        # Script is in scripts/validation/, so repo root is ../..
        script_path = Path(__file__).resolve()
        repo_path = script_path.parent.parent.parent

    try:
        if args.verbose:
            print(f"🔍 Validating root directory: {repo_path}")
            print("")

        result = validate_root_directory(repo_path, verbose=args.verbose)
        report = generate_report(result, repo_path)
        print(report)

        if args.fix and not result.is_valid:
            print("SUGGESTED COMMANDS TO FIX:")
            print("-" * 40)
            for violation in result.violations:
                if violation.is_directory:
                    print("# Review and remove directory:")
                    print(f"rm -rf {violation.path}")
                elif "docs/" in violation.suggestion:
                    print("# Move to docs/:")
                    print(f"mv {violation.path} {repo_path / 'docs/'}")
                else:
                    print("# Delete file:")
                    print(f"rm {violation.path}")
                print()

        return 0 if result.is_valid else 1

    except (FileNotFoundError, ValueError) as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
