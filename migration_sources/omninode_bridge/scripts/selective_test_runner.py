#!/usr/bin/env python3
"""
Selective Test Runner for Git Hooks
Optimized to run only tests relevant to changed files for faster feedback.
"""

import argparse
import fnmatch
import subprocess
import sys
from pathlib import Path

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class SelectiveTestRunner:
    """Runs tests selectively based on changed files."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.project_root = Path(__file__).parent.parent

        # Map file patterns to test categories
        self.test_patterns = {
            "unit": {
                "src/omninode_bridge/models/**/*.py": ["tests/unit/models/"],
                "src/omninode_bridge/services/**/*.py": ["tests/unit/services/"],
                "src/omninode_bridge/ci/**/*.py": ["tests/unit/ci/"],
                "src/omninode_bridge/utils/**/*.py": ["tests/unit/utils/"],
                "src/omninode_bridge/**/*.py": ["tests/unit/"],
            },
            "integration": {
                "src/omninode_bridge/services/**/*.py": ["tests/integration/"],
                "src/omninode_bridge/models/hooks.py": ["tests/integration/"],
                "docker-compose*.yml": ["tests/integration/"],
                "Dockerfile*": ["tests/integration/"],
            },
            "security": {
                "src/omninode_bridge/services/**/*.py": ["tests/test_security*.py"],
                "src/omninode_bridge/models/security.py": ["tests/test_security*.py"],
                "src/omninode_bridge/**/*.py": ["tests/test_security*.py"],
            },
            "ci": {
                "src/omninode_bridge/ci/**/*.py": [
                    "tests/unit/ci/",
                    "tests/integration/",
                ],
                ".github/workflows/**/*.yml": ["tests/unit/ci/"],
                ".pre-commit-config.yaml": ["tests/unit/ci/"],
            },
        }

    def get_changed_files(self, staged_only: bool = True) -> list[str]:
        """Get list of changed files from git."""
        try:
            if staged_only:
                # Get staged files for pre-commit
                cmd = ["git", "diff", "--cached", "--name-only"]
            else:
                # Get all changed files for pre-push
                cmd = ["git", "diff", "HEAD~1", "--name-only"]

            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=self.project_root
            )

            if result.returncode != 0:
                if self.verbose:
                    print(f"Warning: git command failed: {result.stderr}")
                return []

            files = [f.strip() for f in result.stdout.split("\n") if f.strip()]
            return files

        except Exception as e:
            if self.verbose:
                print(f"Error getting changed files: {e}")
            return []

    def match_pattern(self, file_path: str, pattern: str) -> bool:
        """Check if file matches a pattern (supports ** and * wildcards)."""
        # Convert glob pattern to match file path
        if "**" in pattern:
            # Handle ** patterns
            parts = pattern.split("**")
            if len(parts) == 2:
                start, end = parts
                return file_path.startswith(start) and file_path.endswith(
                    end.lstrip("/")
                )

        return fnmatch.fnmatch(file_path, pattern)

    def determine_test_categories(self, changed_files: list[str]) -> set[str]:
        """Determine which test categories to run based on changed files."""
        categories = set()

        for file_path in changed_files:
            for category, patterns in self.test_patterns.items():
                for pattern in patterns:
                    if self.match_pattern(file_path, pattern):
                        categories.add(category)
                        if self.verbose:
                            print(
                                f"File {file_path} matches {category} pattern {pattern}"
                            )

        # Always run unit tests for Python changes
        if any(f.endswith(".py") and f.startswith("src/") for f in changed_files):
            categories.add("unit")

        return categories

    def get_specific_test_files(self, changed_files: list[str]) -> list[str]:
        """Get specific test files that should run for changed files."""
        test_files = set()

        for file_path in changed_files:
            for category, patterns in self.test_patterns.items():
                for pattern, test_paths in patterns.items():
                    if self.match_pattern(file_path, pattern):
                        for test_path in test_paths:
                            if Path(self.project_root / test_path).exists():
                                test_files.add(test_path)

        return list(test_files)

    def run_quick_import_test(self) -> bool:
        """Run quick import test to verify basic functionality."""
        try:
            cmd = [
                "poetry",
                "run",
                "python",
                "-c",
                'import omninode_bridge; print("✅ Core imports successful")',
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=self.project_root
            )

            if result.returncode == 0:
                if self.verbose:
                    print("✅ Quick import test passed")
                return True
            else:
                print(f"❌ Quick import test failed: {result.stderr}")
                return False

        except Exception as e:
            print(f"❌ Error running import test: {e}")
            return False

    def run_selective_tests(self, categories: set[str], max_time: int = 300) -> bool:
        """Run tests for specific categories with timeout."""
        if not categories:
            if self.verbose:
                print("[INFO] No test categories determined, skipping tests")
            return True

        success = True

        for category in sorted(categories):
            if not self._run_category_tests(category, max_time):
                success = False

        return success

    def _run_category_tests(self, category: str, max_time: int) -> bool:
        """Run tests for a specific category."""
        try:
            # Define test commands for each category
            test_commands = {
                "unit": [
                    "poetry",
                    "run",
                    "pytest",
                    "tests/unit/",
                    "-v",
                    "--tb=short",
                    f"--timeout={max_time}",
                    "--maxfail=5",
                ],
                "integration": [
                    "poetry",
                    "run",
                    "pytest",
                    "tests/integration/",
                    "-v",
                    "--tb=short",
                    f"--timeout={max_time}",
                    "--maxfail=3",
                ],
                "security": [
                    "poetry",
                    "run",
                    "pytest",
                    "tests/test_security*.py",
                    "-v",
                    "--tb=short",
                    f"--timeout={max_time}",
                    "--maxfail=1",
                ],
                "ci": [
                    "poetry",
                    "run",
                    "pytest",
                    "tests/unit/ci/",
                    "-v",
                    "--tb=short",
                    f"--timeout={max_time}",
                    "--maxfail=3",
                ],
            }

            if category not in test_commands:
                if self.verbose:
                    print(f"⚠️  Unknown test category: {category}")
                return True

            cmd = test_commands[category]

            if self.verbose:
                print(f"🔍 Running {category} tests...")
                print(f"Command: {' '.join(cmd)}")

            result = subprocess.run(cmd, cwd=self.project_root, timeout=max_time)

            if result.returncode == 0:
                if self.verbose:
                    print(f"✅ {category} tests passed")
                return True
            else:
                print(f"❌ {category} tests failed")
                return False

        except subprocess.TimeoutExpired:
            print(f"⏰ {category} tests timed out after {max_time}s")
            return False
        except Exception as e:
            print(f"❌ Error running {category} tests: {e}")
            return False

    def run_formatting_checks(self, files: list[str] = None) -> bool:
        """Run formatting checks on specified files or all files."""
        checks = [
            (
                ["poetry", "run", "black", "--check"] + (files or ["."]),
                "Black formatting",
            ),
            (
                ["poetry", "run", "isort", "--check-only"] + (files or ["."]),
                "isort import sorting",
            ),
            (["poetry", "run", "ruff", "check"] + (files or ["."]), "Ruff linting"),
        ]

        success = True

        for cmd, name in checks:
            try:
                if self.verbose:
                    print(f"🔍 Running {name}...")

                result = subprocess.run(
                    cmd, capture_output=True, text=True, cwd=self.project_root
                )

                if result.returncode == 0:
                    if self.verbose:
                        print(f"✅ {name} passed")
                else:
                    print(f"❌ {name} failed")
                    if result.stdout:
                        print(result.stdout)
                    if result.stderr:
                        print(result.stderr)
                    success = False

            except Exception as e:
                print(f"❌ Error running {name}: {e}")
                success = False

        return success


def main():
    """Main entry point for selective test runner."""
    parser = argparse.ArgumentParser(description="Selective test runner for git hooks")
    parser.add_argument(
        "--staged",
        action="store_true",
        help="Run tests for staged files (pre-commit mode)",
    )
    parser.add_argument(
        "--push", action="store_true", help="Run tests for push (pre-push mode)"
    )
    parser.add_argument(
        "--format-only", action="store_true", help="Run only formatting checks"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run only quick import tests"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--timeout", type=int, default=300, help="Timeout for test execution (seconds)"
    )
    parser.add_argument(
        "files", nargs="*", help="Specific files to check (for formatting)"
    )

    args = parser.parse_args()

    runner = SelectiveTestRunner(verbose=args.verbose)

    try:
        # Quick mode - just import test
        if args.quick:
            success = runner.run_quick_import_test()
            sys.exit(0 if success else 1)

        # Format-only mode
        if args.format_only:
            success = runner.run_formatting_checks(args.files)
            sys.exit(0 if success else 1)

        # Determine which files changed
        if args.push:
            changed_files = runner.get_changed_files(staged_only=False)
            if args.verbose:
                print(f"📁 Changed files for push: {changed_files}")
        elif args.staged:
            changed_files = runner.get_changed_files(staged_only=True)
            if args.verbose:
                print(f"📁 Staged files: {changed_files}")
        else:
            # Default to staged files
            changed_files = runner.get_changed_files(staged_only=True)
            if args.verbose:
                print(f"📁 Changed files: {changed_files}")

        if not changed_files:
            if args.verbose:
                print("[INFO] No changed files detected")
            # Still run import test
            success = runner.run_quick_import_test()
            sys.exit(0 if success else 1)

        # Run formatting checks first
        if not runner.run_formatting_checks():
            print("❌ Formatting checks failed")
            sys.exit(1)

        # Run import test
        if not runner.run_quick_import_test():
            print("❌ Import test failed")
            sys.exit(1)

        # Determine test categories
        test_categories = runner.determine_test_categories(changed_files)

        if args.verbose:
            print(f"🎯 Test categories to run: {sorted(test_categories)}")

        # Run selective tests
        if test_categories:
            success = runner.run_selective_tests(test_categories, args.timeout)
            if not success:
                print("❌ Some tests failed")
                sys.exit(1)

        print("✅ All checks passed!")
        sys.exit(0)

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        sys.exit(2)
    except Exception as e:
        print(f"❌ Script error: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
