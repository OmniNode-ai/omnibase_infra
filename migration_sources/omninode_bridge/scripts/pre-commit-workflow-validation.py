#!/usr/bin/env python3
"""
Pre-commit hook for validating GitHub Actions workflows.

This script validates all workflow files in the repository using the
Pydantic CI models to ensure they are syntactically correct and follow
best practices.

Usage:
    python scripts/pre-commit-workflow-validation.py [files...]

Exit codes:
    0: All workflows are valid
    1: One or more workflows have validation errors
    2: Script execution error
"""

import argparse
import sys
from pathlib import Path

import yaml

# Secure package import - use proper Python package installation instead of path manipulation
# The package should be properly installed using 'pip install -e .' for development
# or via PYTHONPATH environment variable configuration

try:
    from omninode_bridge.cli.workflow_ci import WorkflowLinter, WorkflowValidator
except ImportError as e:
    print(f"❌ Failed to import workflow validation tools: {e}")
    print(
        "Make sure you're running this from the repository root with dependencies installed."
    )
    sys.exit(2)


class PreCommitWorkflowValidator:
    """Pre-commit hook for workflow validation."""

    def __init__(self, strict_mode: bool = False, enable_linting: bool = True):
        self.strict_mode = strict_mode
        self.enable_linting = enable_linting
        self.validator = WorkflowValidator(strict_mode=strict_mode)
        self.linter = WorkflowLinter() if enable_linting else None
        self.total_errors = 0
        self.total_warnings = 0

    def validate_files(self, file_paths: list[str]) -> bool:
        """Validate multiple workflow files."""
        workflow_files = self._filter_workflow_files(file_paths)

        if not workflow_files:
            print("INFO: No workflow files to validate")
            return True

        print(f"🔍 Validating {len(workflow_files)} workflow file(s)...")

        all_valid = True

        for file_path in workflow_files:
            if not self._validate_single_file(file_path):
                all_valid = False

        # Print summary
        if all_valid:
            print(f"✅ All {len(workflow_files)} workflow files are valid")
            if self.total_warnings > 0:
                print(f"⚠️  Found {self.total_warnings} warnings (not blocking)")
        else:
            print(
                f"❌ Found {self.total_errors} errors across {len(workflow_files)} files"
            )

        return all_valid

    def _filter_workflow_files(self, file_paths: list[str]) -> list[Path]:
        """Filter to only include workflow files."""
        workflow_files = []

        for file_path in file_paths:
            path = Path(file_path)

            # Check if it's a workflow file
            if (
                path.suffix in [".yml", ".yaml"]
                and (".github/workflows" in str(path) or "workflows" in path.parts)
            ) or (path.suffix in [".yml", ".yaml"] and self._looks_like_workflow(path)):
                workflow_files.append(path)

        return workflow_files

    def _looks_like_workflow(self, file_path: Path) -> bool:
        """Check if a YAML file looks like a GitHub Actions workflow."""
        try:
            with open(file_path) as f:
                data = yaml.safe_load(f)

            if not data:
                return False

            # Check for workflow-specific keys
            workflow_keys = ["name", "on", "jobs"]
            return all(key in data for key in workflow_keys)

        except Exception:
            return False

    def _validate_single_file(self, file_path: Path) -> bool:
        """Validate a single workflow file."""
        print(f"  📋 {file_path}")

        # Reset validator state
        self.validator.errors = []
        self.validator.warnings = []

        # Validate workflow structure
        is_valid, data = self.validator.validate_yaml_file(file_path)

        # Get validation report
        report = self.validator.get_validation_report()

        # Track totals
        self.total_errors += report["error_count"]
        self.total_warnings += report["warning_count"]

        # Print errors
        for error in report["errors"]:
            print(f"    ❌ {error}")

        # Print warnings (non-blocking)
        for warning in report["warnings"]:
            print(f"    ⚠️  {warning}")

        # Run linting if enabled
        if self.linter and is_valid:
            lint_result = self.linter.lint_file(file_path)
            lint_issues = lint_result.get("issues", [])

            # Categorize lint issues
            lint_errors = [i for i in lint_issues if i["severity"] == "error"]
            lint_warnings = [
                i for i in lint_issues if i["severity"] in ["warning", "suggestion"]
            ]

            # Print lint errors (blocking)
            for issue in lint_errors:
                line_info = f"Line {issue['line']}: " if "line" in issue else ""
                print(f"    ❌ {line_info}{issue['message']}")
                self.total_errors += 1
                is_valid = False

            # Print lint warnings (non-blocking)
            for issue in lint_warnings:
                line_info = f"Line {issue['line']}: " if "line" in issue else ""
                print(f"    ⚠️  {line_info}{issue['message']}")
                self.total_warnings += 1

        if is_valid and not report["errors"]:
            print("    ✅ Valid")

        return is_valid and len(report["errors"]) == 0

    def validate_workflow_directory(self, directory: Path = None) -> bool:
        """Validate all workflows in .github/workflows directory."""
        if directory is None:
            directory = Path(".github/workflows")

        if not directory.exists():
            print(f"INFO: No workflow directory found: {directory}")
            return True

        workflow_files = list(directory.glob("*.yml")) + list(directory.glob("*.yaml"))

        if not workflow_files:
            print(f"INFO: No workflow files found in {directory}")
            return True

        return self.validate_files([str(f) for f in workflow_files])


def create_sample_workflow_files():
    """Create sample workflow files for testing if none exist."""
    workflows_dir = Path(".github/workflows")

    if workflows_dir.exists() and list(workflows_dir.glob("*.yml")):
        return  # Already have workflows

    workflows_dir.mkdir(parents=True, exist_ok=True)

    # Create a basic CI workflow
    ci_workflow = {
        "name": "CI",
        "on": {
            "push": {"branches": ["main", "develop"]},
            "pull_request": {"branches": ["main"]},
        },
        "jobs": {
            "test": {
                "name": "Run Tests",
                "runs-on": "ubuntu-latest",
                "steps": [
                    {"name": "Checkout", "uses": "actions/checkout@v4"},
                    {
                        "name": "Setup Python",
                        "uses": "actions/setup-python@v5",
                        "with": {"python-version": "3.12"},
                    },
                    {
                        "name": "Install dependencies",
                        "run": "pip install -r requirements.txt",
                    },
                    {"name": "Run tests", "run": "pytest"},
                ],
            }
        },
    }

    with open(workflows_dir / "ci.yml", "w") as f:
        yaml.dump(ci_workflow, f, default_flow_style=False, sort_keys=False)

    print(f"✅ Created sample workflow: {workflows_dir / 'ci.yml'}")


def main():
    """Main entry point for pre-commit hook."""
    parser = argparse.ArgumentParser(
        description="Pre-commit hook for GitHub Actions workflow validation"
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Files to validate (if not provided, validates all workflows)",
    )
    parser.add_argument(
        "--strict", action="store_true", help="Enable strict validation mode"
    )
    parser.add_argument("--no-lint", action="store_true", help="Disable linting checks")
    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Create sample workflow files if none exist",
    )
    parser.add_argument(
        "--all-workflows",
        action="store_true",
        help="Validate all workflows in .github/workflows directory",
    )

    args = parser.parse_args()

    try:
        # Create sample workflows if requested
        if args.create_sample:
            create_sample_workflow_files()
            return 0

        # Initialize validator
        validator = PreCommitWorkflowValidator(
            strict_mode=args.strict, enable_linting=not args.no_lint
        )

        # Validate workflows
        if args.all_workflows:
            success = validator.validate_workflow_directory()
        elif args.files:
            success = validator.validate_files(args.files)
        else:
            # Default: validate all workflows in standard directory
            success = validator.validate_workflow_directory()

        return 0 if success else 1

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        return 2
    except Exception as e:
        print(f"❌ Script error: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
