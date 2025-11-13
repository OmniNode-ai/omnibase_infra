#!/usr/bin/env python3
"""
CI Workflow Management CLI Tool

Comprehensive command-line interface for managing, validating, and testing
GitHub Actions workflows using Pydantic models.

Features:
- Workflow validation and linting
- YAML generation from Pydantic models
- Schema validation against GitHub Actions
- Testing and dry-run capabilities
- Workflow templates and examples

Usage:
    python -m omninode_bridge.cli.workflow_ci validate workflow.yml
    python -m omninode_bridge.cli.workflow_ci generate --template ci-python
    python -m omninode_bridge.cli.workflow_ci test workflow.yml
    python -m omninode_bridge.cli.workflow_ci lint workflows/
"""

import argparse
import asyncio
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, ClassVar

import yaml

from ..ci.models.workflow import (
    EventType,
    MatrixStrategy,
    PermissionLevel,
    PermissionSet,
    WorkflowConfig,
    WorkflowJob,
    WorkflowStep,
)

logger = logging.getLogger(__name__)


class WorkflowValidator:
    """Comprehensive workflow validation engine."""

    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def validate_yaml_file(self, file_path: Path) -> tuple[bool, dict[str, Any]]:
        """Validate a YAML workflow file."""
        try:
            with open(file_path) as f:
                data = yaml.safe_load(f)

            if not data:
                self.errors.append("Empty workflow file")
                return False, {}

            return self._validate_workflow_structure(data)

        except yaml.YAMLError as e:
            self.errors.append(f"Invalid YAML syntax: {e}")
            return False, {}
        except FileNotFoundError:
            self.errors.append(f"File not found: {file_path}")
            return False, {}
        except Exception as e:
            self.errors.append(f"Unexpected error: {e}")
            return False, {}

    def validate_pydantic_workflow(
        self, workflow: WorkflowConfig
    ) -> tuple[bool, dict[str, Any]]:
        """Validate a Pydantic workflow model."""
        try:
            # Convert to dict and validate structure
            yaml_dict = workflow.to_yaml_dict()
            return self._validate_workflow_structure(yaml_dict)

        except Exception as e:
            self.errors.append(f"Pydantic model error: {e}")
            return False, {}

    def _validate_workflow_structure(
        self, data: dict[str, Any]
    ) -> tuple[bool, dict[str, Any]]:
        """Validate workflow structure against GitHub Actions schema."""
        is_valid = True

        # Required top-level fields
        required_fields = ["name", "on", "jobs"]
        for field in required_fields:
            if field not in data:
                self.errors.append(f"Missing required field: {field}")
                is_valid = False

        # Validate workflow name
        if "name" in data and not isinstance(data["name"], str):
            self.errors.append("Workflow name must be a string")
            is_valid = False

        # Validate triggers
        if "on" in data:
            is_valid &= self._validate_triggers(data["on"])

        # Validate jobs
        if "jobs" in data:
            if not data["jobs"]:
                self.errors.append("Workflow must have at least one job")
                is_valid = False
            else:
                is_valid &= self._validate_jobs(data["jobs"])

        # Validate permissions
        if "permissions" in data:
            is_valid &= self._validate_permissions(data["permissions"])

        # Validate environment variables
        if "env" in data:
            is_valid &= self._validate_env_vars(data["env"])

        return is_valid, data

    def _validate_triggers(self, triggers: Any) -> bool:
        """Validate workflow triggers."""
        valid_events = [
            "push",
            "pull_request",
            "issue_comment",
            "pull_request_review",
            "pull_request_review_comment",
            "issues",
            "schedule",
            "workflow_dispatch",
            "workflow_run",
            "workflow_call",
            "repository_dispatch",
            "release",
            "create",
            "delete",
            "fork",
            "gollum",
            "page_build",
            "public",
            "watch",
            "discussion",
            "discussion_comment",
        ]

        if isinstance(triggers, str):
            if triggers not in valid_events:
                self.errors.append(f"Invalid trigger event: {triggers}")
                return False

        elif isinstance(triggers, list):
            for trigger in triggers:
                if trigger not in valid_events:
                    self.errors.append(f"Invalid trigger event: {trigger}")
                    return False

        elif isinstance(triggers, dict):
            for event, config in triggers.items():
                if event not in valid_events:
                    self.errors.append(f"Invalid trigger event: {event}")
                    return False

                # Validate event-specific configuration
                if event == "schedule" and isinstance(config, list):
                    for schedule in config:
                        if "cron" not in schedule:
                            self.errors.append(
                                "Schedule trigger must have 'cron' field"
                            )
                            return False
                        if not self._validate_cron(schedule["cron"]):
                            return False

        else:
            self.errors.append("Invalid trigger format")
            return False

        return True

    def _validate_cron(self, cron: str) -> bool:
        """Validate cron expression."""
        parts = cron.split()
        if len(parts) != 5:
            self.errors.append(f"Invalid cron expression: {cron} (must have 5 parts)")
            return False

        # Basic cron validation (could be more sophisticated)
        for i, part in enumerate(parts):
            if not (
                part.isdigit()
                or part == "*"
                or "/" in part
                or "-" in part
                or "," in part
            ):
                self.warnings.append(f"Potentially invalid cron part: {part}")

        return True

    def _validate_jobs(self, jobs: dict[str, Any]) -> bool:
        """Validate workflow jobs."""
        is_valid = True

        for job_name, job_data in jobs.items():
            # Validate job name
            if not job_name.replace("_", "").replace("-", "").isalnum():
                self.warnings.append(
                    f"Job name '{job_name}' should only contain alphanumeric characters, hyphens, and underscores"
                )

            # Required job fields
            if "runs-on" not in job_data:
                self.errors.append(f"Job '{job_name}' missing required 'runs-on' field")
                is_valid = False

            if "steps" not in job_data:
                self.errors.append(f"Job '{job_name}' missing required 'steps' field")
                is_valid = False
            elif not job_data["steps"]:
                self.errors.append(f"Job '{job_name}' must have at least one step")
                is_valid = False
            else:
                is_valid &= self._validate_steps(job_data["steps"], job_name)

            # Validate runner
            if "runs-on" in job_data:
                is_valid &= self._validate_runner(job_data["runs-on"], job_name)

            # Validate matrix strategy
            if "strategy" in job_data and "matrix" in job_data["strategy"]:
                is_valid &= self._validate_matrix(job_data["strategy"], job_name)

            # Validate needs
            if "needs" in job_data:
                is_valid &= self._validate_needs(job_data["needs"], job_name, jobs)

        return is_valid

    def _validate_steps(self, steps: list[dict[str, Any]], job_name: str) -> bool:
        """Validate job steps."""
        is_valid = True

        for i, step in enumerate(steps):
            step_ref = f"Job '{job_name}', step {i+1}"

            # Required step name
            if "name" not in step:
                self.errors.append(f"{step_ref}: missing required 'name' field")
                is_valid = False

            # Must have either 'run' or 'uses'
            has_run = "run" in step
            has_uses = "uses" in step

            if not has_run and not has_uses:
                self.errors.append(
                    f"{step_ref}: must have either 'run' or 'uses' field"
                )
                is_valid = False

            if has_run and has_uses:
                self.warnings.append(
                    f"{step_ref}: has both 'run' and 'uses' fields (uses will take precedence)"
                )

            # Validate action references
            if has_uses:
                is_valid &= self._validate_action_reference(step["uses"], step_ref)

            # Validate step ID
            if "id" in step:
                step_id = step["id"]
                if not step_id.replace("_", "").replace("-", "").isalnum():
                    self.warnings.append(
                        f"{step_ref}: step ID should only contain alphanumeric characters, hyphens, and underscores"
                    )

        return is_valid

    def _validate_action_reference(self, action_ref: str, step_ref: str) -> bool:
        """Validate action reference format."""
        # Action format: owner/repo@ref or ./path/to/action
        if action_ref.startswith("./"):
            # Local action
            return True

        if "@" not in action_ref:
            self.errors.append(
                f"{step_ref}: action reference '{action_ref}' must include version (@ref)"
            )
            return False

        parts = action_ref.split("@")
        if len(parts) != 2:
            self.errors.append(
                f"{step_ref}: invalid action reference format: {action_ref}"
            )
            return False

        action_name, version = parts
        if "/" not in action_name:
            self.warnings.append(
                f"{step_ref}: action reference '{action_name}' should include owner/repo format"
            )

        # Check for common security issues
        if version in ["master", "main"] and not self.strict_mode:
            self.warnings.append(
                f"{step_ref}: using branch reference '{version}' instead of tagged version may be insecure"
            )

        return True

    def _validate_runner(self, runner: Any, job_name: str) -> bool:
        """Validate job runner specification."""
        github_hosted_runners = [
            "ubuntu-latest",
            "ubuntu-22.04",
            "ubuntu-20.04",
            "windows-latest",
            "windows-2022",
            "windows-2019",
            "macos-latest",
            "macos-13",
            "macos-12",
            "macos-11",
        ]

        if isinstance(runner, str):
            if runner not in github_hosted_runners and not runner.startswith("${{"):
                self.warnings.append(
                    f"Job '{job_name}': runner '{runner}' is not a standard GitHub-hosted runner"
                )

        elif isinstance(runner, list):
            if not runner:
                self.errors.append(f"Job '{job_name}': runner list cannot be empty")
                return False

            # Self-hosted runner labels
            if len(runner) > 1 and "self-hosted" not in runner:
                self.warnings.append(
                    f"Job '{job_name}': multi-label runner should include 'self-hosted'"
                )

        else:
            self.errors.append(f"Job '{job_name}': invalid runner format")
            return False

        return True

    def _validate_matrix(self, strategy: dict[str, Any], job_name: str) -> bool:
        """Validate matrix strategy."""
        matrix = strategy["matrix"]

        if not matrix:
            self.errors.append(f"Job '{job_name}': matrix cannot be empty")
            return False

        # Calculate matrix size
        total_combinations = 1
        for key, values in matrix.items():
            if isinstance(values, list):
                total_combinations *= len(values)

        # Subtract exclusions
        if "exclude" in strategy:
            total_combinations -= len(strategy["exclude"])

        # Add inclusions
        if "include" in strategy:
            total_combinations += len(strategy["include"])

        # Warn about large matrices
        if total_combinations > 20:
            self.warnings.append(
                f"Job '{job_name}': large matrix ({total_combinations} combinations) may consume many runner minutes"
            )

        return True

    def _validate_needs(
        self, needs: Any, job_name: str, all_jobs: dict[str, Any]
    ) -> bool:
        """Validate job dependencies."""
        if isinstance(needs, str):
            needs = [needs]

        if isinstance(needs, list):
            for needed_job in needs:
                if needed_job not in all_jobs:
                    self.errors.append(
                        f"Job '{job_name}': dependency '{needed_job}' does not exist"
                    )
                    return False

                if needed_job == job_name:
                    self.errors.append(f"Job '{job_name}': cannot depend on itself")
                    return False

        return True

    def _validate_permissions(self, permissions: Any) -> bool:
        """Validate workflow permissions."""
        valid_permissions = ["read", "write", "none"]
        valid_scopes = [
            "actions",
            "checks",
            "contents",
            "deployments",
            "id-token",
            "issues",
            "metadata",
            "packages",
            "pages",
            "pull-requests",
            "repository-projects",
            "security-events",
            "statuses",
        ]

        if isinstance(permissions, str):
            if permissions not in ["read-all", "write-all"]:
                self.errors.append(f"Invalid global permission: {permissions}")
                return False

        elif isinstance(permissions, dict):
            for scope, permission in permissions.items():
                if scope not in valid_scopes:
                    self.warnings.append(f"Unknown permission scope: {scope}")

                if permission not in valid_permissions:
                    self.errors.append(
                        f"Invalid permission level '{permission}' for scope '{scope}'"
                    )
                    return False

        return True

    def _validate_env_vars(self, env_vars: dict[str, Any]) -> bool:
        """Validate environment variables."""
        for key, value in env_vars.items():
            if not isinstance(key, str):
                self.errors.append("Environment variable names must be strings")
                return False

            if not key.replace("_", "").isalnum():
                self.warnings.append(
                    f"Environment variable '{key}' contains special characters"
                )

            if not isinstance(value, str | int | float | bool):
                self.warnings.append(
                    f"Environment variable '{key}' has non-scalar value"
                )

        return True

    def get_validation_report(self) -> dict[str, Any]:
        """Get comprehensive validation report."""
        return {
            "valid": len(self.errors) == 0,
            "errors": self.errors,
            "warnings": self.warnings,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
        }


class WorkflowGenerator:
    """Generate workflows from templates and Pydantic models."""

    TEMPLATES: ClassVar[dict[str, Any]] = {
        "ci-python": {
            "name": "Python CI",
            "description": "Continuous integration for Python projects",
            "template": lambda: WorkflowConfig(
                name="Python CI",
                on={
                    "push": {"branches": ["main", "develop"]},
                    "pull_request": {"branches": ["main"]},
                },
                jobs={
                    "test": WorkflowJob(
                        name="Run Tests",
                        runs_on="ubuntu-latest",
                        strategy=MatrixStrategy(
                            matrix={"python-version": ["3.11", "3.12"]}, fail_fast=False
                        ),
                        steps=[
                            WorkflowStep(name="Checkout", uses="actions/checkout@v4"),
                            WorkflowStep(
                                name="Setup Python",
                                uses="actions/setup-python@v5",
                                with_={
                                    "python-version": "${{ matrix.python-version }}"
                                },
                            ),
                            WorkflowStep(
                                name="Cache dependencies",
                                uses="actions/cache@v4",
                                with_={
                                    "path": "~/.cache/pip",
                                    "key": "pip-${{ hashFiles('requirements.txt') }}",
                                },
                            ),
                            WorkflowStep(
                                name="Install dependencies",
                                run="pip install -r requirements.txt",
                            ),
                            WorkflowStep(
                                name="Run tests",
                                run="pytest --cov=src --cov-report=xml",
                            ),
                            WorkflowStep(
                                name="Upload coverage",
                                uses="actions/upload-artifact@v4",
                                with_={
                                    "name": "coverage-report",
                                    "path": "coverage.xml",
                                },
                            ),
                        ],
                    ),
                    "lint": WorkflowJob(
                        name="Code Quality",
                        runs_on="ubuntu-latest",
                        steps=[
                            WorkflowStep(name="Checkout", uses="actions/checkout@v4"),
                            WorkflowStep(
                                name="Setup Python", uses="actions/setup-python@v5"
                            ),
                            WorkflowStep(
                                name="Install tools",
                                run="pip install black flake8 mypy",
                            ),
                            WorkflowStep(name="Format check", run="black --check ."),
                            WorkflowStep(name="Lint", run="flake8 ."),
                            WorkflowStep(name="Type check", run="mypy src/"),
                        ],
                    ),
                },
            ),
        },
        "ci-node": {
            "name": "Node.js CI",
            "description": "Continuous integration for Node.js projects",
            "template": lambda: WorkflowConfig(
                name="Node.js CI",
                on=[EventType.PUSH, EventType.PULL_REQUEST],
                jobs={
                    "test": WorkflowJob(
                        name="Test Node.js Application",
                        runs_on="ubuntu-latest",
                        strategy=MatrixStrategy(
                            matrix={"node-version": ["18", "20"]}, fail_fast=False
                        ),
                        steps=[
                            WorkflowStep(name="Checkout", uses="actions/checkout@v4"),
                            WorkflowStep(
                                name="Setup Node.js",
                                uses="actions/setup-node@v4",
                                with_={
                                    "node-version": "${{ matrix.node-version }}",
                                    "cache": "npm",
                                },
                            ),
                            WorkflowStep(name="Install dependencies", run="npm ci"),
                            WorkflowStep(name="Run tests", run="npm test"),
                            WorkflowStep(name="Build", run="npm run build"),
                        ],
                    )
                },
            ),
        },
        "deploy-production": {
            "name": "Production Deployment",
            "description": "Deploy to production environment",
            "template": lambda: WorkflowConfig(
                name="Deploy to Production",
                on={"push": {"branches": ["main"]}},
                permissions=PermissionSet(
                    contents=PermissionLevel.READ, deployments=PermissionLevel.WRITE
                ),
                jobs={
                    "deploy": WorkflowJob(
                        name="Deploy Application",
                        runs_on="ubuntu-latest",
                        environment="production",
                        steps=[
                            WorkflowStep(name="Checkout", uses="actions/checkout@v4"),
                            WorkflowStep(name="Build application", run="npm run build"),
                            WorkflowStep(
                                name="Deploy",
                                run="echo 'Deploy to production'",
                                env={"DEPLOY_TOKEN": "${{ secrets.DEPLOY_TOKEN }}"},
                            ),
                        ],
                    )
                },
            ),
        },
        "security-scan": {
            "name": "Security Scanning",
            "description": "Automated security scanning workflow",
            "template": lambda: WorkflowConfig(
                name="Security Scan",
                on={
                    "push": {"branches": ["main"]},
                    "pull_request": {"branches": ["main"]},
                    "schedule": [{"cron": "0 2 * * 1"}],
                },
                permissions=PermissionSet(
                    contents=PermissionLevel.READ, security_events=PermissionLevel.WRITE
                ),
                jobs={
                    "security": WorkflowJob(
                        name="Security Analysis",
                        runs_on="ubuntu-latest",
                        steps=[
                            WorkflowStep(name="Checkout", uses="actions/checkout@v4"),
                            WorkflowStep(
                                name="Security scan",
                                run="bandit -r src/ -f json -o security-report.json",
                                continue_on_error=True,
                            ),
                            WorkflowStep(
                                name="Upload security report",
                                uses="actions/upload-artifact@v4",
                                with_={
                                    "name": "security-report",
                                    "path": "security-report.json",
                                },
                                if_="always()",
                            ),
                        ],
                    )
                },
            ),
        },
    }

    def generate_from_template(
        self, template_name: str, output_file: Path | None = None
    ) -> WorkflowConfig:
        """Generate workflow from template."""
        if template_name not in self.TEMPLATES:
            available = ", ".join(self.TEMPLATES.keys())
            raise ValueError(
                f"Unknown template '{template_name}'. Available: {available}"
            )

        template = self.TEMPLATES[template_name]
        workflow = template["template"]()

        if output_file:
            self.save_workflow(workflow, output_file)

        return workflow

    def save_workflow(self, workflow: WorkflowConfig, output_file: Path) -> None:
        """Save workflow to YAML file."""
        yaml_dict = workflow.to_yaml_dict()

        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            yaml.dump(yaml_dict, f, default_flow_style=False, sort_keys=False)

    def list_templates(self) -> dict[str, str]:
        """List available templates."""
        return {name: info["description"] for name, info in self.TEMPLATES.items()}


class WorkflowTester:
    """Test workflow configurations without execution."""

    def __init__(self):
        self.validator = WorkflowValidator()

    def dry_run_test(self, workflow_file: Path) -> dict[str, Any]:
        """Perform dry-run test of workflow."""
        print(f"🔍 Testing workflow: {workflow_file}")

        # Validate workflow
        is_valid, data = self.validator.validate_yaml_file(workflow_file)

        if not is_valid:
            return {
                "success": False,
                "validation": self.validator.get_validation_report(),
                "test_results": [],
            }

        # Simulate workflow execution
        test_results = []

        if "jobs" in data:
            for job_name, job_data in data["jobs"].items():
                job_result = self._test_job(job_name, job_data)
                test_results.append(job_result)

        return {
            "success": True,
            "validation": self.validator.get_validation_report(),
            "test_results": test_results,
        }

    def _test_job(self, job_name: str, job_data: dict[str, Any]) -> dict[str, Any]:
        """Test individual job configuration."""
        print(f"  📋 Testing job: {job_name}")

        issues = []

        # Test runner availability
        runner = job_data.get("runs-on")
        if isinstance(runner, str) and runner.startswith("${{"):
            issues.append(f"Dynamic runner '{runner}' cannot be validated")

        # Test matrix combinations
        if "strategy" in job_data and "matrix" in job_data["strategy"]:
            matrix = job_data["strategy"]["matrix"]
            combinations = self._calculate_matrix_combinations(matrix)
            if combinations > 10:
                issues.append(f"Large matrix with {combinations} combinations")

        # Test steps
        step_results = []
        if "steps" in job_data:
            for i, step in enumerate(job_data["steps"]):
                step_result = self._test_step(i + 1, step)
                step_results.append(step_result)

        return {
            "job_name": job_name,
            "runner": runner,
            "issues": issues,
            "steps": step_results,
            "estimated_duration": self._estimate_job_duration(job_data),
        }

    def _test_step(self, step_num: int, step_data: dict[str, Any]) -> dict[str, Any]:
        """Test individual step configuration."""
        print(f"    ⚡ Testing step {step_num}: {step_data.get('name', 'Unnamed')}")

        issues = []

        # Test action availability (mock)
        if "uses" in step_data:
            action = step_data["uses"]
            if not self._is_known_action(action):
                issues.append(f"Unknown action: {action}")

        # Test shell commands (basic validation)
        if "run" in step_data:
            command = step_data["run"]
            if "rm -rf" in command and not step_data.get("working-directory"):
                issues.append("Potentially dangerous command without working directory")

        return {
            "step_num": step_num,
            "name": step_data.get("name"),
            "type": "action" if "uses" in step_data else "run",
            "issues": issues,
        }

    def _calculate_matrix_combinations(self, matrix: dict[str, Any]) -> int:
        """Calculate number of matrix combinations."""
        total = 1
        for values in matrix.values():
            if isinstance(values, list):
                total *= len(values)
        return total

    def _estimate_job_duration(self, job_data: dict[str, Any]) -> str:
        """Estimate job duration based on steps."""
        step_count = len(job_data.get("steps", []))

        # Simple heuristic
        if step_count <= 3:
            return "< 5 minutes"
        elif step_count <= 10:
            return "5-15 minutes"
        else:
            return "> 15 minutes"

    def _is_known_action(self, action: str) -> bool:
        """Check if action is a known GitHub action."""
        known_actions = [
            "actions/checkout",
            "actions/setup-python",
            "actions/setup-node",
            "actions/upload-artifact",
            "actions/download-artifact",
            "actions/cache",
            "anthropics/claude-code-action",
        ]

        action_name = action.split("@")[0]
        return any(action_name.startswith(known) for known in known_actions)


class WorkflowLinter:
    """Lint and format workflow files."""

    def __init__(self, fix_issues: bool = False):
        self.fix_issues = fix_issues
        self.issues: list[dict[str, Any]] = []

    def lint_file(self, file_path: Path) -> dict[str, Any]:
        """Lint a single workflow file."""
        print(f"🔍 Linting: {file_path}")

        self.issues = []

        try:
            with open(file_path) as f:
                content = f.read()
                lines = content.split("\n")

            # YAML formatting issues
            self._check_yaml_formatting(lines, file_path)

            # GitHub Actions best practices
            self._check_best_practices(file_path)

            # Security issues
            self._check_security_issues(lines)

            return {
                "file": str(file_path),
                "issues": self.issues,
                "issue_count": len(self.issues),
            }

        except Exception as e:
            return {
                "file": str(file_path),
                "error": f"Failed to lint file: {e}",
                "issues": [],
                "issue_count": 0,
            }

    def lint_directory(self, directory: Path) -> dict[str, Any]:
        """Lint all workflow files in directory."""
        workflow_files = list(directory.glob("**/*.yml")) + list(
            directory.glob("**/*.yaml")
        )

        results = []
        total_issues = 0

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self.lint_file, f) for f in workflow_files]

            for future in futures:
                result = future.result()
                results.append(result)
                total_issues += result.get("issue_count", 0)

        return {
            "directory": str(directory),
            "files_checked": len(workflow_files),
            "total_issues": total_issues,
            "results": results,
        }

    def _check_yaml_formatting(self, lines: list[str], file_path: Path) -> None:
        """Check YAML formatting issues."""
        for i, line in enumerate(lines, 1):
            # Tab characters
            if "\t" in line:
                self.issues.append(
                    {
                        "line": i,
                        "type": "formatting",
                        "severity": "error",
                        "message": "Use spaces instead of tabs for indentation",
                        "fixable": True,
                    }
                )

            # Trailing whitespace
            if line.rstrip() != line:
                self.issues.append(
                    {
                        "line": i,
                        "type": "formatting",
                        "severity": "warning",
                        "message": "Trailing whitespace",
                        "fixable": True,
                    }
                )

            # Inconsistent indentation
            if line.strip() and line.startswith(" "):
                leading_spaces = len(line) - len(line.lstrip())
                if leading_spaces % 2 != 0:
                    self.issues.append(
                        {
                            "line": i,
                            "type": "formatting",
                            "severity": "warning",
                            "message": "Inconsistent indentation (should be multiple of 2)",
                            "fixable": True,
                        }
                    )

    def _check_best_practices(self, file_path: Path) -> None:
        """Check GitHub Actions best practices."""
        try:
            with open(file_path) as f:
                data = yaml.safe_load(f)

            if not data:
                return

            # Check for pinned action versions
            self._check_action_versions(data)

            # Check for secrets usage
            self._check_secrets_usage(data)

            # Check for concurrency settings
            if "concurrency" not in data and "push" in str(data.get("on", {})):
                self.issues.append(
                    {
                        "type": "best_practice",
                        "severity": "suggestion",
                        "message": "Consider adding concurrency settings for push events",
                        "fixable": False,
                    }
                )

        except (KeyError, ValueError, TypeError, AttributeError):
            # Already handled by validation - missing keys, invalid values, type errors, or attribute access on wrong types
            pass

    def _check_action_versions(self, data: dict[str, Any]) -> None:
        """Check for pinned action versions."""

        def check_steps(steps):
            for step in steps:
                if "uses" in step:
                    action = step["uses"]
                    if "@" in action:
                        _, version = action.split("@", 1)
                        if version in ["main", "master"]:
                            self.issues.append(
                                {
                                    "type": "security",
                                    "severity": "warning",
                                    "message": f"Action uses branch reference '{version}' instead of tagged version",
                                    "action": action,
                                    "fixable": False,
                                }
                            )

        if "jobs" in data:
            for job_data in data["jobs"].values():
                if "steps" in job_data:
                    check_steps(job_data["steps"])

    def _check_secrets_usage(self, data: dict[str, Any]) -> None:
        """Check for proper secrets usage."""
        content_str = yaml.dump(data)

        # Look for hardcoded credentials patterns
        dangerous_patterns = ["password", "token", "key", "secret"]

        for pattern in dangerous_patterns:
            if f"{pattern}:" in content_str.lower() and "${{" not in content_str:
                self.issues.append(
                    {
                        "type": "security",
                        "severity": "error",
                        "message": f"Potential hardcoded credential: {pattern}",
                        "fixable": False,
                    }
                )

    def _check_security_issues(self, lines: list[str]) -> None:
        """Check for security issues."""
        for i, line in enumerate(lines, 1):
            # Dangerous commands
            dangerous_commands = ["rm -rf", "curl | sh", "wget | sh"]
            for cmd in dangerous_commands:
                if cmd in line:
                    self.issues.append(
                        {
                            "line": i,
                            "type": "security",
                            "severity": "warning",
                            "message": f"Potentially dangerous command: {cmd}",
                            "fixable": False,
                        }
                    )


async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CI Workflow Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate workflow file
  python -m omninode_bridge.cli.workflow_ci validate workflow.yml

  # Generate workflow from template
  python -m omninode_bridge.cli.workflow_ci generate --template ci-python --output ci.yml

  # Test workflow configuration
  python -m omninode_bridge.cli.workflow_ci test workflow.yml

  # Lint workflow files
  python -m omninode_bridge.cli.workflow_ci lint .github/workflows/

  # List available templates
  python -m omninode_bridge.cli.workflow_ci templates
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate workflow files")
    validate_parser.add_argument("files", nargs="+", help="Workflow files to validate")
    validate_parser.add_argument(
        "--strict", action="store_true", help="Enable strict validation"
    )
    validate_parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )

    # Generate command
    generate_parser = subparsers.add_parser(
        "generate", help="Generate workflow from template"
    )
    generate_parser.add_argument("--template", required=True, help="Template name")
    generate_parser.add_argument("--output", "-o", type=Path, help="Output file path")
    generate_parser.add_argument(
        "--list-templates", action="store_true", help="List available templates"
    )

    # Test command
    test_parser = subparsers.add_parser("test", help="Test workflow configuration")
    test_parser.add_argument("files", nargs="+", help="Workflow files to test")
    test_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose output"
    )

    # Lint command
    lint_parser = subparsers.add_parser("lint", help="Lint workflow files")
    lint_parser.add_argument("paths", nargs="+", help="Files or directories to lint")
    lint_parser.add_argument(
        "--fix", action="store_true", help="Automatically fix issues where possible"
    )
    lint_parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )

    # Templates command
    templates_parser = subparsers.add_parser(
        "templates", help="List available templates"
    )

    # Common options
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress output")

    args = parser.parse_args()

    # Setup logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    elif args.quiet:
        logging.basicConfig(level=logging.ERROR)
    else:
        logging.basicConfig(level=logging.INFO)

    try:
        if args.command == "validate":
            await handle_validate_command(args)
        elif args.command == "generate":
            await handle_generate_command(args)
        elif args.command == "test":
            await handle_test_command(args)
        elif args.command == "lint":
            await handle_lint_command(args)
        elif args.command == "templates":
            await handle_templates_command(args)
        else:
            parser.print_help()

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


async def handle_validate_command(args):
    """Handle validate command."""
    validator = WorkflowValidator(strict_mode=args.strict)
    results = []

    for file_path in args.files:
        path = Path(file_path)
        if not path.exists():
            print(f"❌ File not found: {file_path}")
            continue

        print(f"🔍 Validating: {file_path}")
        is_valid, data = validator.validate_yaml_file(path)

        report = validator.get_validation_report()
        results.append({"file": file_path, "valid": is_valid, "report": report})

        if args.format == "text":
            if is_valid:
                print(f"✅ {file_path} is valid")
            else:
                print(f"❌ {file_path} has {report['error_count']} errors")
                for error in report["errors"]:
                    print(f"   Error: {error}")

            if report["warnings"]:
                print(f"⚠️  {len(report['warnings'])} warnings:")
                for warning in report["warnings"]:
                    print(f"   Warning: {warning}")

        # Reset validator for next file
        validator.errors = []
        validator.warnings = []

    if args.format == "json":
        print(json.dumps(results, indent=2))


async def handle_generate_command(args):
    """Handle generate command."""
    generator = WorkflowGenerator()

    if args.list_templates:
        templates = generator.list_templates()
        print("Available templates:")
        for name, description in templates.items():
            print(f"  {name}: {description}")
        return

    try:
        workflow = generator.generate_from_template(args.template, args.output)

        if args.output:
            print(f"✅ Generated workflow: {args.output}")
        else:
            # Print to stdout
            yaml_dict = workflow.to_yaml_dict()
            print(yaml.dump(yaml_dict, default_flow_style=False, sort_keys=False))

    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(1)


async def handle_test_command(args):
    """Handle test command."""
    tester = WorkflowTester()

    for file_path in args.files:
        path = Path(file_path)
        if not path.exists():
            print(f"❌ File not found: {file_path}")
            continue

        result = tester.dry_run_test(path)

        if result["success"]:
            print(f"✅ {file_path} test passed")

            if args.verbose:
                for job_result in result["test_results"]:
                    print(
                        f"  Job '{job_result['job_name']}': {job_result['estimated_duration']}"
                    )
                    if job_result["issues"]:
                        for issue in job_result["issues"]:
                            print(f"    ⚠️  {issue}")
        else:
            print(f"❌ {file_path} test failed")
            validation = result["validation"]
            for error in validation["errors"]:
                print(f"   Error: {error}")


async def handle_lint_command(args):
    """Handle lint command."""
    linter = WorkflowLinter(fix_issues=args.fix)
    all_results = []

    for path_str in args.paths:
        path = Path(path_str)

        if path.is_file():
            result = linter.lint_file(path)
            all_results.append(result)
        elif path.is_dir():
            result = linter.lint_directory(path)
            all_results.extend(result["results"])
        else:
            print(f"❌ Path not found: {path_str}")

    if args.format == "text":
        total_issues = 0
        for result in all_results:
            file_path = result["file"]
            issues = result.get("issues", [])
            total_issues += len(issues)

            if issues:
                print(f"📄 {file_path}:")
                for issue in issues:
                    severity_icon = {"error": "❌", "warning": "⚠️", "suggestion": "💡"}
                    icon = severity_icon.get(issue["severity"], "i")
                    line_info = f"Line {issue['line']}: " if "line" in issue else ""
                    print(f"  {icon} {line_info}{issue['message']}")
            else:
                print(f"✅ {file_path}: No issues")

        print(f"\n📊 Total issues found: {total_issues}")

    elif args.format == "json":
        print(json.dumps(all_results, indent=2))


async def handle_templates_command(args):
    """Handle templates command."""
    generator = WorkflowGenerator()
    templates = generator.list_templates()

    print("Available workflow templates:")
    print()

    for name, description in templates.items():
        print(f"📋 {name}")
        print(f"   {description}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
