"""Comprehensive workflow validation for GitHub Actions."""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Union

import yaml

from ..models.workflow import WorkflowConfig

logger = logging.getLogger(__name__)


class ValidationSeverity(str, Enum):
    """Validation issue severity levels."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """Represents a validation issue found in a workflow."""

    severity: ValidationSeverity
    message: str
    location: str
    suggestion: str | None = None
    rule_id: str | None = None


@dataclass
class ValidationResult:
    """Result of workflow validation."""

    is_valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    warnings_count: int = 0
    errors_count: int = 0

    def __post_init__(self):
        """Calculate issue counts."""
        self.errors_count = sum(
            1 for issue in self.issues if issue.severity == ValidationSeverity.ERROR
        )
        self.warnings_count = sum(
            1 for issue in self.issues if issue.severity == ValidationSeverity.WARNING
        )
        self.is_valid = self.errors_count == 0


class ValidationRule:
    """Base class for validation rules."""

    def __init__(self, rule_id: str, description: str):
        self.rule_id = rule_id
        self.description = description

    def validate(self, workflow: WorkflowConfig) -> list[ValidationIssue]:
        """Validate workflow against this rule.

        Args:
            workflow: Workflow configuration to validate

        Returns:
            List of validation issues
        """
        raise NotImplementedError


class WorkflowStructureRule(ValidationRule):
    """Validates basic workflow structure."""

    def __init__(self):
        super().__init__("WS001", "Workflow must have valid structure")

    def validate(self, workflow: WorkflowConfig) -> list[ValidationIssue]:
        issues = []

        # Check workflow name
        if not workflow.name or not workflow.name.strip():
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message="Workflow name cannot be empty",
                    location="workflow.name",
                    rule_id=self.rule_id,
                )
            )

        # Check jobs exist
        if not workflow.jobs:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message="Workflow must have at least one job",
                    location="workflow.jobs",
                    rule_id=self.rule_id,
                )
            )

        return issues


class JobDependencyRule(ValidationRule):
    """Validates job dependencies are valid."""

    def __init__(self):
        super().__init__("JD001", "Job dependencies must reference existing jobs")

    def validate(self, workflow: WorkflowConfig) -> list[ValidationIssue]:
        issues = []
        job_names = set(workflow.jobs.keys())

        for job_id, job in workflow.jobs.items():
            if job.needs:
                needs = job.needs if isinstance(job.needs, list) else [job.needs]

                for needed_job in needs:
                    if needed_job not in job_names:
                        issues.append(
                            ValidationIssue(
                                severity=ValidationSeverity.ERROR,
                                message=f"Job '{job_id}' depends on non-existent job '{needed_job}'",
                                location=f"jobs.{job_id}.needs",
                                suggestion=f"Available jobs: {', '.join(job_names)}",
                                rule_id=self.rule_id,
                            )
                        )

        return issues


class CircularDependencyRule(ValidationRule):
    """Detects circular dependencies in job needs."""

    def __init__(self):
        super().__init__("CD001", "Jobs cannot have circular dependencies")

    def validate(self, workflow: WorkflowConfig) -> list[ValidationIssue]:
        issues = []

        # Build dependency graph
        dependencies = {}
        for job_id, job in workflow.jobs.items():
            if job.needs:
                needs = job.needs if isinstance(job.needs, list) else [job.needs]
                dependencies[job_id] = needs
            else:
                dependencies[job_id] = []

        # Check for cycles using DFS
        visited = set()
        rec_stack = set()

        def has_cycle(node: str, path: list[str]) -> list[str] | None:
            if node in rec_stack:
                cycle_start = path.index(node)
                return path[cycle_start:] + [node]

            if node in visited:
                return None

            visited.add(node)
            rec_stack.add(node)

            for dependency in dependencies.get(node, []):
                cycle = has_cycle(dependency, path + [node])
                if cycle:
                    return cycle

            rec_stack.remove(node)
            return None

        for job_id in workflow.jobs:
            if job_id not in visited:
                cycle = has_cycle(job_id, [])
                if cycle:
                    cycle_str = " -> ".join(cycle)
                    issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            message=f"Circular dependency detected: {cycle_str}",
                            location=f"jobs.{job_id}.needs",
                            rule_id=self.rule_id,
                        )
                    )

        return issues


class ActionVersionRule(ValidationRule):
    """Validates GitHub Actions use current versions."""

    def __init__(self):
        super().__init__("AV001", "Actions should use current versions")

        # Define current recommended versions
        self.recommended_versions = {
            "actions/checkout": "v4",
            "actions/setup-python": "v5",
            "actions/setup-node": "v4",
            "actions/upload-artifact": "v4",
            "actions/download-artifact": "v4",
            "actions/cache": "v4",
        }

    def validate(self, workflow: WorkflowConfig) -> list[ValidationIssue]:
        issues = []

        for job_id, job in workflow.jobs.items():
            for step_idx, step in enumerate(job.steps):
                if step.uses:
                    action_name, version = self._parse_action_ref(step.uses)

                    if action_name in self.recommended_versions:
                        recommended = self.recommended_versions[action_name]
                        if version != recommended:
                            issues.append(
                                ValidationIssue(
                                    severity=ValidationSeverity.WARNING,
                                    message=f"Action '{action_name}' version '{version}' is outdated",
                                    location=f"jobs.{job_id}.steps[{step_idx}].uses",
                                    suggestion=f"Consider upgrading to {action_name}@{recommended}",
                                    rule_id=self.rule_id,
                                )
                            )

        return issues

    def _parse_action_ref(self, action_ref: str) -> tuple[str, str]:
        """Parse action reference into name and version.

        Args:
            action_ref: Action reference (e.g., 'actions/checkout@v4')

        Returns:
            Tuple of (action_name, version)
        """
        if "@" in action_ref:
            action_name, version = action_ref.rsplit("@", 1)
            return action_name, version
        return action_ref, ""


class SecurityRule(ValidationRule):
    """Validates security best practices."""

    def __init__(self):
        super().__init__("SEC001", "Security best practices validation")

    def validate(self, workflow: WorkflowConfig) -> list[ValidationIssue]:
        issues = []

        for job_id, job in workflow.jobs.items():
            # Check for overly broad permissions
            if hasattr(job, "permissions") and job.permissions:
                self._check_permissions(
                    job.permissions, f"jobs.{job_id}.permissions", issues
                )

            # Check for security issues in steps
            for step_idx, step in enumerate(job.steps):
                step_location = f"jobs.{job_id}.steps[{step_idx}]"

                # Check for hardcoded secrets
                if step.run:
                    self._check_hardcoded_secrets(
                        step.run, f"{step_location}.run", issues
                    )

                # Check for dangerous commands
                if step.run:
                    self._check_dangerous_commands(
                        step.run, f"{step_location}.run", issues
                    )

                # Check third-party actions
                if step.uses and not self._is_trusted_action(step.uses):
                    issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            message=f"Using third-party action: {step.uses}",
                            location=f"{step_location}.uses",
                            suggestion="Verify the action is from a trusted source",
                            rule_id=self.rule_id,
                        )
                    )

        return issues

    def _check_permissions(
        self, permissions: Any, location: str, issues: list[ValidationIssue]
    ):
        """Check for overly broad permissions."""
        if isinstance(permissions, dict):
            for perm, level in permissions.items():
                if level == "write" and perm in ["contents", "actions", "deployments"]:
                    issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            message=f"Broad '{perm}: write' permission detected",
                            location=location,
                            suggestion="Consider using more specific permissions",
                            rule_id=self.rule_id,
                        )
                    )

    def _check_hardcoded_secrets(
        self, command: str, location: str, issues: list[ValidationIssue]
    ):
        """Check for hardcoded secrets in commands."""
        # Look for potential hardcoded secrets
        secret_patterns = [
            r'(?i)(password|token|key|secret)\s*=\s*["\'][^"\']{8,}["\']',
            r'(?i)(api_key|apikey)\s*=\s*["\'][^"\']{8,}["\']',
            r"(?i)Bearer\s+[A-Za-z0-9\-._~+/]+=*",
        ]

        for pattern in secret_patterns:
            if re.search(pattern, command):
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message="Potential hardcoded secret detected",
                        location=location,
                        suggestion="Use GitHub secrets or environment variables",
                        rule_id=self.rule_id,
                    )
                )

    def _check_dangerous_commands(
        self, command: str, location: str, issues: list[ValidationIssue]
    ):
        """Check for potentially dangerous commands."""
        dangerous_patterns = [
            r"rm\s+-rf\s+/",
            r"curl\s+.*\|\s*sh",
            r"wget\s+.*\|\s*sh",
            r"sudo\s+",
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, command):
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message="Potentially dangerous command detected",
                        location=location,
                        suggestion="Review command for security implications",
                        rule_id=self.rule_id,
                    )
                )

    def _is_trusted_action(self, action_ref: str) -> bool:
        """Check if action is from a trusted source."""
        trusted_orgs = ["actions", "github", "microsoft", "google-github-actions"]
        action_name = action_ref.split("@")[0]
        org = action_name.split("/")[0] if "/" in action_name else ""
        return org in trusted_orgs


class PerformanceRule(ValidationRule):
    """Validates performance best practices."""

    def __init__(self):
        super().__init__("PERF001", "Performance best practices validation")

    def validate(self, workflow: WorkflowConfig) -> list[ValidationIssue]:
        issues = []

        for job_id, job in workflow.jobs.items():
            # Check for missing caching
            has_cache = any(
                step.uses and "cache" in step.uses.lower() for step in job.steps
            )

            has_dependencies = any(
                step.run
                and any(
                    cmd in step.run
                    for cmd in ["pip install", "npm install", "yarn install"]
                )
                for step in job.steps
            )

            if has_dependencies and not has_cache:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message="Job installs dependencies but doesn't use caching",
                        location=f"jobs.{job_id}",
                        suggestion="Consider adding caching to improve performance",
                        rule_id=self.rule_id,
                    )
                )

            # Check for excessive timeout
            if job.timeout_minutes and job.timeout_minutes > 360:  # 6 hours
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Job timeout is very long: {job.timeout_minutes} minutes",
                        location=f"jobs.{job_id}.timeout-minutes",
                        suggestion="Consider breaking into smaller jobs or reducing timeout",
                        rule_id=self.rule_id,
                    )
                )

        return issues


class WorkflowValidator:
    """Comprehensive workflow validator."""

    def __init__(self, rules: list[ValidationRule] | None = None):
        """Initialize validator with rules.

        Args:
            rules: Custom validation rules (uses defaults if None)
        """
        self.rules = rules or self._default_rules()

    def validate(self, workflow: WorkflowConfig) -> ValidationResult:
        """Validate workflow configuration.

        Args:
            workflow: Workflow to validate

        Returns:
            Validation result with issues
        """
        all_issues = []

        for rule in self.rules:
            try:
                issues = rule.validate(workflow)
                all_issues.extend(issues)
            except Exception as e:
                logger.error(f"Rule {rule.rule_id} failed: {e}")
                all_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"Validation rule {rule.rule_id} failed: {e}",
                        location="workflow",
                        rule_id=rule.rule_id,
                    )
                )

        return ValidationResult(issues=all_issues, is_valid=True)

    def validate_yaml(self, yaml_content: str) -> ValidationResult:
        """Validate workflow from YAML content.

        Args:
            yaml_content: YAML workflow content

        Returns:
            Validation result
        """
        try:
            # Parse YAML
            workflow_data = yaml.safe_load(yaml_content)

            # Convert to Pydantic model
            workflow = WorkflowConfig(**workflow_data)

            return self.validate(workflow)

        except yaml.YAMLError as e:
            return ValidationResult(
                is_valid=False,
                issues=[
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"Invalid YAML: {e}",
                        location="yaml",
                        rule_id="YAML001",
                    )
                ],
            )
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                issues=[
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"Validation error: {e}",
                        location="workflow",
                        rule_id="VAL001",
                    )
                ],
            )

    def validate_file(self, file_path: Union[str, Path]) -> ValidationResult:
        """Validate workflow from file.

        Args:
            file_path: Path to workflow file

        Returns:
            Validation result
        """
        try:
            with open(file_path, encoding="utf-8") as f:
                yaml_content = f.read()

            return self.validate_yaml(yaml_content)

        except FileNotFoundError:
            return ValidationResult(
                is_valid=False,
                issues=[
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"File not found: {file_path}",
                        location="file",
                        rule_id="FILE001",
                    )
                ],
            )
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                issues=[
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"Error reading file: {e}",
                        location="file",
                        rule_id="FILE002",
                    )
                ],
            )

    def _default_rules(self) -> list[ValidationRule]:
        """Get default validation rules.

        Returns:
            List of default validation rules
        """
        return [
            WorkflowStructureRule(),
            JobDependencyRule(),
            CircularDependencyRule(),
            ActionVersionRule(),
            SecurityRule(),
            PerformanceRule(),
        ]

    def add_rule(self, rule: ValidationRule) -> None:
        """Add a custom validation rule.

        Args:
            rule: Validation rule to add
        """
        self.rules.append(rule)

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a validation rule by ID.

        Args:
            rule_id: ID of rule to remove

        Returns:
            True if rule was removed, False if not found
        """
        original_length = len(self.rules)
        self.rules = [rule for rule in self.rules if rule.rule_id != rule_id]
        return len(self.rules) < original_length


def format_validation_report(result: ValidationResult) -> str:
    """Format validation result as human-readable report.

    Args:
        result: Validation result

    Returns:
        Formatted report string
    """
    lines = []

    # Header
    if result.is_valid:
        lines.append("✅ Workflow validation PASSED")
    else:
        lines.append("❌ Workflow validation FAILED")

    lines.append(f"   Errors: {result.errors_count}")
    lines.append(f"   Warnings: {result.warnings_count}")
    lines.append("")

    # Issues
    if result.issues:
        lines.append("Issues found:")
        lines.append("")

        for issue in result.issues:
            icon = (
                "❌"
                if issue.severity == ValidationSeverity.ERROR
                else "⚠️" if issue.severity == ValidationSeverity.WARNING else "i"
            )
            lines.append(f"{icon} [{issue.rule_id}] {issue.message}")
            lines.append(f"   Location: {issue.location}")

            if issue.suggestion:
                lines.append(f"   Suggestion: {issue.suggestion}")

            lines.append("")

    return "\n".join(lines)
