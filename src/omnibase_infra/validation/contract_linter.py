# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
ONEX Infrastructure Contract Linter.

Validates contract.yaml files against ONEX infrastructure requirements:
- Required fields: name, node_type, contract_version, input_model, output_model
- Type consistency: input_model/output_model module references are importable
- YAML syntax validity
- Node type constraints (EFFECT, COMPUTE, REDUCER, ORCHESTRATOR)

This linter complements omnibase_core.validation.validate_contracts by adding
infrastructure-specific validation that is not covered by the base validator.

Usage:
    from omnibase_infra.validation.contract_linter import (
        ContractLinter,
        lint_contracts_in_directory,
        lint_contract_file,
    )

    # Lint all contracts in a directory
    result = lint_contracts_in_directory("src/omnibase_infra/nodes/")

    # Lint a single contract file
    result = lint_contract_file("path/to/contract.yaml")

Exit Codes (for CI):
    0: All contracts valid
    1: Validation failures found
    2: Runtime error (file not found, YAML parse error, etc.)
"""

import importlib
import logging
import re
from enum import Enum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

# Module-level logger
logger = logging.getLogger(__name__)


class EnumContractViolationSeverity(str, Enum):
    """Severity levels for contract violations."""

    ERROR = "error"  # Must be fixed before merge
    WARNING = "warning"  # Should be fixed, but not blocking
    INFO = "info"  # Informational, best practice suggestion


class ModelContractViolation(BaseModel):
    """A single contract validation violation."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    file_path: str = Field(description="Path to the contract file")
    field_path: str = Field(
        description="JSON path to the violating field (e.g., 'input_model.module')"
    )
    message: str = Field(description="Human-readable violation description")
    severity: EnumContractViolationSeverity = Field(
        default=EnumContractViolationSeverity.ERROR,
        description="Violation severity level",
    )
    suggestion: str | None = Field(
        default=None,
        description="Suggested fix for the violation",
    )

    def __str__(self) -> str:
        """Format violation as human-readable string."""
        prefix = f"[{self.severity.value.upper()}]"
        location = f"{self.file_path}:{self.field_path}"
        msg = f"{prefix} {location}: {self.message}"
        if self.suggestion:
            msg += f" (suggestion: {self.suggestion})"
        return msg


class ModelContractLintResult(BaseModel):
    """Result of linting one or more contract files."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    is_valid: bool = Field(description="True if no ERROR-level violations found")
    violations: list[ModelContractViolation] = Field(
        default_factory=list,
        description="All violations found during linting",
    )
    files_checked: int = Field(
        default=0, description="Number of contract files checked"
    )
    files_valid: int = Field(
        default=0, description="Number of contract files with no errors"
    )
    files_with_errors: int = Field(
        default=0, description="Number of contract files with errors"
    )

    @property
    def error_count(self) -> int:
        """Count of ERROR-level violations."""
        return sum(
            1
            for v in self.violations
            if v.severity == EnumContractViolationSeverity.ERROR
        )

    @property
    def warning_count(self) -> int:
        """Count of WARNING-level violations."""
        return sum(
            1
            for v in self.violations
            if v.severity == EnumContractViolationSeverity.WARNING
        )

    def __str__(self) -> str:
        """Format result summary as human-readable string."""
        status = "PASS" if self.is_valid else "FAIL"
        summary = f"Contract Lint: {status} ({self.files_checked} files, {self.error_count} errors, {self.warning_count} warnings)"
        return summary


# Valid node types per ONEX 4-node architecture
VALID_NODE_TYPES = frozenset({"EFFECT", "COMPUTE", "REDUCER", "ORCHESTRATOR"})


class ContractLinter:
    """
    ONEX Infrastructure Contract Linter.

    Validates contract.yaml files for required fields, type consistency,
    and ONEX compliance. Designed for CI integration with clear exit codes.

    Required Fields:
        - name: Node identifier (snake_case)
        - node_type: One of EFFECT, COMPUTE, REDUCER, ORCHESTRATOR
        - contract_version: Semantic version dict with major, minor, patch
        - input_model: Dict with name and module fields
        - output_model: Dict with name and module fields

    Optional but Recommended Fields:
        - description: Human-readable description
        - node_version: Semantic version string
        - dependencies: List of dependency declarations
        - consumed_events: Event topics the node subscribes to
        - published_events: Event topics the node publishes to
    """

    def __init__(
        self,
        *,
        check_imports: bool = True,
        strict_mode: bool = False,
    ):
        """
        Initialize the contract linter.

        Args:
            check_imports: Whether to verify input_model/output_model modules
                          are importable. Disable for faster validation when
                          modules may not be in the Python path.
            strict_mode: If True, treat warnings as errors.
        """
        self.check_imports = check_imports
        self.strict_mode = strict_mode

    def lint_file(self, file_path: Path) -> ModelContractLintResult:
        """
        Lint a single contract.yaml file.

        Args:
            file_path: Path to the contract.yaml file.

        Returns:
            ModelContractLintResult with violations found.
        """
        violations: list[ModelContractViolation] = []
        file_str = str(file_path)

        # Check file exists
        if not file_path.exists():
            violations.append(
                ModelContractViolation(
                    file_path=file_str,
                    field_path="",
                    message=f"Contract file not found: {file_path}",
                    severity=EnumContractViolationSeverity.ERROR,
                )
            )
            return ModelContractLintResult(
                is_valid=False,
                violations=violations,
                files_checked=1,
                files_with_errors=1,
            )

        # Parse YAML
        try:
            with file_path.open("r", encoding="utf-8") as f:
                content = yaml.safe_load(f)
        except yaml.YAMLError as e:
            violations.append(
                ModelContractViolation(
                    file_path=file_str,
                    field_path="",
                    message=f"YAML parse error: {e}",
                    severity=EnumContractViolationSeverity.ERROR,
                )
            )
            return ModelContractLintResult(
                is_valid=False,
                violations=violations,
                files_checked=1,
                files_with_errors=1,
            )

        if not isinstance(content, dict):
            violations.append(
                ModelContractViolation(
                    file_path=file_str,
                    field_path="",
                    message="Contract must be a YAML mapping (dict), not a scalar or list",
                    severity=EnumContractViolationSeverity.ERROR,
                )
            )
            return ModelContractLintResult(
                is_valid=False,
                violations=violations,
                files_checked=1,
                files_with_errors=1,
            )

        # Validate required fields
        violations.extend(self._validate_required_fields(file_str, content))

        # Validate node_type
        violations.extend(self._validate_node_type(file_str, content))

        # Validate contract_version format
        violations.extend(self._validate_contract_version(file_str, content))

        # Validate input_model and output_model
        violations.extend(
            self._validate_model_reference(file_str, content, "input_model")
        )
        violations.extend(
            self._validate_model_reference(file_str, content, "output_model")
        )

        # Validate naming convention (name should be snake_case)
        violations.extend(self._validate_name_convention(file_str, content))

        # Check for recommended fields
        violations.extend(self._check_recommended_fields(file_str, content))

        # Calculate result
        has_errors = any(
            v.severity == EnumContractViolationSeverity.ERROR for v in violations
        )
        if self.strict_mode:
            has_errors = has_errors or any(
                v.severity == EnumContractViolationSeverity.WARNING for v in violations
            )

        return ModelContractLintResult(
            is_valid=not has_errors,
            violations=violations,
            files_checked=1,
            files_valid=0 if has_errors else 1,
            files_with_errors=1 if has_errors else 0,
        )

    def lint_directory(
        self,
        directory: Path,
        *,
        recursive: bool = True,
    ) -> ModelContractLintResult:
        """
        Lint all contract.yaml files in a directory.

        Args:
            directory: Directory to search for contract.yaml files.
            recursive: Whether to search subdirectories.

        Returns:
            ModelContractLintResult with aggregated violations.
        """
        if not directory.exists():
            return ModelContractLintResult(
                is_valid=False,
                violations=[
                    ModelContractViolation(
                        file_path=str(directory),
                        field_path="",
                        message=f"Directory not found: {directory}",
                        severity=EnumContractViolationSeverity.ERROR,
                    )
                ],
                files_checked=0,
                files_with_errors=0,
            )

        # Find all contract.yaml files
        pattern = "**/contract.yaml" if recursive else "contract.yaml"
        contract_files = list(directory.glob(pattern))

        if not contract_files:
            # No contracts found - this is informational, not an error
            logger.info("No contract.yaml files found in %s", directory)
            return ModelContractLintResult(
                is_valid=True,
                violations=[],
                files_checked=0,
                files_valid=0,
                files_with_errors=0,
            )

        # Lint each file and aggregate results
        all_violations: list[ModelContractViolation] = []
        files_valid = 0
        files_with_errors = 0

        for contract_file in sorted(contract_files):
            result = self.lint_file(contract_file)
            all_violations.extend(result.violations)
            files_valid += result.files_valid
            files_with_errors += result.files_with_errors

        has_errors = any(
            v.severity == EnumContractViolationSeverity.ERROR for v in all_violations
        )
        if self.strict_mode:
            has_errors = has_errors or any(
                v.severity == EnumContractViolationSeverity.WARNING
                for v in all_violations
            )

        return ModelContractLintResult(
            is_valid=not has_errors,
            violations=all_violations,
            files_checked=len(contract_files),
            files_valid=files_valid,
            files_with_errors=files_with_errors,
        )

    def _validate_required_fields(
        self,
        file_path: str,
        content: dict,
    ) -> list[ModelContractViolation]:
        """Validate that all required top-level fields are present."""
        violations: list[ModelContractViolation] = []
        required_fields = [
            "name",
            "node_type",
            "contract_version",
            "input_model",
            "output_model",
        ]

        for field in required_fields:
            if field not in content:
                violations.append(
                    ModelContractViolation(
                        file_path=file_path,
                        field_path=field,
                        message=f"Required field '{field}' is missing",
                        severity=EnumContractViolationSeverity.ERROR,
                        suggestion=f"Add '{field}:' to your contract.yaml",
                    )
                )

        return violations

    def _validate_node_type(
        self,
        file_path: str,
        content: dict,
    ) -> list[ModelContractViolation]:
        """Validate node_type is one of the valid ONEX 4-node types."""
        violations: list[ModelContractViolation] = []
        node_type = content.get("node_type")

        if node_type is None:
            return violations  # Already caught by required fields check

        if not isinstance(node_type, str):
            violations.append(
                ModelContractViolation(
                    file_path=file_path,
                    field_path="node_type",
                    message=f"node_type must be a string, got {type(node_type).__name__}",
                    severity=EnumContractViolationSeverity.ERROR,
                )
            )
            return violations

        if node_type not in VALID_NODE_TYPES:
            violations.append(
                ModelContractViolation(
                    file_path=file_path,
                    field_path="node_type",
                    message=f"Invalid node_type '{node_type}'. Must be one of: {', '.join(sorted(VALID_NODE_TYPES))}",
                    severity=EnumContractViolationSeverity.ERROR,
                    suggestion=f"Change node_type to one of: {', '.join(sorted(VALID_NODE_TYPES))}",
                )
            )

        return violations

    def _validate_contract_version(
        self,
        file_path: str,
        content: dict,
    ) -> list[ModelContractViolation]:
        """Validate contract_version has proper semver structure."""
        violations: list[ModelContractViolation] = []
        version = content.get("contract_version")

        if version is None:
            return violations  # Already caught by required fields check

        if not isinstance(version, dict):
            violations.append(
                ModelContractViolation(
                    file_path=file_path,
                    field_path="contract_version",
                    message="contract_version must be a dict with 'major', 'minor', 'patch' keys",
                    severity=EnumContractViolationSeverity.ERROR,
                    suggestion="Use format: contract_version:\\n  major: 1\\n  minor: 0\\n  patch: 0",
                )
            )
            return violations

        for key in ["major", "minor", "patch"]:
            if key not in version:
                violations.append(
                    ModelContractViolation(
                        file_path=file_path,
                        field_path=f"contract_version.{key}",
                        message=f"contract_version missing required field '{key}'",
                        severity=EnumContractViolationSeverity.ERROR,
                    )
                )
            elif not isinstance(version[key], int):
                violations.append(
                    ModelContractViolation(
                        file_path=file_path,
                        field_path=f"contract_version.{key}",
                        message=f"contract_version.{key} must be an integer, got {type(version[key]).__name__}",
                        severity=EnumContractViolationSeverity.ERROR,
                    )
                )
            elif version[key] < 0:
                violations.append(
                    ModelContractViolation(
                        file_path=file_path,
                        field_path=f"contract_version.{key}",
                        message=f"contract_version.{key} must be non-negative, got {version[key]}",
                        severity=EnumContractViolationSeverity.ERROR,
                    )
                )

        return violations

    def _validate_model_reference(
        self,
        file_path: str,
        content: dict,
        field_name: Literal["input_model", "output_model"],
    ) -> list[ModelContractViolation]:
        """Validate input_model or output_model reference structure and importability."""
        violations: list[ModelContractViolation] = []
        model_ref = content.get(field_name)

        if model_ref is None:
            return violations  # Already caught by required fields check

        if not isinstance(model_ref, dict):
            violations.append(
                ModelContractViolation(
                    file_path=file_path,
                    field_path=field_name,
                    message=f"{field_name} must be a dict with 'name' and 'module' keys",
                    severity=EnumContractViolationSeverity.ERROR,
                    suggestion=f"Use format: {field_name}:\\n  name: ModelName\\n  module: package.module",
                )
            )
            return violations

        # Check required sub-fields
        for key in ["name", "module"]:
            if key not in model_ref:
                violations.append(
                    ModelContractViolation(
                        file_path=file_path,
                        field_path=f"{field_name}.{key}",
                        message=f"{field_name} missing required field '{key}'",
                        severity=EnumContractViolationSeverity.ERROR,
                    )
                )
            elif not isinstance(model_ref[key], str):
                violations.append(
                    ModelContractViolation(
                        file_path=file_path,
                        field_path=f"{field_name}.{key}",
                        message=f"{field_name}.{key} must be a string",
                        severity=EnumContractViolationSeverity.ERROR,
                    )
                )

        # Validate model name follows ONEX naming convention (Model* prefix)
        model_name = model_ref.get("name")
        if isinstance(model_name, str) and not model_name.startswith("Model"):
            violations.append(
                ModelContractViolation(
                    file_path=file_path,
                    field_path=f"{field_name}.name",
                    message=f"{field_name}.name should start with 'Model' prefix per ONEX conventions",
                    severity=EnumContractViolationSeverity.WARNING,
                    suggestion=f"Rename to 'Model{model_name}'",
                )
            )

        # Check if module is importable (optional, can be slow)
        if self.check_imports:
            violations.extend(
                self._check_module_importable(file_path, field_name, model_ref)
            )

        return violations

    def _check_module_importable(
        self,
        file_path: str,
        field_name: str,
        model_ref: dict,
    ) -> list[ModelContractViolation]:
        """Check if the model's module is importable."""
        violations: list[ModelContractViolation] = []
        module_name = model_ref.get("module")
        class_name = model_ref.get("name")

        if not isinstance(module_name, str) or not isinstance(class_name, str):
            return violations  # Type errors already reported

        try:
            module = importlib.import_module(module_name)
            if not hasattr(module, class_name):
                violations.append(
                    ModelContractViolation(
                        file_path=file_path,
                        field_path=f"{field_name}.name",
                        message=f"Class '{class_name}' not found in module '{module_name}'",
                        severity=EnumContractViolationSeverity.ERROR,
                        suggestion=f"Verify the class name exists in {module_name}",
                    )
                )
        except ImportError as e:
            violations.append(
                ModelContractViolation(
                    file_path=file_path,
                    field_path=f"{field_name}.module",
                    message=f"Cannot import module '{module_name}': {e}",
                    severity=EnumContractViolationSeverity.WARNING,
                    suggestion="Verify module path and ensure it's installed",
                )
            )

        return violations

    def _validate_name_convention(
        self,
        file_path: str,
        content: dict,
    ) -> list[ModelContractViolation]:
        """Validate name follows snake_case convention."""
        violations: list[ModelContractViolation] = []
        name = content.get("name")

        if name is None or not isinstance(name, str):
            return violations  # Already caught by required fields check

        # Check snake_case pattern
        if not re.match(r"^[a-z][a-z0-9_]*$", name):
            violations.append(
                ModelContractViolation(
                    file_path=file_path,
                    field_path="name",
                    message=f"Node name '{name}' should be snake_case (lowercase with underscores)",
                    severity=EnumContractViolationSeverity.WARNING,
                    suggestion="Use snake_case: e.g., 'node_registration_orchestrator'",
                )
            )

        return violations

    def _check_recommended_fields(
        self,
        file_path: str,
        content: dict,
    ) -> list[ModelContractViolation]:
        """Check for recommended but optional fields."""
        violations: list[ModelContractViolation] = []
        recommended_fields = ["description", "node_version"]

        for field in recommended_fields:
            if field not in content:
                violations.append(
                    ModelContractViolation(
                        file_path=file_path,
                        field_path=field,
                        message=f"Recommended field '{field}' is missing",
                        severity=EnumContractViolationSeverity.INFO,
                        suggestion=f"Consider adding '{field}:' for better documentation",
                    )
                )

        return violations


def lint_contract_file(
    file_path: str | Path,
    *,
    check_imports: bool = True,
    strict_mode: bool = False,
) -> ModelContractLintResult:
    """
    Lint a single contract.yaml file.

    Convenience function that creates a ContractLinter and lints the file.

    Args:
        file_path: Path to the contract.yaml file.
        check_imports: Whether to verify model modules are importable.
        strict_mode: If True, treat warnings as errors.

    Returns:
        ModelContractLintResult with violations found.
    """
    linter = ContractLinter(check_imports=check_imports, strict_mode=strict_mode)
    return linter.lint_file(Path(file_path))


def lint_contracts_in_directory(
    directory: str | Path,
    *,
    recursive: bool = True,
    check_imports: bool = True,
    strict_mode: bool = False,
) -> ModelContractLintResult:
    """
    Lint all contract.yaml files in a directory.

    Convenience function that creates a ContractLinter and lints the directory.

    Args:
        directory: Directory to search for contract.yaml files.
        recursive: Whether to search subdirectories.
        check_imports: Whether to verify model modules are importable.
        strict_mode: If True, treat warnings as errors.

    Returns:
        ModelContractLintResult with aggregated violations.
    """
    linter = ContractLinter(check_imports=check_imports, strict_mode=strict_mode)
    return linter.lint_directory(Path(directory), recursive=recursive)


def lint_contracts_ci(
    directory: str | Path = "src/omnibase_infra/nodes/",
    *,
    check_imports: bool = True,
    strict_mode: bool = False,
    verbose: bool = False,
) -> tuple[bool, ModelContractLintResult]:
    """
    Lint contracts with CI-friendly output.

    Returns a tuple of (success, result) for easy integration with CI scripts.
    Prints violations to stdout for CI visibility.

    Args:
        directory: Directory to lint.
        check_imports: Whether to verify model modules are importable.
        strict_mode: If True, treat warnings as errors.
        verbose: If True, print all violations including INFO level.

    Returns:
        Tuple of (success: bool, result: ModelContractLintResult).
        success is True if no errors found (and no warnings if strict_mode).
    """
    result = lint_contracts_in_directory(
        directory,
        check_imports=check_imports,
        strict_mode=strict_mode,
    )

    # Print summary
    print(str(result))

    # Print violations
    for violation in result.violations:
        if verbose or violation.severity != EnumContractViolationSeverity.INFO:
            print(f"  {violation}")

    return result.is_valid, result
