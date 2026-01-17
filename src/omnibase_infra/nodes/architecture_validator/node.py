# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Architecture Validator Node - Declarative COMPUTE_GENERIC node for architecture validation.

This COMPUTE_GENERIC node follows the ONEX declarative pattern:
    - DECLARATIVE node driven by contract.yaml
    - Zero custom routing logic - all behavior from validation_rules
    - Lightweight shell that delegates to validator implementations
    - Pattern: "Contract-driven, validators wired externally"

Extends NodeCompute from omnibase_core for pure computation.
All validation rules are 100% driven by contract.yaml, not Python code.

Validation Rules:
    ARCH-001: No Direct Handler Dispatch
        Handlers MUST NOT be invoked directly bypassing the RuntimeHost.
        All event routing MUST go through the MessageDispatchEngine.

    ARCH-002: No Handler Publishing Events
        Handlers MUST NOT have direct event bus access. Only orchestrators
        may publish events. Handlers return events; orchestrators publish.

    ARCH-003: No Workflow FSM in Orchestrators
        Orchestrators MUST NOT duplicate reducer FSM transitions in workflow
        steps. Reducers own FSM; orchestrators coordinate workflow execution.

Design Decisions:
    - 100% Contract-Driven: All validation rules in YAML, not Python
    - Zero Custom Routing: Validators selected based on rule_id
    - Declarative Execution: Rule definitions in validation_rules section
    - Pure Computation: No I/O operations in this node

Related Modules:
    - contract.yaml: Validation rule definitions
    - validators/: Rule-specific validator implementations
    - models/: Input/output model definitions

Ticket: OMN-1099
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from omnibase_core.nodes.node_compute import NodeCompute
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError
from omnibase_infra.nodes.architecture_validator.models.model_architecture_violation import (
    ModelArchitectureViolation,
)
from omnibase_infra.nodes.architecture_validator.models.model_validation_request import (
    ModelArchitectureValidationRequest,
)
from omnibase_infra.nodes.architecture_validator.models.model_validation_result import (
    ModelFileValidationResult,
)
from omnibase_infra.nodes.architecture_validator.validators import (
    validate_no_direct_dispatch,
    validate_no_handler_publishing,
    validate_no_orchestrator_fsm,
)

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer

# Type alias for validator functions
_ValidatorFunc = Callable[[str], ModelFileValidationResult]


class NodeArchitectureValidator(NodeCompute):
    """Architecture Validator - COMPUTE_GENERIC node for architecture pattern validation.

    This node validates ONEX architecture patterns by analyzing Python source
    code and contract files. It detects violations of three core rules:
    - ARCH-001: No direct handler dispatch bypassing runtime
    - ARCH-002: No handler publishing events (only orchestrators may publish)
    - ARCH-003: No workflow FSM in orchestrators duplicating reducer transitions

    All validation logic, rule definitions, and detection strategies are handled
    by the contract.yaml configuration and pluggable validators.

    Example YAML Contract:
        ```yaml
        validation_rules:
          - rule_id: "ARCH-001"
            name: "No Direct Handler Dispatch"
            severity: "CRITICAL"
            detection_strategy:
              type: "ast_pattern"
              patterns:
                - "direct_handler_call"
        ```

    Usage:
        ```python
        from omnibase_core.models.container import ModelONEXContainer

        # Create and initialize
        container = ModelONEXContainer()
        validator = NodeArchitectureValidator(container)

        # Validate paths
        request = ModelArchitectureValidationRequest(
            paths=["src/omnibase_infra/"],
        )
        result = await validator.compute(request)
        print(f"Validation passed: {result.valid}")
        ```

    Attributes:
        _validators: List of (rule_id, validator_func) tuples for each rule.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the architecture validator.

        Args:
            container: ONEX dependency injection container.

        Note:
            All three validators are wired at initialization:
            - ARCH-001: validate_no_direct_dispatch
            - ARCH-002: validate_no_handler_publishing
            - ARCH-003: validate_no_orchestrator_fsm
        """
        super().__init__(container)
        self._validators: list[tuple[str, _ValidatorFunc]] = [
            ("ARCH-001", validate_no_direct_dispatch),
            ("ARCH-002", validate_no_handler_publishing),
            ("ARCH-003", validate_no_orchestrator_fsm),
        ]

    def compute(
        self,
        request: ModelArchitectureValidationRequest,
    ) -> ModelFileValidationResult:
        """Compute architecture validation results.

        Analyzes the specified paths for architecture violations based on
        the rules defined in contract.yaml. Returns a structured result
        with validation status and any violations found.

        Args:
            request: Validation request containing paths to analyze,
                optional rule filters, and configuration options.

        Returns:
            ModelArchitectureValidationResult with valid status,
            violations list, files checked count, and rules applied.

        Raises:
            OnexError: If validation encounters an unrecoverable error.

        Algorithm:
            1. Collect Python files from paths (files, directories, globs)
            2. Filter validators by requested rule_ids (if specified)
            3. Run each validator on each file
            4. Aggregate violations and track file/rule counts
            5. If fail_fast=True, stop on first violation
        """
        all_violations: list[ModelArchitectureViolation] = []
        rules_checked: list[str] = []

        # Collect Python files from paths
        files_to_check = self._collect_files(request.paths)

        # Count files once, not per validator
        files_checked = len(files_to_check)

        # Determine which rules to run
        validators_to_run = self._validators
        if request.rule_ids:
            validators_to_run = [
                (rule_id, validator)
                for rule_id, validator in self._validators
                if rule_id in request.rule_ids
            ]

        # Run validators
        for rule_id, validator in validators_to_run:
            rules_checked.append(rule_id)

            for file_path in files_to_check:
                result = validator(str(file_path))
                all_violations.extend(result.violations)

                # Fail fast if requested
                if request.fail_fast and all_violations:
                    return ModelFileValidationResult(
                        valid=False,
                        violations=all_violations,
                        files_checked=files_checked,
                        rules_checked=rules_checked,
                    )

        return ModelFileValidationResult(
            valid=len(all_violations) == 0,
            violations=all_violations,
            files_checked=files_checked,
            rules_checked=rules_checked,
        )

    def _collect_files(self, paths: list[str]) -> list[Path]:
        """Collect Python files from paths.

        Args:
            paths: List of file paths, directory paths, or glob patterns.

        Returns:
            List of Python file paths to validate.

        Raises:
            ProtocolConfigurationError: If path attempts directory traversal or is absolute outside cwd.

        Examples:
            - File: "src/module/file.py" -> [Path("src/module/file.py")]
            - Directory: "src/" -> [all .py files recursively]
            - Glob: "src/**/*.py" -> [matched files]
        """
        files: list[Path] = []
        cwd = Path.cwd().resolve()

        for path_str in paths:
            # Security: reject path traversal attempts
            if ".." in path_str:
                context = ModelInfraErrorContext.with_correlation(
                    transport_type=EnumInfraTransportType.FILESYSTEM,
                    operation="collect_files",
                )
                raise ProtocolConfigurationError(
                    f"Path traversal not allowed: {path_str}",
                    context=context,
                )

            path = Path(path_str)

            # Security: resolve and verify path is within cwd or is relative
            resolved = path.resolve() if path.exists() else (cwd / path).resolve()
            if path.is_absolute() and not str(resolved).startswith(str(cwd)):
                context = ModelInfraErrorContext.with_correlation(
                    transport_type=EnumInfraTransportType.FILESYSTEM,
                    operation="collect_files",
                )
                raise ProtocolConfigurationError(
                    f"Absolute path outside working directory not allowed: {path_str}",
                    context=context,
                )

            if path.is_file() and path.suffix == ".py":
                files.append(path)
            elif path.is_dir():
                files.extend(path.rglob("*.py"))
            elif "*" in path_str:
                # Glob pattern
                files.extend(Path().glob(path_str))

        return files


__all__ = ["NodeArchitectureValidator"]
