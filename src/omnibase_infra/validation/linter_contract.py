# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""
ONEX Infrastructure Contract Linter.

Validates contract.yaml files against ONEX infrastructure requirements:
- Required fields: name, node_type, contract_version, input_model, output_model
- Type consistency: input_model/output_model module references are importable
- YAML syntax validity
- Node type constraints (EFFECT_GENERIC, COMPUTE_GENERIC, REDUCER_GENERIC, ORCHESTRATOR_GENERIC)
- Line-number error reporting for YAML validation errors (OMN-517)
- Contract dependency validation (imports, references) (OMN-517)
- Contract version compatibility validation (OMN-517)
- "Did you mean?" suggestions for unrecognized fields (OMN-517)

This linter complements omnibase_core.validation.validate_contracts by adding
infrastructure-specific validation that is not covered by the base validator.

Integration with Structured Error Reporting (OMN-1091):
    The linter now supports converting contract violations to structured
    ModelHandlerValidationError instances with unique rule IDs, handler
    identities, and remediation hints. Use ModelContractLintResult.to_handler_errors()
    to convert violations to structured errors.

    Rule ID Mapping:
        CONTRACT-001: YAML parse error
        CONTRACT-002: Missing required field
        CONTRACT-003: Invalid node_type
        CONTRACT-004: Invalid field type
        CONTRACT-005: Import error for models
        CONTRACT-006: Invalid contract_version format
        CONTRACT-007: Invalid model reference
        CONTRACT-008: Invalid name convention
        CONTRACT-009: File not found
        CONTRACT-010: Non-dict contract
        CONTRACT-011: Model not found in module
        CONTRACT-012: Encoding error
        CONTRACT-013: Unknown field (with "did you mean?" suggestion)
        CONTRACT-014: Invalid dependency declaration
        CONTRACT-015: Version compatibility error

Usage:
    from omnibase_infra.validation.linter_contract import (
        ContractLinter,
        lint_contracts_in_directory,
        lint_contract_file,
        convert_violation_to_handler_error,
    )

    # Lint all contracts in a directory
    result = lint_contracts_in_directory("src/omnibase_infra/nodes/")

    # Lint a single contract file
    result = lint_contract_file("path/to/contract.yaml")

    # Convert violations to structured errors
    errors = result.to_handler_errors()
    for error in errors:
        logger.error(error.format_for_logging())

Exit Codes (for CI):
    0: All contracts valid
    1: Validation failures found
    2: Runtime error (file not found, YAML parse error, etc.)
"""

import difflib
import importlib
import logging
import re
from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid4

import yaml

from omnibase_infra.enums import EnumValidationSeverity
from omnibase_infra.models.errors import ModelHandlerValidationError
from omnibase_infra.models.handlers import ModelHandlerIdentifier
from omnibase_infra.types import PathInput
from omnibase_infra.validation.enums.enum_contract_violation_severity import (
    EnumContractViolationSeverity,
)
from omnibase_infra.validation.models.model_contract_lint_result import (
    ModelContractLintResult,
)
from omnibase_infra.validation.models.model_contract_violation import (
    ModelContractViolation,
)

# Module-level logger
logger = logging.getLogger(__name__)


# Valid node types per ONEX 4-node architecture (omnibase_core 0.7.0+)
VALID_NODE_TYPES = frozenset(
    {"EFFECT_GENERIC", "COMPUTE_GENERIC", "REDUCER_GENERIC", "ORCHESTRATOR_GENERIC"}
)

# All known valid top-level contract fields (for "did you mean?" suggestions).
# This set includes all fields found across existing contracts in the codebase.
KNOWN_CONTRACT_FIELDS = frozenset(
    {
        # Core required fields
        "name",
        "contract_name",
        "node_name",
        "node_type",
        "contract_version",
        "node_version",
        "description",
        "input_model",
        "output_model",
        # Routing and handler configuration
        "handler_routing",
        "handler_id",
        "operation_bindings",
        # Event and intent configuration
        "consumed_events",
        "published_events",
        "produced_events",
        "event_bus",
        "intent_consumption",
        "intent_emission",
        "input_subscriptions",
        # Workflow and coordination
        "workflow_coordination",
        "time_injection",
        "projection_reader",
        # Infrastructure and operations
        "capabilities",
        "dependencies",
        "metadata",
        "error_handling",
        "health_check",
        "mcp",
        "tags",
        # Domain-specific contract extensions
        "config",
        "database_schema",
        "definitions",
        "descriptor",
        "idempotency",
        "io_operations",
        "performance",
        "staleness",
        "state_machine",
        "state_model",
        "supported_rules",
        "validation",
        "validation_rules",
    }
)

# Required fields in a valid contract.yaml
REQUIRED_CONTRACT_FIELDS = [
    "name",
    "node_type",
    "contract_version",
    "input_model",
    "output_model",
]

# Valid dependency types (includes all types used across existing contracts)
VALID_DEPENDENCY_TYPES = frozenset(
    {
        "protocol",
        "node",
        "environment",
        "service",
        "handler",
        "library",
        "mixin",
        "path",
        "ModelONEXContainer",
    }
)


# Rule ID mapping for contract violations
class ContractRuleId:
    """Rule IDs for contract validation errors.

    These IDs provide unique identifiers for each type of contract validation
    failure, enabling structured error tracking and remediation guidance.
    """

    YAML_PARSE_ERROR = "CONTRACT-001"
    MISSING_REQUIRED_FIELD = "CONTRACT-002"
    INVALID_NODE_TYPE = "CONTRACT-003"
    INVALID_FIELD_TYPE = "CONTRACT-004"
    IMPORT_ERROR = "CONTRACT-005"
    INVALID_CONTRACT_VERSION = "CONTRACT-006"
    INVALID_MODEL_REFERENCE = "CONTRACT-007"
    INVALID_NAME_CONVENTION = "CONTRACT-008"
    FILE_NOT_FOUND = "CONTRACT-009"
    NON_DICT_CONTRACT = "CONTRACT-010"
    MODEL_NOT_FOUND = "CONTRACT-011"
    ENCODING_ERROR = "CONTRACT-012"
    UNKNOWN_FIELD = "CONTRACT-013"
    INVALID_DEPENDENCY = "CONTRACT-014"
    VERSION_COMPATIBILITY = "CONTRACT-015"


def _get_yaml_line_numbers(file_path: Path) -> dict[str, int]:
    """Extract line numbers for top-level YAML keys from a contract file.

    Parses the raw YAML text to find line numbers for each top-level key.
    This enables line-number error reporting without requiring a custom YAML
    loader, which keeps compatibility with yaml.safe_load.

    Args:
        file_path: Path to the YAML file.

    Returns:
        Dict mapping top-level key names to their 1-based line numbers.
        Nested keys use dot notation (e.g., "contract_version.major").

    Example:
        >>> lines = _get_yaml_line_numbers(Path("contract.yaml"))
        >>> lines["node_type"]
        5
        >>> lines["contract_version.major"]
        7
    """
    line_map: dict[str, int] = {}
    try:
        with file_path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
    except (OSError, UnicodeDecodeError):
        return line_map

    # Track current top-level key for nested key detection
    current_top_key: str | None = None

    for line_num, line in enumerate(lines, start=1):
        stripped = line.rstrip()

        # Skip comments and empty lines
        if not stripped or stripped.startswith("#"):
            continue

        # Top-level key: no leading whitespace, ends with ":"
        match = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*):", line)
        if match:
            key = match.group(1)
            line_map[key] = line_num
            current_top_key = key
            continue

        # Nested key (indented): only track one level deep under current top key
        if current_top_key:
            nested_match = re.match(r"^\s+([a-zA-Z_][a-zA-Z0-9_]*):", line)
            if nested_match:
                nested_key = nested_match.group(1)
                line_map[f"{current_top_key}.{nested_key}"] = line_num

    return line_map


def _suggest_similar_field(
    unknown_field: str, known_fields: frozenset[str]
) -> str | None:
    """Find the closest matching known field for a typo suggestion.

    Uses difflib.get_close_matches with a cutoff of 0.6 to find plausible
    "did you mean?" suggestions for unrecognized contract fields.

    Args:
        unknown_field: The unrecognized field name.
        known_fields: Set of valid field names to match against.

    Returns:
        The closest matching field name, or None if no close match found.

    Example:
        >>> _suggest_similar_field("node_typ", KNOWN_CONTRACT_FIELDS)
        'node_type'
        >>> _suggest_similar_field("input_modle", KNOWN_CONTRACT_FIELDS)
        'input_model'
        >>> _suggest_similar_field("zzz_nonexistent", KNOWN_CONTRACT_FIELDS)
        None
    """
    matches = difflib.get_close_matches(
        unknown_field, sorted(known_fields), n=1, cutoff=0.6
    )
    return matches[0] if matches else None


def convert_violation_to_handler_error(
    violation: ModelContractViolation,
    correlation_id: UUID | None = None,
) -> ModelHandlerValidationError:
    """Convert contract violation to structured handler validation error.

    Maps ModelContractViolation to ModelHandlerValidationError with appropriate
    rule IDs, handler identity, and remediation hints for structured error reporting.

    Args:
        violation: Contract violation to convert.
        correlation_id: Optional correlation ID for distributed tracing.

    Returns:
        ModelHandlerValidationError with structured error information.

    Example:
        >>> violation = ModelContractViolation(
        ...     file_path="nodes/registration/contract.yaml",
        ...     field_path="node_type",
        ...     message="Invalid node_type 'INVALID'",
        ...     severity=EnumContractViolationSeverity.ERROR,
        ... )
        >>> error = convert_violation_to_handler_error(violation)
        >>> error.rule_id
        'CONTRACT-003'
    """
    # Derive handler_id from file path (e.g., nodes/registration/contract.yaml -> registration)
    file_path = violation.file_path
    handler_id = _derive_handler_id_from_path(file_path)

    # Create handler identifier
    handler_identity = ModelHandlerIdentifier.from_handler_id(handler_id)

    # Map violation to rule ID based on field_path and message
    rule_id = _map_violation_to_rule_id(violation)

    # Map severity
    severity = (
        EnumValidationSeverity.ERROR
        if violation.severity == EnumContractViolationSeverity.ERROR
        else EnumValidationSeverity.WARNING
    )

    # Use suggestion as remediation hint, or provide default
    remediation_hint = (
        violation.suggestion or "Review contract.yaml and fix the validation error"
    )

    return ModelHandlerValidationError.from_contract_error(
        rule_id=rule_id,
        message=violation.message,
        file_path=file_path,
        remediation_hint=remediation_hint,
        handler_identity=handler_identity,
        line_number=violation.line_number,
        correlation_id=correlation_id or uuid4(),
        severity=severity,
    )


def _derive_handler_id_from_path(file_path: str) -> str:
    """Derive handler ID from contract file path.

    Extracts the node name from the contract file path to use as handler_id.
    For root-level contract.yaml files without a meaningful parent directory,
    falls back to using the filename stem as the handler ID.

    Args:
        file_path: Path to contract.yaml file.

    Returns:
        Derived handler ID (e.g., "registration" from "nodes/registration/contract.yaml").

    Example:
        >>> _derive_handler_id_from_path("nodes/registration/contract.yaml")
        'registration'
        >>> _derive_handler_id_from_path("contract.yaml")
        'contract'
        >>> _derive_handler_id_from_path("./contract.yaml")
        'contract'
        >>> _derive_handler_id_from_path("/tmp/contract.yaml")
        'tmp'
    """
    # Extract parent directory name as handler ID
    path = Path(file_path)
    if path.name == "contract.yaml":
        # Get parent directory name, but only if it's meaningful (not ".", "", or just "/")
        parent_name = path.parent.name
        if parent_name and parent_name not in {".", "", "/"}:
            return parent_name
    # Fallback to filename without extension for root-level files
    return path.stem


def _map_violation_to_rule_id(violation: ModelContractViolation) -> str:  # stub-ok
    """Map contract violation to appropriate rule ID using keyword heuristics.

    This function uses a decision tree based on keyword matching in the violation's
    field_path and message to determine the appropriate CONTRACT-xxx rule ID.
    The mapping provides unique, stable rule IDs for structured error reporting.

    Mapping Logic (evaluated in order):

    1. **YAML/File-Level Errors** (no field_path):
       - "not found" -> CONTRACT-009 (FILE_NOT_FOUND)
       - "yaml" + "parse" -> CONTRACT-001 (YAML_PARSE_ERROR)
       - "encoding" or "binary" -> CONTRACT-012 (ENCODING_ERROR)
       - "must be a yaml mapping" or "must be a dict" -> CONTRACT-010 (NON_DICT_CONTRACT)

    2. **Missing Fields**:
       - "missing" + "required field" -> CONTRACT-002 (MISSING_REQUIRED_FIELD)

    3. **Field-Specific Errors** (based on field_path):
       - field_path == "node_type":
         - "invalid node_type" -> CONTRACT-003 (INVALID_NODE_TYPE)
         - "must be a string" -> CONTRACT-004 (INVALID_FIELD_TYPE)
       - field_path starts with "contract_version" -> CONTRACT-006 (INVALID_CONTRACT_VERSION)
       - field_path starts with "input_model" or "output_model":
         - "cannot import" -> CONTRACT-005 (IMPORT_ERROR)
         - "not found in module" -> CONTRACT-011 (MODEL_NOT_FOUND)
         - Otherwise -> CONTRACT-007 (INVALID_MODEL_REFERENCE)
       - field_path == "name" -> CONTRACT-008 (INVALID_NAME_CONVENTION)

    4. **New in OMN-517**:
       - "unknown field" or "did you mean" -> CONTRACT-013 (UNKNOWN_FIELD)
       - field_path starts with "dependencies" -> CONTRACT-014 (INVALID_DEPENDENCY)
       - "version" + "compatibility" -> CONTRACT-015 (VERSION_COMPATIBILITY)

    5. **Default Fallback**:
       - CONTRACT-004 (INVALID_FIELD_TYPE) for any unmatched violations

    Args:
        violation: Contract violation to map.

    Returns:
        Rule ID string (e.g., "CONTRACT-001").
    """
    field_path = violation.field_path
    message_lower = violation.message.lower()

    # YAML/file errors
    if "not found" in message_lower and not field_path:
        return ContractRuleId.FILE_NOT_FOUND
    if "yaml" in message_lower and "parse" in message_lower:
        return ContractRuleId.YAML_PARSE_ERROR
    if "encoding" in message_lower or "binary" in message_lower:
        return ContractRuleId.ENCODING_ERROR
    if "must be a yaml mapping" in message_lower or "must be a dict" in message_lower:
        return ContractRuleId.NON_DICT_CONTRACT

    # Field-specific errors
    if not field_path:
        return ContractRuleId.YAML_PARSE_ERROR

    # Unknown field with "did you mean?" (OMN-517)
    if "unknown field" in message_lower or "did you mean" in message_lower:
        return ContractRuleId.UNKNOWN_FIELD

    # Missing required fields
    if "missing" in message_lower and "required field" in message_lower:
        return ContractRuleId.MISSING_REQUIRED_FIELD

    # Dependency validation (OMN-517)
    if field_path.startswith("dependencies"):
        return ContractRuleId.INVALID_DEPENDENCY

    # Version compatibility (OMN-517)
    if "version" in message_lower and "compatibility" in message_lower:
        return ContractRuleId.VERSION_COMPATIBILITY

    # Node type validation
    if field_path == "node_type":
        if "invalid node_type" in message_lower:
            return ContractRuleId.INVALID_NODE_TYPE
        if "must be a string" in message_lower:
            return ContractRuleId.INVALID_FIELD_TYPE

    # Contract version validation
    if field_path.startswith("contract_version"):
        return ContractRuleId.INVALID_CONTRACT_VERSION

    # Model reference validation
    if field_path.startswith(("input_model", "output_model")):
        if "cannot import" in message_lower:
            return ContractRuleId.IMPORT_ERROR
        if "not found in module" in message_lower:
            return ContractRuleId.MODEL_NOT_FOUND
        return ContractRuleId.INVALID_MODEL_REFERENCE

    # Name convention validation
    if field_path == "name":
        return ContractRuleId.INVALID_NAME_CONVENTION

    # Default to invalid field type
    return ContractRuleId.INVALID_FIELD_TYPE


class ContractLinter:
    """ONEX Infrastructure Contract Linter.

    Validates contract.yaml files for required fields, type consistency,
    and ONEX compliance. Designed for CI integration with clear exit codes.

    Enhanced in OMN-517 with:
    - Line-number error reporting for YAML validation errors
    - "Did you mean?" suggestions for unrecognized fields
    - Contract dependency validation (imports, references)
    - Contract version compatibility validation

    Required Fields:
        - name: Node identifier (snake_case)
        - node_type: One of EFFECT_GENERIC, COMPUTE_GENERIC, REDUCER_GENERIC, ORCHESTRATOR_GENERIC
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
        check_unknown_fields: bool = True,
        check_dependencies: bool = True,
        check_version_compatibility: bool = True,
    ):
        """Initialize the contract linter.

        Args:
            check_imports: Whether to verify input_model/output_model modules
                          are importable. Disable for faster validation when
                          modules may not be in the Python path.
            strict_mode: If True, treat warnings as errors.
            check_unknown_fields: If True, warn about unrecognized top-level
                          fields with "did you mean?" suggestions.
            check_dependencies: If True, validate dependency declarations
                          for structure and importability.
            check_version_compatibility: If True, validate that contract_version
                          is compatible with the current runtime.
        """
        self.check_imports = check_imports
        self.strict_mode = strict_mode
        self.check_unknown_fields = check_unknown_fields
        self.check_dependencies = check_dependencies
        self.check_version_compatibility = check_version_compatibility

    def lint_file(self, file_path: Path) -> ModelContractLintResult:
        """Lint a single contract.yaml file.

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

        # Extract line numbers for error reporting (OMN-517)
        line_map = _get_yaml_line_numbers(file_path)

        # Parse YAML
        try:
            with file_path.open("r", encoding="utf-8") as f:
                content = yaml.safe_load(f)
        except yaml.YAMLError as e:
            # Extract line number from YAML error if available
            yaml_line: int | None = None
            if hasattr(e, "problem_mark") and e.problem_mark is not None:
                yaml_line = e.problem_mark.line + 1  # 0-based to 1-based
            violations.append(
                ModelContractViolation(
                    file_path=file_str,
                    field_path="",
                    message=f"YAML parse error: {e}",
                    severity=EnumContractViolationSeverity.ERROR,
                    line_number=yaml_line,
                )
            )
            return ModelContractLintResult(
                is_valid=False,
                violations=violations,
                files_checked=1,
                files_with_errors=1,
            )
        except UnicodeDecodeError as e:
            violations.append(
                ModelContractViolation(
                    file_path=file_str,
                    field_path="",
                    message=f"Contract file contains binary or non-UTF-8 content: "
                    f"encoding error at position {e.start}-{e.end}: {e.reason}",
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
        violations.extend(self._validate_required_fields(file_str, content, line_map))

        # Validate node_type
        violations.extend(self._validate_node_type(file_str, content, line_map))

        # Validate contract_version format
        violations.extend(self._validate_contract_version(file_str, content, line_map))

        # Validate input_model and output_model
        violations.extend(
            self._validate_model_reference(file_str, content, "input_model", line_map)
        )
        violations.extend(
            self._validate_model_reference(file_str, content, "output_model", line_map)
        )

        # Validate naming convention (name should be snake_case)
        violations.extend(self._validate_name_convention(file_str, content, line_map))

        # Check for recommended fields
        violations.extend(self._check_recommended_fields(file_str, content, line_map))

        # Validate unknown fields with "did you mean?" suggestions (OMN-517)
        if self.check_unknown_fields:
            violations.extend(
                self._validate_unknown_fields(file_str, content, line_map)
            )

        # Validate dependencies (OMN-517)
        if self.check_dependencies:
            violations.extend(self._validate_dependencies(file_str, content, line_map))

        # Validate version compatibility (OMN-517)
        if self.check_version_compatibility:
            violations.extend(
                self._validate_version_compatibility(file_str, content, line_map)
            )

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
        """Lint all contract.yaml files in a directory.

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

    # ONEX_EXCLUDE: any_type - YAML contract content is heterogeneous dict from yaml.safe_load
    def _validate_required_fields(
        self,
        file_path: str,
        content: dict[str, Any],
        line_map: dict[str, int],
    ) -> list[ModelContractViolation]:
        """Validate that all required top-level fields are present."""
        violations: list[ModelContractViolation] = []

        for field in REQUIRED_CONTRACT_FIELDS:
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

    # ONEX_EXCLUDE: any_type - YAML contract content is heterogeneous dict from yaml.safe_load
    def _validate_node_type(
        self,
        file_path: str,
        content: dict[str, Any],
        line_map: dict[str, int],
    ) -> list[ModelContractViolation]:
        """Validate node_type is one of the valid ONEX 4-node types."""
        violations: list[ModelContractViolation] = []
        node_type = content.get("node_type")

        if node_type is None:
            return violations  # Already caught by required fields check

        line_num = line_map.get("node_type")

        if not isinstance(node_type, str):
            violations.append(
                ModelContractViolation(
                    file_path=file_path,
                    field_path="node_type",
                    message=f"node_type must be a string, got {type(node_type).__name__}",
                    severity=EnumContractViolationSeverity.ERROR,
                    line_number=line_num,
                )
            )
            return violations

        if node_type not in VALID_NODE_TYPES:
            # Provide "did you mean?" for near-matches
            suggestion_text = (
                f"Change node_type to one of: {', '.join(sorted(VALID_NODE_TYPES))}"
            )
            similar = _suggest_similar_field(node_type, VALID_NODE_TYPES)
            if similar:
                suggestion_text = f"Did you mean '{similar}'? Valid values: {', '.join(sorted(VALID_NODE_TYPES))}"

            violations.append(
                ModelContractViolation(
                    file_path=file_path,
                    field_path="node_type",
                    message=f"Invalid node_type '{node_type}'. Must be one of: {', '.join(sorted(VALID_NODE_TYPES))}",
                    severity=EnumContractViolationSeverity.ERROR,
                    suggestion=suggestion_text,
                    line_number=line_num,
                )
            )

        return violations

    # ONEX_EXCLUDE: any_type - YAML contract content is heterogeneous dict from yaml.safe_load
    def _validate_contract_version(
        self,
        file_path: str,
        content: dict[str, Any],
        line_map: dict[str, int],
    ) -> list[ModelContractViolation]:
        """Validate contract_version has proper semver structure."""
        violations: list[ModelContractViolation] = []
        version = content.get("contract_version")

        if version is None:
            return violations  # Already caught by required fields check

        line_num = line_map.get("contract_version")

        if not isinstance(version, dict):
            violations.append(
                ModelContractViolation(
                    file_path=file_path,
                    field_path="contract_version",
                    message="contract_version must be a dict with 'major', 'minor', 'patch' keys",
                    severity=EnumContractViolationSeverity.ERROR,
                    suggestion="Use format: contract_version:\\n  major: 1\\n  minor: 0\\n  patch: 0",
                    line_number=line_num,
                )
            )
            return violations

        for key in ["major", "minor", "patch"]:
            nested_line = line_map.get(f"contract_version.{key}")
            if key not in version:
                violations.append(
                    ModelContractViolation(
                        file_path=file_path,
                        field_path=f"contract_version.{key}",
                        message=f"contract_version missing required field '{key}'",
                        severity=EnumContractViolationSeverity.ERROR,
                        line_number=line_num,
                    )
                )
            elif not isinstance(version[key], int):
                violations.append(
                    ModelContractViolation(
                        file_path=file_path,
                        field_path=f"contract_version.{key}",
                        message=f"contract_version.{key} must be an integer, got {type(version[key]).__name__}",
                        severity=EnumContractViolationSeverity.ERROR,
                        line_number=nested_line or line_num,
                    )
                )
            elif version[key] < 0:
                violations.append(
                    ModelContractViolation(
                        file_path=file_path,
                        field_path=f"contract_version.{key}",
                        message=f"contract_version.{key} must be non-negative, got {version[key]}",
                        severity=EnumContractViolationSeverity.ERROR,
                        line_number=nested_line or line_num,
                    )
                )

        return violations

    # ONEX_EXCLUDE: any_type - YAML contract content is heterogeneous dict from yaml.safe_load
    def _validate_model_reference(
        self,
        file_path: str,
        content: dict[str, Any],
        field_name: Literal["input_model", "output_model"],
        line_map: dict[str, int],
    ) -> list[ModelContractViolation]:
        """Validate input_model or output_model reference structure and importability."""
        violations: list[ModelContractViolation] = []
        model_ref = content.get(field_name)

        if model_ref is None:
            return violations  # Already caught by required fields check

        line_num = line_map.get(field_name)

        if not isinstance(model_ref, dict):
            violations.append(
                ModelContractViolation(
                    file_path=file_path,
                    field_path=field_name,
                    message=f"{field_name} must be a dict with 'name' and 'module' keys",
                    severity=EnumContractViolationSeverity.ERROR,
                    suggestion=f"Use format: {field_name}:\\n  name: ModelName\\n  module: package.module",
                    line_number=line_num,
                )
            )
            return violations

        # Check required sub-fields
        for key in ["name", "module"]:
            nested_line = line_map.get(f"{field_name}.{key}")
            if key not in model_ref:
                violations.append(
                    ModelContractViolation(
                        file_path=file_path,
                        field_path=f"{field_name}.{key}",
                        message=f"{field_name} missing required field '{key}'",
                        severity=EnumContractViolationSeverity.ERROR,
                        line_number=line_num,
                    )
                )
            elif not isinstance(model_ref[key], str):
                violations.append(
                    ModelContractViolation(
                        file_path=file_path,
                        field_path=f"{field_name}.{key}",
                        message=f"{field_name}.{key} must be a string",
                        severity=EnumContractViolationSeverity.ERROR,
                        line_number=nested_line or line_num,
                    )
                )

        # Validate model name follows ONEX naming convention (Model* prefix)
        model_name = model_ref.get("name")
        if isinstance(model_name, str) and not model_name.startswith("Model"):
            name_line = line_map.get(f"{field_name}.name")
            violations.append(
                ModelContractViolation(
                    file_path=file_path,
                    field_path=f"{field_name}.name",
                    message=f"{field_name}.name should start with 'Model' prefix per ONEX conventions",
                    severity=EnumContractViolationSeverity.WARNING,
                    suggestion=f"Rename to 'Model{model_name}'",
                    line_number=name_line or line_num,
                )
            )

        # Check if module is importable (optional, can be slow)
        if self.check_imports:
            violations.extend(
                self._check_module_importable(
                    file_path, field_name, model_ref, line_map
                )
            )

        return violations

    # ONEX_EXCLUDE: any_type - YAML model reference is heterogeneous dict from yaml.safe_load
    def _check_module_importable(
        self,
        file_path: str,
        field_name: str,
        model_ref: dict[str, Any],
        line_map: dict[str, int],
    ) -> list[ModelContractViolation]:
        """Check if the model's module is importable."""
        violations: list[ModelContractViolation] = []
        module_name = model_ref.get("module")
        class_name = model_ref.get("name")

        if not isinstance(module_name, str) or not isinstance(class_name, str):
            return violations  # Type errors already reported

        module_line = line_map.get(f"{field_name}.module")

        try:
            module = importlib.import_module(module_name)
            if not hasattr(module, class_name):
                name_line = line_map.get(f"{field_name}.name")
                violations.append(
                    ModelContractViolation(
                        file_path=file_path,
                        field_path=f"{field_name}.name",
                        message=f"Class '{class_name}' not found in module '{module_name}'",
                        severity=EnumContractViolationSeverity.ERROR,
                        suggestion=f"Verify the class name exists in {module_name}",
                        line_number=name_line or module_line,
                    )
                )
        except ImportError as e:
            # Import failures are ERROR severity because they violate the type
            # consistency guarantee documented in the module docstring: contracts
            # must reference importable modules to ensure type safety.
            # Use check_imports=False to skip this check in environments where
            # dependencies may not be available.
            violations.append(
                ModelContractViolation(
                    file_path=file_path,
                    field_path=f"{field_name}.module",
                    message=f"Cannot import module '{module_name}': {e}",
                    severity=EnumContractViolationSeverity.ERROR,
                    suggestion="Verify module path and ensure it's installed, or use check_imports=False",
                    line_number=module_line,
                )
            )

        return violations

    # ONEX_EXCLUDE: any_type - YAML contract content is heterogeneous dict from yaml.safe_load
    def _validate_name_convention(
        self,
        file_path: str,
        content: dict[str, Any],
        line_map: dict[str, int],
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
                    line_number=line_map.get("name"),
                )
            )

        return violations

    # ONEX_EXCLUDE: any_type - YAML contract content is heterogeneous dict from yaml.safe_load
    def _check_recommended_fields(
        self,
        file_path: str,
        content: dict[str, Any],
        line_map: dict[str, int],
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

    # ONEX_EXCLUDE: any_type - YAML contract content is heterogeneous dict from yaml.safe_load
    def _validate_unknown_fields(
        self,
        file_path: str,
        content: dict[str, Any],
        line_map: dict[str, int],
    ) -> list[ModelContractViolation]:
        """Detect unrecognized top-level fields and suggest corrections.

        Provides "Did you mean?" suggestions for fields that are close matches
        to known contract fields, helping catch typos early.

        Example output:
            ContractValidationError: Unknown field 'node_typ' at line 5
              Did you mean 'node_type'?
              Valid fields: name, node_type, contract_version, ...
        """
        violations: list[ModelContractViolation] = []

        for field in content:
            if field not in KNOWN_CONTRACT_FIELDS:
                line_num = line_map.get(field)
                similar = _suggest_similar_field(field, KNOWN_CONTRACT_FIELDS)

                if similar:
                    suggestion = f"Did you mean '{similar}'?"
                    at_line = f" at line {line_num}" if line_num else ""
                    message = (
                        f"Unknown field '{field}'{at_line}. "
                        f"Did you mean '{similar}'? "
                        f"Valid fields: {', '.join(sorted(KNOWN_CONTRACT_FIELDS))}"
                    )
                else:
                    suggestion = (
                        f"Valid fields: {', '.join(sorted(KNOWN_CONTRACT_FIELDS))}"
                    )
                    at_line = f" at line {line_num}" if line_num else ""
                    message = (
                        f"Unknown field '{field}'{at_line}. "
                        f"Valid fields: {', '.join(sorted(KNOWN_CONTRACT_FIELDS))}"
                    )

                violations.append(
                    ModelContractViolation(
                        file_path=file_path,
                        field_path=field,
                        message=message,
                        severity=EnumContractViolationSeverity.WARNING,
                        suggestion=suggestion,
                        line_number=line_num,
                    )
                )

        return violations

    # ONEX_EXCLUDE: any_type - YAML contract content is heterogeneous dict from yaml.safe_load
    def _validate_dependencies(
        self,
        file_path: str,
        content: dict[str, Any],
        line_map: dict[str, int],
    ) -> list[ModelContractViolation]:
        """Validate contract dependency declarations.

        Checks that each dependency in the ``dependencies`` list has:
        - A ``name`` field (required, string)
        - A ``type`` field (required, one of: protocol, node, environment, service)
        - A ``module`` field when type is "protocol" or "node" (importable)
        - A ``description`` field (recommended)

        Args:
            file_path: Path to the contract file.
            content: Parsed YAML content.
            line_map: Line number map for error reporting.

        Returns:
            List of violations found in dependency declarations.
        """
        violations: list[ModelContractViolation] = []
        deps = content.get("dependencies")

        if deps is None:
            return violations  # Dependencies are optional

        deps_line = line_map.get("dependencies")

        if not isinstance(deps, list):
            violations.append(
                ModelContractViolation(
                    file_path=file_path,
                    field_path="dependencies",
                    message=f"dependencies must be a list, got {type(deps).__name__}",
                    severity=EnumContractViolationSeverity.ERROR,
                    line_number=deps_line,
                )
            )
            return violations

        for idx, dep in enumerate(deps):
            dep_path = f"dependencies[{idx}]"

            if not isinstance(dep, dict):
                violations.append(
                    ModelContractViolation(
                        file_path=file_path,
                        field_path=dep_path,
                        message=f"Dependency at index {idx} must be a dict, got {type(dep).__name__}",
                        severity=EnumContractViolationSeverity.ERROR,
                        line_number=deps_line,
                    )
                )
                continue

            # Validate required dependency fields
            dep_name = dep.get("name")
            if dep_name is None:
                violations.append(
                    ModelContractViolation(
                        file_path=file_path,
                        field_path=f"{dep_path}.name",
                        message=f"Dependency at index {idx} missing required field 'name'",
                        severity=EnumContractViolationSeverity.ERROR,
                        line_number=deps_line,
                    )
                )
            elif not isinstance(dep_name, str):
                violations.append(
                    ModelContractViolation(
                        file_path=file_path,
                        field_path=f"{dep_path}.name",
                        message=f"Dependency name must be a string, got {type(dep_name).__name__}",
                        severity=EnumContractViolationSeverity.ERROR,
                        line_number=deps_line,
                    )
                )

            dep_type = dep.get("type")
            if dep_type is None:
                violations.append(
                    ModelContractViolation(
                        file_path=file_path,
                        field_path=f"{dep_path}.type",
                        message=f"Dependency '{dep_name or idx}' missing required field 'type'",
                        severity=EnumContractViolationSeverity.ERROR,
                        suggestion=f"Add 'type:' with one of: {', '.join(sorted(VALID_DEPENDENCY_TYPES))}",
                        line_number=deps_line,
                    )
                )
            elif not isinstance(dep_type, str):
                violations.append(
                    ModelContractViolation(
                        file_path=file_path,
                        field_path=f"{dep_path}.type",
                        message=f"Dependency type must be a string, got {type(dep_type).__name__}",
                        severity=EnumContractViolationSeverity.ERROR,
                        line_number=deps_line,
                    )
                )
            elif dep_type not in VALID_DEPENDENCY_TYPES:
                violations.append(
                    ModelContractViolation(
                        file_path=file_path,
                        field_path=f"{dep_path}.type",
                        message=(
                            f"Dependency '{dep_name or idx}' has invalid type '{dep_type}'. "
                            f"Must be one of: {', '.join(sorted(VALID_DEPENDENCY_TYPES))}"
                        ),
                        severity=EnumContractViolationSeverity.ERROR,
                        line_number=deps_line,
                    )
                )

            # Validate module importability for protocol/node dependencies
            if (
                self.check_imports
                and isinstance(dep_type, str)
                and dep_type in {"protocol", "node"}
            ):
                dep_module = dep.get("module")
                if dep_module is not None and isinstance(dep_module, str):
                    try:
                        importlib.import_module(dep_module)
                    except ImportError as e:
                        violations.append(
                            ModelContractViolation(
                                file_path=file_path,
                                field_path=f"{dep_path}.module",
                                message=f"Cannot import dependency module '{dep_module}': {e}",
                                severity=EnumContractViolationSeverity.ERROR,
                                suggestion="Verify the module path is correct and the package is installed",
                                line_number=deps_line,
                            )
                        )

            # Recommend description
            if "description" not in dep:
                violations.append(
                    ModelContractViolation(
                        file_path=file_path,
                        field_path=f"{dep_path}.description",
                        message=f"Dependency '{dep_name or idx}' is missing a description",
                        severity=EnumContractViolationSeverity.INFO,
                        suggestion="Add a 'description:' field for better documentation",
                        line_number=deps_line,
                    )
                )

        return violations

    # ONEX_EXCLUDE: any_type - YAML contract content is heterogeneous dict from yaml.safe_load
    def _validate_version_compatibility(
        self,
        file_path: str,
        content: dict[str, Any],
        line_map: dict[str, int],
    ) -> list[ModelContractViolation]:
        """Validate contract version is compatible with the current runtime.

        Checks that the contract_version follows semantic versioning constraints:
        - major version must be >= 1 for production contracts
        - node_version (if present) must be a valid semver string

        This is a structural validation, not a runtime compatibility check.
        Runtime compatibility is handled by version_compatibility.py.
        """
        violations: list[ModelContractViolation] = []
        version = content.get("contract_version")

        if not isinstance(version, dict):
            return violations  # Structure errors already reported

        # Validate node_version format if present
        # node_version can be either a string ("1.0.0") or a dict ({major, minor, patch})
        node_version = content.get("node_version")
        if node_version is not None:
            nv_line = line_map.get("node_version")
            if isinstance(node_version, dict):
                # Dict format: validate it has major/minor/patch like contract_version
                for key in ["major", "minor", "patch"]:
                    nested_line = line_map.get(f"node_version.{key}")
                    if key not in node_version:
                        violations.append(
                            ModelContractViolation(
                                file_path=file_path,
                                field_path=f"node_version.{key}",
                                message=f"node_version missing required field '{key}'",
                                severity=EnumContractViolationSeverity.ERROR,
                                line_number=nv_line,
                            )
                        )
                    elif not isinstance(node_version[key], int):
                        violations.append(
                            ModelContractViolation(
                                file_path=file_path,
                                field_path=f"node_version.{key}",
                                message=f"node_version.{key} must be an integer, got {type(node_version[key]).__name__}",
                                severity=EnumContractViolationSeverity.ERROR,
                                line_number=nested_line or nv_line,
                            )
                        )
            elif isinstance(node_version, str):
                if not re.match(r"^\d+\.\d+\.\d+(-[a-zA-Z0-9.]+)?$", node_version):
                    violations.append(
                        ModelContractViolation(
                            file_path=file_path,
                            field_path="node_version",
                            message=f"node_version '{node_version}' is not a valid semantic version",
                            severity=EnumContractViolationSeverity.WARNING,
                            suggestion="Use semver format: e.g., '1.0.0' or '0.1.0-beta'",
                            line_number=nv_line,
                        )
                    )
            else:
                violations.append(
                    ModelContractViolation(
                        file_path=file_path,
                        field_path="node_version",
                        message=f"node_version must be a string or dict, got {type(node_version).__name__}",
                        severity=EnumContractViolationSeverity.ERROR,
                        suggestion="Use string format '1.0.0' or dict format {major: 1, minor: 0, patch: 0}",
                        line_number=nv_line,
                    )
                )

        # Check version consistency: contract_version and node_version should both exist
        if node_version is not None:
            # Extract node_version major regardless of format (string or dict)
            nv_major: int | None = None
            nv_display: str = ""
            if isinstance(node_version, str):
                nv_parts = node_version.split(".")
                if nv_parts:
                    try:
                        nv_major = int(nv_parts[0])
                        nv_display = node_version
                    except ValueError:
                        pass
            elif isinstance(node_version, dict):
                nv_major_val = node_version.get("major")
                if isinstance(nv_major_val, int):
                    nv_major = nv_major_val
                    nv_display = f"{nv_major_val}.{node_version.get('minor', 0)}.{node_version.get('patch', 0)}"

            major = version.get("major")
            if (
                isinstance(major, int)
                and major == 0
                and nv_major is not None
                and nv_major >= 1
            ):
                cv_line = line_map.get("contract_version")
                violations.append(
                    ModelContractViolation(
                        file_path=file_path,
                        field_path="contract_version",
                        message=(
                            f"Version compatibility warning: node_version is {nv_display} "
                            f"(major >= 1) but contract_version major is 0. "
                            f"Consider bumping contract_version to 1.0.0."
                        ),
                        severity=EnumContractViolationSeverity.INFO,
                        suggestion="Bump contract_version major to 1 when the contract is stable",
                        line_number=cv_line,
                    )
                )

        return violations


def lint_contract_file(
    file_path: PathInput,
    *,
    check_imports: bool = True,
    strict_mode: bool = False,
    check_unknown_fields: bool = True,
    check_dependencies: bool = True,
    check_version_compatibility: bool = True,
) -> ModelContractLintResult:
    """Lint a single contract.yaml file.

    Convenience function that creates a ContractLinter and lints the file.

    Args:
        file_path: Path to the contract.yaml file.
        check_imports: Whether to verify model modules are importable.
        strict_mode: If True, treat warnings as errors.
        check_unknown_fields: If True, warn about unrecognized top-level fields.
        check_dependencies: If True, validate dependency declarations.
        check_version_compatibility: If True, validate version compatibility.

    Returns:
        ModelContractLintResult with violations found.
    """
    linter = ContractLinter(
        check_imports=check_imports,
        strict_mode=strict_mode,
        check_unknown_fields=check_unknown_fields,
        check_dependencies=check_dependencies,
        check_version_compatibility=check_version_compatibility,
    )
    return linter.lint_file(Path(file_path))


def lint_contracts_in_directory(
    directory: PathInput,
    *,
    recursive: bool = True,
    check_imports: bool = True,
    strict_mode: bool = False,
    check_unknown_fields: bool = True,
    check_dependencies: bool = True,
    check_version_compatibility: bool = True,
) -> ModelContractLintResult:
    """Lint all contract.yaml files in a directory.

    Convenience function that creates a ContractLinter and lints the directory.

    Args:
        directory: Directory to search for contract.yaml files.
        recursive: Whether to search subdirectories.
        check_imports: Whether to verify model modules are importable.
        strict_mode: If True, treat warnings as errors.
        check_unknown_fields: If True, warn about unrecognized top-level fields.
        check_dependencies: If True, validate dependency declarations.
        check_version_compatibility: If True, validate version compatibility.

    Returns:
        ModelContractLintResult with aggregated violations.
    """
    linter = ContractLinter(
        check_imports=check_imports,
        strict_mode=strict_mode,
        check_unknown_fields=check_unknown_fields,
        check_dependencies=check_dependencies,
        check_version_compatibility=check_version_compatibility,
    )
    return linter.lint_directory(Path(directory), recursive=recursive)


def lint_contracts_ci(
    directory: PathInput = "src/omnibase_infra/nodes/",
    *,
    check_imports: bool = True,
    strict_mode: bool = False,
    verbose: bool = False,
) -> tuple[bool, ModelContractLintResult]:
    """Lint contracts with CI-friendly output.

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
