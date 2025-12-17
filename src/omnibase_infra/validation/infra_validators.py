"""
Infrastructure-specific validation wrappers.

Provides validators from omnibase_core with sensible defaults for infrastructure code.
All wrappers maintain strong typing and follow ONEX validation patterns.
"""

import re
from pathlib import Path
from typing import Literal, TypedDict

from omnibase_core.validation import (
    CircularImportValidationResult,
    CircularImportValidator,
    ModelContractValidationResult,
    ModelValidationResult,
    ProtocolContractValidator,
    validate_architecture,
    validate_contracts,
    validate_patterns,
    validate_union_usage,
)

# Type alias for cleaner return types in infrastructure validators
# Most validation results return None as the data payload (validation only)
# Using Python 3.12+ type keyword for modern type alias syntax
type ValidationResult = ModelValidationResult[None]


class ExemptionPattern(TypedDict, total=False):
    """
    Structure for validation exemption patterns.

    Uses regex-based matching to handle code evolution gracefully without
    hardcoded line numbers that break when code changes.

    Fields:
        file_pattern: Regex pattern matching the filename (e.g., r"kafka_event_bus\\.py")
        class_pattern: Optional regex for class name (e.g., r"Class 'KafkaEventBus'")
        method_pattern: Optional regex for method name (e.g., r"Function '__init__'")
        violation_pattern: Regex matching the violation type (e.g., r"too many (methods|parameters)")

    Example:
        {
            "file_pattern": r"kafka_event_bus\\.py",
            "class_pattern": r"Class 'KafkaEventBus'",
            "violation_pattern": r"has \\d+ methods"
        }

    Notes:
        - Patterns are matched using re.search() for flexibility
        - All specified patterns must match for an exemption to apply
        - Omitted optional fields are not checked
        - Use raw strings (r"...") for regex patterns
    """

    file_pattern: str
    class_pattern: str
    method_pattern: str
    violation_pattern: str


# Default paths for infrastructure validation
INFRA_SRC_PATH = "src/omnibase_infra/"
INFRA_NODES_PATH = "src/omnibase_infra/nodes/"

# Maximum allowed complex union types in infrastructure code.
# TECH DEBT (OMN-871): Baseline as of 2025-12-17, target: reduce incrementally
#
# Current count breakdown (154 unions as of 2025-12-17):
# - Infrastructure handlers (~90): Consul, Kafka, Vault, PostgreSQL adapters
# - Runtime components (~40): RuntimeHostProcess, handler/policy registries, wiring
# - Models (~24): Event bus models, error context, runtime config, registration events
#
# OMN-891 registration event models contribute 10 unions:
# - model_node_heartbeat_event.py (3): memory_usage_mb, cpu_usage_percent, correlation_id
# - model_node_introspection_event.py (5): node_role, correlation_id, network_id,
#     deployment_id, epoch
# - model_node_registration.py (2): health_endpoint, last_heartbeat
#
# Note: The validator counts X | None (PEP 604) patterns as unions, which is
# the ONEX-preferred syntax per CLAUDE.md. Threshold set to 185 to allow for
# organic growth while maintaining awareness of union complexity.
INFRA_MAX_UNIONS = 185

# Maximum allowed architecture violations in infrastructure code.
# Set to 0 (strict enforcement) to ensure one-model-per-file principle is always followed.
# Infrastructure nodes require strict architecture compliance for maintainability and
# contract-driven code generation.
INFRA_MAX_VIOLATIONS = 0

# Strict mode for pattern validation in infrastructure code.
# Set to True to enforce strict pattern compliance per ONEX CLAUDE.md mandates.
# Specific documented exemptions (KafkaEventBus, RuntimeHostProcess) are handled via the
# exempted_patterns list in validate_infra_patterns(), NOT via global relaxation.
# All other infrastructure code must comply with standard ONEX pattern thresholds.
INFRA_PATTERNS_STRICT = True

# Strict mode for union usage validation in infrastructure code.
# Set to False to allow necessary unions for protocol implementations and service adapters
# while still preventing overly complex union types via INFRA_MAX_UNIONS limit.
INFRA_UNIONS_STRICT = False


def validate_infra_architecture(
    directory: str | Path = INFRA_SRC_PATH,
    max_violations: int = INFRA_MAX_VIOLATIONS,
) -> ValidationResult:
    """
    Validate infrastructure architecture with strict defaults.

    Enforces ONEX one-model-per-file principle critical for infrastructure nodes.

    Args:
        directory: Directory to validate. Defaults to infrastructure source.
        max_violations: Maximum allowed violations. Defaults to INFRA_MAX_VIOLATIONS (0).

    Returns:
        ModelValidationResult with validation status and any errors.
    """
    return validate_architecture(str(directory), max_violations=max_violations)


def validate_infra_contracts(
    directory: str | Path = INFRA_NODES_PATH,
) -> ValidationResult:
    """
    Validate all infrastructure node contracts.

    Validates YAML contract files for Consul, Kafka, Vault, PostgreSQL adapters.

    Args:
        directory: Directory containing node contracts. Defaults to nodes path.

    Returns:
        ModelValidationResult with validation status and any errors.
    """
    return validate_contracts(str(directory))


def validate_infra_patterns(
    directory: str | Path = INFRA_SRC_PATH,
    strict: bool = INFRA_PATTERNS_STRICT,
) -> ValidationResult:
    """
    Validate infrastructure code patterns with infrastructure-specific exemptions.

    Enforces:
    - Model prefix naming (Model*)
    - snake_case file naming
    - Anti-pattern detection (no *Manager, *Handler, *Helper)

    Exemptions:
        KafkaEventBus (kafka_event_bus.py) - Documented infrastructure pattern exception:
        - Class has many methods (threshold: 10) - Event bus lifecycle, pub/sub, circuit breaker
        - __init__ has many parameters (threshold: 5) - Backwards compatibility during config migration

        RuntimeHostProcess (runtime_host_process.py) - Documented infrastructure pattern exception:
        - Class has many methods (threshold: 10) - Lifecycle, message handling, graceful shutdown
        - Central coordinator requiring: start/stop, health_check, _on_message, _handle_envelope,
          shutdown_ready, register_handler, get_handler, and supporting methods (OMN-756)

        These violations are intentional infrastructure patterns documented in:
        - kafka_event_bus.py / runtime_host_process.py class/method docstrings
        - CLAUDE.md "Accepted Pattern Exceptions" section
        - This validator's docstring

    Exemption Pattern Format:
        Uses regex-based matching instead of hardcoded line numbers for resilience
        to code changes. See ExemptionPattern TypedDict for structure details.

        Example:
            {
                "file_pattern": r"kafka_event_bus\\.py",
                "class_pattern": r"Class 'KafkaEventBus'",
                "violation_pattern": r"has \\d+ methods"
            }

    Args:
        directory: Directory to validate. Defaults to infrastructure source.
        strict: Enable strict mode. Defaults to INFRA_PATTERNS_STRICT (True).

    Returns:
        ModelValidationResult with validation status and filtered errors.
        Documented exemptions are filtered from error list but logged for transparency.
    """
    # Run base validation
    base_result = validate_patterns(str(directory), strict=strict)

    # Filter known infrastructure pattern exemptions using regex-based matching
    # Patterns match class/method names and violation types without hardcoded line numbers
    exempted_patterns: list[ExemptionPattern] = [
        # KafkaEventBus method count exemption
        {
            "file_pattern": r"kafka_event_bus\.py",
            "class_pattern": r"Class 'KafkaEventBus'",
            "violation_pattern": r"has \d+ methods",
        },
        # KafkaEventBus __init__ parameter count exemption
        {
            "file_pattern": r"kafka_event_bus\.py",
            "method_pattern": r"Function '__init__'",
            "violation_pattern": r"has \d+ parameters",
        },
        # Protocol method 'execute' exemption (standard plugin architecture pattern)
        {
            "file_pattern": r"protocol_plugin_compute\.py",
            "violation_pattern": r"Function name 'execute' is too generic",
        },
        # Base class method 'execute' exemption (implements protocol pattern)
        {
            "file_pattern": r"plugin_compute_base\.py",
            "violation_pattern": r"Function name 'execute' is too generic",
        },
        # RuntimeHostProcess method count exemption (OMN-756)
        # Central coordinator class that legitimately requires multiple methods for:
        # - Lifecycle management (start, stop, health_check)
        # - Message handling (_on_message, _handle_envelope)
        # - Graceful shutdown (shutdown_ready, drain logic)
        # - Handler management (register_handler, get_handler)
        # This complexity is intentional for the runtime coordinator pattern.
        {
            "file_pattern": r"runtime_host_process\.py",
            "class_pattern": r"Class 'RuntimeHostProcess'",
            "violation_pattern": r"has \d+ methods",
        },
        # RuntimeHostProcess __init__ parameter count exemption (OMN-756)
        # Central coordinator requires multiple configuration parameters:
        # - event_bus, handler_registry, correlation_id_provider, message_router
        # - health_checker, shutdown_handler for lifecycle management
        {
            "file_pattern": r"runtime_host_process\.py",
            "method_pattern": r"Function '__init__'",
            "violation_pattern": r"has \d+ parameters",
        },
        # PolicyRegistry method count exemption
        # Central registry class requires comprehensive policy management:
        # - CRUD operations (register, get, update, remove)
        # - Query operations (list, filter, search)
        # - Lifecycle operations (enable, disable, validate)
        # This is a domain registry pattern, not a code smell.
        {
            "file_pattern": r"policy_registry\.py",
            "class_pattern": r"Class 'PolicyRegistry'",
            "violation_pattern": r"has \d+ methods",
        },
        # PolicyRegistry.register_policy parameter count exemption
        # Policy registration requires multiple fields for complete policy definition
        {
            "file_pattern": r"policy_registry\.py",
            "method_pattern": r"Function 'register_policy'",
            "violation_pattern": r"has \d+ parameters",
        },
        # ModelPolicyKey.policy_id exemption (OMN-812)
        # policy_id is intentionally a human-readable string identifier (e.g., 'exponential_backoff'),
        # NOT a UUID. The _id suffix triggers false positive UUID suggestions.
        {
            "file_pattern": r"model_policy_key\.py",
            "violation_pattern": r"Field 'policy_id' should use UUID",
        },
        # ModelPolicyRegistration.policy_id exemption (OMN-812)
        # Same rationale as ModelPolicyKey - semantic identifier, not UUID
        {
            "file_pattern": r"model_policy_registration\.py",
            "violation_pattern": r"Field 'policy_id' should use UUID",
        },
    ]

    # Filter errors using regex-based pattern matching
    filtered_errors = _filter_exempted_errors(base_result.errors, exempted_patterns)

    # Create wrapper result (avoid mutation)
    return _create_filtered_result(base_result, filtered_errors)


def _filter_exempted_errors(
    errors: list[str],
    exempted_patterns: list[ExemptionPattern],
) -> list[str]:
    """
    Filter errors based on regex exemption patterns.

    Uses regex-based matching to identify exempted violations without relying on
    hardcoded line numbers or exact counts. This makes exemptions resilient to
    code changes while still precisely targeting specific violations.

    Pattern Matching Logic:
        - All specified pattern fields must match for exemption to apply
        - Unspecified optional fields are not checked (e.g., missing method_pattern)
        - Uses re.search() for flexible substring matching
        - Case-sensitive matching for precision

    Args:
        errors: List of error messages from validation.
        exempted_patterns: List of ExemptionPattern dictionaries with regex patterns.

    Returns:
        Filtered list of errors excluding exempted patterns.

    Example:
        Pattern:
            {
                "file_pattern": r"kafka_event_bus\\.py",
                "class_pattern": r"Class 'KafkaEventBus'",
                "violation_pattern": r"has \\d+ methods"
            }

        Matches error:
            "kafka_event_bus.py:123: Class 'KafkaEventBus' has 14 methods (threshold: 10)"

        Does not match:
            "kafka_event_bus.py:50: Function 'connect' has 7 parameters" (no class_pattern)
            "other_file.py:10: Class 'KafkaEventBus' has 14 methods" (wrong file)
    """
    filtered = []
    for err in errors:
        is_exempted = False

        for pattern in exempted_patterns:
            # Extract pattern fields (all are optional except file_pattern in practice)
            file_pattern = pattern.get("file_pattern", "")
            class_pattern = pattern.get("class_pattern", "")
            method_pattern = pattern.get("method_pattern", "")
            violation_pattern = pattern.get("violation_pattern", "")

            # Check if all specified patterns match
            # Skip unspecified (empty) patterns - they match everything
            matches_file = not file_pattern or re.search(file_pattern, err)
            matches_class = not class_pattern or re.search(class_pattern, err)
            matches_method = not method_pattern or re.search(method_pattern, err)
            matches_violation = not violation_pattern or re.search(
                violation_pattern, err
            )

            # All specified patterns must match for exemption
            if matches_file and matches_class and matches_method and matches_violation:
                is_exempted = True
                break

        if not is_exempted:
            filtered.append(err)

    return filtered


def _create_filtered_result(
    base_result: ValidationResult,
    filtered_errors: list[str],
) -> ValidationResult:
    """
    Create a new validation result with filtered errors (wrapper approach).

    Avoids mutating the original result object for better functional programming practices.
    Creates new metadata using model_validate to prevent mutation of Pydantic models.

    Args:
        base_result: Original validation result.
        filtered_errors: Filtered error list.

    Returns:
        New ValidationResult with filtered errors and updated metadata.
    """
    # Calculate filtering statistics
    violations_filtered = len(base_result.errors) - len(filtered_errors)
    all_violations_exempted = violations_filtered > 0 and len(filtered_errors) == 0

    # Create new metadata if present (avoid mutation)
    new_metadata = None
    if base_result.metadata:
        # Use model_copy for deep copy with updates (Pydantic v2 pattern)
        # This works with both real Pydantic models and test mocks
        try:
            new_metadata = base_result.metadata.model_copy(deep=True)
            # Update violations_found if the field exists
            if hasattr(new_metadata, "violations_found"):
                new_metadata.violations_found = len(filtered_errors)
        except AttributeError:
            # Fallback for test mocks that don't support model_copy
            new_metadata = base_result.metadata

    # Create new result (wrapper pattern - no mutation)
    return ModelValidationResult(
        is_valid=all_violations_exempted or base_result.is_valid,
        validated_value=base_result.validated_value,
        issues=base_result.issues,
        errors=filtered_errors,
        warnings=base_result.warnings,
        suggestions=base_result.suggestions,
        summary=base_result.summary,
        details=base_result.details,
        metadata=new_metadata,
    )


def validate_infra_contract_deep(
    contract_path: str | Path,
    contract_type: Literal["effect", "compute", "reducer", "orchestrator"] = "effect",
) -> ModelContractValidationResult:
    """
    Perform deep contract validation for ONEX compliance.

    Uses ProtocolContractValidator for comprehensive contract checking
    suitable for autonomous code generation.

    Args:
        contract_path: Path to the contract YAML file.
        contract_type: Type of contract to validate. Defaults to "effect".

    Returns:
        ModelContractValidationResult with validation status, score, and any errors.
    """
    validator = ProtocolContractValidator()
    return validator.validate_contract_file(Path(contract_path), contract_type)


def validate_infra_union_usage(
    directory: str | Path = INFRA_SRC_PATH,
    max_unions: int = INFRA_MAX_UNIONS,
    strict: bool = INFRA_UNIONS_STRICT,
) -> ValidationResult:
    """
    Validate Union type usage in infrastructure code.

    Prevents overly complex union types that complicate infrastructure code.

    Args:
        directory: Directory to validate. Defaults to infrastructure source.
        max_unions: Maximum allowed complex unions. Defaults to INFRA_MAX_UNIONS (185).
        strict: Enable strict mode for union validation. Defaults to INFRA_UNIONS_STRICT (False).

    Returns:
        ModelValidationResult with validation status and any errors.
    """
    return validate_union_usage(str(directory), max_unions=max_unions, strict=strict)


def validate_infra_circular_imports(
    directory: str | Path = INFRA_SRC_PATH,
) -> CircularImportValidationResult:
    """
    Check for circular imports in infrastructure code.

    Infrastructure packages have complex dependencies; circular imports
    cause runtime issues that are hard to debug.

    Args:
        directory: Directory to check. Defaults to infrastructure source.

    Returns:
        CircularImportValidationResult with detailed import validation results.
        Use result.has_circular_imports to check for issues.
    """
    validator = CircularImportValidator(source_path=Path(directory))
    return validator.validate()


def validate_infra_all(
    directory: str | Path = INFRA_SRC_PATH,
    nodes_directory: str | Path = INFRA_NODES_PATH,
) -> dict[str, ValidationResult | CircularImportValidationResult]:
    """
    Run all validations on infrastructure code.

    Executes all 5 validators with infrastructure-appropriate defaults:
    - Architecture (strict, 0 violations)
    - Contracts (nodes directory)
    - Patterns (strict mode)
    - Union usage (max INFRA_MAX_UNIONS)
    - Circular imports

    Args:
        directory: Main source directory. Defaults to infrastructure source.
        nodes_directory: Nodes directory for contract validation.

    Returns:
        Dictionary mapping validator name to result.
    """
    results: dict[str, ValidationResult | CircularImportValidationResult] = {}

    # HIGH priority validators
    results["architecture"] = validate_infra_architecture(directory)
    results["contracts"] = validate_infra_contracts(nodes_directory)
    results["patterns"] = validate_infra_patterns(directory)

    # MEDIUM priority validators
    results["union_usage"] = validate_infra_union_usage(directory)
    results["circular_imports"] = validate_infra_circular_imports(directory)

    return results


def get_validation_summary(
    results: dict[str, ValidationResult | CircularImportValidationResult],
) -> dict[str, int | list[str]]:
    """
    Generate a summary of validation results.

    Args:
        results: Dictionary of validation results from validate_infra_all().

    Returns:
        Dictionary with summary statistics including passed/failed counts and failed validators.
    """
    passed = 0
    failed = 0
    failed_validators: list[str] = []

    for name, result in results.items():
        # NOTE: isinstance usage here is justified as CircularImportValidationResult
        # and ValidationResult have different APIs (has_circular_imports vs is_valid).
        # Duck typing would require protocol definitions in omnibase_core.
        # This is acceptable for result type discrimination in summary generation.
        if isinstance(result, CircularImportValidationResult):
            # Circular import validator uses has_circular_imports
            if not result.has_circular_imports:
                passed += 1
            else:
                failed += 1
                failed_validators.append(name)
        # Standard ModelValidationResult uses is_valid
        elif result.is_valid:
            passed += 1
        else:
            failed += 1
            failed_validators.append(name)

    return {
        "total_validators": passed + failed,
        "passed": passed,
        "failed": failed,
        "failed_validators": failed_validators,
    }


__all__ = [
    # Type aliases
    "ValidationResult",
    "ExemptionPattern",
    # Re-exported types from omnibase_core.validation
    "CircularImportValidationResult",
    # Constants
    "INFRA_SRC_PATH",
    "INFRA_NODES_PATH",
    "INFRA_MAX_UNIONS",
    "INFRA_MAX_VIOLATIONS",
    "INFRA_PATTERNS_STRICT",
    "INFRA_UNIONS_STRICT",
    # Validators
    "validate_infra_architecture",
    "validate_infra_contracts",
    "validate_infra_patterns",
    "validate_infra_contract_deep",
    "validate_infra_union_usage",
    "validate_infra_circular_imports",
    "validate_infra_all",
    "get_validation_summary",
]
