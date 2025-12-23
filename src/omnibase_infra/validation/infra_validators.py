"""
Infrastructure-specific validation wrappers.

Provides validators from omnibase_core with sensible defaults for infrastructure code.
All wrappers maintain strong typing and follow ONEX validation patterns.

Exemption System:
    This module uses a YAML-based exemption system for managing validation exceptions.
    Exemption patterns are defined in `validation_exemptions.yaml` alongside this module.

    The exemption system provides:
    - Centralized management of all validation exemptions
    - Clear documentation of rationale and ticket references
    - Regex-based matching resilient to code changes (no line numbers)
    - Separation of exemption configuration from validation logic

    See validation_exemptions.yaml for:
    - pattern_exemptions: Method count, parameter count, naming violations
    - union_exemptions: Complex union type violations

    Adding new exemptions:
    1. Identify the exact violation message from validator output
    2. Add entry to appropriate section in validation_exemptions.yaml
    3. Document the rationale and link to relevant tickets
    4. Run tests to verify the exemption works
"""

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Literal, TypedDict

import yaml

# Module-level logger for validation operations
logger = logging.getLogger(__name__)
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


# Path to the exemptions YAML file (alongside this module)
EXEMPTIONS_YAML_PATH = Path(__file__).parent / "validation_exemptions.yaml"


@lru_cache(maxsize=1)
def _load_exemptions_yaml() -> dict[str, list[ExemptionPattern]]:
    """
    Load and cache exemption patterns from YAML configuration.

    The exemption patterns are cached to avoid repeated file I/O during validation.
    Cache is cleared when the module is reloaded.

    Returns:
        Dictionary with 'pattern_exemptions' and 'union_exemptions' keys,
        each containing a list of ExemptionPattern dictionaries.
        Returns empty lists if file is missing or malformed.

    Note:
        The YAML file is expected to be at validation_exemptions.yaml alongside
        this module. See that file for schema documentation and exemption rationale.
    """
    if not EXEMPTIONS_YAML_PATH.exists():
        # Fallback to empty exemptions if file is missing
        return {"pattern_exemptions": [], "union_exemptions": []}

    try:
        with EXEMPTIONS_YAML_PATH.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            return {"pattern_exemptions": [], "union_exemptions": []}

        # Extract exemption lists, converting YAML structure to ExemptionPattern format
        pattern_exemptions = _convert_yaml_exemptions(
            data.get("pattern_exemptions", [])
        )
        union_exemptions = _convert_yaml_exemptions(data.get("union_exemptions", []))

        return {
            "pattern_exemptions": pattern_exemptions,
            "union_exemptions": union_exemptions,
        }
    except (yaml.YAMLError, OSError) as e:
        # Log warning but continue with empty exemptions
        logger.warning(
            "Failed to load validation exemptions from %s: %s. Using empty exemptions.",
            EXEMPTIONS_YAML_PATH,
            e,
        )
        return {"pattern_exemptions": [], "union_exemptions": []}


def _convert_yaml_exemptions(yaml_list: list[dict]) -> list[ExemptionPattern]:
    """
    Convert YAML exemption entries to ExemptionPattern format.

    The YAML format includes additional metadata (reason, ticket) that is used
    for documentation but not for pattern matching. This function extracts only
    the pattern fields needed for matching.

    Regex patterns are validated at load time to prevent runtime errors during
    validation. Entries with invalid regex patterns are skipped with a warning.

    Args:
        yaml_list: List of exemption entries from YAML.

    Returns:
        List of ExemptionPattern dictionaries with only pattern fields.
        Entries with invalid regex patterns are excluded.

    Invalid Entry Handling:
        This function is defensive and skips invalid entries to ensure
        validation continues even with malformed exemption configuration:

        - If yaml_list is not a list: returns empty list (no exemptions applied)
        - If an entry is not a dict: entry is skipped silently
        - If entry lacks required fields (file_pattern AND violation_pattern):
          entry is skipped silently (both fields are required for meaningful matching)
        - If any pattern field contains an invalid regex: entry is skipped
          with a warning log (prevents runtime errors during pattern matching)
        - All pattern field values are coerced to str via str() to handle
          non-string values gracefully

    Design Rationale:
        Skipping invalid entries (rather than raising exceptions) is intentional:
        1. Validation should not fail due to exemption configuration issues
        2. Missing exemptions result in stricter validation (safer default)
        3. Errors in exemption config are detected during exemption testing
        4. Production validation continues even with partial exemption config
        5. Invalid regex patterns are logged to aid debugging
    """
    if not isinstance(yaml_list, list):
        return []

    result: list[ExemptionPattern] = []
    for entry in yaml_list:
        if not isinstance(entry, dict):
            continue

        # Extract only pattern fields (ignore reason, ticket metadata)
        # Validate each regex pattern before adding to prevent runtime errors
        pattern: ExemptionPattern = {}
        entry_valid = True

        if "file_pattern" in entry:
            file_pattern = str(entry["file_pattern"])
            try:
                re.compile(file_pattern)
                pattern["file_pattern"] = file_pattern
            except re.error as e:
                logger.warning(
                    "Invalid regex in file_pattern '%s': %s. Skipping exemption entry.",
                    file_pattern,
                    e,
                )
                entry_valid = False

        if entry_valid and "class_pattern" in entry:
            class_pattern = str(entry["class_pattern"])
            try:
                re.compile(class_pattern)
                pattern["class_pattern"] = class_pattern
            except re.error as e:
                logger.warning(
                    "Invalid regex in class_pattern '%s': %s. Skipping exemption entry.",
                    class_pattern,
                    e,
                )
                entry_valid = False

        if entry_valid and "method_pattern" in entry:
            method_pattern = str(entry["method_pattern"])
            try:
                re.compile(method_pattern)
                pattern["method_pattern"] = method_pattern
            except re.error as e:
                logger.warning(
                    "Invalid regex in method_pattern '%s': %s. Skipping exemption entry.",
                    method_pattern,
                    e,
                )
                entry_valid = False

        if entry_valid and "violation_pattern" in entry:
            violation_pattern = str(entry["violation_pattern"])
            try:
                re.compile(violation_pattern)
                pattern["violation_pattern"] = violation_pattern
            except re.error as e:
                logger.warning(
                    "Invalid regex in violation_pattern '%s': %s. Skipping exemption entry.",
                    violation_pattern,
                    e,
                )
                entry_valid = False

        # Only include if entry is valid and has required patterns
        if entry_valid and "file_pattern" in pattern and "violation_pattern" in pattern:
            result.append(pattern)

    return result


def get_pattern_exemptions() -> list[ExemptionPattern]:
    """
    Get pattern validator exemptions from YAML configuration.

    Returns:
        List of ExemptionPattern dictionaries for pattern validation.
    """
    return _load_exemptions_yaml()["pattern_exemptions"]


def get_union_exemptions() -> list[ExemptionPattern]:
    """
    Get union validator exemptions from YAML configuration.

    Returns:
        List of ExemptionPattern dictionaries for union validation.
    """
    return _load_exemptions_yaml()["union_exemptions"]


# Default paths for infrastructure validation
INFRA_SRC_PATH = "src/omnibase_infra/"
INFRA_NODES_PATH = "src/omnibase_infra/nodes/"

# ============================================================================
# Pattern Validator Threshold Reference (from omnibase_core.validation)
# ============================================================================
# These thresholds are defined in omnibase_core and applied by validate_patterns().
# Documented here for reference and to explain infrastructure exemptions.
#
# See CLAUDE.md "Accepted Pattern Exceptions" section for full rationale.
# Ticket: OMN-934 (message dispatch engine implementation)
# Updated: PR #61 review feedback - added explicit threshold documentation
#
# DEFAULT_MAX_METHODS = 10     # Maximum methods per class
# DEFAULT_MAX_INIT_PARAMS = 5  # Maximum __init__ parameters
#
# Infrastructure Pattern Exemptions (OMN-934, PR #61):
# ----------------------------------------------------
# KafkaEventBus (14 methods, 10 __init__ params):
#   - Event bus pattern requires: lifecycle (start/stop/health), pub/sub
#     (subscribe/unsubscribe/publish), circuit breaker, protocol compatibility
#   - Backwards compatibility during config migration requires multiple __init__ params
#   - See: kafka_event_bus.py class docstring, CLAUDE.md "Accepted Pattern Exceptions"
#
# RuntimeHostProcess (11+ methods, 6+ __init__ params):
#   - Central coordinator requires: lifecycle management, message handling,
#     graceful shutdown, handler management
#   - See: runtime_host_process.py class docstring, CLAUDE.md "Accepted Pattern Exceptions"
#
# These exemptions are handled via exempted_patterns in validate_infra_patterns(),
# NOT by modifying global thresholds.
#
# Exemption Pattern Examples (explicit format):
# ---------------------------------------------
# KafkaEventBus method count:
#   {"file_pattern": r"kafka_event_bus\.py", "class_pattern": r"Class 'KafkaEventBus'",
#    "violation_pattern": r"has \d+ methods"}
#
# KafkaEventBus __init__ params:
#   {"file_pattern": r"kafka_event_bus\.py", "method_pattern": r"Function '__init__'",
#    "violation_pattern": r"has \d+ parameters"}
#
# RuntimeHostProcess method count:
#   {"file_pattern": r"runtime_host_process\.py", "class_pattern": r"Class 'RuntimeHostProcess'",
#    "violation_pattern": r"has \d+ methods"}
#
# RuntimeHostProcess __init__ params:
#   {"file_pattern": r"runtime_host_process\.py", "method_pattern": r"Function '__init__'",
#    "violation_pattern": r"has \d+ parameters"}
#
# See exempted_patterns list in validate_infra_patterns() for complete definitions.
# ============================================================================

# Maximum allowed union count in infrastructure code.
# This is a COUNT threshold, not a violation threshold. The validator counts all
# unions including the ONEX-preferred `X | None` patterns, which are valid.
#
# Current baseline (515 unions as of 2025-12-22):
# - Most unions are legitimate `X | None` nullable patterns
# - These are NOT flagged as violations, just counted
# - Actual violations (primitive soup, Union[X,None] syntax) are reported separately
#
# Threshold history:
# - 491 (2025-12-21): Initial baseline with DispatcherFunc | ContextAwareDispatcherFunc
# - 515 (2025-12-22): OMN-990 MessageDispatchEngine + OMN-947 snapshots (~24 unions added)
# - 540 (2025-12-23): OMN-950 comprehensive reducer tests (~25 unions from type annotations)
#
# Target: Reduce to <200 through dict[str, object] -> JsonValue migration.
INFRA_MAX_UNIONS = 540

# Maximum allowed architecture violations in infrastructure code.
# Set to 0 (strict enforcement) to ensure one-model-per-file principle is always followed.
# Infrastructure nodes require strict architecture compliance for maintainability and
# contract-driven code generation.
INFRA_MAX_VIOLATIONS = 0

# Strict mode for pattern validation.
# Enabled: All violations must be exempted or fixed.
# See validate_infra_patterns() exempted_patterns list for documented exemptions.
INFRA_PATTERNS_STRICT = True

# Strict mode for union usage validation.
# Enabled: The validator will flag actual violations (not just count unions).
INFRA_UNIONS_STRICT = True


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
        Exemption patterns are loaded from validation_exemptions.yaml (pattern_exemptions section).
        See that file for the complete list of exemptions with rationale and ticket references.

        Key exemption categories:
        - KafkaEventBus: Event bus pattern with many methods/params (OMN-934)
        - RuntimeHostProcess: Central coordinator pattern (OMN-756)
        - PolicyRegistry: Domain registry pattern
        - ExecutionShapeValidator: AST analysis validator pattern (OMN-958)
        - MixinNodeIntrospection: Introspection mixin pattern (OMN-958)

    Exemption Pattern Format:
        Uses regex-based matching instead of hardcoded line numbers for resilience
        to code changes. See ExemptionPattern TypedDict for structure details.

    Args:
        directory: Directory to validate. Defaults to infrastructure source.
        strict: Enable strict mode. Defaults to INFRA_PATTERNS_STRICT (True).

    Returns:
        ModelValidationResult with validation status and filtered errors.
        Documented exemptions are filtered from error list but logged for transparency.
    """
    # Run base validation
    base_result = validate_patterns(str(directory), strict=strict)

    # Load exemption patterns from YAML configuration
    # See validation_exemptions.yaml for pattern definitions and rationale
    exempted_patterns = get_pattern_exemptions()

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
        Returns empty list if inputs are not the expected types.

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
    # Defensive type checks for list inputs
    if not isinstance(errors, list):
        return []
    if not isinstance(exempted_patterns, list):
        # If no valid exemption patterns, return errors as-is (no filtering)
        return [err for err in errors if isinstance(err, str)]

    filtered = []
    for err in errors:
        # Skip non-string errors
        if not isinstance(err, str):
            continue
        is_exempted = False

        for pattern in exempted_patterns:
            # Skip non-dict patterns
            if not isinstance(pattern, dict):
                continue

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

    Performance Note:
        This function uses a cached singleton ProtocolContractValidator instance
        for optimal performance in hot paths. The validator is stateless after
        initialization, making it safe to reuse across calls.

    Args:
        contract_path: Path to the contract YAML file.
        contract_type: Type of contract to validate. Defaults to "effect".

    Returns:
        ModelContractValidationResult with validation status, score, and any errors.
    """
    return _contract_validator.validate_contract_file(
        Path(contract_path), contract_type
    )


# ==============================================================================
# Module-Level Singleton Validators
# ==============================================================================
#
# Performance Optimization: The ProtocolContractValidator is stateless after
# initialization. Creating new instances on every validation call is wasteful
# in hot paths. Instead, we use a module-level singleton.
#
# Why a singleton is safe here:
# - The validator has no mutable state after initialization
# - All validation state is created fresh for each file
# - No per-validation state is stored in the validator instance

_contract_validator = ProtocolContractValidator()


def validate_infra_union_usage(
    directory: str | Path = INFRA_SRC_PATH,
    max_unions: int = INFRA_MAX_UNIONS,
    strict: bool = INFRA_UNIONS_STRICT,
) -> ValidationResult:
    """
    Validate Union type usage in infrastructure code.

    Prevents overly complex union types that complicate infrastructure code.

    Exemptions:
        Exemption patterns are loaded from validation_exemptions.yaml (union_exemptions section).
        See that file for the complete list of exemptions with rationale.

        Key exemption categories:
        - ModelNodeCapabilities.config: JSON-like configuration pattern with primitive unions

    Args:
        directory: Directory to validate. Defaults to infrastructure source.
        max_unions: Maximum union count threshold. Defaults to INFRA_MAX_UNIONS.
        strict: Enable strict mode. Defaults to INFRA_UNIONS_STRICT (True).

    Returns:
        ModelValidationResult with validation status and any errors.
    """
    # Run base validation
    base_result = validate_union_usage(
        str(directory), max_unions=max_unions, strict=strict
    )

    # Load exemption patterns from YAML configuration
    # See validation_exemptions.yaml for pattern definitions and rationale
    exempted_patterns = get_union_exemptions()

    # Filter errors using regex-based pattern matching
    filtered_errors = _filter_exempted_errors(base_result.errors, exempted_patterns)

    # Create wrapper result (avoid mutation)
    return _create_filtered_result(base_result, filtered_errors)


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
        Returns zero counts if input is not a dictionary.
    """
    # Defensive type check for dict input
    if not isinstance(results, dict):
        return {
            "total_validators": 0,
            "passed": 0,
            "failed": 0,
            "failed_validators": [],
        }

    passed = 0
    failed = 0
    failed_validators: list[str] = []

    for name, result in results.items():
        # Skip entries with non-string keys
        if not isinstance(name, str):
            continue
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
    "EXEMPTIONS_YAML_PATH",
    # Exemption loaders
    "get_pattern_exemptions",
    "get_union_exemptions",
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
