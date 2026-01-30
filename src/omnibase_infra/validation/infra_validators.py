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

# Standard library imports
import ast
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

# Third-party imports
import yaml

from omnibase_core.models.common import ModelValidationMetadata
from omnibase_core.models.primitives import ModelSemVer
from omnibase_core.models.validation.model_union_pattern import ModelUnionPattern
from omnibase_core.validation import (
    CircularImportValidator,
    ModelContractValidationResult,
    ModelImportValidationResult,
    ModelModuleImportResult,
    ModelValidationResult,
    validate_architecture,
    validate_contracts,
    validate_patterns,
    validate_union_usage_file,
    validate_yaml_file,
)

# Local imports
from omnibase_infra.types import PathInput

# Module-level initialization (AFTER all imports)
logger = logging.getLogger(__name__)

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
        file_pattern: Regex pattern matching the filename (e.g., r"event_bus_kafka\\.py")
        class_pattern: Optional regex for class name (e.g., r"Class 'EventBusKafka'")
        method_pattern: Optional regex for method name (e.g., r"Function '__init__'")
        violation_pattern: Regex matching the violation type (e.g., r"too many (methods|parameters)")

    Example:
        {
            "file_pattern": r"event_bus_kafka\\.py",
            "class_pattern": r"Class 'EventBusKafka'",
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
        Dictionary with 'pattern_exemptions', 'union_exemptions', and
        'architecture_exemptions' keys, each containing a list of
        ExemptionPattern dictionaries.
        Returns empty lists if file is missing or malformed.

    Note:
        The YAML file is expected to be at validation_exemptions.yaml alongside
        this module. See that file for schema documentation and exemption rationale.
    """
    if not EXEMPTIONS_YAML_PATH.exists():
        # Fallback to empty exemptions if file is missing
        return {
            "pattern_exemptions": [],
            "union_exemptions": [],
            "architecture_exemptions": [],
        }

    try:
        with EXEMPTIONS_YAML_PATH.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            return {
                "pattern_exemptions": [],
                "union_exemptions": [],
                "architecture_exemptions": [],
            }

        # Extract exemption lists, converting YAML structure to ExemptionPattern format
        pattern_exemptions = _convert_yaml_exemptions(
            data.get("pattern_exemptions", [])
        )
        union_exemptions = _convert_yaml_exemptions(data.get("union_exemptions", []))
        architecture_exemptions = _convert_yaml_exemptions(
            data.get("architecture_exemptions", [])
        )

        return {
            "pattern_exemptions": pattern_exemptions,
            "union_exemptions": union_exemptions,
            "architecture_exemptions": architecture_exemptions,
        }
    except (yaml.YAMLError, OSError) as e:
        # Log warning but continue with empty exemptions
        logger.warning(
            "Failed to load validation exemptions from %s: %s. Using empty exemptions.",
            EXEMPTIONS_YAML_PATH,
            e,
        )
        return {
            "pattern_exemptions": [],
            "union_exemptions": [],
            "architecture_exemptions": [],
        }


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


def get_architecture_exemptions() -> list[ExemptionPattern]:
    """
    Get architecture validator exemptions from YAML configuration.

    Returns:
        List of ExemptionPattern dictionaries for architecture validation.
    """
    exemptions = _load_exemptions_yaml()
    return exemptions.get("architecture_exemptions", [])


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
# EventBusKafka (14 methods, 10 __init__ params):
#   - Event bus pattern requires: lifecycle (start/stop/health), pub/sub
#     (subscribe/unsubscribe/publish), circuit breaker, protocol compatibility
#   - Backwards compatibility during config migration requires multiple __init__ params
#   - See: event_bus_kafka.py class docstring, CLAUDE.md "Accepted Pattern Exceptions"
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
# EventBusKafka method count:
#   {"file_pattern": r"event_bus_kafka\.py", "class_pattern": r"Class 'EventBusKafka'",
#    "violation_pattern": r"has \d+ methods"}
#
# EventBusKafka __init__ params:
#   {"file_pattern": r"event_bus_kafka\.py", "method_pattern": r"Function '__init__'",
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
# This threshold counts ONLY complex type annotation unions.
#
# Excluded patterns (NOT counted toward threshold):
# 1. Simple optionals (`X | None`) - idiomatic nullable pattern (PEP 604)
# 2. isinstance() unions (`isinstance(x, A | B)`) - runtime type checks, not annotations
#
# What IS counted (threshold applies to):
# - Multi-type unions in annotations: `def foo(x: str | int)`
# - Complex patterns: `dict[str, str | int]` (nested unions)
# - Unions with 3+ types (potential "primitive soup")
#
# What is NOT counted (excluded from threshold):
# - Simple optionals: `str | None`, `int | None`, `ModelFoo | None`
#   - These are idiomatic Python nullable patterns, not complexity concerns
# - isinstance() unions: `isinstance(x, str | int)` (ruff UP038 modern syntax)
#   - These are runtime type checks, not type annotations
#   - Encouraged by ruff UP038 over isinstance(x, (str, int))
#
# Threshold history (after exclusion logic):
# - 120 (2025-12-25): Initial threshold after excluding ~470 `X | None` patterns
#   - ~568 total unions in codebase
#   - ~468 are simple `X | None` optionals (82%)
#   - ~100 non-optional unions remain
#   - Buffer of 20 above baseline for codebase growth
# - 121 (2025-12-25): OMN-881 introspection feature (+1 non-optional union)
# - 121 (2025-12-25): OMN-949 DLQ, OMN-816, OMN-811, OMN-1006 merges (all used X | None patterns, excluded)
# - 121 (2025-12-26): OMN-1007 registry pattern + merge with main (X | None patterns excluded)
# - 122 (2026-01-15): OMN-1203 corpus capture service, OMN-1346 extract registration domain plugin
# - 142 (2026-01-16): OMN-1305 ruff UP038 isinstance union syntax modernization (+20 unions)
# - 121 (2026-01-16): OMN-1305 isinstance union exclusion (excluding 21 isinstance unions)
#   - Updated validator to exclude isinstance(x, A | B) patterns
#   - These are runtime checks, not type annotations
# - 70 (2026-01-16): OMN-1358 type alias replacements reduced from 122 to 63 non-optional unions
#                    Applied HandlerMap, NodeId, PayloadDict, EventMetadata, MetadataDict type aliases
# - 81 (2026-01-16): OMN-1305 PR #151 merge with main - combined changes
#                    isinstance exclusion + type alias refactoring + PR #151 fixes
# - 83 (2026-01-16): OMN-1181 structured errors merge with main
#                    (+2 unions for EnumPolicyType | str in validate_policy_type_value)
# - 82 (2026-01-16): OMN-1181 fix PolicyTypeInput validator coercion
#                    (-1 union: changed return type from str | EnumPolicyType to EnumPolicyType)
#                    Validators now coerce strings to EnumPolicyType, ensuring type-safe access.
#
# Soft ceiling guidance:
# - 100-120: Healthy range, minor increments OK for legitimate features
# - 120-140: Caution zone, consider refactoring before incrementing
# - 140+: Refactor required - extract type aliases or use domain models from omnibase_core
#
# When incrementing threshold:
# 1. Document the ticket/PR that added unions in threshold history above
# 2. Verify new unions are not simple X | None (those should be excluded automatically)
# 3. Verify new unions are not isinstance() patterns (also excluded automatically)
# 4. Consider if a domain-specific type from omnibase_core would be cleaner
#
# Target: Keep below 150 - if this grows, consider typed patterns from omnibase_core.
# - 95 (2026-01-16): OMN-1142 Qdrant/Graph handlers (+14 legitimate union patterns)
#                    - str | int for graph node IDs (5 occurrences in handler_graph.py)
#                    - UUID | str for Qdrant point IDs (2 occurrences in Qdrant models)
#                    - float | int for score fields (1 occurrence)
# - 96 (2026-01-16): OMN-1181 structured errors merge with main (+1 net)
#                    (+2 unions for EnumPolicyType | str in validate_policy_type_value)
#                    (-1 union: fix PolicyTypeInput validator coercion, changed return
#                    type from str | EnumPolicyType to EnumPolicyType)
# - 98 (2026-01-20): OMN-1277 security validator contract refactoring (+2 unions)
#                    ast.FunctionDef | ast.AsyncFunctionDef for AST method type checking
# - 105 (2026-01-21): Contract-driven handler config loading (+4 unions)
#                     ModelHandlerContract transport config fields and lifecycle types
# - 108 (2026-01-27): OMN-1518 declarative operation bindings (+3 unions)
#                     ModelEventEnvelope[object] | dict[str, object] for materialized
#                     envelopes in dispatch engine (3 occurrences in type aliases)
# - 105 (2026-01-27): OMN-1518 simplify to always-dict envelope format (-3 unions)
#                     Removed hybrid union types by always materializing to dict format
#                     Dispatchers now receive consistent dict[str, object] with __bindings
# - 112 (2026-01-27): OMN-1610 emit daemon for persistent Kafka connections (+7 unions)
#                     BoundedEventQueue, EmitClient, EventRegistry return types
# - 112 (2026-01-29): OMN-1610 emit daemon + refactor to strongly typed input model
#                     BoundedEventQueue, EmitClient, EventRegistry, socket_permissions
#                     Replaced dict unions with ModelEmitDaemonConfigInput
# - 113 (2026-01-29): OMN-1610 properly typed daemon protocol models (+1 union)
#                     Added ModelDaemonRequest, ModelDaemonResponse discriminated unions
#                     Replaced dict[str, object] soup with strongly typed Pydantic models
# - 115 (2026-01-29): OMN-1653 contract registry reducer (+2 unions)
#                     ContractRegistryEvent: 4-type union for event routing
#                     contract_yaml: dict | str for flexible YAML handling
INFRA_MAX_UNIONS = 115

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
    directory: PathInput = INFRA_SRC_PATH,
    max_violations: int = INFRA_MAX_VIOLATIONS,
) -> ValidationResult:
    """
    Validate infrastructure architecture with strict defaults.

    Enforces ONEX one-model-per-file principle critical for infrastructure nodes.

    Exemptions:
        Exemption patterns are loaded from validation_exemptions.yaml (architecture_exemptions section).
        See that file for the complete list of exemptions with rationale and ticket references.

        Key exemption categories:
        - contract_linter.py: Domain-grouped validation models (PR-57)
        - protocols.py: Domain-grouped protocols per CLAUDE.md convention (OMN-888)

    Args:
        directory: Directory to validate. Defaults to infrastructure source.
        max_violations: Maximum allowed violations. Defaults to INFRA_MAX_VIOLATIONS (0).

    Returns:
        ModelValidationResult with validation status and filtered errors.
        Documented exemptions are filtered from error list but logged for transparency.
    """
    # Run base validation
    base_result = validate_architecture(str(directory), max_violations=max_violations)

    # Load exemption patterns from YAML configuration
    # See validation_exemptions.yaml for pattern definitions and rationale
    exempted_patterns = get_architecture_exemptions()

    # Filter errors using regex-based pattern matching
    filtered_errors = _filter_exempted_errors(base_result.errors, exempted_patterns)

    # Create wrapper result (avoid mutation)
    return _create_filtered_result(base_result, filtered_errors)


def validate_infra_contracts(
    directory: PathInput = INFRA_NODES_PATH,
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
    directory: PathInput = INFRA_SRC_PATH,
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
        - EventBusKafka: Event bus pattern with many methods/params (OMN-934)
        - RuntimeHostProcess: Central coordinator pattern (OMN-756)
        - RegistryPolicy: Domain registry pattern
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
                "file_pattern": r"event_bus_kafka\\.py",
                "class_pattern": r"Class 'EventBusKafka'",
                "violation_pattern": r"has \\d+ methods"
            }

        Matches error:
            "event_bus_kafka.py:123: Class 'EventBusKafka' has 14 methods (threshold: 10)"

        Does not match:
            "event_bus_kafka.py:50: Function 'connect' has 7 parameters" (no class_pattern)
            "other_file.py:10: Class 'EventBusKafka' has 14 methods" (wrong file)
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

    Guards against missing attributes on base_result to handle edge cases where
    validation results may have incomplete or missing fields.

    Args:
        base_result: Original validation result.
        filtered_errors: Filtered error list.

    Returns:
        New ValidationResult with filtered errors and updated metadata.
    """
    # Guard against missing errors attribute on base_result
    base_errors = getattr(base_result, "errors", None)
    base_errors_count = len(base_errors) if base_errors is not None else 0

    # Calculate filtering statistics
    violations_filtered = base_errors_count - len(filtered_errors)
    all_violations_exempted = violations_filtered > 0 and len(filtered_errors) == 0

    # Create new metadata if present (avoid mutation)
    # Use getattr to guard against missing metadata attribute on base_result
    new_metadata = None
    base_metadata = getattr(base_result, "metadata", None)
    if base_metadata is not None:
        # Use model_copy for deep copy with updates (Pydantic v2 pattern)
        # This works with both real Pydantic models and test mocks
        try:
            new_metadata = base_metadata.model_copy(deep=True)
            # Update violations_found if the field exists and is writable
            # Guard against None return from model_copy and missing/read-only attributes
            if new_metadata is not None and hasattr(new_metadata, "violations_found"):
                try:
                    new_metadata.violations_found = len(filtered_errors)
                except (AttributeError, TypeError):
                    # violations_found may be a read-only property or frozen field
                    pass
        except AttributeError:
            # Fallback for test mocks that don't support model_copy
            # Use original metadata without modification to avoid mutation
            new_metadata = base_metadata

    # Guard against missing attributes on base_result
    # Use getattr with sensible defaults to handle incomplete validation results
    base_is_valid = getattr(base_result, "is_valid", False)
    base_validated_value = getattr(base_result, "validated_value", None)
    base_issues = getattr(base_result, "issues", [])
    base_warnings = getattr(base_result, "warnings", [])
    base_suggestions = getattr(base_result, "suggestions", [])
    base_summary = getattr(base_result, "summary", None)
    base_details = getattr(base_result, "details", None)

    # Create new result (wrapper pattern - no mutation)
    return ModelValidationResult(
        is_valid=all_violations_exempted or base_is_valid,
        validated_value=base_validated_value,
        issues=base_issues if base_issues is not None else [],
        errors=filtered_errors,
        warnings=base_warnings if base_warnings is not None else [],
        suggestions=base_suggestions if base_suggestions is not None else [],
        summary=base_summary,
        details=base_details,
        metadata=new_metadata,
    )


def validate_infra_contract_deep(
    contract_path: PathInput,
) -> ModelContractValidationResult:
    """
    Perform deep contract validation for ONEX compliance.

    Uses validate_yaml_file() from omnibase_core for comprehensive contract
    checking suitable for autonomous code generation.

    Args:
        contract_path: Path to the contract YAML file.

    Returns:
        ModelContractValidationResult with validation status, score, and any errors.

    Raises:
        OnexError: If YAML validation fails with an unexpected error.
    """
    from uuid import uuid4

    from omnibase_core.enums import EnumCoreErrorCode
    from omnibase_core.errors import OnexError

    correlation_id = uuid4()

    # Use the validation API from omnibase_core 0.6.x directly
    try:
        result = validate_yaml_file(Path(contract_path))
    except Exception as e:
        raise OnexError(
            message=f"YAML validation failed for {contract_path}: {e}",
            error_code=EnumCoreErrorCode.VALIDATION_ERROR,
            correlation_id=correlation_id,
            contract_path=str(contract_path),
        ) from e

    # Return a ModelContractValidationResult
    # The API may return a different type, so we adapt it
    if isinstance(result, ModelContractValidationResult):
        return result

    # If result is a different type, wrap it in ModelContractValidationResult
    # Default to is_valid=False for unknown result types to avoid silently masking failures
    # Check 'is_valid' first, then 'passed' as fallback (some validators use passed)
    return ModelContractValidationResult(
        is_valid=getattr(result, "is_valid", getattr(result, "passed", False)),
        score=getattr(result, "score", 0.0),
        violations=getattr(result, "violations", getattr(result, "errors", [])),
        warnings=getattr(result, "warnings", []),
        interface_version=ModelSemVer(major=1, minor=0, patch=0),
    )


# ==============================================================================
# Skip Directory Configuration
# ==============================================================================
#
# Skip directories are loaded from validation_exemptions.yaml for configurability.
# If the YAML file is missing or doesn't contain skip_directories, we fall back
# to a hardcoded default set.
#
# This follows the same pattern as exemption loading to keep all validation
# configuration in one place.


@lru_cache(maxsize=1)
def load_skip_directories_from_yaml() -> frozenset[str] | None:
    """
    Load skip directory configuration from YAML.

    Returns:
        frozenset of directory names to skip, or None if not configured in YAML.
        Returns None (not empty set) to distinguish "not configured" from
        "explicitly empty".
    """
    if not EXEMPTIONS_YAML_PATH.exists():
        return None

    try:
        with EXEMPTIONS_YAML_PATH.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            return None

        skip_dirs = data.get("skip_directories")
        if skip_dirs is None:
            return None

        # Handle both list and dict formats
        if isinstance(skip_dirs, list):
            # Simple list format: ["archive", "examples", ...]
            return frozenset(str(d) for d in skip_dirs if d)
        elif isinstance(skip_dirs, dict):
            # Dict format with categories: {historical: [...], caches: [...]}
            all_dirs: set[str] = set()
            for category_dirs in skip_dirs.values():
                if isinstance(category_dirs, list):
                    all_dirs.update(str(d) for d in category_dirs if d)
            return frozenset(all_dirs) if all_dirs else None

        return None

    except (yaml.YAMLError, OSError) as e:
        logger.warning(
            "Failed to load skip directories from %s: %s. Using defaults.",
            EXEMPTIONS_YAML_PATH,
            e,
        )
        return None


def get_skip_directories() -> frozenset[str]:
    """
    Get the set of directory names to skip during validation.

    Returns skip directories from YAML configuration if available,
    otherwise falls back to the hardcoded SKIP_DIRECTORY_NAMES default.

    Returns:
        frozenset of directory names that should be excluded from validation.
    """
    yaml_dirs = load_skip_directories_from_yaml()
    if yaml_dirs is not None:
        return yaml_dirs
    return SKIP_DIRECTORY_NAMES


def is_simple_optional(pattern: ModelUnionPattern) -> bool:
    """
    Determine if a union pattern is a simple optional (`X | None`).

    Simple optionals are the ONEX-preferred pattern for nullable types and should
    NOT count toward the union complexity threshold. They represent:
    - `str | None` - nullable string
    - `int | None` - nullable integer
    - `ModelFoo | None` - nullable model
    - `list[str] | None` - nullable list

    These are NOT considered complex unions because:
    1. They are idiomatic Python (PEP 604)
    2. They express optionality, not type ambiguity
    3. They don't require complex type narrowing logic

    Args:
        pattern: The ModelUnionPattern to check.

    Returns:
        True if the pattern is a simple optional (`X | None`), False otherwise.

    Examples:
        >>> is_simple_optional(ModelUnionPattern(["str", "None"], 1, "test.py"))
        True
        >>> is_simple_optional(ModelUnionPattern(["int", "None"], 1, "test.py"))
        True
        >>> is_simple_optional(ModelUnionPattern(["str", "int"], 1, "test.py"))
        False
        >>> is_simple_optional(ModelUnionPattern(["str", "int", "None"], 1, "test.py"))
        False
    """
    # Simple optional: exactly 2 types, one of which is None
    return len(pattern.types) == 2 and "None" in pattern.types


# ==============================================================================
# isinstance() Union Exclusion
# ==============================================================================
#
# Modern Python (PEP 604) allows isinstance(x, A | B) syntax instead of
# isinstance(x, (A, B)). These are runtime type checks, NOT type annotations.
# Ruff UP038 encourages this modern syntax.
#
# The union validator's goal is to limit complex TYPE ANNOTATIONS that indicate
# type ambiguity in function signatures. isinstance() unions are:
# - Runtime expressions, not type hints
# - Used for dynamic type checking, not static typing
# - Encouraged by modern Python style guides (ruff UP038)
#
# Therefore, isinstance() unions should NOT count toward the union threshold.


class IsinstanceUnionVisitor(ast.NodeVisitor):
    """
    AST visitor to find line numbers where unions appear inside isinstance() calls.

    This visitor tracks context when descending into isinstance() call arguments,
    marking any union (BitOr) expressions found in the second argument as
    "isinstance unions" that should be excluded from the complexity threshold.

    The visitor correctly handles:
    - isinstance(x, A | B) - simple isinstance union
    - isinstance(x, A | B | C) - multi-type isinstance union
    - isinstance(x, list[A | B]) - union inside generic (NOT excluded, it's a type hint)

    Attributes:
        isinstance_union_lines: Set of line numbers containing isinstance unions.
    """

    def __init__(self) -> None:
        """Initialize the visitor with empty line tracking."""
        self.isinstance_union_lines: set[int] = set()
        self._in_isinstance_type_arg: bool = False

    def visit_Call(self, node: ast.Call) -> None:
        """
        Visit a Call node and check if it's an isinstance() call.

        When an isinstance() call is found, we mark that we're inside its
        second argument (the type specification) before visiting children.
        Any BitOr (union) found in this context is tracked.

        Args:
            node: The Call AST node to visit.
        """
        # Check if this is an isinstance() call
        is_isinstance_call = (
            isinstance(node.func, ast.Name) and node.func.id == "isinstance"
        )

        if is_isinstance_call and len(node.args) >= 2:
            # Visit the first argument (the object being checked) normally
            self.visit(node.args[0])

            # Mark that we're in the type argument, then visit it
            self._in_isinstance_type_arg = True
            self.visit(node.args[1])
            self._in_isinstance_type_arg = False

            # Visit any remaining arguments normally
            for arg in node.args[2:]:
                self.visit(arg)

            # Visit keyword arguments normally
            for keyword in node.keywords:
                self.visit(keyword)
        else:
            # Not isinstance(), visit normally
            self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        """
        Visit a BinOp node and track if it's a union inside isinstance().

        When we're inside an isinstance() type argument and find a BitOr
        (union) operator, we record this line as an isinstance union.

        Args:
            node: The BinOp AST node to visit.
        """
        if self._in_isinstance_type_arg and isinstance(node.op, ast.BitOr):
            # This is a union inside isinstance() - track the line
            self.isinstance_union_lines.add(node.lineno)

        # Continue visiting children (for nested unions like A | B | C)
        self.generic_visit(node)


@lru_cache(maxsize=128)
def _find_isinstance_union_lines(file_path: Path) -> frozenset[int]:
    """
    Find all line numbers containing unions inside isinstance() calls.

    This function parses the file once and returns all lines where unions
    appear inside isinstance() type arguments. Results are cached to avoid
    repeated parsing when checking multiple patterns from the same file.

    Args:
        file_path: Path to the Python file to analyze.

    Returns:
        Frozenset of line numbers (1-based) containing isinstance unions.
        Returns empty frozenset if file cannot be parsed or doesn't exist.

    Examples:
        >>> # File with: isinstance(x, str | int)  on line 5
        >>> _find_isinstance_union_lines(Path("example.py"))
        frozenset({5})

    Note:
        Uses lru_cache to avoid re-parsing files during the same validation run.
        Cache should be cleared between validation runs if files may have changed.
    """
    try:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(file_path))
    except (OSError, SyntaxError) as e:
        logger.debug("Cannot parse %s for isinstance detection: %s", file_path, e)
        return frozenset()

    visitor = IsinstanceUnionVisitor()
    visitor.visit(tree)
    return frozenset(visitor.isinstance_union_lines)


def is_isinstance_union(pattern: ModelUnionPattern) -> bool:
    """
    Determine if a union pattern is inside an isinstance() call.

    isinstance() unions are runtime type checks, not type annotations.
    They should NOT count toward the union complexity threshold because:
    1. They are runtime expressions, not static type hints
    2. Modern Python (PEP 604) encourages isinstance(x, A | B) syntax
    3. Ruff UP038 recommends this syntax over isinstance(x, (A, B))

    Args:
        pattern: The ModelUnionPattern to check.

    Returns:
        True if the pattern is inside an isinstance() call, False otherwise.

    Examples:
        >>> # Pattern from: isinstance(x, str | int)
        >>> is_isinstance_union(pattern_from_isinstance)
        True
        >>> # Pattern from: def foo(x: str | int)
        >>> is_isinstance_union(pattern_from_annotation)
        False

    Note:
        This function caches file parsing results for efficiency.
    """
    file_path = Path(pattern.file_path)
    isinstance_lines = _find_isinstance_union_lines(file_path)
    return pattern.line in isinstance_lines


# ==============================================================================
# Path Skipping Configuration
# ==============================================================================
#
# These directories are excluded from validation because:
# - archive/archived: Historical code not subject to current validation rules
# - examples: Demo code that may intentionally show anti-patterns
# - __pycache__: Compiled Python bytecode, not source code
# - .git: Git repository metadata
# - .venv/venv: Virtual environment directories
# - .tox: Tox testing directory
# - .mypy_cache: mypy type checking cache
# - .pytest_cache: pytest cache directory
# - build/dist: Build output directories
# - .eggs: setuptools eggs directory
# - node_modules: Node.js dependencies (if any JS in repo)
#
# The set is used for O(1) lookup when checking path components.
#
# Note: Matching is case-sensitive (Linux standard). On case-insensitive
# filesystems (macOS, Windows), "Archive" would NOT match "archive".
# This is intentional for portability and consistency.
SKIP_DIRECTORY_NAMES: frozenset[str] = frozenset(
    {
        # Historical/demo code
        "archive",
        "archived",
        "examples",
        # Bytecode and caches
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        # Virtual environments
        ".venv",
        "venv",
        # Build outputs
        "build",
        "dist",
        ".eggs",
        # Version control
        ".git",
        # Testing
        ".tox",
        # Node.js (if present)
        "node_modules",
    }
)


def is_skip_directory(component: str) -> bool:
    """
    Check if a path component is a directory that should be skipped.

    This predicate is extracted for reuse and testability. It checks if the
    given component matches one of the known skip directory names exactly.

    Uses exact string matching (case-sensitive) via set membership for O(1) lookup.
    This prevents false positives from substring matching.

    Skip directories are loaded from validation_exemptions.yaml if configured,
    otherwise falls back to the hardcoded SKIP_DIRECTORY_NAMES default.
    See get_skip_directories() for the configuration loading logic.

    Args:
        component: A single path component (directory or file name).

    Returns:
        True if the component is a skip directory name, False otherwise.

    Examples:
        Exact matches (skipped):
        >>> is_skip_directory("archived")
        True
        >>> is_skip_directory("archive")
        True
        >>> is_skip_directory("__pycache__")
        True
        >>> is_skip_directory(".venv")
        True
        >>> is_skip_directory(".git")
        True

        Partial/similar names (NOT skipped - prevents false positives):
        >>> is_skip_directory("archived_feature")
        False
        >>> is_skip_directory("my_archive")
        False
        >>> is_skip_directory("Archive")  # Case-sensitive
        False
        >>> is_skip_directory(".git_backup")
        False
    """
    return component in get_skip_directories()


def should_skip_path(path: Path) -> bool:
    """
    Check if a path should be skipped for validation.

    Uses exact path component matching to avoid false positives from substring
    matching. A path is skipped if ANY of its PARENT directory components match
    a known skip directory name exactly. The filename itself is NOT checked
    to avoid false positives from files that happen to share names with skip
    directories (e.g., `archive.py` should not be skipped).

    This approach prevents false positives like:
    - /foo/archived_feature/bar.py - NOT skipped ("archived_feature" != "archived")
    - /foo/archive_manager.py - NOT skipped (only checks parent dirs, not filename)
    - /foo/examples_utils.py - NOT skipped (only checks parent dirs, not filename)
    - /foo/my_archive/bar.py - NOT skipped ("my_archive" != "archive")
    - /foo/.git_backup/bar.py - NOT skipped (".git_backup" != ".git")

    While correctly skipping:
    - /foo/archived/bar.py - Skipped (has "archived" directory component)
    - /foo/archive/bar.py - Skipped (has "archive" directory component)
    - /foo/examples/bar.py - Skipped (has "examples" directory component)
    - /foo/__pycache__/bar.pyc - Skipped (has "__pycache__" directory component)
    - /foo/.venv/lib/bar.py - Skipped (has ".venv" directory component)
    - /foo/.git/hooks/pre-commit - Skipped (has ".git" directory component)
    - /foo/build/lib/bar.py - Skipped (has "build" directory component)

    Args:
        path: The file path to check.

    Returns:
        True if the path should be skipped, False otherwise.

    Note:
        Matching is case-sensitive (Linux standard). On case-insensitive
        filesystems (macOS, Windows), directories like "Build" or "VENV"
        would NOT be skipped. This is intentional for cross-platform
        consistency - use lowercase directory names for skipped directories.
    """
    # Check PARENT directory components only (exclude the filename)
    # This prevents false positives from files named like skip directories
    # (e.g., archive.py, examples.py)
    #
    # path.parts includes all components including filename:
    # "/foo/archived/bar.py" -> ('/', 'foo', 'archived', 'bar.py')
    #
    # path.parent.parts excludes the filename:
    # "/foo/archived/bar.py" -> ('/', 'foo', 'archived')
    #
    # Using parent.parts ensures we only match DIRECTORY names, not filenames
    return any(is_skip_directory(part) for part in path.parent.parts)


def _count_non_optional_unions(
    directory: Path,
) -> tuple[int, int, int, int, list[str]]:
    """
    Count unions in a directory, excluding simple optional and isinstance patterns.

    This function provides accurate union counting for threshold checks by
    excluding:
    - Idiomatic `X | None` patterns (simple optionals) that are valid ONEX style
    - isinstance(x, A | B) patterns that are runtime type checks, not annotations

    Args:
        directory: Directory to scan for Python files.

    Returns:
        Tuple of (threshold_count, total_count, optional_count, isinstance_count, issues):
        - threshold_count: Count of unions that count toward threshold
          (excludes both `X | None` and isinstance patterns)
        - total_count: Total count of all unions (for reporting)
        - optional_count: Count of simple `X | None` patterns excluded
        - isinstance_count: Count of isinstance unions excluded
        - issues: List of validation issues found
    """
    total_unions = 0
    threshold_unions = 0
    optional_unions = 0
    isinstance_unions = 0
    all_issues: list[str] = []

    for py_file in directory.rglob("*.py"):
        # Filter out archived files, examples, and __pycache__
        if should_skip_path(py_file):
            continue

        union_count, issues, patterns = validate_union_usage_file(py_file)
        total_unions += union_count

        # Count and categorize patterns
        for pattern in patterns:
            if is_simple_optional(pattern):
                optional_unions += 1
            elif is_isinstance_union(pattern):
                isinstance_unions += 1
            else:
                threshold_unions += 1

        # Prefix issues with file path
        if issues:
            all_issues.extend([f"{py_file}: {issue}" for issue in issues])

    return (
        threshold_unions,
        total_unions,
        optional_unions,
        isinstance_unions,
        all_issues,
    )


def validate_infra_union_usage(
    directory: PathInput = INFRA_SRC_PATH,
    max_unions: int = INFRA_MAX_UNIONS,
    strict: bool = INFRA_UNIONS_STRICT,
) -> ValidationResult:
    """
    Validate Union type usage in infrastructure code.

    Prevents overly complex union types that complicate infrastructure code.

    This validator EXCLUDES the following patterns from the count:
    - Simple optional patterns (`X | None`) - idiomatic nullable types
    - isinstance() unions (`isinstance(x, A | B)`) - runtime type checks

    Only actual complex TYPE ANNOTATIONS count toward the threshold.

    What IS counted (threshold applies to):
        - Multi-type unions in annotations: `def foo(x: str | int)`
        - Complex patterns: unions with 3+ types in annotations
        - Non-optional type hints: any annotation union without `None`

    What is NOT counted (excluded from threshold):
        - Simple optionals: `X | None` where X is any single type
        - isinstance() unions: `isinstance(x, A | B)` (runtime checks, not annotations)
        - These are either idiomatic Python or runtime expressions, not type complexity

    Exemptions:
        Exemption patterns are loaded from validation_exemptions.yaml (union_exemptions section).
        See that file for the complete list of exemptions with rationale.

        Key exemption categories:
        - ModelNodeCapabilities.config: JSON-like configuration pattern with primitive unions

    Args:
        directory: Directory to validate. Defaults to infrastructure source.
        max_unions: Maximum union count threshold. Defaults to INFRA_MAX_UNIONS.
            Note: This threshold applies only after excluding optionals and isinstance.
        strict: Enable strict mode. Defaults to INFRA_UNIONS_STRICT (True).

    Returns:
        ModelValidationResult with validation status and any errors.
        The metadata includes total_unions (all unions), threshold_unions (what counts),
        and breakdown of excluded patterns for transparency.

    Metadata Extension Fields:
        ModelValidationMetadata uses `extra="allow"` to support domain-specific fields.
        The following extension fields are used by this validator and are properly typed:

        - non_optional_unions (int): Count of unions after all exclusions.
          This is what the threshold check applies to.
        - optional_unions_excluded (int): Count of simple `X | None` optionals excluded.
        - isinstance_unions_excluded (int): Count of isinstance() unions excluded.

        These fields are additional to the base ModelValidationMetadata fields like
        total_unions and max_unions which are formally defined on the model.
    """
    # Convert to Path if string
    dir_path = Path(directory) if isinstance(directory, str) else directory

    # Count unions with exclusion of simple optionals and isinstance patterns
    threshold_count, total_count, optional_count, isinstance_count, issues = (
        _count_non_optional_unions(dir_path)
    )

    # Load exemption patterns from YAML configuration
    exempted_patterns = get_union_exemptions()

    # Filter errors using regex-based pattern matching
    filtered_issues = _filter_exempted_errors(issues, exempted_patterns)

    # Determine validity: threshold count must be within max
    # and no issues in strict mode
    is_valid = (threshold_count <= max_unions) and (not filtered_issues or not strict)

    # Count Python files for metadata (excluding archive, examples, __pycache__)
    python_files = list(dir_path.rglob("*.py"))
    files_processed = len([f for f in python_files if not should_skip_path(f)])

    # Create result with enhanced metadata showing all counts
    # Note: ModelValidationMetadata uses extra="allow", so extension fields
    # are accepted as int values.
    # See docstring "Metadata Extension Fields" section for field documentation.
    #
    # Extension fields are passed via model_construct() to satisfy type checker
    # while preserving runtime behavior with extra="allow".
    metadata_fields: dict[str, object] = {
        # Standard ModelValidationMetadata fields (formally defined)
        "validation_type": "union_usage",
        "files_processed": files_processed,
        "violations_found": len(filtered_issues),
        "total_unions": total_count,  # Base field: all unions found
        "max_unions": max_unions,  # Base field: configured threshold
        "strict_mode": strict,  # Base field: whether strict mode enabled
        # Extension fields (via extra="allow", typed as int)
        # These provide transparency into the exclusion logic:
        "non_optional_unions": threshold_count,  # What threshold actually checks
        "optional_unions_excluded": optional_count,  # X | None patterns
        "isinstance_unions_excluded": isinstance_count,  # isinstance(x, A | B) patterns
    }
    return ModelValidationResult(
        is_valid=is_valid,
        errors=filtered_issues,
        metadata=ModelValidationMetadata.model_construct(**metadata_fields),  # type: ignore[arg-type]
    )


def validate_infra_circular_imports(
    directory: PathInput = INFRA_SRC_PATH,
) -> ModelImportValidationResult:
    """
    Check for circular imports in infrastructure code.

    Infrastructure packages have complex dependencies; circular imports
    cause runtime issues that are hard to debug.

    Args:
        directory: Directory to check. Defaults to infrastructure source.

    Returns:
        ModelImportValidationResult with detailed import validation results.
        Use result.has_circular_imports to check for issues.
    """
    validator = CircularImportValidator(source_path=Path(directory))
    return validator.validate()


def validate_infra_all(
    directory: PathInput = INFRA_SRC_PATH,
    nodes_directory: PathInput = INFRA_NODES_PATH,
) -> dict[str, ValidationResult | ModelImportValidationResult]:
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
    results: dict[str, ValidationResult | ModelImportValidationResult] = {}

    # HIGH priority validators
    results["architecture"] = validate_infra_architecture(directory)
    results["contracts"] = validate_infra_contracts(nodes_directory)
    results["patterns"] = validate_infra_patterns(directory)

    # MEDIUM priority validators
    results["union_usage"] = validate_infra_union_usage(directory)
    results["circular_imports"] = validate_infra_circular_imports(directory)

    return results


def get_validation_summary(
    results: dict[str, ValidationResult | ModelImportValidationResult],
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
        # Use duck typing to determine result API:
        # - ModelModuleImportResult has 'has_circular_imports' attribute
        # - ModelValidationResult has 'is_valid' attribute
        # This follows ONEX convention of duck typing over isinstance for protocols.
        if hasattr(result, "has_circular_imports"):
            # Circular import validator uses has_circular_imports
            if not result.has_circular_imports:
                passed += 1
            else:
                failed += 1
                failed_validators.append(name)
        elif hasattr(result, "is_valid"):
            # Standard ModelValidationResult uses is_valid
            if result.is_valid:
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
    # Constants
    "EXEMPTIONS_YAML_PATH",  # Path to exemptions YAML file
    "INFRA_MAX_UNIONS",  # Maximum union count threshold
    "INFRA_MAX_VIOLATIONS",  # Maximum violations threshold
    "INFRA_NODES_PATH",  # Nodes directory path
    "INFRA_PATTERNS_STRICT",  # Strict pattern validation flag
    "INFRA_SRC_PATH",  # Source directory path
    "INFRA_UNIONS_STRICT",  # Strict union validation flag
    "SKIP_DIRECTORY_NAMES",  # Directories to skip
    # Types
    "ExemptionPattern",  # Exemption pattern TypedDict
    "ModelModuleImportResult",  # Re-export from omnibase_core
    "ValidationResult",  # Type alias for validation result
    # Exemption loaders
    "get_architecture_exemptions",  # Architecture exemption loader
    "get_pattern_exemptions",  # Pattern exemption loader
    "get_skip_directories",  # Skip directory loader
    "get_union_exemptions",  # Union exemption loader
    "get_validation_summary",  # Validation summary generator
    # Path utilities
    "is_isinstance_union",  # Check if union is in isinstance() call
    "is_simple_optional",  # Check if union is X | None
    "is_skip_directory",  # Check if directory should be skipped
    "load_skip_directories_from_yaml",  # Load skip dirs from YAML
    "should_skip_path",  # Check if path should be skipped
    # Validators
    "validate_infra_all",  # Run all validators
    "validate_infra_architecture",  # Architecture validation
    "validate_infra_circular_imports",  # Circular import check
    "validate_infra_contracts",  # Contract validation
    "validate_infra_patterns",  # Pattern validation
    "validate_infra_union_usage",  # Union usage validation
]
