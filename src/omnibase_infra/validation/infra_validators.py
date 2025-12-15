"""
Infrastructure-specific validation wrappers.

Provides validators from omnibase_core with sensible defaults for infrastructure code.
All wrappers maintain strong typing and follow ONEX validation patterns.
"""

from pathlib import Path
from typing import Literal

from omnibase_core.models.common.model_validation_result import ModelValidationResult
from omnibase_core.models.validation.model_contract_validation_result import (
    ModelContractValidationResult,
)
from omnibase_core.models.validation.model_import_validation_result import (
    ModelValidationResult as CircularImportValidationResult,
)
from omnibase_core.validation import (
    validate_architecture,
    validate_contracts,
    validate_patterns,
    validate_union_usage,
)
from omnibase_core.validation.circular_import_validator import CircularImportValidator
from omnibase_core.validation.contract_validator import ProtocolContractValidator

# Type alias for cleaner return types in infrastructure validators
# Most validation results return None as the data payload (validation only)
# Using Python 3.12+ type keyword for modern type alias syntax
type ValidationResult = ModelValidationResult[None]

# Default paths for infrastructure validation
INFRA_SRC_PATH = "src/omnibase_infra/"
INFRA_NODES_PATH = "src/omnibase_infra/nodes/"

# Maximum allowed complex union types in infrastructure code.
# Infrastructure code has many typed handlers (Consul, Kafka, Vault, PostgreSQL adapters)
# which require typed unions for protocol implementations and message routing.
# Set to 30 to accommodate infrastructure service integration patterns including
# RuntimeHostProcess and handler wiring while preventing overly complex union types.
INFRA_MAX_UNIONS = 30

# Maximum allowed architecture violations in infrastructure code.
# Set to 0 (strict enforcement) to ensure one-model-per-file principle is always followed.
# Infrastructure nodes require strict architecture compliance for maintainability and
# contract-driven code generation.
INFRA_MAX_VIOLATIONS = 0

# Strict mode for pattern validation in infrastructure code.
# Set to True to enforce all naming conventions and anti-patterns (no *Manager, *Handler, *Helper).
# Infrastructure code must follow ONEX patterns strictly for consistency across service adapters.
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
    Validate infrastructure code patterns.

    Enforces:
    - Model prefix naming (Model*)
    - snake_case file naming
    - Anti-pattern detection (no *Manager, *Handler, *Helper)

    Args:
        directory: Directory to validate. Defaults to infrastructure source.
        strict: Enable strict mode. Defaults to INFRA_PATTERNS_STRICT (True).

    Returns:
        ModelValidationResult with validation status and any errors.
    """
    return validate_patterns(str(directory), strict=strict)


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
        max_unions: Maximum allowed complex unions. Defaults to INFRA_MAX_UNIONS (20).
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
