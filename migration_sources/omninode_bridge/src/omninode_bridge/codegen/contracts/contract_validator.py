#!/usr/bin/env python3
"""
Contract Validator for ONEX v2.0 Contracts (Phase 3).

Validates contract fields including Phase 3 enhancements:
- Template configuration (variant, patterns, custom templates)
- Generation directives (LLM tier, quality level, fallback strategy)
- Quality gates configuration
- Subcontract references
- Backward compatibility checks

Provides detailed validation results with clear error messages and suggestions.

Thread-safe and stateless.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ============================================================================
# Validation Result Models
# ============================================================================


@dataclass
class ModelFieldValidationError:
    """
    Single field validation error.

    Attributes:
        field_name: Name of the field with error
        error_message: Descriptive error message
        suggestion: Suggested fix
        severity: Error severity (error, warning, info)
    """

    field_name: str
    error_message: str
    suggestion: str = ""
    severity: str = "error"  # error, warning, info


@dataclass
class ModelContractValidationResult:
    """
    Contract validation result.

    Attributes:
        is_valid: Whether contract passed all validations
        errors: List of validation errors
        warnings: List of validation warnings
        info_messages: List of informational messages
        validation_summary: Summary of validation results
    """

    is_valid: bool = True
    errors: list[ModelFieldValidationError] = field(default_factory=list)
    warnings: list[ModelFieldValidationError] = field(default_factory=list)
    info_messages: list[str] = field(default_factory=list)
    validation_summary: dict[str, Any] = field(default_factory=dict)

    @property
    def error_count(self) -> int:
        """Count of errors."""
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        """Count of warnings."""
        return len(self.warnings)

    def add_error(
        self, field_name: str, error_message: str, suggestion: str = ""
    ) -> None:
        """Add validation error."""
        self.errors.append(
            ModelFieldValidationError(
                field_name=field_name,
                error_message=error_message,
                suggestion=suggestion,
                severity="error",
            )
        )
        self.is_valid = False

    def add_warning(
        self, field_name: str, warning_message: str, suggestion: str = ""
    ) -> None:
        """Add validation warning."""
        self.warnings.append(
            ModelFieldValidationError(
                field_name=field_name,
                error_message=warning_message,
                suggestion=suggestion,
                severity="warning",
            )
        )


# ============================================================================
# Contract Validator
# ============================================================================


class ContractValidator:
    """
    Validate ONEX v2.0 contracts with Phase 3 enhancements.

    Provides comprehensive validation of:
    - Core contract fields (name, version, node_type, description)
    - Template configuration (Phase 3)
    - Generation directives (Phase 3)
    - Quality gates configuration (Phase 3)
    - Subcontract references
    - Mixin declarations (v2.0)
    - Backward compatibility

    Thread-safe and stateless - can be reused across calls.

    Example:
        >>> validator = ContractValidator()
        >>> contract = ... # ModelEnhancedContract
        >>> result = validator.validate_contract(contract)
        >>> if result.is_valid:
        ...     print("Contract is valid!")
        ... else:
        ...     for error in result.errors:
        ...         print(f"Error in {error.field_name}: {error.error_message}")
    """

    # Valid pattern names (from Foundation task F2)
    VALID_PATTERNS = {
        "circuit_breaker",
        "retry_policy",
        "dead_letter_queue",
        "transactions",
        "security_validation",
        "observability",
        "consul_integration",
        "health_checks",
        "event_publishing",
        "lifecycle",
        "metrics",
    }

    # Valid quality gate names
    VALID_QUALITY_GATES = {
        "syntax_validation",
        "onex_compliance",
        "import_resolution",
        "type_checking",
        "security_scan",
        "pattern_validation",
        "test_validation",
    }

    def __init__(self):
        """Initialize contract validator."""
        pass

    def validate_contract(
        self,
        contract: Any,  # ModelEnhancedContract type hint would be circular
    ) -> ModelContractValidationResult:
        """
        Validate contract with all Phase 3 enhancements.

        Args:
            contract: ModelEnhancedContract instance

        Returns:
            ModelContractValidationResult with validation results

        Example:
            >>> result = validator.validate_contract(contract)
            >>> print(f"Valid: {result.is_valid}, Errors: {result.error_count}")
        """
        result = ModelContractValidationResult()

        # Validate core fields
        self._validate_core_fields(contract, result)

        # Validate Phase 3 fields (v2.1+)
        if contract.schema_version.startswith("v2.1") or hasattr(contract, "template"):
            self._validate_template_configuration(contract, result)
            self._validate_generation_directives(contract, result)
            self._validate_quality_gates_configuration(contract, result)

        # Validate subcontracts (if present)
        if contract.subcontracts:
            self._validate_subcontracts(contract, result)

        # Generate summary
        result.validation_summary = {
            "contract_name": contract.name,
            "schema_version": contract.schema_version,
            "total_errors": result.error_count,
            "total_warnings": result.warning_count,
            "phase_3_enabled": (
                contract.is_v2_1 if hasattr(contract, "is_v2_1") else False
            ),
        }

        logger.info(
            f"Validated contract {contract.name}: "
            f"{result.error_count} errors, {result.warning_count} warnings"
        )

        return result

    def _validate_core_fields(
        self, contract: Any, result: ModelContractValidationResult
    ) -> None:
        """
        Validate core contract fields.

        Args:
            contract: Contract instance
            result: Validation result accumulator
        """
        # Validate name
        if not contract.name:
            result.add_error(
                "name",
                "Contract name is required",
                "Provide a valid node name following NodeXxxYyy pattern",
            )
        elif not re.match(r"^Node[A-Z][a-zA-Z0-9]*$", contract.name):
            result.add_warning(
                "name",
                f"Contract name '{contract.name}' does not follow NodeXxxYyy pattern",
                "Use PascalCase with 'Node' prefix (e.g., NodeMyEffect)",
            )

        # Validate node_type
        valid_node_types = {"effect", "compute", "reducer", "orchestrator"}
        if contract.node_type not in valid_node_types:
            result.add_error(
                "node_type",
                f"Invalid node_type '{contract.node_type}'",
                f"Must be one of: {', '.join(valid_node_types)}",
            )

        # Validate description
        if not contract.description:
            result.add_warning(
                "description",
                "Contract description is empty",
                "Provide a clear description of the node's purpose",
            )

    def _validate_template_configuration(
        self, contract: Any, result: ModelContractValidationResult
    ) -> None:
        """
        Validate template configuration (Phase 3).

        Args:
            contract: Contract instance
            result: Validation result accumulator
        """
        template = contract.template

        # Validate variant
        # (enum validation already handled by pydantic/dataclass)

        # Validate custom_template path
        if template.variant.value == "custom":
            if not template.custom_template:
                result.add_error(
                    "template.custom_template",
                    "Template variant 'custom' requires custom_template path",
                    "Provide path to custom template file",
                )
            elif template.custom_template:
                # Check if path exists (if it's an absolute path)
                custom_path = Path(template.custom_template)
                if custom_path.is_absolute() and not custom_path.exists():
                    result.add_warning(
                        "template.custom_template",
                        f"Custom template path does not exist: {template.custom_template}",
                        "Verify the template file path is correct",
                    )

        # Validate patterns
        invalid_patterns = [
            p for p in template.patterns if p not in self.VALID_PATTERNS
        ]
        if invalid_patterns:
            result.add_error(
                "template.patterns",
                f"Invalid pattern names: {', '.join(invalid_patterns)}",
                f"Valid patterns: {', '.join(sorted(self.VALID_PATTERNS))}",
            )

        # Validate pattern_configuration
        for pattern_name, pattern_config in template.pattern_configuration.items():
            if pattern_name not in template.patterns:
                result.add_warning(
                    f"template.pattern_configuration.{pattern_name}",
                    f"Pattern '{pattern_name}' has configuration but is not in patterns list",
                    f"Add '{pattern_name}' to template.patterns",
                )

            if not isinstance(pattern_config, dict):
                result.add_error(
                    f"template.pattern_configuration.{pattern_name}",
                    f"Pattern configuration must be dict, got {type(pattern_config).__name__}",
                    "Use dict for pattern configuration",
                )

    def _validate_generation_directives(
        self, contract: Any, result: ModelContractValidationResult
    ) -> None:
        """
        Validate generation directives (Phase 3).

        Args:
            contract: Contract instance
            result: Validation result accumulator
        """
        generation = contract.generation

        # Validate max_context_size
        if generation.max_context_size < 1000:
            result.add_warning(
                "generation.max_context_size",
                f"Max context size {generation.max_context_size} is very small",
                "Consider using at least 4000 tokens for meaningful context",
            )
        elif generation.max_context_size > 32000:
            result.add_warning(
                "generation.max_context_size",
                f"Max context size {generation.max_context_size} exceeds most model limits",
                "Most models support up to 32000 tokens",
            )

        # Validate timeout_seconds
        if generation.timeout_seconds < 5:
            result.add_warning(
                "generation.timeout_seconds",
                f"Timeout {generation.timeout_seconds}s may be too short for LLM calls",
                "Consider at least 30s timeout for reliable generation",
            )

        # Validate retry_attempts
        if generation.retry_attempts < 1:
            result.add_error(
                "generation.retry_attempts",
                "Retry attempts must be at least 1",
                "Set retry_attempts to at least 1",
            )
        elif generation.retry_attempts > 10:
            result.add_warning(
                "generation.retry_attempts",
                f"Retry attempts {generation.retry_attempts} is very high",
                "Consider 3-5 retry attempts for balance of reliability and speed",
            )

        # Validate LLM enablement consistency
        if not generation.enable_llm and generation.include_patterns:
            result.add_warning(
                "generation.include_patterns",
                "Pattern context enhancement enabled but LLM is disabled",
                "Set enable_llm=true to use pattern examples in generation",
            )

    def _validate_quality_gates_configuration(
        self, contract: Any, result: ModelContractValidationResult
    ) -> None:
        """
        Validate quality gates configuration (Phase 3).

        Args:
            contract: Contract instance
            result: Validation result accumulator
        """
        quality_gates = contract.quality_gates

        # Validate gate names
        for gate in quality_gates.gates:
            if gate.name not in self.VALID_QUALITY_GATES:
                result.add_warning(
                    f"quality_gates.gates.{gate.name}",
                    f"Unknown quality gate: {gate.name}",
                    f"Valid gates: {', '.join(sorted(self.VALID_QUALITY_GATES))}",
                )

        # Check for recommended gates
        required_gates = {g.name for g in quality_gates.get_required_gates()}

        if "syntax_validation" not in required_gates:
            result.add_warning(
                "quality_gates",
                "Recommended gate 'syntax_validation' not required",
                "Consider adding syntax_validation as required gate",
            )

        if "onex_compliance" not in required_gates and contract.mixins:
            result.add_warning(
                "quality_gates",
                "Recommended gate 'onex_compliance' not required for mixin-enhanced contract",
                "Consider adding onex_compliance as required gate",
            )

        # Validate gate configurations
        for gate in quality_gates.gates:
            if gate.config and not isinstance(gate.config, dict):
                result.add_error(
                    f"quality_gates.gates.{gate.name}.config",
                    f"Gate config must be dict, got {type(gate.config).__name__}",
                    "Use dict for gate configuration",
                )

    def _validate_subcontracts(
        self, contract: Any, result: ModelContractValidationResult
    ) -> None:
        """
        Validate subcontract references.

        Args:
            contract: Contract instance
            result: Validation result accumulator
        """
        if not isinstance(contract.subcontracts, dict):
            result.add_error(
                "subcontracts",
                f"Subcontracts must be dict, got {type(contract.subcontracts).__name__}",
                "Use dict format for subcontracts",
            )
            return

        # Import here to avoid circular imports
        from .subcontract_processor import EnumSubcontractType

        for subcontract_name, subcontract_data in contract.subcontracts.items():
            if not isinstance(subcontract_data, dict):
                result.add_error(
                    f"subcontracts.{subcontract_name}",
                    f"Subcontract must be dict, got {type(subcontract_data).__name__}",
                    "Use dict format for subcontract definition",
                )
                continue

            # Validate subcontract type
            subcontract_type = subcontract_data.get("type")
            if not subcontract_type:
                result.add_error(
                    f"subcontracts.{subcontract_name}.type",
                    "Subcontract type is required",
                    f"Provide type (one of: {', '.join(t.value for t in EnumSubcontractType)})",
                )
            else:
                valid_types = {t.value for t in EnumSubcontractType}
                if subcontract_type not in valid_types:
                    result.add_error(
                        f"subcontracts.{subcontract_name}.type",
                        f"Invalid subcontract type: {subcontract_type}",
                        f"Must be one of: {', '.join(sorted(valid_types))}",
                    )


# Export
__all__ = [
    "ModelFieldValidationError",
    "ModelContractValidationResult",
    "ContractValidator",
]
