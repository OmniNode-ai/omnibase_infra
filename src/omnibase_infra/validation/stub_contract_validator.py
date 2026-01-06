# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""TEMPORARY STUB: Contract validator removed in omnibase_core 0.6.2.

WARNING: This module is a TEMPORARY workaround and will be removed.
Do not add new dependencies on this module.

This stub provides backwards compatibility for code that depends on
the removed ServiceContractValidator. It implements basic contract
validation using the validation functions from omnibase_core.validation.

Migration Path:
    All code should migrate to using omnibase_core.validation API directly.
    See OMN-1104 for the migration tracking and context.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

from omnibase_core.enums import EnumNodeType
from omnibase_core.models.contracts import ModelContractBase
from omnibase_core.validation import ModelContractValidationResult
from omnibase_core.validation.contract_validator import ProtocolContractValidator

if TYPE_CHECKING:
    from omnibase_core.models import ModelSemVer

# Module-level flag to track if warning has been shown (avoid spam)
# Mutable container for warning state (avoids global statement)
_warning_state: dict[str, bool] = {"shown": False}


class StubContractValidator:
    """TEMPORARY stub replacement for ServiceContractValidator.

    This class provides a minimal implementation to allow existing code to work
    after the omnibase_core 0.6.2 migration removed ServiceContractValidator.

    Warning:
        This is a STUB implementation that performs MINIMAL validation only.
        Using this validator may give a FALSE SENSE OF SECURITY as it does not
        perform full ONEX contract compliance checks.

    Validations Performed (MINIMAL):
        - File existence check
        - Valid YAML parsing
        - Required field presence check (``name``, ``node_type``)
        - Node type matching (if ``contract_type`` argument provided)

    Validations NOT Performed (CRITICAL GAPS):
        - Full ONEX contract schema validation
        - Semantic version compliance (contract_version, node_version)
        - Input/output model validation (type existence, schema correctness)
        - Handler routing configuration validation
        - FSM state machine validation for reducers
        - Cross-reference validation (referenced models exist)
        - Protocol compliance checks
        - Metadata field validation (author, description, etc.)
        - Dependency graph validation
        - Version compatibility checks

    Why This Stub Exists:
        The ``ServiceContractValidator`` class was removed in omnibase_core 0.6.2.
        This stub provides backwards compatibility for code that still imports
        and uses the old validator API. It is NOT a replacement for proper
        contract validation.

    Limitations:
        - Returns 100% compliance score for contracts that only pass basic checks
        - Cannot detect malformed contract structures beyond basic YAML
        - Does not validate against ONEX contract schema
        - May allow invalid contracts to pass validation
        - Should NOT be used in production validation pipelines

    Migration Path:
        1. Migrate to ``omnibase_core.validation`` API directly
        2. Use ``validate_contracts()`` for directory-level validation
        3. Implement proper schema validation using ``ModelContractBase``
        4. See OMN-1104 for migration tracking and detailed guidance

    Example:
        >>> # WARNING: This emits a runtime warning!
        >>> validator = StubContractValidator()
        >>> result = validator.validate_contract_file(Path("contract.yaml"))
        >>> # result.valid may be True even if contract is not fully compliant

    See Also:
        - OMN-1104: Migration tracking issue
        - omnibase_core.validation: Target validation API
        - ModelContractBase: Contract model for proper validation
    """

    # TODO(OMN-1104): Remove this stub after migrating to omnibase_core.validation API
    # This is a temporary workaround for backwards compatibility with code that
    # depends on the removed ServiceContractValidator. The proper fix is to update
    # all validation code to use the new omnibase_core.validation API directly.
    # Tracking: https://linear.app/omninode/issue/OMN-1104

    def validate_contract_file(
        self,
        contract_path: Path,
        contract_type: EnumNodeType | None = None,
    ) -> ModelContractValidationResult:
        """Validate a contract YAML file.

        Warning:
            This method performs MINIMAL validation only. It checks for file
            existence, valid YAML, and basic required fields. It does NOT
            perform full ONEX contract compliance validation.

            A passing result from this stub does NOT guarantee the contract
            is valid for production use.

        Args:
            contract_path: Path to the contract.yaml file.
            contract_type: Expected node type (EFFECT, COMPUTE, REDUCER, ORCHESTRATOR).

        Returns:
            ModelContractValidationResult with validation status and score.
            Note: A score of 100.0 only means basic checks passed, not full compliance.

        Raises:
            No exceptions are raised; all errors are captured in the result.
        """
        # Emit runtime warning (once per session to avoid spam)
        if not _warning_state["shown"]:
            warnings.warn(
                "StubContractValidator is a TEMPORARY stub that performs MINIMAL validation. "
                "Real ONEX contract validation is NOT being performed. "
                "This may allow invalid contracts to pass. "
                "See OMN-1104 for migration to omnibase_core.validation API.",
                UserWarning,
                stacklevel=2,
            )
            _warning_state["shown"] = True

        # Use the validation functions from omnibase_core.validation
        try:
            from omnibase_core.validation import validate_contracts

            # validate_contracts expects a directory, not a single file
            # So we use a simpler validation for now
            if not contract_path.exists():
                return ModelContractValidationResult(
                    valid=False,
                    score=0.0,
                    errors=[f"Contract file not found: {contract_path}"],
                    warnings=[],
                    compliance_score=0.0,
                )

            import yaml

            with contract_path.open("r", encoding="utf-8") as f:
                contract_data = yaml.safe_load(f)

            if not isinstance(contract_data, dict):
                return ModelContractValidationResult(
                    valid=False,
                    score=0.0,
                    errors=["Contract file does not contain valid YAML dict"],
                    warnings=[],
                    compliance_score=0.0,
                )

            # Basic validation - check required fields
            required_fields = ["name", "node_type"]
            missing = [f for f in required_fields if f not in contract_data]

            if missing:
                return ModelContractValidationResult(
                    valid=False,
                    score=50.0,
                    errors=[f"Missing required fields: {missing}"],
                    warnings=[],
                    compliance_score=50.0,
                )

            # Check node_type matches expected if provided
            if contract_type is not None:
                actual_type = contract_data.get("node_type", "").upper()
                expected_type = (
                    contract_type.value.upper()
                    if hasattr(contract_type, "value")
                    else str(contract_type).upper()
                )
                if actual_type != expected_type:
                    return ModelContractValidationResult(
                        valid=False,
                        score=75.0,
                        errors=[
                            f"Node type mismatch: expected {expected_type}, got {actual_type}"
                        ],
                        warnings=[],
                        compliance_score=75.0,
                    )

            return ModelContractValidationResult(
                valid=True,
                score=100.0,
                errors=[],
                warnings=[],
                compliance_score=100.0,
            )

        except Exception as e:
            return ModelContractValidationResult(
                valid=False,
                score=0.0,
                errors=[f"Validation error: {e!s}"],
                warnings=[],
                compliance_score=0.0,
            )


# Alias for backwards compatibility
ServiceContractValidator = StubContractValidator

__all__ = ["ServiceContractValidator", "StubContractValidator"]
