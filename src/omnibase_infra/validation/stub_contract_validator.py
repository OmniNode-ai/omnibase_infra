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

from pathlib import Path
from typing import TYPE_CHECKING

from omnibase_core.enums import EnumNodeType
from omnibase_core.models.contracts import ModelContractBase
from omnibase_core.validation import ModelContractValidationResult
from omnibase_core.validation.contract_validator import ProtocolContractValidator

if TYPE_CHECKING:
    from omnibase_core.models import ModelSemVer


class StubContractValidator:
    """Stub replacement for contract validator.

    This class provides a minimal implementation to allow existing
    code to work after the omnibase_core 0.6.2 migration.

    WARNING: This is a stub implementation. Some methods return
    placeholder results and should be updated with actual validation
    logic if needed.
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

        Args:
            contract_path: Path to the contract.yaml file.
            contract_type: Expected node type (EFFECT, COMPUTE, REDUCER, ORCHESTRATOR).

        Returns:
            ModelContractValidationResult with validation status and score.
        """
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
