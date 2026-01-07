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

Security Note:
    This stub may MASK REAL VALIDATION FAILURES by returning success for
    contracts that would fail full ONEX compliance validation. Do not rely
    on this validator for production contract verification.

    CRITICAL SECURITY CONSIDERATIONS:
    1. This validator performs MINIMAL checks only - see class docstring for gaps
    2. A passing result does NOT guarantee ONEX compliance
    3. Unexpected errors are RAISED (not silently returned) to prevent masking
    4. Use in production pipelines is STRONGLY DISCOURAGED
    5. The 65% compliance score indicates incomplete validation

Appropriate Use Cases:
    - Local development iteration
    - Rapid prototyping
    - Test fixtures that need basic contract structure

Inappropriate Use Cases (SECURITY RISK):
    - Production deployment validation
    - CI/CD pipeline gates
    - Security-sensitive contract verification
    - Any context where false positives could cause harm
"""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from omnibase_core.enums import EnumNodeType
from omnibase_core.errors import OnexError
from omnibase_core.models.contracts import ModelContractBase
from omnibase_core.validation import ModelContractValidationResult
from omnibase_core.validation.contract_validator import ProtocolContractValidator

if TYPE_CHECKING:
    from omnibase_core.models import ModelSemVer

logger = logging.getLogger(__name__)

# Module-level flag to track if warning has been shown (avoid spam)
# Mutable container for warning state (avoids global statement)
_warning_state: dict[str, bool] = {"shown": False}

# Environment variable to detect production context
_PRODUCTION_INDICATORS = frozenset(
    {
        "production",
        "prod",
        "staging",
        "stg",
    }
)


def _is_production_context() -> bool:
    """Detect if running in a production-like environment.

    Checks common environment variables that indicate production deployment.
    This is used to emit stronger warnings when stub validator is used
    inappropriately.

    Returns:
        True if environment appears to be production-like.
    """
    env_value = os.environ.get("ENVIRONMENT", "").lower()
    if env_value in _PRODUCTION_INDICATORS:
        return True

    # Also check common CI/CD indicators
    if os.environ.get("CI") == "true":
        return True

    # Check for Kubernetes/container deployment indicators
    if os.environ.get("KUBERNETES_SERVICE_HOST"):
        return True

    return False


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
        - Returns 65% compliance score for contracts that only pass basic checks
          (reduced from 100% to clearly indicate incomplete validation)
        - Cannot detect malformed contract structures beyond basic YAML
        - Does not validate against ONEX contract schema
        - May allow invalid contracts to pass validation
        - Should NOT be used in production validation pipelines
        - Raises OnexError for unexpected errors (to prevent masking issues)

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
            Note: A score of 65.0 indicates basic checks passed, not full compliance.
            Full compliance (100%) is only returned by the real validator.
            The 65% threshold was chosen to clearly distinguish stub validation
            from real ONEX compliance validation and prevent masking failures.

        Raises:
            OnexError: For unexpected errors that cannot be safely captured in
                a validation result. This follows ONEX error patterns and prevents
                masking of serious issues. Expected errors (YAML parsing, file
                access) are captured in the result to support batch validation.
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

        # SECURITY: Log warning for production-like environments
        if _is_production_context():
            logger.warning(
                "SECURITY WARNING: StubContractValidator used in production-like "
                "environment (ENVIRONMENT=%s, CI=%s). This validator performs "
                "MINIMAL validation and may allow invalid contracts to pass. "
                "Migrate to omnibase_core.validation API immediately. "
                "See OMN-1104 for migration guidance. (path=%s)",
                os.environ.get("ENVIRONMENT", ""),
                os.environ.get("CI", ""),
                contract_path,
            )

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

            # WARNING: Stub validator returns reduced compliance score (65%) to indicate
            # that FULL ONEX compliance validation has NOT been performed. A score of
            # 100% should only be returned by the real validator after complete checks.
            # The 65% threshold was chosen to:
            # 1. Clearly distinguish from real validation (which returns 100%)
            # 2. Prevent masking real validation failures in CI/CD pipelines
            # 3. Signal to consumers that additional validation is required
            # TODO(OMN-1104): Implement real validator that performs full ONEX checks
            return ModelContractValidationResult(
                valid=True,
                score=65.0,
                errors=[],
                warnings=[
                    "STUB VALIDATOR: Only basic checks performed (65% compliance). "
                    "Full ONEX compliance validation was NOT executed. "
                    "This score intentionally low to prevent masking failures. "
                    "See OMN-1104 for migration to real validator."
                ],
                compliance_score=65.0,
            )

        except yaml.YAMLError as e:
            # ONEX Error Pattern: Log with context and return structured error
            # NOTE: Stub validator catches exceptions to return results rather than raise,
            # as this allows callers to collect multiple validation results. However, we
            # still follow ONEX patterns for logging and error context.
            logger.warning(
                "Contract YAML parsing failed: %s (path=%s)",
                str(e),
                contract_path,
                exc_info=True,
            )
            return ModelContractValidationResult(
                valid=False,
                score=0.0,
                errors=[f"YAML parsing error: {e!s}"],
                warnings=[],
                compliance_score=0.0,
            )

        except OSError as e:
            # File system errors (permissions, disk issues, etc.)
            logger.warning(
                "Contract file access error: %s (path=%s)",
                str(e),
                contract_path,
                exc_info=True,
            )
            return ModelContractValidationResult(
                valid=False,
                score=0.0,
                errors=[f"File access error: {e!s}"],
                warnings=[],
                compliance_score=0.0,
            )

        except Exception as e:
            # ONEX Error Pattern: Raise OnexError for unexpected exceptions
            # SECURITY: Unexpected errors MUST be raised, not silently returned,
            # to prevent masking serious issues that could allow invalid contracts
            # to pass validation. Expected errors (YAML, OSError) are handled above.
            logger.exception(
                "Unexpected validation error: %s (path=%s, type=%s)",
                str(e),
                contract_path,
                type(e).__name__,
            )
            raise OnexError(
                message=(
                    f"Unexpected error during contract validation: {e!s}. "
                    "This may indicate a bug in the stub validator or an unexpected "
                    "contract format. See logs for full traceback."
                ),
                error_code="STUB_VALIDATION_UNEXPECTED_ERROR",
                context={
                    "contract_path": str(contract_path),
                    "exception_type": type(e).__name__,
                    "exception_message": str(e),
                    "validator": "StubContractValidator",
                    "migration_ticket": "OMN-1104",
                },
            ) from e


# Alias for backwards compatibility
ServiceContractValidator = StubContractValidator

__all__ = ["ServiceContractValidator", "StubContractValidator"]
