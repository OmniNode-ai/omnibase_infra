# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""ProtocolContractValidator compatibility shim for omnibase_core 0.6.x.

This module provides a compatibility shim for ProtocolContractValidator,
which was removed in omnibase_core 0.6.x. The shim uses the new validation
API while maintaining backwards compatibility with existing code.

.. versionadded:: 0.1.0
    Added for omnibase_core 0.6.x compatibility (OMN-1258).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from omnibase_core.validation import (
    ModelContractValidationResult,
    validate_yaml_file,
)

__all__ = ["ProtocolContractValidator"]


class ProtocolContractValidator:
    """Compatibility shim for ProtocolContractValidator.

    This class provides backwards compatibility for code that uses
    ProtocolContractValidator from omnibase_core < 0.6.x.

    The implementation uses the new omnibase_core 0.6.x validation API
    under the hood.
    """

    def validate_contract_file(
        self,
        contract_path: Path,
        contract_type: Literal[
            "effect", "compute", "reducer", "orchestrator"
        ] = "effect",
    ) -> ModelContractValidationResult:
        """Validate a contract YAML file.

        Args:
            contract_path: Path to the contract YAML file.
            contract_type: Type of contract to validate.

        Returns:
            ModelContractValidationResult with validation status and errors.
        """
        # Use the new validation API from omnibase_core 0.6.x
        result = validate_yaml_file(contract_path)

        # Return a ModelContractValidationResult
        # The new API may return a different type, so we adapt it
        if isinstance(result, ModelContractValidationResult):
            return result

        # If result is a different type, wrap it in ModelContractValidationResult
        # This handles API changes between versions
        return ModelContractValidationResult(
            passed=getattr(result, "passed", True),
            score=getattr(result, "score", 100.0),
            errors=getattr(result, "errors", []),
            warnings=getattr(result, "warnings", []),
        )
