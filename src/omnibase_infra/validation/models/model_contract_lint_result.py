# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Model for contract linting results."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.validation.enums.enum_contract_violation_severity import (
    EnumContractViolationSeverity,
)
from omnibase_infra.validation.models.model_contract_violation import (
    ModelContractViolation,
)

if TYPE_CHECKING:
    from omnibase_infra.models.errors import ModelHandlerValidationError


class ModelContractLintResult(BaseModel):
    """Result of linting one or more contract files."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    is_valid: bool = Field(description="True if no ERROR-level violations found")
    violations: list[ModelContractViolation] = Field(
        default_factory=list,
        description="All violations found during linting",
    )
    files_checked: int = Field(
        default=0, description="Number of contract files checked"
    )
    files_valid: int = Field(
        default=0, description="Number of contract files with no errors"
    )
    files_with_errors: int = Field(
        default=0, description="Number of contract files with errors"
    )

    @property
    def error_count(self) -> int:
        """Count of ERROR-level violations."""
        return sum(
            1
            for v in self.violations
            if v.severity == EnumContractViolationSeverity.ERROR
        )

    @property
    def warning_count(self) -> int:
        """Count of WARNING-level violations."""
        return sum(
            1
            for v in self.violations
            if v.severity == EnumContractViolationSeverity.WARNING
        )

    def __str__(self) -> str:
        """Format result summary as human-readable string."""
        status = "PASS" if self.is_valid else "FAIL"
        summary = f"Contract Lint: {status} ({self.files_checked} files, {self.error_count} errors, {self.warning_count} warnings)"
        return summary

    def to_handler_errors(
        self,
        correlation_id: UUID | None = None,
    ) -> list[ModelHandlerValidationError]:
        """Convert all violations to structured handler validation errors.

        Transforms all ModelContractViolation instances into ModelHandlerValidationError
        instances with appropriate rule IDs, handler identities, and remediation hints.
        This enables integration with the structured validation and error reporting system.

        Args:
            correlation_id: Optional correlation ID to apply to all errors.

        Returns:
            List of ModelHandlerValidationError instances.

        Example:
            >>> linter = ContractLinter()
            >>> result = linter.lint_file(Path("contract.yaml"))
            >>> errors = result.to_handler_errors()
            >>> for error in errors:
            ...     logger.error(error.format_for_logging())
        """
        # Late import to avoid circular dependency
        from omnibase_infra.validation.linter_contract import (
            convert_violation_to_handler_error,
        )

        return [
            convert_violation_to_handler_error(violation, correlation_id)
            for violation in self.violations
        ]
