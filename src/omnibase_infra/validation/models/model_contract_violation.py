# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Model for a single contract validation violation."""

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.validation.enums.enum_contract_violation_severity import (
    EnumContractViolationSeverity,
)


class ModelContractViolation(BaseModel):
    """A single contract validation violation."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    file_path: str = Field(description="Path to the contract file")
    field_path: str = Field(
        description="JSON path to the violating field (e.g., 'input_model.module')"
    )
    message: str = Field(description="Human-readable violation description")
    severity: EnumContractViolationSeverity = Field(
        default=EnumContractViolationSeverity.ERROR,
        description="Violation severity level",
    )
    suggestion: str | None = Field(
        default=None,
        description="Suggested fix for the violation",
    )

    def __str__(self) -> str:
        """Format violation as human-readable string."""
        prefix = f"[{self.severity.value.upper()}]"
        location = f"{self.file_path}:{self.field_path}"
        msg = f"{prefix} {location}: {self.message}"
        if self.suggestion:
            msg += f" (suggestion: {self.suggestion})"
        return msg
