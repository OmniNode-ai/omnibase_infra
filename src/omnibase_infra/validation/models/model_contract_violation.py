# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Model for a single contract validation violation.

Part of OMN-517: Comprehensive contract schema validation with line-number
error reporting and actionable suggestions.
"""

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.validation.enums.enum_contract_violation_severity import (
    EnumContractViolationSeverity,
)


class ModelContractViolation(BaseModel):
    """A single contract validation violation.

    Provides structured error information including optional line numbers
    for YAML source position tracking, and "did you mean?" suggestions
    for common typos.

    Example:
        >>> violation = ModelContractViolation(
        ...     file_path="nodes/my_node/contract.yaml",
        ...     field_path="node_typ",
        ...     message="Invalid field 'node_typ' at line 23",
        ...     line_number=23,
        ...     suggestion="Did you mean 'node_type'?",
        ... )
        >>> print(violation)
        [ERROR] nodes/my_node/contract.yaml:23:node_typ: Invalid field ...
    """

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
    line_number: int | None = Field(
        default=None,
        description="Line number in the YAML file where the violation occurs (1-based)",
    )

    def __str__(self) -> str:
        """Format violation as human-readable string.

        Format: [SEVERITY] file_path:line:field_path: message (suggestion: ...)

        Line number is included when available, omitted otherwise.
        """
        prefix = f"[{self.severity.value.upper()}]"
        if self.line_number is not None:
            location = f"{self.file_path}:{self.line_number}:{self.field_path}"
        else:
            location = f"{self.file_path}:{self.field_path}"
        msg = f"{prefix} {location}: {self.message}"
        if self.suggestion:
            msg += f" (suggestion: {self.suggestion})"
        return msg
