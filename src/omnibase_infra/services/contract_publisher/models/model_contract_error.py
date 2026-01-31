# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Contract Error Model.

Non-fatal error for contract parsing/validation failures.

.. versionadded:: 0.3.0
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelContractError(BaseModel):
    """Non-fatal contract-level error.

    Represents errors that occur during contract parsing, validation, or
    conflict detection. These errors are non-fatal - processing continues
    and the error is collected in the result.

    Error Types:
        yaml_parse: YAML syntax error
        schema_validation: Contract doesn't match ModelHandlerContract schema
        missing_field: Required field missing (handler_id, etc.)
        invalid_handler_class: handler_class not fully qualified Python path
        duplicate_conflict: Same handler_id with different content hash

    Attributes:
        contract_path: Path or identifier of the contract that failed
        handler_id: Handler ID if extracted, None if parsing failed early
        error_type: Category of the error
        message: Human-readable error description

    Example:
        >>> error = ModelContractError(
        ...     contract_path="/app/contracts/handlers/foo/contract.yaml",
        ...     handler_id=None,
        ...     error_type="yaml_parse",
        ...     message="Invalid YAML syntax at line 10",
        ... )

    .. versionadded:: 0.3.0
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    contract_path: str = Field(
        description="Path or identifier of the contract that failed"
    )
    handler_id: str | None = Field(
        default=None,
        description="Handler ID if extracted, None if parsing failed early",
    )
    error_type: Literal[
        "yaml_parse",
        "schema_validation",
        "missing_field",
        "invalid_handler_class",
        "duplicate_conflict",
    ] = Field(description="Category of the error")
    message: str = Field(description="Human-readable error description")


__all__ = ["ModelContractError"]
