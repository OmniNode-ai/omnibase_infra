# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler Validation Error Model for Structured Error Reporting.

This module defines the canonical error model for handler validation failures,
capturing structured information about validation errors from contract parsing,
descriptor validation, security validation, and architecture validation paths.

Part of OMN-1091: Structured Validation & Error Reporting for Handlers.

The model provides:
- Structured error classification with EnumHandlerErrorType
- Handler identification via ModelHandlerIdentifier
- Source attribution via EnumHandlerSourceType
- Location information (file path, line number)
- Actionable remediation hints
- Error chaining for root cause analysis
- Distributed tracing support via correlation_id
- Multiple output formats (logging, CI, structured JSON)

See Also:
    - EnumHandlerErrorType: Error type classification
    - EnumHandlerSourceType: Validation stage identification
    - ModelHandlerIdentifier: Handler identity model
    - ModelExecutionShapeViolationResult: Related validation result model
"""

from __future__ import annotations

from typing import Self
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import (
    EnumHandlerErrorType,
    EnumHandlerSourceType,
    EnumValidationSeverity,
)
from omnibase_infra.models.handlers import ModelHandlerIdentifier


class ModelHandlerValidationError(BaseModel):
    """Canonical error model for handler validation failures.

    This model captures structured information about validation errors
    from contract parsing, descriptor validation, security validation,
    and architecture validation paths. It provides multiple output
    formats for logging, CI integration, and structured reporting.

    Attributes:
        error_type: Type of validation error (contract, descriptor, security, etc.).
        rule_id: Unique identifier for the validation rule (e.g., "CONTRACT-001").
        handler_identity: Handler identification information.
        source_type: Where the error originated (contract, descriptor, static analysis, etc.).
        message: Human-readable error message describing the failure.
        remediation_hint: Actionable fix suggestion for developers.
        file_path: File where error occurred (optional).
        line_number: Line number if applicable (optional, 1-indexed).
        details: Additional context (optional, uses object for generic payloads).
        caused_by: Chained error for root cause analysis (optional).
        correlation_id: Request correlation ID for distributed tracing (optional).
        severity: Severity level ('error' blocks startup, 'warning' is advisory).

    Example:
        >>> # Contract parsing error
        >>> error = ModelHandlerValidationError.from_contract_error(
        ...     rule_id="CONTRACT-001",
        ...     message="Invalid YAML syntax in contract.yaml",
        ...     file_path="nodes/registration/contract.yaml",
        ...     remediation_hint="Check YAML indentation and syntax",
        ...     handler_identity=ModelHandlerIdentifier.from_handler_id("registration-orchestrator"),
        ... )
        >>> error.is_blocking()
        True
        >>> print(error.format_for_ci())
        ::error file=nodes/registration/contract.yaml,line=1::[CONTRACT-001] Invalid YAML syntax...

        >>> # Security violation
        >>> error = ModelHandlerValidationError.from_security_violation(
        ...     rule_id="SECURITY-002",
        ...     message="Handler exposes sensitive method names",
        ...     remediation_hint="Prefix internal methods with underscore",
        ...     handler_identity=ModelHandlerIdentifier.from_node(
        ...         node_path="nodes/auth/node.py",
        ...         handler_type=EnumHandlerType.INFRA_HANDLER,
        ...     ),
        ...     line_number=42,
        ... )

    Note:
        This model is frozen to ensure immutability in error contexts.
        Attempting to mutate any field after instantiation will raise a
        pydantic.ValidationError. The severity field defaults to 'error'
        for blocking validation failures.

    .. versionadded:: 0.6.1
        Created as part of OMN-1091 structured validation and error reporting.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
    )

    error_type: EnumHandlerErrorType = Field(
        ...,
        description="Type of validation error (contract, descriptor, security, etc.)",
    )
    rule_id: str = Field(
        ...,
        min_length=1,
        pattern=r"^[A-Z]+-\d+$",
        description="Unique identifier for the validation rule (e.g., 'CONTRACT-001')",
    )
    handler_identity: ModelHandlerIdentifier = Field(
        ...,
        description="Handler identification information",
    )
    source_type: EnumHandlerSourceType = Field(
        ...,
        description="Where the error originated (contract, descriptor, static analysis, etc.)",
    )
    message: str = Field(
        ...,
        min_length=1,
        description="Human-readable error message describing the failure",
    )
    remediation_hint: str = Field(
        ...,
        min_length=1,
        description="Actionable fix suggestion for developers",
    )
    file_path: str | None = Field(
        default=None,
        description="File where error occurred (optional)",
    )
    line_number: int | None = Field(
        default=None,
        ge=1,
        description="Line number if applicable (optional, 1-indexed)",
    )
    details: dict[str, object] | None = Field(
        default=None,
        description="Additional context (uses object for generic payloads, not Any)",
    )
    caused_by: ModelHandlerValidationError | None = Field(
        default=None,
        description="Chained error for root cause analysis (optional)",
    )
    correlation_id: UUID | None = Field(
        default=None,
        description="Request correlation ID for distributed tracing (optional)",
    )
    severity: EnumValidationSeverity = Field(
        default=EnumValidationSeverity.ERROR,
        description="Severity level: 'error' blocks startup, 'warning' is advisory",
    )

    def is_blocking(self) -> bool:
        """Check if this error should block handler startup or CI.

        Returns:
            True if severity is 'error', False for 'warning'.

        Example:
            >>> error = ModelHandlerValidationError.from_contract_error(...)
            >>> if error.is_blocking():
            ...     from omnibase_infra.errors import ProtocolConfigurationError
            ...     raise ProtocolConfigurationError(
            ...         "Cannot start with validation errors",
            ...         code="HANDLER_VALIDATION_FAILED",
            ...         errors=[error.to_structured_dict()],
            ...     )
        """
        return self.severity == EnumValidationSeverity.ERROR

    def format_for_logging(self) -> str:
        """Format error for structured logging output.

        Produces a multi-line formatted string suitable for logging systems,
        including all relevant error context and remediation hints.

        Returns:
            Formatted string for logging with error details and remediation.

        Example:
            >>> error = ModelHandlerValidationError.from_contract_error(...)
            >>> logger.error(error.format_for_logging())
            Handler Validation Error [CONTRACT-001]
            Error Type: CONTRACT_PARSE_ERROR (EnumHandlerErrorType.CONTRACT_PARSE_ERROR)
            Source: CONTRACT (EnumHandlerSourceType.CONTRACT)
            Handler: registration-orchestrator
            File: nodes/registration/contract.yaml:1
            Message: Invalid YAML syntax in contract.yaml
            Remediation: Check YAML indentation and syntax
        """
        # Include enum names for clarity in logs
        error_type_display = (
            f"{self.error_type.name} (EnumHandlerErrorType.{self.error_type.name})"
        )
        source_type_display = (
            f"{self.source_type.name} (EnumHandlerSourceType.{self.source_type.name})"
        )

        lines = [
            f"Handler Validation Error [{self.rule_id}]",
            f"Severity: {self.severity.upper()}",
            f"Error Type: {error_type_display}",
            f"Source: {source_type_display}",
            f"Handler: {self.handler_identity.format_for_error()}",
        ]

        # Add location if available
        if self.file_path:
            location = self.file_path
            if self.line_number:
                location = f"{location}:{self.line_number}"
            lines.append(f"Location: {location}")

        lines.extend(
            [
                f"Message: {self.message}",
                f"Remediation: {self.remediation_hint}",
            ]
        )

        # Add correlation ID if present
        if self.correlation_id:
            lines.append(f"Correlation ID: {self.correlation_id}")

        # Add chained error if present
        if self.caused_by:
            lines.append(
                f"Caused by: [{self.caused_by.rule_id}] {self.caused_by.message}"
            )

        return "\n".join(lines)

    def format_for_ci(self) -> str:
        """Format error for CI output (GitHub Actions compatible).

        Produces a GitHub Actions annotation format string that will
        be rendered as an inline annotation in pull requests.

        Returns:
            Formatted string in GitHub Actions annotation format.

        Example:
            >>> error = ModelHandlerValidationError.from_contract_error(...)
            >>> print(error.format_for_ci())
            ::error file=nodes/registration/contract.yaml,line=1::[CONTRACT-001] (CONTRACT_PARSE_ERROR) Invalid YAML...
        """
        annotation_type = "error" if self.is_blocking() else "warning"

        # Build file location string
        file_location = f"file={self.file_path or 'unknown'}"
        if self.line_number:
            file_location += f",line={self.line_number}"

        # Build error message with rule ID and error type for clarity
        error_message = f"[{self.rule_id}] ({self.error_type.name}) {self.message}"

        # Include remediation hint in CI output
        full_message = f"{error_message}. Remediation: {self.remediation_hint}"

        return f"::{annotation_type} {file_location}::{full_message}"

    def to_structured_dict(self) -> dict[str, object]:
        """Convert error to structured dictionary for JSON serialization.

        Produces a dictionary representation suitable for JSON APIs,
        logging systems, and error aggregation services.

        Returns:
            Dictionary representation with all error fields.

        Example:
            >>> error = ModelHandlerValidationError.from_contract_error(...)
            >>> json.dumps(error.to_structured_dict())
            '{"error_type": "contract_parse_error", "rule_id": "CONTRACT-001", ...}'
        """
        result: dict[str, object] = {
            "error_type": self.error_type.value,
            "rule_id": self.rule_id,
            "handler_identity": {
                "handler_id": self.handler_identity.handler_id,
                "handler_type": (
                    self.handler_identity.handler_type.value
                    if self.handler_identity.handler_type
                    else None
                ),
                "handler_name": self.handler_identity.handler_name,
                "node_path": self.handler_identity.node_path,
            },
            "source_type": self.source_type.value,
            "message": self.message,
            "remediation_hint": self.remediation_hint,
            "severity": self.severity.value,
        }

        # Add optional fields if present
        if self.file_path:
            result["file_path"] = self.file_path
        if self.line_number:
            result["line_number"] = self.line_number
        if self.details:
            result["details"] = self.details
        if self.correlation_id:
            result["correlation_id"] = str(self.correlation_id)
        if self.caused_by:
            result["caused_by"] = self.caused_by.to_structured_dict()

        return result

    @classmethod
    def from_contract_error(
        cls,
        rule_id: str,
        message: str,
        file_path: str,
        remediation_hint: str,
        handler_identity: ModelHandlerIdentifier,
        line_number: int | None = None,
        details: dict[str, object] | None = None,
        caused_by: ModelHandlerValidationError | None = None,
        correlation_id: UUID | None = None,
        severity: EnumValidationSeverity = EnumValidationSeverity.ERROR,
    ) -> Self:
        """Create error from contract validation failure.

        Factory method for creating errors detected during contract.yaml
        parsing and validation. Automatically sets error_type to
        CONTRACT_PARSE_ERROR or CONTRACT_VALIDATION_ERROR based on context,
        and source_type to CONTRACT.

        Args:
            rule_id: Unique identifier for the validation rule.
            message: Human-readable error message.
            file_path: Path to contract.yaml file.
            remediation_hint: Actionable fix suggestion.
            handler_identity: Handler identification information.
            line_number: Line number in contract file (optional).
            details: Additional context (optional).
            caused_by: Chained error for root cause (optional).
            correlation_id: Request correlation ID (optional).
            severity: Severity level (defaults to 'error').

        Returns:
            ModelHandlerValidationError configured for contract errors.

        Example:
            >>> error = ModelHandlerValidationError.from_contract_error(
            ...     rule_id="CONTRACT-001",
            ...     message="Missing required field 'handler_routing'",
            ...     file_path="nodes/registration/contract.yaml",
            ...     remediation_hint="Add handler_routing section to contract",
            ...     handler_identity=ModelHandlerIdentifier.from_handler_id("registration"),
            ...     line_number=5,
            ... )

        Note:
            Error Type Heuristic - The method automatically classifies errors into
            CONTRACT_PARSE_ERROR or CONTRACT_VALIDATION_ERROR based on message content:

            - CONTRACT_PARSE_ERROR: Triggered when message contains "parse" or "yaml"
              (case-insensitive). These are syntax-level errors preventing the contract
              from being read (e.g., "YAML parse error", "Invalid YAML syntax").

            - CONTRACT_VALIDATION_ERROR: Default for all other contract errors. These
              are semantic errors where the contract is readable but violates schema
              requirements (e.g., "Missing required field", "Invalid node_type").

            This keyword-based heuristic provides automatic error classification without
            requiring callers to explicitly specify the error type. The approach is
            intentionally simple and robust - YAML/parse errors have distinct
            terminology that rarely appears in validation error messages.
        """
        # Determine error type based on message content using keyword heuristic.
        # See docstring Note section above for classification rules.
        error_type = (
            EnumHandlerErrorType.CONTRACT_PARSE_ERROR
            if "parse" in message.lower() or "yaml" in message.lower()
            else EnumHandlerErrorType.CONTRACT_VALIDATION_ERROR
        )

        return cls(
            error_type=error_type,
            rule_id=rule_id,
            handler_identity=handler_identity,
            source_type=EnumHandlerSourceType.CONTRACT,
            message=message,
            remediation_hint=remediation_hint,
            file_path=file_path,
            line_number=line_number,
            details=details,
            caused_by=caused_by,
            correlation_id=correlation_id or uuid4(),
            severity=severity,
        )

    @classmethod
    def from_security_violation(
        cls,
        rule_id: str,
        message: str,
        remediation_hint: str,
        handler_identity: ModelHandlerIdentifier,
        file_path: str | None = None,
        line_number: int | None = None,
        details: dict[str, object] | None = None,
        caused_by: ModelHandlerValidationError | None = None,
        correlation_id: UUID | None = None,
        severity: EnumValidationSeverity = EnumValidationSeverity.ERROR,
    ) -> Self:
        """Create error from security constraint violation.

        Factory method for creating errors detected during security
        validation (introspection restrictions, input validation,
        secret handling). Automatically sets error_type to
        SECURITY_VALIDATION_ERROR and source_type to STATIC_ANALYSIS.

        Args:
            rule_id: Unique identifier for the validation rule.
            message: Human-readable error message.
            remediation_hint: Actionable fix suggestion.
            handler_identity: Handler identification information.
            file_path: Path to source file (optional).
            line_number: Line number in source file (optional).
            details: Additional context (optional).
            caused_by: Chained error for root cause (optional).
            correlation_id: Request correlation ID (optional).
            severity: Severity level (defaults to 'error').

        Returns:
            ModelHandlerValidationError configured for security violations.

        Example:
            >>> error = ModelHandlerValidationError.from_security_violation(
            ...     rule_id="SECURITY-002",
            ...     message="Handler exposes sensitive method 'get_api_key'",
            ...     remediation_hint="Prefix method with underscore: '_get_api_key'",
            ...     handler_identity=ModelHandlerIdentifier.from_node(
            ...         node_path="nodes/auth/node.py",
            ...         handler_type=EnumHandlerType.INFRA_HANDLER,
            ...     ),
            ...     file_path="nodes/auth/handlers/handler_authenticate.py",
            ...     line_number=42,
            ... )
        """
        return cls(
            error_type=EnumHandlerErrorType.SECURITY_VALIDATION_ERROR,
            rule_id=rule_id,
            handler_identity=handler_identity,
            source_type=EnumHandlerSourceType.STATIC_ANALYSIS,
            message=message,
            remediation_hint=remediation_hint,
            file_path=file_path,
            line_number=line_number,
            details=details,
            caused_by=caused_by,
            correlation_id=correlation_id or uuid4(),
            severity=severity,
        )

    @classmethod
    def from_descriptor_error(
        cls,
        rule_id: str,
        message: str,
        remediation_hint: str,
        handler_identity: ModelHandlerIdentifier,
        file_path: str | None = None,
        line_number: int | None = None,
        details: dict[str, object] | None = None,
        caused_by: ModelHandlerValidationError | None = None,
        correlation_id: UUID | None = None,
        severity: EnumValidationSeverity = EnumValidationSeverity.ERROR,
    ) -> Self:
        """Create error from handler descriptor validation failure.

        Factory method for creating errors detected during handler
        descriptor validation (signature, protocol compliance, etc.).
        Automatically sets error_type to DESCRIPTOR_VALIDATION_ERROR
        and source_type to DESCRIPTOR.

        Args:
            rule_id: Unique identifier for the validation rule.
            message: Human-readable error message.
            remediation_hint: Actionable fix suggestion.
            handler_identity: Handler identification information.
            file_path: Path to handler file (optional).
            line_number: Line number in handler file (optional).
            details: Additional context (optional).
            caused_by: Chained error for root cause (optional).
            correlation_id: Request correlation ID (optional).
            severity: Severity level (defaults to 'error').

        Returns:
            ModelHandlerValidationError configured for descriptor errors.

        Example:
            >>> error = ModelHandlerValidationError.from_descriptor_error(
            ...     rule_id="DESCRIPTOR-001",
            ...     message="Handler missing required 'handle' method",
            ...     remediation_hint="Add async def handle(self, event) method",
            ...     handler_identity=ModelHandlerIdentifier.from_node(
            ...         node_path="nodes/compute/node.py",
            ...         handler_type=EnumHandlerType.COMPUTE_HANDLER,
            ...     ),
            ... )
        """
        return cls(
            error_type=EnumHandlerErrorType.DESCRIPTOR_VALIDATION_ERROR,
            rule_id=rule_id,
            handler_identity=handler_identity,
            source_type=EnumHandlerSourceType.DESCRIPTOR,
            message=message,
            remediation_hint=remediation_hint,
            file_path=file_path,
            line_number=line_number,
            details=details,
            caused_by=caused_by,
            correlation_id=correlation_id or uuid4(),
            severity=severity,
        )

    @classmethod
    def from_architecture_error(
        cls,
        rule_id: str,
        message: str,
        remediation_hint: str,
        handler_identity: ModelHandlerIdentifier,
        file_path: str | None = None,
        line_number: int | None = None,
        details: dict[str, object] | None = None,
        caused_by: ModelHandlerValidationError | None = None,
        correlation_id: UUID | None = None,
        severity: EnumValidationSeverity = EnumValidationSeverity.ERROR,
    ) -> Self:
        """Create error from architecture pattern violation.

        Factory method for creating errors detected during architecture
        validation (layering, dependency injection, archetype patterns).
        Automatically sets error_type to ARCHITECTURE_VALIDATION_ERROR
        and source_type to STATIC_ANALYSIS.

        Args:
            rule_id: Unique identifier for the validation rule.
            message: Human-readable error message.
            remediation_hint: Actionable fix suggestion.
            handler_identity: Handler identification information.
            file_path: Path to source file (optional).
            line_number: Line number in source file (optional).
            details: Additional context (optional).
            caused_by: Chained error for root cause (optional).
            correlation_id: Request correlation ID (optional).
            severity: Severity level (defaults to 'error').

        Returns:
            ModelHandlerValidationError configured for architecture violations.

        Example:
            >>> error = ModelHandlerValidationError.from_architecture_error(
            ...     rule_id="ARCH-001",
            ...     message="COMPUTE_HANDLER performs I/O operation",
            ...     remediation_hint="Move I/O logic to INFRA_HANDLER",
            ...     handler_identity=ModelHandlerIdentifier.from_node(
            ...         node_path="nodes/compute/node.py",
            ...         handler_type=EnumHandlerType.COMPUTE_HANDLER,
            ...     ),
            ...     line_number=85,
            ... )
        """
        return cls(
            error_type=EnumHandlerErrorType.ARCHITECTURE_VALIDATION_ERROR,
            rule_id=rule_id,
            handler_identity=handler_identity,
            source_type=EnumHandlerSourceType.STATIC_ANALYSIS,
            message=message,
            remediation_hint=remediation_hint,
            file_path=file_path,
            line_number=line_number,
            details=details,
            caused_by=caused_by,
            correlation_id=correlation_id or uuid4(),
            severity=severity,
        )


__all__ = ["ModelHandlerValidationError"]
