# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Validation error aggregation and reporting for startup.

Collects and formats handler validation errors for clear presentation
during application startup.

This module provides the ValidationAggregator class for collecting handler
validation errors from multiple validation paths (contract, security,
architecture) and formatting them for console output, CI integration,
and structured reporting.

Part of OMN-1091: Structured Validation & Error Reporting for Handlers.

Example:
    >>> aggregator = ValidationAggregator()
    >>> aggregator.add_error(error1)
    >>> aggregator.add_errors([error2, error3])
    >>> if aggregator.has_blocking_errors:
    ...     print(aggregator.format_for_console())
    ...     aggregator.raise_if_blocking()

See Also:
    - ModelHandlerValidationError: Structured error model
    - EnumHandlerErrorType: Error type classification
    - EnumHandlerSourceType: Validation stage identification
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from omnibase_infra.enums import EnumHandlerErrorType, EnumHandlerSourceType
from omnibase_infra.models.errors import ModelHandlerValidationError


class ValidationAggregator:
    """Aggregates handler validation errors for startup reporting.

    Collects errors from multiple validation paths (contract, security,
    architecture) and provides formatted output for logging and CI.

    This class is the central point for collecting all handler validation
    errors during application startup. It provides multiple output formats
    for different contexts (console, CI, structured logging) and can raise
    exceptions when blocking errors are present.

    Attributes:
        _errors: Internal list of collected validation errors.

    Example:
        >>> # Collect errors from multiple validators
        >>> aggregator = ValidationAggregator()
        >>>
        >>> # Add contract errors
        >>> contract_errors = validate_contracts(node_directory)
        >>> aggregator.add_errors(contract_errors)
        >>>
        >>> # Add security errors
        >>> security_errors = validate_security(node_directory)
        >>> aggregator.add_errors(security_errors)
        >>>
        >>> # Check for blocking errors
        >>> if aggregator.has_blocking_errors:
        ...     logger.error(aggregator.format_for_console())
        ...     aggregator.raise_if_blocking()
        >>> else:
        ...     logger.warning(aggregator.format_for_console())
        >>>
        >>> # Export for CI
        >>> print(aggregator.format_for_ci())

    .. versionadded:: 0.6.1
        Created as part of OMN-1091 structured validation and error reporting.
    """

    def __init__(self) -> None:
        """Initialize an empty validation aggregator."""
        self._errors: list[ModelHandlerValidationError] = []

    def add_error(self, error: ModelHandlerValidationError) -> None:
        """Add a validation error to the collection.

        Args:
            error: The validation error to add.

        Example:
            >>> aggregator = ValidationAggregator()
            >>> error = ModelHandlerValidationError.from_contract_error(...)
            >>> aggregator.add_error(error)
        """
        self._errors.append(error)

    def add_errors(self, errors: Sequence[ModelHandlerValidationError]) -> None:
        """Add multiple validation errors.

        Args:
            errors: Sequence of validation errors to add.

        Example:
            >>> aggregator = ValidationAggregator()
            >>> errors = validate_all_handlers(directory)
            >>> aggregator.add_errors(errors)
        """
        self._errors.extend(errors)

    @property
    def has_errors(self) -> bool:
        """Check if any errors have been collected.

        Returns:
            True if at least one error exists, False otherwise.

        Example:
            >>> aggregator = ValidationAggregator()
            >>> aggregator.has_errors
            False
            >>> aggregator.add_error(error)
            >>> aggregator.has_errors
            True
        """
        return len(self._errors) > 0

    @property
    def has_blocking_errors(self) -> bool:
        """Check if any blocking errors exist.

        Blocking errors have severity='error' and should prevent startup.

        Returns:
            True if at least one blocking error exists, False otherwise.

        Example:
            >>> aggregator = ValidationAggregator()
            >>> aggregator.add_error(warning_error)
            >>> aggregator.has_blocking_errors
            False
            >>> aggregator.add_error(blocking_error)
            >>> aggregator.has_blocking_errors
            True
        """
        return any(e.is_blocking() for e in self._errors)

    @property
    def error_count(self) -> int:
        """Get total error count.

        Returns:
            Total number of errors (both blocking and non-blocking).

        Example:
            >>> aggregator = ValidationAggregator()
            >>> aggregator.add_errors([error1, error2, error3])
            >>> aggregator.error_count
            3
        """
        return len(self._errors)

    @property
    def blocking_error_count(self) -> int:
        """Get blocking error count.

        Returns:
            Number of blocking errors (severity='error').

        Example:
            >>> aggregator = ValidationAggregator()
            >>> aggregator.add_errors([blocking_error, warning_error])
            >>> aggregator.blocking_error_count
            1
        """
        return sum(1 for e in self._errors if e.is_blocking())

    def get_errors_by_type(
        self,
    ) -> dict[EnumHandlerErrorType, list[ModelHandlerValidationError]]:
        """Group errors by error type.

        Returns:
            Dictionary mapping error types to lists of errors.

        Example:
            >>> aggregator = ValidationAggregator()
            >>> aggregator.add_errors(errors)
            >>> by_type = aggregator.get_errors_by_type()
            >>> contract_errors = by_type[EnumHandlerErrorType.CONTRACT_PARSE_ERROR]
        """
        result: dict[EnumHandlerErrorType, list[ModelHandlerValidationError]] = (
            defaultdict(list)
        )
        for error in self._errors:
            result[error.error_type].append(error)
        return dict(result)

    def get_errors_by_source(
        self,
    ) -> dict[EnumHandlerSourceType, list[ModelHandlerValidationError]]:
        """Group errors by source type.

        Returns:
            Dictionary mapping source types to lists of errors.

        Example:
            >>> aggregator = ValidationAggregator()
            >>> aggregator.add_errors(errors)
            >>> by_source = aggregator.get_errors_by_source()
            >>> contract_errors = by_source[EnumHandlerSourceType.CONTRACT]
        """
        result: dict[EnumHandlerSourceType, list[ModelHandlerValidationError]] = (
            defaultdict(list)
        )
        for error in self._errors:
            result[error.source_type].append(error)
        return dict(result)

    def format_for_console(self) -> str:
        """Format errors for console output with colors/symbols.

        Produces a human-readable multi-line output suitable for console
        display, with section headers, error grouping, and clear status
        indicators.

        Returns:
            Formatted string for console output.

        Example:
            >>> aggregator = ValidationAggregator()
            >>> aggregator.add_errors(errors)
            >>> print(aggregator.format_for_console())
            ============================================================
            HANDLER VALIDATION ERRORS (3 total, 2 blocking)
            ============================================================

            [CONTRACT] (EnumHandlerSourceType.CONTRACT)
            ----------------------------------------
              ERROR [CONTRACT-001] (CONTRACT_PARSE_ERROR) Invalid YAML syntax
                Handler: registration-orchestrator
                Location: nodes/registration/contract.yaml:5
                Remediation: Check YAML indentation
            ...
        """
        if not self._errors:
            return "No validation errors found"

        warning_count = self.error_count - self.blocking_error_count
        lines = [
            "",
            "=" * 70,
            f"HANDLER VALIDATION ERRORS ({self.error_count} total: "
            f"{self.blocking_error_count} blocking, {warning_count} warnings)",
            "=" * 70,
        ]

        # Group by source type
        by_source = self.get_errors_by_source()
        for source_type, errors in sorted(by_source.items(), key=lambda x: x[0].value):
            lines.append(
                f"\n[{source_type.name}] (EnumHandlerSourceType.{source_type.name})"
            )
            lines.append("-" * 50)
            for error in errors:
                severity_label = "ERROR" if error.is_blocking() else "WARNING"
                # Include error type name for clarity
                lines.append(
                    f"  {severity_label} [{error.rule_id}] "
                    f"({error.error_type.name}) {error.message}"
                )
                # Include handler identity
                lines.append(
                    f"    Handler: {error.handler_identity.format_for_error()}"
                )
                # Include file location
                if error.file_path:
                    loc = f"{error.file_path}"
                    if error.line_number:
                        loc += f":{error.line_number}"
                    lines.append(f"    Location: {loc}")
                # Include remediation hint
                lines.append(f"    Remediation: {error.remediation_hint}")
                lines.append("")

        lines.append("=" * 70)
        if self.has_blocking_errors:
            lines.append(
                f"BLOCKED: {self.blocking_error_count} blocking error(s) must be "
                f"resolved before startup can proceed."
            )
        else:
            lines.append(
                f"PROCEEDING WITH WARNINGS: {warning_count} warning(s) detected. "
                f"Consider addressing these for improved code quality."
            )
        lines.append("=" * 70)

        return "\n".join(lines)

    def format_for_ci(self) -> str:
        """Format errors for CI output (GitHub Actions annotations).

        Produces GitHub Actions-compatible annotation format that will
        render as inline annotations in pull requests.

        Returns:
            Formatted string for CI output (one annotation per line).

        Example:
            >>> aggregator = ValidationAggregator()
            >>> aggregator.add_errors(errors)
            >>> print(aggregator.format_for_ci())
            ::error file=nodes/registration/contract.yaml,line=5::[CONTRACT-001] Invalid...
            ::warning file=nodes/compute/node.py,line=42::[ARCH-001] Missing...
        """
        if not self._errors:
            return ""

        lines = []
        for error in self._errors:
            lines.append(error.format_for_ci())
        return "\n".join(lines)

    def format_summary(self) -> str:
        """Format a short summary line.

        Returns:
            One-line summary of validation status.

        Example:
            >>> aggregator = ValidationAggregator()
            >>> aggregator.format_summary()
            'Handler Validation: PASSED (0 errors)'
            >>> aggregator.add_errors(errors)
            >>> aggregator.format_summary()
            'Handler Validation: FAILED (3 total: 2 blocking, 1 warning)'
        """
        if not self._errors:
            return "Handler Validation: PASSED (0 errors)"

        warning_count = self.error_count - self.blocking_error_count
        status = "FAILED" if self.has_blocking_errors else "PASSED WITH WARNINGS"
        return (
            f"Handler Validation: {status} "
            f"({self.error_count} total: {self.blocking_error_count} blocking, "
            f"{warning_count} warning{'s' if warning_count != 1 else ''})"
        )

    def raise_if_blocking(self) -> None:
        """Raise exception if blocking errors exist.

        Raises:
            ProtocolConfigurationError: If any blocking errors are present.

        Example:
            >>> aggregator = ValidationAggregator()
            >>> aggregator.add_errors(errors)
            >>> aggregator.raise_if_blocking()  # Raises if blocking errors exist
            Traceback (most recent call last):
                ...
            ProtocolConfigurationError: Handler validation failed: 2 blocking errors
        """
        if self.has_blocking_errors:
            from uuid import uuid4

            from omnibase_infra.enums import EnumInfraTransportType
            from omnibase_infra.errors import ProtocolConfigurationError
            from omnibase_infra.models.errors import ModelInfraErrorContext

            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="startup_validation",
                correlation_id=uuid4(),
            )
            raise ProtocolConfigurationError(
                f"Handler validation failed: {self.blocking_error_count} blocking errors\n{self.format_for_console()}",
                context=context,
            )

    def clear(self) -> None:
        """Clear all collected errors.

        Resets the aggregator to an empty state.

        Example:
            >>> aggregator = ValidationAggregator()
            >>> aggregator.add_errors(errors)
            >>> aggregator.error_count
            3
            >>> aggregator.clear()
            >>> aggregator.error_count
            0
        """
        self._errors.clear()


__all__ = ["ValidationAggregator"]
