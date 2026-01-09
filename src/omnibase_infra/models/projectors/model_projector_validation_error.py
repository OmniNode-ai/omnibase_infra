# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Projector validation error model.

Provides structured error information for projector contract parsing
and validation failures.

Part of OMN-1168: ProjectorPluginLoader contract discovery loading.
"""

from __future__ import annotations


class ModelProjectorValidationError:
    """Structured error for projector validation failures.

    Captures detailed information about contract parsing and validation
    errors to support graceful mode error collection.

    Attributes:
        error_type: Category of error (parse, validation, io, size).
        contract_path: Path to the failing contract file.
        message: Human-readable error message.
        remediation_hint: Suggested fix for the error.
    """

    def __init__(
        self,
        error_type: str,
        contract_path: str,
        message: str,
        remediation_hint: str | None = None,
    ) -> None:
        """Initialize validation error.

        Args:
            error_type: Category of error (parse, validation, io, size).
            contract_path: Path to the failing contract file.
            message: Human-readable error message.
            remediation_hint: Suggested fix for the error.
        """
        self.error_type = error_type
        self.contract_path = contract_path
        self.message = message
        self.remediation_hint = remediation_hint

    def __repr__(self) -> str:
        """Return string representation."""
        return f"ModelProjectorValidationError(type={self.error_type!r}, path={self.contract_path!r})"


__all__ = ["ModelProjectorValidationError"]
