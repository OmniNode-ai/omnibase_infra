# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""LocalHandler Violation Model.

Defines the result structure for detected LocalHandler import violations.
Used by the LocalHandler validator to report policy breaches with
full context for debugging and CI gate integration.

LocalHandler is a test-only handler that must NEVER be imported in production
code (src/omnibase_infra/). This validator enforces that policy.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ModelLocalHandlerViolation(BaseModel):
    """Result of a LocalHandler import violation detection.

    Contains full context about a detected violation for debugging,
    reporting, and CI gate integration. Each violation includes the
    source location, the violating import line, and suggested fix.

    Attributes:
        file_path: Path to the source file containing the violation.
        line_number: Line number where the violation was detected (1-indexed).
        import_line: The actual import statement that violates the policy.

    Example:
        >>> violation = ModelLocalHandlerViolation(
        ...     file_path=Path("/src/omnibase_infra/handlers/bad_handler.py"),
        ...     line_number=5,
        ...     import_line="from omnibase_core.handlers import LocalHandler",
        ... )
        >>> print(violation.format_for_ci())
        ::error file=.../bad_handler.py,line=5::LocalHandler import forbidden...

    Note:
        All LocalHandler violations are blocking (error severity).
        LocalHandler is only permitted in test code (tests/ directory).
    """

    file_path: Path = Field(
        ...,
        description="Path to the source file containing the violation",
    )
    line_number: int = Field(
        ...,
        ge=1,
        description="Line number where the violation was detected (1-indexed)",
    )
    import_line: str = Field(
        ...,
        min_length=1,
        description="The actual import statement that violates the policy",
    )

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
    )

    def format_for_ci(self) -> str:
        """Format violation for CI output (GitHub Actions compatible).

        Returns:
            Formatted string in GitHub Actions annotation format.

        Example:
            ::error file=src/handler.py,line=5::LocalHandler import forbidden...
        """
        return (
            f"::error file={self.file_path},line={self.line_number}::"
            f"LocalHandler import forbidden in production code: {self.import_line}"
        )

    def format_human_readable(self) -> str:
        """Format violation for human-readable console output.

        Returns:
            Formatted string with file location and violation details.

        Example:
            src/omnibase_infra/handlers/bad_handler.py:5
              Import: from omnibase_core.handlers import LocalHandler
              Fix: Remove LocalHandler - use only in tests/
        """
        return (
            f"{self.file_path}:{self.line_number}\n"
            f"  Import: {self.import_line}\n"
            f"  Fix: Remove LocalHandler import - LocalHandler is for test-only use"
        )


__all__ = ["ModelLocalHandlerViolation"]
