# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Reporting mixin for Any type detection.

This mixin provides methods for tracking and reporting Any type violations.
"""

from __future__ import annotations

from pathlib import Path

from omnibase_infra.enums import EnumAnyTypeViolation, EnumValidationSeverity
from omnibase_infra.models.validation.model_any_type_violation import (
    ModelAnyTypeViolation,
)


class MixinAnyTypeReporting:
    """Mixin providing reporting methods for Any type detection.

    This mixin extracts reporting logic from AnyTypeDetector to reduce
    method count while maintaining functionality.

    Required attributes (from main class):
        filepath: str - Path to the file being analyzed.
        source_lines: list[str] - Source code lines for snippet extraction.
        violations: list[ModelAnyTypeViolation] - List of detected violations.
        allowed_lines: set[int] - Set of line numbers exempted.

    Methods:
        _get_code_snippet: Extract code snippet for a line.
        _add_violation: Add a violation to the list.
    """

    # These attributes are expected to exist on the main class
    filepath: str
    source_lines: list[str]
    violations: list[ModelAnyTypeViolation]
    allowed_lines: set[int]

    def _get_code_snippet(self, line_number: int, max_length: int = 80) -> str:
        """Get a code snippet for a violation, with smart truncation.

        Truncates at word/syntax boundaries to avoid cutting mid-identifier.

        Args:
            line_number: 1-based line number to extract.
            max_length: Maximum length of the returned snippet.

        Returns:
            The code snippet, truncated smartly if necessary.
        """
        if not (0 < line_number <= len(self.source_lines)):
            return ""

        snippet = self.source_lines[line_number - 1].strip()
        if len(snippet) <= max_length:
            return snippet

        # Find last space or punctuation before max_length
        truncate_at = snippet.rfind(" ", 0, max_length - 3)
        if truncate_at == -1:
            # No space found, try common delimiters
            for delim in [",", ":", "(", "[", "="]:
                pos = snippet.rfind(delim, 0, max_length - 3)
                truncate_at = max(pos, truncate_at)
        if truncate_at > 0:
            return snippet[:truncate_at] + "..."
        return snippet[: max_length - 3] + "..."

    def _add_violation(
        self,
        line_number: int,
        column: int,
        violation_type: EnumAnyTypeViolation,
        context_name: str = "",
    ) -> None:
        """Add a violation to the list.

        Args:
            line_number: Line number where violation was detected.
            column: Column offset where Any appears.
            violation_type: The type of violation.
            context_name: Context identifier (function/variable name).
        """
        if line_number in self.allowed_lines:
            return

        # Get code snippet with smart truncation
        snippet = self._get_code_snippet(line_number)

        # Get suggestion from enum
        suggestion = violation_type.suggestion

        self.violations.append(
            ModelAnyTypeViolation(
                file_path=Path(self.filepath),
                line_number=line_number,
                column=column,
                violation_type=violation_type,
                code_snippet=snippet,
                suggestion=suggestion,
                severity=EnumValidationSeverity.ERROR,
                context_name=context_name,
            )
        )
