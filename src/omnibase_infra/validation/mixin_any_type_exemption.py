# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Exemption mixin for Any type detection.

This mixin provides methods for managing exemptions via decorators,
comments, and file-level NOTE patterns.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

# Decorator names that allow Any type usage
_ALLOW_DECORATORS: frozenset[str] = frozenset({"allow_any", "allow_any_type"})

# Comment patterns for exemptions
_NOTE_PATTERN = "NOTE:"

# Number of lines to look back for NOTE comments.
# 5 lines is sufficient because NOTE comments typically appear:
# - Immediately above the field (1-2 lines)
# - After blank lines and decorators (3-4 lines)
# - With docstring gap (5 lines max)
# Larger values would risk matching unrelated NOTE comments.
_NOTE_LOOKBACK_LINES = 5


class MixinAnyTypeExemption:
    """Mixin providing exemption management for Any type detection.

    This mixin extracts exemption logic from AnyTypeDetector to reduce
    method count while maintaining functionality.

    Required attributes (from main class):
        source_lines: list[str] - Source code lines for context checking.
        allowed_lines: set[int] - Set of line numbers exempted.
        has_file_level_note: bool - Whether file has NOTE near Any import.

    Methods:
        _check_file_level_note: Check for file-level NOTE comment.
        _has_allow_decorator: Check for @allow_any decorator.
        _allow_all_lines_in_node: Exempt all lines in an AST node.
        _is_field_with_note: Check for Field() with NOTE comment.
    """

    # These attributes are expected to exist on the main class
    source_lines: list[str]
    allowed_lines: set[int]
    has_file_level_note: bool

    def _check_file_level_note(self) -> bool:
        """Check if file has a NOTE comment near the Any import.

        A file-level NOTE comment (within 5 lines of 'from typing import Any'
        or 'import typing') exempts all Pydantic Field() usages with Any in
        that file. This supports the common pattern of documenting the
        workaround once at the import rather than at each field.

        Returns:
            True if file has a valid file-level NOTE comment.

        Note:
            Only scans first 100 lines for performance. Imports should
            always appear near the top of Python files per PEP 8.
        """
        # Only scan first 100 lines - imports must be at top per PEP 8
        max_lines = min(100, len(self.source_lines))
        for i, line in enumerate(self.source_lines[:max_lines]):
            # Look for Any import
            if "from typing import" in line and "Any" in line:
                # Check surrounding lines (5 before and 5 after)
                start = max(0, i - 5)
                end = min(len(self.source_lines), i + 6)
                context = "\n".join(self.source_lines[start:end])
                if _NOTE_PATTERN in context and "Any" in context:
                    return True
            elif "import typing" in line:
                # Also check for 'import typing' style
                start = max(0, i - 5)
                end = min(len(self.source_lines), i + 6)
                context = "\n".join(self.source_lines[start:end])
                if _NOTE_PATTERN in context and "Any" in context:
                    return True
        return False

    def _has_allow_decorator(self, decorator_list: list[ast.expr]) -> bool:
        """Check if any decorator allows Any type usage.

        Args:
            decorator_list: List of decorator AST expressions.

        Returns:
            True if an allow decorator is present.
        """
        for decorator in decorator_list:
            # Direct decorator: @allow_any
            if isinstance(decorator, ast.Name) and decorator.id in _ALLOW_DECORATORS:
                return True
            # Call decorator: @allow_any("reason")
            if isinstance(decorator, ast.Call):
                if (
                    isinstance(decorator.func, ast.Name)
                    and decorator.func.id in _ALLOW_DECORATORS
                ):
                    return True
        return False

    def _allow_all_lines_in_node(self, node: ast.AST) -> None:
        """Allow Any usage in all lines within a node.

        Args:
            node: The AST node whose lines should be exempted.
        """
        for stmt in ast.walk(node):
            if hasattr(stmt, "lineno"):
                self.allowed_lines.add(stmt.lineno)

    def _is_field_with_note(self, node: ast.AnnAssign) -> bool:
        """Check if an annotated assignment is a Field() with NOTE comment.

        The NOTE comment must appear:
        - At file level (near the Any import), OR
        - On the same line as the field (inline comment), OR
        - In the contiguous comment block immediately preceding the field

        File-level NOTE comments (within 5 lines of Any import) exempt all
        Field() usages in that file, supporting the common pattern of
        documenting the workaround once at the import.

        Args:
            node: The annotated assignment node.

        Returns:
            True if this is a Field() call with a proper NOTE comment.

        Note:
            This method requires _is_field_call and _contains_any from
            MixinAnyTypeClassification and the main class respectively.
        """
        # Check if value is a Field() call
        # _is_field_call is provided by MixinAnyTypeClassification
        if node.value is None or not self._is_field_call(node.value):  # type: ignore[attr-defined]
            return False

        # Check if annotation contains Any
        # _contains_any is provided by the main class
        if not self._contains_any(node.annotation):  # type: ignore[attr-defined]
            return False

        # File-level NOTE comment exempts all Field() usages
        if self.has_file_level_note:
            return True

        line_num = node.lineno

        # First, check for inline NOTE comment on the same line
        if 0 < line_num <= len(self.source_lines):
            current_line = self.source_lines[line_num - 1]
            if _NOTE_PATTERN in current_line and "Any" in current_line:
                return True

        # Look for NOTE comment in contiguous comment block immediately above
        # Stop as soon as we hit a non-comment, non-blank line
        for i in range(line_num - 2, max(-1, line_num - _NOTE_LOOKBACK_LINES - 2), -1):
            if i < 0 or i >= len(self.source_lines):
                break

            line_content = self.source_lines[i].strip()

            # Skip blank lines
            if not line_content:
                continue

            # Check if this is a comment line
            if line_content.startswith("#"):
                if _NOTE_PATTERN in line_content and "Any" in line_content:
                    return True
                # Continue looking in the comment block
                continue

            # Non-comment, non-blank line - stop looking
            break

        return False
