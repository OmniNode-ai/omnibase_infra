# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Exemption mixin for Any type detection.

This mixin provides methods for managing exemptions via decorators,
comments, and file-level NOTE patterns.
"""

from __future__ import annotations

import ast

# Decorator names that allow Any type usage
_ALLOW_DECORATORS: frozenset[str] = frozenset({"allow_any", "allow_any_type"})

# Comment patterns for exemptions
_NOTE_PATTERN = "NOTE:"

# Specific patterns for file-level Any exemption notes
# These patterns must appear in a comment near the Any import to indicate
# intentional Any usage. Generic "NOTE:" comments don't qualify.
# The pattern is: "# NOTE: Any <keyword>" where keyword indicates intent
_ANY_EXEMPTION_KEYWORDS: frozenset[str] = frozenset(
    {
        "required",  # "NOTE: Any required for..."
        "needed",  # "NOTE: Any needed for..."
        "necessary",  # "NOTE: Any necessary for..."
        "used",  # "NOTE: Any used for..."
        "workaround",  # "NOTE: Any workaround for..."
        # "type" intentionally omitted - too generic, matches "the Any type"
    }
)

# Number of lines to look back for NOTE comments.
# 5 lines is sufficient because NOTE comments typically appear:
# - Immediately above the field (1-2 lines)
# - After blank lines and decorators (3-4 lines)
# - With docstring gap (5 lines max)
# Larger values would risk matching unrelated NOTE comments.
_NOTE_LOOKBACK_LINES = 5


def _extract_comment_portion(line: str) -> str | None:
    """Extract the comment portion of a line, ignoring # in string literals.

    This function finds the first # that is NOT inside a string literal
    (single-quoted, double-quoted, or triple-quoted).

    Args:
        line: A single line of source code.

    Returns:
        The comment portion (including #) if found, None otherwise.

    Examples:
        >>> _extract_comment_portion("x = 1  # comment")
        '# comment'
        >>> _extract_comment_portion('x = "# not comment"  # real comment')
        '# real comment'
        >>> _extract_comment_portion('x = "# not comment"')
        None
    """
    in_single_quote = False
    in_double_quote = False
    i = 0
    n = len(line)

    while i < n:
        char = line[i]

        # Check for triple quotes (must check before single quotes)
        if i + 2 < n:
            triple = line[i : i + 3]
            if triple == '"""' and not in_single_quote:
                # Triple double quote - skip to end or toggle
                if in_double_quote:
                    in_double_quote = False
                else:
                    in_double_quote = True
                i += 3
                continue
            if triple == "'''" and not in_double_quote:
                # Triple single quote - skip to end or toggle
                if in_single_quote:
                    in_single_quote = False
                else:
                    in_single_quote = True
                i += 3
                continue

        # Check for escaped quotes
        if char == "\\" and i + 1 < n:
            # Skip escaped character
            i += 2
            continue

        # Check for single quote
        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
            i += 1
            continue

        # Check for double quote
        if char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
            i += 1
            continue

        # Check for comment (only if not in string)
        if char == "#" and not in_single_quote and not in_double_quote:
            return line[i:]

        i += 1

    return None


def _is_any_exemption_note(text: str) -> bool:
    """Check if text contains a valid Any type exemption note.

    A valid exemption note must have:
    1. A comment marker (#)
    2. "NOTE:" immediately after the comment marker (with optional whitespace)
    3. "Any" after NOTE:
    4. One of the exemption keywords (required, needed, necessary, etc.) after "Any"

    This prevents generic NOTE comments from accidentally matching and ensures
    NOTE: is at the start of a comment, not buried in the middle.

    Valid examples:
        - "# NOTE: Any required for Pydantic discriminated union"
        - "code  # NOTE: Any type needed for dynamic payloads"
        - "# NOTE: Any workaround - see ADR-123"

    Invalid examples:
        - "# Some text NOTE: Any required" (NOTE not at comment start)
        - "NOTE: Any required" (no comment marker)
        - "# NOTE: This handles Any input" (no keyword after Any)
        - "# NOTE: Any questions?" (not an exemption)
        - "x = 'NOTE: Any required'" (NOTE in string, not comment)

    Args:
        text: Text to check (can be single line or multi-line context).

    Returns:
        True if text contains a valid Any exemption note at comment start.
    """
    # Must contain a comment marker
    comment_idx = text.find("#")
    if comment_idx == -1:
        return False

    # Extract the comment portion (after #) and strip leading whitespace
    comment = text[comment_idx + 1 :].lstrip()
    comment_lower = comment.lower()

    # NOTE: must be at the start of the comment (after # and whitespace)
    if not comment_lower.startswith("note:"):
        return False

    # Look for "any" AFTER the "note:" prefix
    text_after_note = comment_lower[5:]  # Start after "note:"
    any_pos = text_after_note.find("any")
    if any_pos == -1:
        return False

    # Check for exemption keyword after "any" (within reasonable distance)
    # Look at the text after "any" (limited window to avoid false matches)
    after_any = text_after_note[any_pos + 3 : any_pos + 50]

    for keyword in _ANY_EXEMPTION_KEYWORDS:
        if keyword in after_any:
            return True

    return False


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

        The NOTE comment must follow the specific format:
            # NOTE: Any required for <reason>
            # NOTE: Any needed for <reason>
            # NOTE: Any necessary for <reason>

        The keyword after "Any" must indicate intent (required, needed, necessary,
        used, workaround). Generic NOTE comments do not qualify.

        Supports both single-line and multi-line import blocks:
            from typing import Any  # NOTE: Any required for...

            from typing import (
                Any,  # NOTE: Any required for...
                Dict,
            )

        Returns:
            True if file has a valid file-level NOTE comment.

        Note:
            Only scans first 100 lines for performance. Imports should
            always appear near the top of Python files per PEP 8.

            IMPORTANT: Each line is checked individually rather than joining
            context into a single string. This prevents false positives where
            "NOTE:" appears on one line and "Any required" on a different,
            unrelated line.

            Uses _extract_comment_portion to avoid false positives from
            # characters inside string literals.
        """
        # Only scan first 100 lines - imports must be at top per PEP 8
        max_lines = min(100, len(self.source_lines))

        # Track if we're inside a multi-line import block
        in_multiline_import = False
        multiline_import_start = -1

        for i, line in enumerate(self.source_lines[:max_lines]):
            # Detect start of multi-line import: "from typing import ("
            if "from typing import" in line and "(" in line and ")" not in line:
                in_multiline_import = True
                multiline_import_start = i
                continue

            # Inside multi-line import block
            if in_multiline_import:
                # Check if this line contains Any
                if "Any" in line:
                    # Check this line and surrounding lines for NOTE
                    start = max(0, multiline_import_start - 5)
                    end = min(len(self.source_lines), i + 6)
                    for check_line in self.source_lines[start:end]:
                        comment = _extract_comment_portion(check_line)
                        if comment and _is_any_exemption_note(comment):
                            return True

                # End of multi-line import block
                if ")" in line:
                    in_multiline_import = False
                    multiline_import_start = -1
                continue

            # Single-line import: "from typing import Any"
            if "from typing import" in line and "Any" in line:
                # Check surrounding lines (5 before and 5 after) INDIVIDUALLY
                # to prevent cross-line false positives
                start = max(0, i - 5)
                end = min(len(self.source_lines), i + 6)
                for check_line in self.source_lines[start:end]:
                    # Use _extract_comment_portion to avoid string literal false positives
                    comment = _extract_comment_portion(check_line)
                    if comment and _is_any_exemption_note(comment):
                        return True

            # Also check for 'import typing' style
            elif "import typing" in line:
                start = max(0, i - 5)
                end = min(len(self.source_lines), i + 6)
                for check_line in self.source_lines[start:end]:
                    comment = _extract_comment_portion(check_line)
                    if comment and _is_any_exemption_note(comment):
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

        The NOTE comment must follow the specific format:
            # NOTE: Any required for <reason>
            # NOTE: Any needed for <reason>
            # NOTE: Any necessary for <reason>

        The keyword after "Any" must indicate intent (required, needed, necessary,
        used, workaround). Generic NOTE comments do not qualify.

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
        # NOTE: OMN-1305 - _is_field_call is provided by MixinAnyTypeClassification
        # via multiple inheritance. Mypy cannot verify mixin method availability.
        if node.value is None or not self._is_field_call(node.value):  # type: ignore[attr-defined]
            return False

        # Check if annotation contains Any
        # NOTE: OMN-1305 - _contains_any is provided by the main AnyTypeDetector
        # class via multiple inheritance. Mypy cannot verify mixin composition.
        if not self._contains_any(node.annotation):  # type: ignore[attr-defined]
            return False

        # File-level NOTE comment exempts all Field() usages
        if self.has_file_level_note:
            return True

        line_num = node.lineno

        # First, check for inline NOTE comment on the same line
        # Extract ONLY the comment portion to avoid matching "Any" in code
        # e.g., for "field: Any = Field(...)  # NOTE: Any required for..."
        # we only want to check "# NOTE: Any required for..."
        # Uses _extract_comment_portion to avoid false positives from
        # # characters inside string literals.
        if 0 < line_num <= len(self.source_lines):
            current_line = self.source_lines[line_num - 1]
            # Use _extract_comment_portion to properly handle # in strings
            comment_portion = _extract_comment_portion(current_line)
            if comment_portion and _is_any_exemption_note(comment_portion):
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
                # Use strict matching - must have exemption keyword after "Any"
                if _is_any_exemption_note(line_content):
                    return True
                # Continue looking in the comment block
                continue

            # Non-comment, non-blank line - stop looking
            break

        return False
