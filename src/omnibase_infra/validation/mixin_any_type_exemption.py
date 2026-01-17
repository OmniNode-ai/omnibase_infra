# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Exemption mixin for Any type detection.

This mixin provides methods for managing exemptions via decorators,
comments, and file-level NOTE patterns.

Exemption Format Requirements:
    All NOTE comments must include an OMN ticket reference:
        # NOTE: OMN-1234 - Reason for Any usage

    This applies to:
        - File-level NOTE comments (within 2 lines of Any import)
        - Field-level NOTE comments (above or inline with Field())
        - Type-ignore directive justifications (NOTE must be on line above)

    Invalid formats (will NOT be accepted):
        - # NOTE: Required for...  (missing OMN-XXXX ticket code)
        - # reason  (missing NOTE: prefix and ticket code)
        - # NOTE: reason  (missing OMN-XXXX ticket code)

    Valid formats:
        - # NOTE: OMN-1234 - Required for Pydantic discriminated union
        - # NOTE: OMN-5678 - Dynamic JSON payload handling
        - # NOTE: omn-9999 - Case-insensitive ticket codes accepted
"""

from __future__ import annotations

import ast
import re

# Decorator names that allow Any type usage
_ALLOW_DECORATORS: frozenset[str] = frozenset({"allow_any", "allow_any_type"})

# Regex pattern for OMN ticket codes
# Matches: OMN-1234, OMN-12345, etc.
_OMN_TICKET_PATTERN = re.compile(r"OMN-\d+", re.IGNORECASE)

# Type-ignore pattern for detecting type: ignore comments
_TYPE_IGNORE_PATTERN = re.compile(r"#\s*type:\s*ignore", re.IGNORECASE)

# Number of lines to look back for NOTE comments.
# 3 lines is sufficient because NOTE comments should appear:
# - Immediately above the field (1 line)
# - With a blank line gap (2 lines)
# - After a decorator (3 lines max)
# Larger values risk matching unrelated NOTE comments, so keep this tight.
_NOTE_LOOKBACK_LINES = 3

# Number of lines to look around file-level Any imports for NOTE comments.
# This is intentionally tight (2 lines before/after) to ensure NOTE comments
# are closely associated with the import they document.
_FILE_LEVEL_NOTE_WINDOW = 2


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
    """Check if text contains a valid Any exemption note.

    A valid exemption note must have:
    1. A comment marker (#)
    2. "NOTE:" immediately after the comment marker (with optional whitespace)
    3. An OMN ticket code (e.g., OMN-1234)
    4. A reason/description after the ticket code

    This prevents generic NOTE comments from accidentally matching and ensures
    NOTE: is at the start of a comment, not buried in the middle.

    Valid examples:
        - "# NOTE: OMN-1234 - Required for Pydantic discriminated union"
        - "code  # NOTE: OMN-5678 - Dynamic JSON payload handling"
        - "# NOTE: OMN-9999 - Workaround for typing limitations"
        - "# NOTE: omn-1234 - Case-insensitive ticket codes accepted"

    Invalid examples:
        - "# Some text NOTE: OMN-1234 - reason" (NOTE not at comment start)
        - "NOTE: OMN-1234 - reason" (no comment marker)
        - "# NOTE: Required for..." (missing OMN-XXXX ticket code)
        - "# NOTE: reason" (missing OMN-XXXX ticket code)
        - "# reason" (missing NOTE: prefix and ticket code)
        - "x = 'NOTE: OMN-1234'" (NOTE in string, not comment)

    Args:
        text: Text to check. Can be a single line of source code (with or without
            code before the comment) or a pre-extracted comment portion starting
            with '#'. The function uses _extract_comment_portion internally to
            properly handle '#' characters inside string literals.

    Returns:
        True if text contains a valid Any exemption note at comment start with
        an OMN ticket reference.
    """
    # Use _extract_comment_portion to properly handle # in string literals
    # e.g., x = "# not a comment"  # real comment
    # This prevents false negatives where text.find("#") finds the wrong #
    comment_portion = _extract_comment_portion(text)
    if comment_portion is None:
        return False

    # Extract the text after # and strip leading whitespace
    # comment_portion is guaranteed to start with #
    comment = comment_portion[1:].lstrip()
    comment_lower = comment.lower()

    # NOTE: must be at the start of the comment (after # and whitespace)
    if not comment_lower.startswith("note:"):
        return False

    # Extract text after "NOTE:" for OMN ticket validation
    text_after_note = comment[5:]  # Keep original case for regex matching

    # Check for OMN ticket code (e.g., OMN-1234)
    if _OMN_TICKET_PATTERN.search(text_after_note):
        return True

    return False


def _has_type_ignore_with_valid_note(source_lines: list[str], line_num: int) -> bool:
    """Check if a type: ignore comment has a valid NOTE on the line above.

    When using `# type: ignore` to suppress type checker warnings for Any usage,
    the preceding line MUST have a NOTE comment with an OMN ticket reference.

    Valid pattern:
        # NOTE: OMN-1234 - Required for Pydantic discriminated union
        field: Any = Field(...)  # type: ignore[valid-type]

    Invalid patterns:
        field: Any = Field(...)  # type: ignore[valid-type]  (no NOTE above)

        # NOTE: Required for discriminated union  (no OMN ticket)
        field: Any = Field(...)  # type: ignore[valid-type]

    Args:
        source_lines: List of source code lines.
        line_num: 1-based line number of the line with type: ignore.

    Returns:
        True if the type: ignore has a valid NOTE with OMN ticket on preceding line.
    """
    # Check if current line has type: ignore
    if line_num < 1 or line_num > len(source_lines):
        return False

    current_line = source_lines[line_num - 1]
    # Extract only the comment portion to check for type: ignore
    # This prevents false matches where "type: ignore" appears in string literals
    comment_portion = _extract_comment_portion(current_line)
    if comment_portion is None or not _TYPE_IGNORE_PATTERN.search(comment_portion):
        return False

    # Check the line above for a valid NOTE comment
    if line_num < 2:
        return False

    preceding_line = source_lines[line_num - 2]
    # Extract comment portion from preceding line before checking
    # This ensures we only match NOTE in actual comments, not code/strings
    preceding_comment = _extract_comment_portion(preceding_line)
    if preceding_comment is None:
        return False
    return _is_any_exemption_note(preceding_comment)


def _is_in_multiline_import(
    source_lines: list[str], start_idx: int, max_lines: int
) -> tuple[bool, int, int]:
    """Check if we're at the start of a multi-line import block.

    Handles both parenthesized imports and backslash-continuation imports:
        from typing import (
            Any,
            Dict,
        )

        from typing import \\
            Any, \\
            Dict

    Args:
        source_lines: List of source code lines.
        start_idx: Index to start checking from.
        max_lines: Maximum number of lines to scan.

    Returns:
        Tuple of (is_multiline, start_line_idx, end_line_idx).
        If not multiline, returns (False, -1, -1).
    """
    if start_idx >= len(source_lines) or start_idx >= max_lines:
        return (False, -1, -1)

    line = source_lines[start_idx]

    # Check for parenthesized import: "from typing import ("
    if "from typing import" in line and "(" in line:
        if ")" in line:
            # Single line: from typing import (Any, Dict)
            return (False, -1, -1)
        # Find closing parenthesis
        for end_idx in range(start_idx + 1, min(len(source_lines), max_lines)):
            if ")" in source_lines[end_idx]:
                return (True, start_idx, end_idx)
        return (True, start_idx, min(len(source_lines), max_lines) - 1)

    # Check for backslash continuation: "from typing import \"
    if "from typing import" in line and line.rstrip().endswith("\\"):
        end_idx = start_idx
        for i in range(start_idx + 1, min(len(source_lines), max_lines)):
            if not source_lines[i].rstrip().endswith("\\"):
                end_idx = i
                break
            end_idx = i
        return (True, start_idx, end_idx)

    return (False, -1, -1)


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

        A file-level NOTE comment (within 2 lines of 'from typing import Any'
        or 'import typing') exempts all Pydantic Field() usages with Any in
        that file. This supports the common pattern of documenting the
        workaround once at the import rather than at each field.

        The NOTE comment must follow the specific format with OMN ticket:
            # NOTE: OMN-1234 - Required for Pydantic discriminated union
            # NOTE: OMN-5678 - Dynamic JSON payload handling

        Generic NOTE comments without OMN ticket codes do NOT qualify:
            # NOTE: Required for...  (INVALID - missing ticket)
            # NOTE: reason  (INVALID - missing ticket)

        Supports both single-line and multi-line import blocks:
            from typing import Any  # NOTE: OMN-1234 - JSON payload fields

            from typing import (
                Any,  # NOTE: OMN-1234 - JSON payload fields
                Dict,
            )

            from typing import \\
                Any  # NOTE: OMN-1234 - Backslash continuation import

        Returns:
            True if file has a valid file-level NOTE comment with OMN ticket.

        Note:
            Only scans first 100 lines for performance. Imports should
            always appear near the top of Python files per PEP 8.

            IMPORTANT: Each line is checked individually rather than joining
            context into a single string. This prevents false positives where
            "NOTE:" appears on one line and "OMN-1234" on a different,
            unrelated line.

            Uses _extract_comment_portion to avoid false positives from
            # characters inside string literals.

            The window is intentionally tight (2 lines before/after) to ensure
            NOTE comments are closely associated with the import they document.
        """
        # Only scan first 100 lines - imports must be at top per PEP 8
        max_lines = min(100, len(self.source_lines))

        # Track if we're inside a multi-line import block
        in_multiline_import = False
        multiline_import_start = -1
        multiline_import_end = -1

        i = 0
        while i < max_lines:
            line = self.source_lines[i]

            # Skip if we're inside a known multi-line block
            if in_multiline_import and i <= multiline_import_end:
                # Check if this line contains Any (case-sensitive for import)
                if "Any" in line:
                    # Check the entire multi-line block and surrounding lines for NOTE
                    # Use tight window for lines before/after the block
                    start = max(0, multiline_import_start - _FILE_LEVEL_NOTE_WINDOW)
                    end = min(
                        len(self.source_lines),
                        multiline_import_end + _FILE_LEVEL_NOTE_WINDOW + 1,
                    )
                    for check_line in self.source_lines[start:end]:
                        comment = _extract_comment_portion(check_line)
                        if comment and _is_any_exemption_note(comment):
                            return True

                # Check if we've reached the end of the multi-line block
                if i >= multiline_import_end:
                    in_multiline_import = False
                    multiline_import_start = -1
                    multiline_import_end = -1
                i += 1
                continue

            # Check for multi-line import (parenthesized or backslash continuation)
            is_multiline, start_idx, end_idx = _is_in_multiline_import(
                self.source_lines, i, max_lines
            )
            if is_multiline:
                in_multiline_import = True
                multiline_import_start = start_idx
                multiline_import_end = end_idx
                # Check if Any is on the first line of the multi-line import
                if "Any" in line:
                    start = max(0, start_idx - _FILE_LEVEL_NOTE_WINDOW)
                    end = min(
                        len(self.source_lines),
                        end_idx + _FILE_LEVEL_NOTE_WINDOW + 1,
                    )
                    for check_line in self.source_lines[start:end]:
                        comment = _extract_comment_portion(check_line)
                        if comment and _is_any_exemption_note(comment):
                            return True
                i += 1
                continue

            # Single-line import: "from typing import Any"
            if "from typing import" in line and "Any" in line:
                # Check surrounding lines INDIVIDUALLY using tight window
                # to prevent cross-line false positives
                start = max(0, i - _FILE_LEVEL_NOTE_WINDOW)
                end = min(len(self.source_lines), i + _FILE_LEVEL_NOTE_WINDOW + 1)
                for check_line in self.source_lines[start:end]:
                    # Use _extract_comment_portion to avoid string literal false positives
                    comment = _extract_comment_portion(check_line)
                    if comment and _is_any_exemption_note(comment):
                        return True

            # Also check for 'import typing' style
            elif "import typing" in line:
                start = max(0, i - _FILE_LEVEL_NOTE_WINDOW)
                end = min(len(self.source_lines), i + _FILE_LEVEL_NOTE_WINDOW + 1)
                for check_line in self.source_lines[start:end]:
                    comment = _extract_comment_portion(check_line)
                    if comment and _is_any_exemption_note(comment):
                        return True

            i += 1

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
        - In the contiguous comment block immediately preceding the field, OR
        - As a NOTE comment above a type: ignore directive

        The NOTE comment must follow the specific format with OMN ticket:
            # NOTE: OMN-1234 - Required for Pydantic discriminated union
            # NOTE: OMN-5678 - Dynamic JSON payload handling

        Generic NOTE comments without OMN ticket codes do NOT qualify:
            # NOTE: Required for...  (INVALID - missing ticket)
            # NOTE: reason  (INVALID - missing ticket)
            # Any reason here  (INVALID - wrong format, must start with NOTE:)

        File-level NOTE comments (within 2 lines of Any import) exempt all
        Field() usages in that file, supporting the common pattern of
        documenting the workaround once at the import.

        Type-ignore pattern (requires NOTE on preceding line):
            # NOTE: OMN-1234 - Required for Pydantic discriminated union
            field: Any = Field(...)  # type: ignore[valid-type]

        Args:
            node: The annotated assignment node.

        Returns:
            True if this is a Field() call with a proper NOTE comment
            containing an OMN ticket reference.

        Note:
            This method requires _is_field_call and _contains_any from
            MixinAnyTypeClassification and the main class respectively.

            For inline comments, ONLY the comment portion is inspected.
            This prevents false matches where NOTE appears in code/strings.
        """
        # Check if value is a Field() call
        # NOTE: OMN-1305 - _is_field_call from MixinAnyTypeClassification (mixin composition)
        if node.value is None or not self._is_field_call(node.value):  # type: ignore[attr-defined]
            return False

        # Check if annotation contains Any
        # NOTE: OMN-1305 - _contains_any from AnyTypeDetector (mixin composition)
        if not self._contains_any(node.annotation):  # type: ignore[attr-defined]
            return False

        # File-level NOTE comment exempts all Field() usages
        if self.has_file_level_note:
            return True

        line_num = node.lineno

        # First, check for inline NOTE comment on the same line
        # CRITICAL: Extract ONLY the comment portion to avoid matching NOTE in code
        # This handles cases like:
        #   field: Any = Field(NOTE="something")  # NOTE: OMN-1234 - reason
        # We only check "# NOTE: OMN-1234 - reason", not "NOTE=" in the code.
        #
        # _extract_comment_portion correctly handles:
        #   - Hash (#) characters inside string literals
        #   - Single, double, and triple quoted strings
        #   - Escaped quotes within strings
        if 0 < line_num <= len(self.source_lines):
            current_line = self.source_lines[line_num - 1]
            # Extract only the actual comment, ignoring # in strings
            comment_portion = _extract_comment_portion(current_line)
            if comment_portion and _is_any_exemption_note(comment_portion):
                return True

        # Check for type: ignore with valid NOTE on preceding line
        # This handles the pattern:
        #   # NOTE: OMN-1234 - Required for Pydantic discriminated union
        #   field: Any = Field(...)  # type: ignore[valid-type]
        if _has_type_ignore_with_valid_note(self.source_lines, line_num):
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
                # Use strict matching - must have NOTE: and OMN ticket code
                if _is_any_exemption_note(line_content):
                    return True
                # Continue looking in the comment block
                continue

            # Non-comment, non-blank line - stop looking
            break

        return False
