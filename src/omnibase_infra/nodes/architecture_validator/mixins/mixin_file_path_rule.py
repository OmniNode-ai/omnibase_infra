# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Mixin for file-path-based architecture rules.

This module provides a mixin that extracts file paths from validation targets
using duck typing. Rules that validate file paths can inherit from this mixin
to avoid duplicating the path extraction logic.

Thread Safety:
    MixinFilePathRule is stateless and safe for concurrent use.
    All methods are pure functions with no side effects.

Example:
    >>> class RuleFilePath(MixinFilePathRule):
    ...     @property
    ...     def rule_id(self) -> str:
    ...         return "FILE-001"
    ...
    ...     def check(self, target: object) -> ModelRuleCheckResult:
    ...         file_path = self._extract_file_path(target)
    ...         if file_path is None:
    ...             return ModelRuleCheckResult(
    ...                 passed=True,
    ...                 skipped=True,
    ...                 rule_id=self.rule_id,
    ...                 reason="Target is not a valid file path",
    ...             )
    ...         # Continue with file-based validation...
"""

from __future__ import annotations

__all__ = ["MixinFilePathRule"]


class MixinFilePathRule:
    """Mixin providing file path extraction for architecture rules.

    This mixin provides a standardized method for extracting file paths from
    validation targets using duck typing. It handles the common case where
    a rule expects a file path but may receive other types of objects.

    The extraction uses ``str()`` conversion and validates that the result
    looks like a file path rather than an object repr string.

    Thread Safety:
        This mixin is stateless and safe for concurrent use. The
        ``_extract_file_path`` method is a pure function with no side effects.

    Example:
        >>> mixin = MixinFilePathRule()
        >>> mixin._extract_file_path("/path/to/file.py")
        '/path/to/file.py'
        >>> mixin._extract_file_path(object())  # Returns None for repr strings
        >>> mixin._extract_file_path("")  # Returns None for empty strings
    """

    def _extract_file_path(self, target: object) -> str | None:
        """Extract file path from target or return None if invalid.

        Uses duck typing to convert the target to a string and validates
        that the result looks like a file path. Returns None for:

        - Empty strings
        - Object repr strings (e.g., "<object at 0x...>")
        - Targets that cannot be converted to string

        Args:
            target: Object to extract file path from. Expected to be a string
                or Path-like object, but any object is accepted.

        Returns:
            File path string if valid, None otherwise.

        Example:
            >>> self._extract_file_path("/path/to/file.py")
            '/path/to/file.py'
            >>> self._extract_file_path(Path("/path/to/file.py"))
            '/path/to/file.py'
            >>> self._extract_file_path(object())
            None
            >>> self._extract_file_path("")
            None
        """
        try:
            file_path = str(target)
            # Skip object repr strings (e.g., "<object at 0x...>")
            if not file_path or file_path.startswith("<"):
                return None
            return file_path
        except (TypeError, ValueError):
            return None
