# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Mixin for file-path-based architecture rules.

This module provides a mixin that extracts file paths from validation targets
using explicit isinstance checks. Rules that validate file paths can inherit
from this mixin to avoid duplicating the path extraction logic.

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

from pathlib import Path

__all__ = ["MixinFilePathRule"]


class MixinFilePathRule:
    """Mixin providing file path extraction for architecture rules.

    This mixin provides a standardized method for extracting file paths from
    validation targets using explicit isinstance checks. It handles the common
    case where a rule expects a file path but may receive other types of objects.

    The extraction uses explicit type checking to ensure the target is a string
    or Path object before processing.

    Thread Safety:
        This mixin is stateless and safe for concurrent use. The
        ``_extract_file_path`` method is a pure function with no side effects.

    Example:
        >>> mixin = MixinFilePathRule()
        >>> mixin._extract_file_path("/path/to/file.py")
        '/path/to/file.py'
        >>> mixin._extract_file_path(object())  # Returns None for non-path objects
        >>> mixin._extract_file_path("")  # Returns None for empty strings
    """

    def _extract_file_path(self, target: object) -> str | None:
        """Extract file path from target or return None if invalid.

        Uses explicit isinstance check to validate the target is a string
        or Path object. Returns None for:

        - Non-string, non-Path objects
        - Empty strings

        Args:
            target: Object to extract file path from. Must be a string
                or Path object; other types return None.

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
        # Only handle string or Path-like targets
        if isinstance(target, str | Path):
            file_path = str(target)
            if not file_path:
                return None
            return file_path
        return None
