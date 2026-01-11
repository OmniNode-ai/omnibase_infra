"""Shared fixtures for CI tests.

This module provides common fixtures used across CI test modules,
promoting code reuse and consistent test patterns.

Ticket: OMN-255
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture
def create_test_file(tmp_path: Path) -> Callable[[str, str], Path]:
    """Factory fixture for creating temporary Python test files.

    Uses pytest's tmp_path fixture for automatic cleanup after tests.
    This eliminates the need for try/finally cleanup patterns.

    Args:
        tmp_path: Pytest fixture providing a temporary directory unique to each test.

    Returns:
        A callable that takes content and optional filename, returns the file path.

    Example:
        def test_something(create_test_file):
            test_file = create_test_file("import kafka\\n")
            # File is automatically cleaned up after test
    """

    def _create(content: str, filename: str = "test_module.py") -> Path:
        """Create a temporary Python file with given content.

        Args:
            content: The content to write to the file.
            filename: Optional filename (default: test_module.py).

        Returns:
            Path to the created temporary file.
        """
        file_path = tmp_path / filename
        file_path.write_text(content, encoding="utf-8")
        return file_path

    return _create


@pytest.fixture
def forbidden_patterns() -> list[str]:
    """Standard forbidden import patterns for architecture compliance testing.

    Returns:
        List containing the standard test pattern ["kafka"].
    """
    return ["kafka"]
