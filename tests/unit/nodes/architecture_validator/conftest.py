# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Shared pytest fixtures for architecture validator tests.

This module provides common fixtures used across multiple architecture validator
test files, including helpers for creating temporary Python files with specific
code patterns.

Fixtures Provided:
    - create_temp_python_file: Factory fixture for creating temp Python files
    - temp_python_file: Simple fixture that returns a file creator function
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest


@pytest.fixture
def create_temp_python_file(tmp_path: Path) -> Callable[[str, str], Path]:
    """Factory fixture for creating temporary Python files with code patterns.

    This fixture returns a function that creates a temporary Python file
    with the specified filename and content. Useful for testing code analysis
    and validation.

    Args:
        tmp_path: Pytest's built-in tmp_path fixture.

    Returns:
        A callable that takes (filename, content) and returns the Path
        to the created file.

    Example::

        def test_something(create_temp_python_file):
            path = create_temp_python_file("service.py", "class MyService: pass")
            result = validator.validate(str(path))
            assert result.valid

    """

    def _create_file(filename: str, content: str) -> Path:
        """Create a temporary Python file with the given content.

        Args:
            filename: Name of the file to create (e.g., "service.py").
            content: Python source code to write to the file.

        Returns:
            Path to the created file.

        """
        file_path = tmp_path / filename
        file_path.write_text(content, encoding="utf-8")
        return file_path

    return _create_file


@pytest.fixture
def temp_python_file(tmp_path: Path) -> Callable[[str], Path]:
    """Simplified fixture for creating a single temp Python file.

    This fixture returns a function that creates a temporary Python file
    with a default name. The function takes only the content as argument.

    Args:
        tmp_path: Pytest's built-in tmp_path fixture.

    Returns:
        A callable that takes content and returns the Path to the created file.

    Example::

        def test_something(temp_python_file):
            path = temp_python_file("class MyService: pass")
            result = validator.validate(str(path))
            assert result.valid

    """

    def _create_file(content: str, filename: str = "code.py") -> Path:
        """Create a temporary Python file with the given content.

        Args:
            content: Python source code to write to the file.
            filename: Name of the file (default: "code.py").

        Returns:
            Path to the created file.

        """
        file_path = tmp_path / filename
        file_path.write_text(content, encoding="utf-8")
        return file_path

    return _create_file


__all__ = [
    "create_temp_python_file",
    "temp_python_file",
]
