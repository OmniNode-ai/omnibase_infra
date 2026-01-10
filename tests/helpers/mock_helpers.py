# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Shared mock helpers for omnibase_infra tests.

This module provides reusable mock utilities for filesystem and I/O testing,
reducing code duplication across test files.

Available Utilities:
    MockStatResult: NamedTuple matching os.stat_result interface for file stat mocking
    create_mock_stat_result: Factory function for creating MockStatResult with overrides

Usage Example:
    >>> from tests.helpers.mock_helpers import MockStatResult, create_mock_stat_result
    >>>
    >>> original_stat = Path.stat
    >>> def mock_stat(self: Path, **kwargs: object) -> object:
    ...     result = original_stat(self, **kwargs)
    ...     if self.name == "handler_contract.yaml":
    ...         return create_mock_stat_result(result, override_size=10 * 1024 * 1024 + 1)
    ...     return result
"""

from __future__ import annotations

from typing import NamedTuple


class MockStatResult(NamedTuple):
    """Mock stat result with typed fields matching os.stat_result.

    Uses NamedTuple for clarity and type safety. Provides the same interface
    as os.stat_result for file stat operations in tests.

    This class is used to mock filesystem stat operations without actually
    creating large files on disk, particularly useful for testing file size
    limits and DoS protection.

    Attributes:
        st_size: File size in bytes.
        st_mode: File mode (permissions and file type).
        st_ino: Inode number.
        st_dev: Device identifier.
        st_nlink: Number of hard links.
        st_uid: User ID of owner.
        st_gid: Group ID of owner.
        st_atime: Time of last access.
        st_mtime: Time of last modification.
        st_ctime: Time of last status change.
    """

    st_size: int
    st_mode: int
    st_ino: int
    st_dev: int
    st_nlink: int
    st_uid: int
    st_gid: int
    st_atime: float
    st_mtime: float
    st_ctime: float


def create_mock_stat_result(
    real_stat_result: object,
    override_size: int,
) -> MockStatResult:
    """Create a mock stat result object with overridden st_size.

    This factory reduces duplication in file size limit tests by creating
    mock stat result objects that report a specific file size while preserving
    all other stat attributes from the original file.

    Args:
        real_stat_result: The actual os.stat_result from Path.stat().
            Must have all standard stat attributes (st_mode, st_ino, etc.).
        override_size: The file size (st_size) to report in the mock result.

    Returns:
        A MockStatResult with st_size set to override_size and all other
        attributes copied from real_stat_result.

    Example:
        >>> from pathlib import Path
        >>> from unittest.mock import patch
        >>> from tests.helpers.mock_helpers import create_mock_stat_result
        >>>
        >>> MAX_SIZE = 10 * 1024 * 1024  # 10MB
        >>> original_stat = Path.stat
        >>>
        >>> def mock_stat(self: Path, **kwargs: object) -> object:
        ...     result = original_stat(self, **kwargs)
        ...     if self.name.endswith("_contract.yaml"):
        ...         return create_mock_stat_result(result, MAX_SIZE + 1)
        ...     return result
        >>>
        >>> with patch.object(Path, "stat", mock_stat):
        ...     stat = Path("test_contract.yaml").stat()
        ...     # stat.st_size will be MAX_SIZE + 1

    Note:
        The type: ignore comments are necessary because real_stat_result is
        typed as object to allow duck typing, but we access stat-specific
        attributes that exist on os.stat_result.
    """
    return MockStatResult(
        st_size=override_size,
        st_mode=real_stat_result.st_mode,  # type: ignore[attr-defined]
        st_ino=real_stat_result.st_ino,  # type: ignore[attr-defined]
        st_dev=real_stat_result.st_dev,  # type: ignore[attr-defined]
        st_nlink=real_stat_result.st_nlink,  # type: ignore[attr-defined]
        st_uid=real_stat_result.st_uid,  # type: ignore[attr-defined]
        st_gid=real_stat_result.st_gid,  # type: ignore[attr-defined]
        st_atime=real_stat_result.st_atime,  # type: ignore[attr-defined]
        st_mtime=real_stat_result.st_mtime,  # type: ignore[attr-defined]
        st_ctime=real_stat_result.st_ctime,  # type: ignore[attr-defined]
    )
