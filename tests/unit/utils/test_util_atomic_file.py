# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for util_atomic_file module.

This test suite provides comprehensive coverage of the atomic file write utilities
in omnibase_infra.utils.util_atomic_file:
    - write_atomic_bytes: Synchronous atomic file write
    - write_atomic_bytes_async: Async wrapper for atomic file write

Test Organization:
    - TestWriteAtomicBytesBasic: Core functionality
    - TestWriteAtomicBytesParameters: Parameter handling (prefix, suffix)
    - TestWriteAtomicBytesCleanup: Temp file cleanup on failure
    - TestWriteAtomicBytesLogging: Correlation ID logging
    - TestWriteAtomicBytesAsync: Async wrapper tests

Coverage Goals:
    - Full coverage of write_atomic_bytes function
    - Verification that async wrapper delegates to sync
    - Cleanup behavior on various failure modes
    - Logging with correlation context
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from omnibase_infra.errors import InfraConnectionError
from omnibase_infra.utils.util_atomic_file import (
    write_atomic_bytes,
    write_atomic_bytes_async,
)


@pytest.mark.unit
class TestWriteAtomicBytesBasic:
    """Test suite for basic write_atomic_bytes functionality.

    Tests verify core behavior:
    - Successful write returns bytes written
    - File contains correct data after write
    - File is created if it doesn't exist
    - File is overwritten if it exists
    """

    def test_successful_write_returns_bytes_written(self, tmp_path: Path) -> None:
        """Test successful write returns the number of bytes written."""
        target = tmp_path / "test.txt"
        data = b"Hello, World!"

        result = write_atomic_bytes(target, data)

        assert result == len(data)
        assert result == 13

    def test_file_contains_correct_data(self, tmp_path: Path) -> None:
        """Test file contains the exact data that was written."""
        target = tmp_path / "test.txt"
        data = b"Test data with special chars: \x00\xff\n\t"

        write_atomic_bytes(target, data)

        assert target.read_bytes() == data

    def test_file_created_if_not_exists(self, tmp_path: Path) -> None:
        """Test file is created if it doesn't exist."""
        target = tmp_path / "new_file.txt"
        assert not target.exists()

        write_atomic_bytes(target, b"new content")

        assert target.exists()

    def test_file_overwritten_if_exists(self, tmp_path: Path) -> None:
        """Test existing file is overwritten with new content."""
        target = tmp_path / "existing.txt"
        target.write_bytes(b"old content")

        write_atomic_bytes(target, b"new content")

        assert target.read_bytes() == b"new content"

    def test_empty_data_creates_empty_file(self, tmp_path: Path) -> None:
        """Test writing empty bytes creates an empty file."""
        target = tmp_path / "empty.txt"

        result = write_atomic_bytes(target, b"")

        assert result == 0
        assert target.exists()
        assert target.read_bytes() == b""

    def test_binary_data_preserved(self, tmp_path: Path) -> None:
        """Test binary data (including null bytes) is preserved exactly."""
        target = tmp_path / "binary.bin"
        data = bytes(range(256))  # All possible byte values

        write_atomic_bytes(target, data)

        assert target.read_bytes() == data

    def test_large_data_write(self, tmp_path: Path) -> None:
        """Test writing large data (1MB) works correctly."""
        target = tmp_path / "large.bin"
        data = b"x" * (1024 * 1024)  # 1MB

        result = write_atomic_bytes(target, data)

        assert result == 1024 * 1024
        assert target.read_bytes() == data


@pytest.mark.unit
class TestWriteAtomicBytesParameters:
    """Test suite for write_atomic_bytes parameter handling.

    Tests verify:
    - Custom temp_prefix is used
    - Custom temp_suffix is used
    - Default values work correctly
    """

    def test_custom_prefix_used_in_temp_file(self, tmp_path: Path) -> None:
        """Test custom prefix appears in temp file name during write."""
        target = tmp_path / "test.txt"
        data = b"test"

        # Mock mkstemp to capture the prefix argument
        original_mkstemp = tempfile.mkstemp

        captured_kwargs: dict[str, object] = {}

        def capturing_mkstemp(**kwargs: object) -> tuple[int, str]:
            captured_kwargs.update(kwargs)
            return original_mkstemp(**kwargs)  # type: ignore[arg-type]

        with patch("tempfile.mkstemp", side_effect=capturing_mkstemp):
            write_atomic_bytes(target, data, temp_prefix="myprefix_")

        assert captured_kwargs.get("prefix") == "myprefix_"

    def test_custom_suffix_used_in_temp_file(self, tmp_path: Path) -> None:
        """Test custom suffix appears in temp file name during write."""
        target = tmp_path / "test.txt"
        data = b"test"

        original_mkstemp = tempfile.mkstemp
        captured_kwargs: dict[str, object] = {}

        def capturing_mkstemp(**kwargs: object) -> tuple[int, str]:
            captured_kwargs.update(kwargs)
            return original_mkstemp(**kwargs)  # type: ignore[arg-type]

        with patch("tempfile.mkstemp", side_effect=capturing_mkstemp):
            write_atomic_bytes(target, data, temp_suffix=".partial")

        assert captured_kwargs.get("suffix") == ".partial"

    def test_default_prefix_is_empty(self, tmp_path: Path) -> None:
        """Test default prefix is empty string."""
        target = tmp_path / "test.txt"
        data = b"test"

        original_mkstemp = tempfile.mkstemp
        captured_kwargs: dict[str, object] = {}

        def capturing_mkstemp(**kwargs: object) -> tuple[int, str]:
            captured_kwargs.update(kwargs)
            return original_mkstemp(**kwargs)  # type: ignore[arg-type]

        with patch("tempfile.mkstemp", side_effect=capturing_mkstemp):
            write_atomic_bytes(target, data)

        assert captured_kwargs.get("prefix") == ""

    def test_default_suffix_is_tmp(self, tmp_path: Path) -> None:
        """Test default suffix is '.tmp'."""
        target = tmp_path / "test.txt"
        data = b"test"

        original_mkstemp = tempfile.mkstemp
        captured_kwargs: dict[str, object] = {}

        def capturing_mkstemp(**kwargs: object) -> tuple[int, str]:
            captured_kwargs.update(kwargs)
            return original_mkstemp(**kwargs)  # type: ignore[arg-type]

        with patch("tempfile.mkstemp", side_effect=capturing_mkstemp):
            write_atomic_bytes(target, data)

        assert captured_kwargs.get("suffix") == ".tmp"

    def test_temp_file_created_in_target_directory(self, tmp_path: Path) -> None:
        """Test temp file is created in same directory as target (atomicity requirement)."""
        target = tmp_path / "test.txt"
        data = b"test"

        original_mkstemp = tempfile.mkstemp
        captured_kwargs: dict[str, object] = {}

        def capturing_mkstemp(**kwargs: object) -> tuple[int, str]:
            captured_kwargs.update(kwargs)
            return original_mkstemp(**kwargs)  # type: ignore[arg-type]

        with patch("tempfile.mkstemp", side_effect=capturing_mkstemp):
            write_atomic_bytes(target, data)

        assert captured_kwargs.get("dir") == tmp_path


@pytest.mark.unit
class TestWriteAtomicBytesCleanup:
    """Test suite for temp file cleanup on failure.

    Tests verify:
    - Temp file is removed on write failure
    - Temp file is removed on rename failure
    - Original file is preserved on failure
    """

    def test_temp_file_cleaned_up_on_write_failure(self, tmp_path: Path) -> None:
        """Test temp file is removed when write fails."""
        target = tmp_path / "test.txt"

        # Mock fdopen to fail during write
        with patch("os.fdopen") as mock_fdopen:
            mock_file = MagicMock()
            mock_file.__enter__ = MagicMock(return_value=mock_file)
            mock_file.__exit__ = MagicMock(return_value=False)
            mock_file.write.side_effect = OSError("Disk full")
            mock_fdopen.return_value = mock_file

            with pytest.raises(InfraConnectionError) as exc_info:
                write_atomic_bytes(target, b"test data")

            # Verify OSError is chained
            assert isinstance(exc_info.value.__cause__, OSError)
            assert "Disk full" in str(exc_info.value.__cause__)

        # Verify no temp files left behind
        temp_files = list(tmp_path.glob("*.tmp"))
        assert len(temp_files) == 0

    def test_temp_file_cleaned_up_on_rename_failure(self, tmp_path: Path) -> None:
        """Test temp file is removed when rename fails."""
        target = tmp_path / "test.txt"

        with patch("os.replace") as mock_replace:
            mock_replace.side_effect = OSError("Permission denied")

            with pytest.raises(InfraConnectionError) as exc_info:
                write_atomic_bytes(target, b"test data")

            # Verify OSError is chained
            assert isinstance(exc_info.value.__cause__, OSError)
            assert "Permission denied" in str(exc_info.value.__cause__)

        # Verify no temp files left behind
        temp_files = list(tmp_path.glob("*.tmp"))
        assert len(temp_files) == 0

    def test_original_file_preserved_on_failure(self, tmp_path: Path) -> None:
        """Test original file content is preserved when write fails."""
        target = tmp_path / "existing.txt"
        original_content = b"original content"
        target.write_bytes(original_content)

        with patch("os.replace") as mock_replace:
            mock_replace.side_effect = OSError("Permission denied")

            with pytest.raises(InfraConnectionError):
                write_atomic_bytes(target, b"new content")

        # Original content should be preserved
        assert target.read_bytes() == original_content

    def test_raises_infra_connection_error_with_chained_oserror(
        self, tmp_path: Path
    ) -> None:
        """Test InfraConnectionError is raised with chained OSError on failure."""
        target = tmp_path / "test.txt"

        with patch("os.replace") as mock_replace:
            mock_replace.side_effect = OSError("Test error")

            with pytest.raises(InfraConnectionError) as exc_info:
                write_atomic_bytes(target, b"test")

            # Should be InfraConnectionError wrapping OSError
            assert isinstance(exc_info.value, InfraConnectionError)
            assert exc_info.value.__cause__ is not None
            assert isinstance(exc_info.value.__cause__, OSError)
            assert "Test error" in str(exc_info.value.__cause__)


@pytest.mark.unit
class TestWriteAtomicBytesLogging:
    """Test suite for correlation ID logging behavior.

    Tests verify:
    - Error logged with correlation_id when provided
    - No logging when correlation_id is None
    - Log contains expected context fields
    """

    def test_error_logged_with_correlation_id(self, tmp_path: Path) -> None:
        """Test error is logged with correlation context when correlation_id provided."""
        target = tmp_path / "test.txt"
        corr_id = uuid4()

        with (
            patch("os.replace") as mock_replace,
            patch("omnibase_infra.utils.util_atomic_file.logger") as mock_logger,
        ):
            mock_replace.side_effect = OSError("Test error")

            with pytest.raises(InfraConnectionError):
                write_atomic_bytes(target, b"test", correlation_id=corr_id)

            mock_logger.exception.assert_called_once()
            call_args = mock_logger.exception.call_args
            assert "Atomic write failed" in call_args[0][0]

    def test_log_contains_correlation_id(self, tmp_path: Path) -> None:
        """Test log extra contains correlation_id."""
        target = tmp_path / "test.txt"
        corr_id = uuid4()

        with (
            patch("os.replace") as mock_replace,
            patch("omnibase_infra.utils.util_atomic_file.logger") as mock_logger,
        ):
            mock_replace.side_effect = OSError("Test error")

            with pytest.raises(InfraConnectionError):
                write_atomic_bytes(target, b"test", correlation_id=corr_id)

            call_kwargs = mock_logger.exception.call_args[1]
            assert "extra" in call_kwargs
            assert call_kwargs["extra"]["correlation_id"] == str(corr_id)

    def test_log_contains_target_path(self, tmp_path: Path) -> None:
        """Test log extra contains target_path."""
        target = tmp_path / "test.txt"
        corr_id = uuid4()

        with (
            patch("os.replace") as mock_replace,
            patch("omnibase_infra.utils.util_atomic_file.logger") as mock_logger,
        ):
            mock_replace.side_effect = OSError("Test error")

            with pytest.raises(InfraConnectionError):
                write_atomic_bytes(target, b"test", correlation_id=corr_id)

            call_kwargs = mock_logger.exception.call_args[1]
            assert call_kwargs["extra"]["target_path"] == str(target)

    def test_log_contains_prefix_and_suffix(self, tmp_path: Path) -> None:
        """Test log extra contains temp_prefix and temp_suffix."""
        target = tmp_path / "test.txt"
        corr_id = uuid4()

        with (
            patch("os.replace") as mock_replace,
            patch("omnibase_infra.utils.util_atomic_file.logger") as mock_logger,
        ):
            mock_replace.side_effect = OSError("Test error")

            with pytest.raises(InfraConnectionError):
                write_atomic_bytes(
                    target,
                    b"test",
                    temp_prefix="myprefix_",
                    temp_suffix=".partial",
                    correlation_id=corr_id,
                )

            call_kwargs = mock_logger.exception.call_args[1]
            assert call_kwargs["extra"]["temp_prefix"] == "myprefix_"
            assert call_kwargs["extra"]["temp_suffix"] == ".partial"

    def test_no_logging_when_correlation_id_none(self, tmp_path: Path) -> None:
        """Test no error logging when correlation_id is None."""
        target = tmp_path / "test.txt"

        with (
            patch("os.replace") as mock_replace,
            patch("omnibase_infra.utils.util_atomic_file.logger") as mock_logger,
        ):
            mock_replace.side_effect = OSError("Test error")

            with pytest.raises(InfraConnectionError):
                write_atomic_bytes(target, b"test", correlation_id=None)

            mock_logger.exception.assert_not_called()

    def test_no_logging_on_success(self, tmp_path: Path) -> None:
        """Test no logging occurs on successful write."""
        target = tmp_path / "test.txt"
        corr_id = uuid4()

        with patch("omnibase_infra.utils.util_atomic_file.logger") as mock_logger:
            write_atomic_bytes(target, b"test", correlation_id=corr_id)

            mock_logger.exception.assert_not_called()


@pytest.mark.unit
class TestWriteAtomicBytesAsync:
    """Test suite for async wrapper write_atomic_bytes_async.

    Tests verify:
    - Async wrapper calls sync function
    - Return value is propagated correctly
    - Parameters are passed through
    - Exceptions are propagated
    """

    def test_async_wrapper_calls_sync_function(self, tmp_path: Path) -> None:
        """Test async wrapper delegates to sync implementation."""
        target = tmp_path / "test.txt"
        data = b"async test"

        with patch(
            "omnibase_infra.utils.util_atomic_file.write_atomic_bytes"
        ) as mock_sync:
            mock_sync.return_value = len(data)

            result = asyncio.run(write_atomic_bytes_async(target, data))

            mock_sync.assert_called_once_with(
                target,
                data,
                temp_prefix="",
                temp_suffix=".tmp",
                correlation_id=None,
            )
            assert result == len(data)

    def test_async_wrapper_returns_correct_value(self, tmp_path: Path) -> None:
        """Test async wrapper returns bytes written from sync function."""
        target = tmp_path / "test.txt"
        data = b"async test data"

        result = asyncio.run(write_atomic_bytes_async(target, data))

        assert result == len(data)
        assert target.read_bytes() == data

    def test_async_wrapper_passes_parameters(self, tmp_path: Path) -> None:
        """Test async wrapper passes all parameters to sync function."""
        target = tmp_path / "test.txt"
        data = b"test"
        corr_id = uuid4()

        with patch(
            "omnibase_infra.utils.util_atomic_file.write_atomic_bytes"
        ) as mock_sync:
            mock_sync.return_value = len(data)

            asyncio.run(
                write_atomic_bytes_async(
                    target,
                    data,
                    temp_prefix="async_",
                    temp_suffix=".async",
                    correlation_id=corr_id,
                )
            )

            mock_sync.assert_called_once_with(
                target,
                data,
                temp_prefix="async_",
                temp_suffix=".async",
                correlation_id=corr_id,
            )

    def test_async_wrapper_propagates_exceptions(self, tmp_path: Path) -> None:
        """Test async wrapper propagates InfraConnectionError from sync function."""
        target = tmp_path / "test.txt"

        with patch(
            "omnibase_infra.utils.util_atomic_file.write_atomic_bytes"
        ) as mock_sync:
            mock_sync.side_effect = InfraConnectionError("Async error")

            with pytest.raises(InfraConnectionError, match="Async error"):
                asyncio.run(write_atomic_bytes_async(target, b"test"))

    def test_async_wrapper_uses_to_thread(self, tmp_path: Path) -> None:
        """Test async wrapper uses asyncio.to_thread for non-blocking I/O."""
        target = tmp_path / "test.txt"
        data = b"test"

        with patch("asyncio.to_thread") as mock_to_thread:
            mock_to_thread.return_value = len(data)

            asyncio.run(write_atomic_bytes_async(target, data))

            mock_to_thread.assert_called_once()
            # First positional arg should be the sync function
            assert mock_to_thread.call_args[0][0] is write_atomic_bytes

    def test_async_actually_writes_file(self, tmp_path: Path) -> None:
        """Test async wrapper actually writes file content correctly."""
        target = tmp_path / "async_test.txt"
        data = b"async file content with special chars: \x00\n\t"

        result = asyncio.run(write_atomic_bytes_async(target, data))

        assert result == len(data)
        assert target.exists()
        assert target.read_bytes() == data


@pytest.mark.unit
class TestWriteAtomicBytesAtomicity:
    """Test suite for atomicity guarantees.

    Tests verify:
    - os.replace is used for atomic rename
    - Temp file in same directory as target
    """

    def test_uses_path_replace_for_atomic_rename(self, tmp_path: Path) -> None:
        """Test Path.replace is used for cross-platform atomic rename.

        Path.replace() (which wraps os.replace()) is atomic on both POSIX and
        Windows 3.3+. Unlike Path.rename(), it will replace the target file
        if it already exists.
        """
        target = tmp_path / "test.txt"

        # Path.replace() internally uses os.replace(), so we mock that
        with patch("os.replace") as mock_replace:
            write_atomic_bytes(target, b"test")

            mock_replace.assert_called_once()
            # Second arg should be the target path
            assert mock_replace.call_args[0][1] == target

    def test_temp_file_same_filesystem_as_target(self, tmp_path: Path) -> None:
        """Test temp file is created on same filesystem as target.

        This is critical for POSIX atomicity - rename across filesystems
        becomes a copy operation.
        """
        target = tmp_path / "subdir" / "test.txt"
        target.parent.mkdir(parents=True)

        original_mkstemp = tempfile.mkstemp
        captured_dir: Path | None = None

        def capturing_mkstemp(**kwargs: object) -> tuple[int, str]:
            nonlocal captured_dir
            captured_dir = kwargs.get("dir")  # type: ignore[assignment]
            return original_mkstemp(**kwargs)  # type: ignore[arg-type]

        with patch("tempfile.mkstemp", side_effect=capturing_mkstemp):
            write_atomic_bytes(target, b"test")

        assert captured_dir == target.parent
