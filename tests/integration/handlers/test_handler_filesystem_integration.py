# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for HandlerFileSystem.

This module tests HandlerFileSystem behavior under realistic conditions including:
- Circuit breaker recovery patterns
- Concurrent filesystem operations
- Permission and filesystem error handling

Test Categories:
    - TestCircuitBreakerRecovery: Circuit breaker state transitions
    - TestConcurrentOperations: Parallel read/write behavior
    - TestFilesystemErrors: Permission denied and read-only handling

Note on Permission Tests:
    Permission-based tests are skipped in the following environments:
    - Windows (no Unix permission model)
    - Running as root (root bypasses file permission checks)
    - Docker containers typically run as root by default
"""

from __future__ import annotations

import asyncio
import os
import stat
import sys
import tempfile
from collections.abc import AsyncGenerator
from pathlib import Path
from uuid import uuid4

import pytest

from omnibase_infra.errors import (
    InfraConnectionError,
    InfraUnavailableError,
)
from omnibase_infra.handlers.handler_filesystem import HandlerFileSystem

# ============================================================================
# Platform Detection
# ============================================================================

# Check if running as root - permissions don't work as expected for root
IS_ROOT = os.geteuid() == 0 if hasattr(os, "geteuid") else False
IS_WINDOWS = sys.platform == "win32"


def _permissions_are_enforced() -> bool:
    """Check if the filesystem actually enforces Unix permissions.

    Some environments (Docker volume mounts, certain filesystems) may not
    enforce permissions even when not running as root. This function tests
    actual permission enforcement.

    Returns:
        bool: True if permissions are enforced, False otherwise.
    """
    if IS_WINDOWS:
        return False
    if IS_ROOT:
        return False

    # Test actual permission enforcement
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "permission_test.txt"
            test_file.write_text("test")
            original_mode = test_file.stat().st_mode

            try:
                # Remove all permissions
                test_file.chmod(0o000)

                # Try to read - if this succeeds, permissions aren't enforced
                try:
                    test_file.read_text()
                    # Read succeeded despite chmod 000 - permissions not enforced
                    return False
                except PermissionError:
                    # PermissionError raised - permissions ARE enforced
                    return True
            finally:
                # Restore permissions for cleanup
                test_file.chmod(original_mode)
    except OSError:
        # Filesystem-related errors (permissions, missing files, etc.)
        # indicate permissions aren't reliably enforceable
        return False


# Check permission enforcement at module import time
PERMISSIONS_ENFORCED = _permissions_are_enforced()

# Combined skip condition for permission tests
SKIP_PERMISSION_TESTS = IS_WINDOWS or IS_ROOT or not PERMISSIONS_ENFORCED
SKIP_PERMISSION_REASON = (
    "Permission tests require Unix environment with enforced permissions "
    "(root bypasses checks, Windows lacks Unix permissions, "
    "Docker volume mounts may not enforce permissions)"
)


# ============================================================================
# Helper Functions
# ============================================================================


def create_read_envelope(path: str) -> dict[str, object]:
    """Create envelope for filesystem.read_file operation.

    Args:
        path: File path to read.

    Returns:
        Envelope dict for execute() method.
    """
    return {
        "id": str(uuid4()),
        "operation": "filesystem.read_file",
        "payload": {"path": path},
        "correlation_id": str(uuid4()),
    }


def create_write_envelope(
    path: str, content: str, create_dirs: bool = False
) -> dict[str, object]:
    """Create envelope for filesystem.write_file operation.

    Args:
        path: File path to write.
        content: Content to write.
        create_dirs: Whether to create parent directories.

    Returns:
        Envelope dict for execute() method.
    """
    return {
        "id": str(uuid4()),
        "operation": "filesystem.write_file",
        "payload": {
            "path": path,
            "content": content,
            "create_dirs": create_dirs,
        },
        "correlation_id": str(uuid4()),
    }


def create_list_envelope(path: str, pattern: str | None = None) -> dict[str, object]:
    """Create envelope for filesystem.list_directory operation.

    Args:
        path: Directory path to list.
        pattern: Optional glob pattern.

    Returns:
        Envelope dict for execute() method.
    """
    payload: dict[str, object] = {"path": path}
    if pattern:
        payload["pattern"] = pattern
    return {
        "id": str(uuid4()),
        "operation": "filesystem.list_directory",
        "payload": payload,
        "correlation_id": str(uuid4()),
    }


def create_delete_envelope(path: str, missing_ok: bool = False) -> dict[str, object]:
    """Create envelope for filesystem.delete_file operation.

    Args:
        path: File path to delete.
        missing_ok: Whether to ignore missing files.

    Returns:
        Envelope dict for execute() method.
    """
    return {
        "id": str(uuid4()),
        "operation": "filesystem.delete_file",
        "payload": {
            "path": path,
            "missing_ok": missing_ok,
        },
        "correlation_id": str(uuid4()),
    }


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_dir() -> Path:
    """Create a temporary directory for tests.

    Yields:
        Path: Temporary directory that exists for the test duration.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
async def handler(temp_dir: Path) -> AsyncGenerator[HandlerFileSystem, None]:
    """Create and initialize HandlerFileSystem with temp_dir as allowed path.

    Args:
        temp_dir: Temporary directory fixture.

    Yields:
        Initialized HandlerFileSystem with temp_dir allowed.
    """
    h = HandlerFileSystem()
    await h.initialize(
        {
            "allowed_paths": [str(temp_dir)],
            "max_read_size": 10 * 1024 * 1024,  # 10 MB
            "max_write_size": 10 * 1024 * 1024,  # 10 MB
        }
    )
    yield h
    await h.shutdown()


@pytest.fixture
async def handler_low_threshold(
    temp_dir: Path,
) -> AsyncGenerator[HandlerFileSystem, None]:
    """Create handler with low circuit breaker threshold for testing.

    Uses threshold=3 for faster circuit breaker testing.

    Args:
        temp_dir: Temporary directory fixture.

    Yields:
        Initialized HandlerFileSystem with low threshold.
    """
    h = HandlerFileSystem()
    # Initialize handler first
    await h.initialize(
        {
            "allowed_paths": [str(temp_dir)],
            "max_read_size": 10 * 1024 * 1024,
            "max_write_size": 10 * 1024 * 1024,
        }
    )
    # Override circuit breaker threshold for testing
    # This is done after initialize since init sets default threshold=5
    h.circuit_breaker_threshold = 3
    yield h
    await h.shutdown()


# ============================================================================
# TestCircuitBreakerRecovery
# ============================================================================


class TestCircuitBreakerRecovery:
    """Test circuit breaker behavior across multiple operations."""

    @pytest.mark.asyncio
    async def test_circuit_opens_after_threshold_failures(
        self, handler_low_threshold: HandlerFileSystem, temp_dir: Path
    ) -> None:
        """Verify circuit opens after consecutive failures reaching threshold.

        The circuit breaker should transition from CLOSED to OPEN after
        the configured number of consecutive failures (threshold=3 for this test).
        """
        handler = handler_low_threshold

        # Cause failures by reading non-existent files
        non_existent_base = temp_dir / "nonexistent"

        # First 3 failures should record but not open circuit
        for i in range(3):
            with pytest.raises(InfraConnectionError):
                await handler.execute(
                    create_read_envelope(str(non_existent_base / f"file_{i}.txt"))
                )

        # Circuit should now be open - next operation should raise InfraUnavailableError
        with pytest.raises(InfraUnavailableError) as exc_info:
            await handler.execute(
                create_read_envelope(str(non_existent_base / "file_4.txt"))
            )

        error_msg = str(exc_info.value).lower()
        assert "circuit breaker" in error_msg or "unavailable" in error_msg

    @pytest.mark.asyncio
    async def test_circuit_resets_on_success(
        self, handler_low_threshold: HandlerFileSystem, temp_dir: Path
    ) -> None:
        """Verify successful operation resets failure count.

        After some failures (but less than threshold), a successful operation
        should reset the failure counter to zero.
        """
        handler = handler_low_threshold

        # Cause 2 failures (below threshold of 3)
        non_existent = temp_dir / "nonexistent"
        for i in range(2):
            with pytest.raises(InfraConnectionError):
                await handler.execute(
                    create_read_envelope(str(non_existent / f"file_{i}.txt"))
                )

        # Verify failure count is 2
        assert handler._circuit_breaker_failures == 2

        # Create and read a valid file (successful operation)
        test_file = temp_dir / "valid_file.txt"
        test_file.write_text("test content")
        result = await handler.execute(create_read_envelope(str(test_file)))

        assert result.result["status"] == "success"
        # Failure count should be reset to 0
        assert handler._circuit_breaker_failures == 0

    @pytest.mark.asyncio
    async def test_half_open_state_transitions_to_closed_on_success(
        self, handler_low_threshold: HandlerFileSystem, temp_dir: Path
    ) -> None:
        """Verify HALF_OPEN state transitions to CLOSED on successful operation.

        When the reset timeout elapses, the circuit transitions to HALF_OPEN.
        A successful operation in HALF_OPEN state should close the circuit.
        """
        handler = handler_low_threshold

        # Open the circuit by causing threshold failures
        non_existent = temp_dir / "nonexistent"
        for i in range(3):
            with pytest.raises(InfraConnectionError):
                await handler.execute(
                    create_read_envelope(str(non_existent / f"file_{i}.txt"))
                )

        # Verify circuit is open
        assert handler._circuit_breaker_open is True

        # Manually set the open_until to past to simulate timeout elapsed
        handler._circuit_breaker_open_until = 0.0

        # Create a valid file for successful operation
        test_file = temp_dir / "recovery_test.txt"
        test_file.write_text("recovery content")

        # This should succeed and close the circuit
        result = await handler.execute(create_read_envelope(str(test_file)))

        assert result.result["status"] == "success"
        assert handler._circuit_breaker_open is False
        assert handler._circuit_breaker_failures == 0

    @pytest.mark.asyncio
    async def test_circuit_stays_open_until_timeout(
        self, handler_low_threshold: HandlerFileSystem, temp_dir: Path
    ) -> None:
        """Verify circuit stays open until reset timeout elapses.

        Operations should fail immediately with InfraUnavailableError
        while the circuit is open.
        """
        handler = handler_low_threshold

        # Open the circuit
        non_existent = temp_dir / "nonexistent"
        for i in range(3):
            with pytest.raises(InfraConnectionError):
                await handler.execute(
                    create_read_envelope(str(non_existent / f"file_{i}.txt"))
                )

        # Circuit should be open
        assert handler._circuit_breaker_open is True

        # Create a valid file
        test_file = temp_dir / "valid.txt"
        test_file.write_text("content")

        # Even valid operations should fail while circuit is open
        with pytest.raises(InfraUnavailableError):
            await handler.execute(create_read_envelope(str(test_file)))


# ============================================================================
# TestConcurrentOperations
# ============================================================================


class TestConcurrentOperations:
    """Test behavior under concurrent access."""

    @pytest.mark.asyncio
    async def test_concurrent_reads_same_file(
        self, handler: HandlerFileSystem, temp_dir: Path
    ) -> None:
        """Multiple concurrent reads of the same file should all succeed.

        This tests that the handler correctly handles concurrent read
        operations without data corruption or race conditions.
        """
        test_file = temp_dir / "concurrent_read.txt"
        test_content = "concurrent read test content"
        test_file.write_text(test_content)

        async def read_file() -> dict[str, object]:
            result = await handler.execute(create_read_envelope(str(test_file)))
            return result.result

        # Run 10 concurrent reads
        tasks = [read_file() for _ in range(10)]
        results = await asyncio.gather(*tasks)

        # All reads should succeed with correct content
        assert all(r["status"] == "success" for r in results)
        assert all(r["payload"]["content"] == test_content for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_writes_different_files(
        self, handler: HandlerFileSystem, temp_dir: Path
    ) -> None:
        """Concurrent writes to different files should all succeed.

        Each concurrent write targets a unique file to avoid conflicts.
        """

        async def write_file(index: int) -> dict[str, object]:
            path = temp_dir / f"concurrent_write_{index}.txt"
            content = f"content for file {index}"
            result = await handler.execute(create_write_envelope(str(path), content))
            return {"result": result.result, "index": index, "path": str(path)}

        # Run 10 concurrent writes to different files
        tasks = [write_file(i) for i in range(10)]
        results = await asyncio.gather(*tasks)

        # All writes should succeed
        assert all(r["result"]["status"] == "success" for r in results)

        # Verify all files exist with correct content
        for r in results:
            path = Path(r["path"])
            assert path.exists()
            assert path.read_text() == f"content for file {r['index']}"

    @pytest.mark.asyncio
    async def test_concurrent_reads_and_writes_different_files(
        self, handler: HandlerFileSystem, temp_dir: Path
    ) -> None:
        """Mixed concurrent reads and writes to different files should succeed."""
        # Create initial files for reading
        for i in range(5):
            (temp_dir / f"read_file_{i}.txt").write_text(f"read content {i}")

        async def read_file(index: int) -> dict[str, object]:
            path = temp_dir / f"read_file_{index}.txt"
            result = await handler.execute(create_read_envelope(str(path)))
            return {"type": "read", "result": result.result, "index": index}

        async def write_file(index: int) -> dict[str, object]:
            path = temp_dir / f"write_file_{index}.txt"
            result = await handler.execute(
                create_write_envelope(str(path), f"write content {index}")
            )
            return {"type": "write", "result": result.result, "index": index}

        # Mix reads and writes
        tasks = []
        for i in range(5):
            tasks.append(read_file(i))
            tasks.append(write_file(i))

        results = await asyncio.gather(*tasks)

        # All operations should succeed
        assert all(r["result"]["status"] == "success" for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_list_operations(
        self, handler: HandlerFileSystem, temp_dir: Path
    ) -> None:
        """Concurrent directory listing operations should all succeed."""
        # Create test files
        for i in range(10):
            (temp_dir / f"list_file_{i}.txt").write_text(f"content {i}")

        async def list_directory() -> dict[str, object]:
            result = await handler.execute(create_list_envelope(str(temp_dir)))
            return result.result

        # Run 5 concurrent list operations
        tasks = [list_directory() for _ in range(5)]
        results = await asyncio.gather(*tasks)

        # All list operations should succeed and return same count
        assert all(r["status"] == "success" for r in results)
        assert all(r["payload"]["count"] == 10 for r in results)


# ============================================================================
# TestFilesystemErrors
# ============================================================================


class TestFilesystemErrors:
    """Test error handling for filesystem conditions."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        SKIP_PERMISSION_TESTS,
        reason=SKIP_PERMISSION_REASON,
    )
    async def test_permission_denied_read(
        self, handler: HandlerFileSystem, temp_dir: Path
    ) -> None:
        """Verify proper error on permission denied during read.

        Creates a file, removes read permissions, then attempts to read.
        The handler should raise InfraConnectionError.
        """
        test_file = temp_dir / "no_read.txt"
        test_file.write_text("secret content")

        # Remove read permissions
        original_mode = test_file.stat().st_mode
        test_file.chmod(0o000)

        try:
            with pytest.raises(InfraConnectionError) as exc_info:
                await handler.execute(create_read_envelope(str(test_file)))

            error_msg = str(exc_info.value).lower()
            assert (
                "permission" in error_msg
                or "denied" in error_msg
                or "failed to read" in error_msg
            )
        finally:
            # Restore permissions for cleanup
            test_file.chmod(original_mode)

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        SKIP_PERMISSION_TESTS,
        reason=SKIP_PERMISSION_REASON,
    )
    async def test_write_to_readonly_directory(
        self, handler: HandlerFileSystem, temp_dir: Path
    ) -> None:
        """Verify proper error when directory is read-only.

        Creates a directory, removes write permissions, then attempts to write.
        The handler should raise InfraConnectionError.
        """
        readonly_dir = temp_dir / "readonly"
        readonly_dir.mkdir()

        # Remove write permissions from directory
        original_mode = readonly_dir.stat().st_mode
        readonly_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)

        try:
            with pytest.raises(InfraConnectionError) as exc_info:
                await handler.execute(
                    create_write_envelope(str(readonly_dir / "test.txt"), "content")
                )

            error_msg = str(exc_info.value).lower()
            assert (
                "permission" in error_msg
                or "denied" in error_msg
                or "failed to write" in error_msg
                or "read-only" in error_msg
            )
        finally:
            # Restore permissions for cleanup
            readonly_dir.chmod(original_mode)

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        SKIP_PERMISSION_TESTS,
        reason=SKIP_PERMISSION_REASON,
    )
    async def test_delete_file_without_permission(
        self, handler: HandlerFileSystem, temp_dir: Path
    ) -> None:
        """Verify proper error when delete is not permitted.

        Creates a file in a directory, then removes write permissions from directory.
        Deleting files requires write permission on the directory.
        """
        protected_dir = temp_dir / "protected"
        protected_dir.mkdir()
        test_file = protected_dir / "protected_file.txt"
        test_file.write_text("protected content")

        # Remove write permissions from directory (needed to delete files)
        original_mode = protected_dir.stat().st_mode
        protected_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)

        try:
            with pytest.raises(InfraConnectionError) as exc_info:
                await handler.execute(create_delete_envelope(str(test_file)))

            error_msg = str(exc_info.value).lower()
            assert (
                "permission" in error_msg
                or "denied" in error_msg
                or "failed to delete" in error_msg
            )
        finally:
            # Restore permissions for cleanup
            protected_dir.chmod(original_mode)

    @pytest.mark.asyncio
    async def test_read_directory_as_file_raises_error(
        self, handler: HandlerFileSystem, temp_dir: Path
    ) -> None:
        """Verify proper error when attempting to read a directory as a file."""
        subdir = temp_dir / "subdir"
        subdir.mkdir()

        with pytest.raises(InfraConnectionError) as exc_info:
            await handler.execute(create_read_envelope(str(subdir)))

        error_msg = str(exc_info.value).lower()
        assert "not a file" in error_msg or "is not a file" in error_msg

    @pytest.mark.asyncio
    async def test_list_file_as_directory_raises_error(
        self, handler: HandlerFileSystem, temp_dir: Path
    ) -> None:
        """Verify proper error when attempting to list a file as a directory."""
        test_file = temp_dir / "not_a_dir.txt"
        test_file.write_text("content")

        with pytest.raises(InfraConnectionError) as exc_info:
            await handler.execute(create_list_envelope(str(test_file)))

        error_msg = str(exc_info.value).lower()
        assert (
            "not a directory" in error_msg
            or "is not a directory" in error_msg
            or "not found" in error_msg
        )

    @pytest.mark.asyncio
    async def test_delete_directory_as_file_raises_error(
        self, handler: HandlerFileSystem, temp_dir: Path
    ) -> None:
        """Verify proper error when attempting to delete a directory with delete_file."""
        subdir = temp_dir / "delete_me_dir"
        subdir.mkdir()

        with pytest.raises(InfraConnectionError) as exc_info:
            await handler.execute(create_delete_envelope(str(subdir)))

        error_msg = str(exc_info.value).lower()
        assert (
            "directory" in error_msg
            or "rmdir" in error_msg
            or "is a directory" in error_msg
        )


# ============================================================================
# TestFilesystemRecoveryScenarios
# ============================================================================


class TestFilesystemRecoveryScenarios:
    """Test recovery scenarios after transient errors."""

    @pytest.mark.asyncio
    async def test_recovery_after_file_created(
        self, handler: HandlerFileSystem, temp_dir: Path
    ) -> None:
        """Test that reads succeed after file is created.

        Simulates a scenario where a file doesn't exist initially
        but is created by another process, then read succeeds.
        """
        test_file = temp_dir / "eventual_file.txt"

        # First read should fail
        with pytest.raises(InfraConnectionError):
            await handler.execute(create_read_envelope(str(test_file)))

        # File is created (simulating external process)
        test_file.write_text("eventually available")

        # Second read should succeed
        result = await handler.execute(create_read_envelope(str(test_file)))
        assert result.result["status"] == "success"
        assert result.result["payload"]["content"] == "eventually available"

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        SKIP_PERMISSION_TESTS,
        reason=SKIP_PERMISSION_REASON,
    )
    async def test_recovery_after_permissions_restored(
        self, handler: HandlerFileSystem, temp_dir: Path
    ) -> None:
        """Test that operations succeed after permissions are restored.

        Simulates temporary permission issues that are later resolved.
        """
        test_file = temp_dir / "temp_locked.txt"
        test_file.write_text("locked content")
        original_mode = test_file.stat().st_mode

        # Remove permissions
        test_file.chmod(0o000)

        # Read should fail
        try:
            with pytest.raises(InfraConnectionError):
                await handler.execute(create_read_envelope(str(test_file)))
        finally:
            # Restore permissions
            test_file.chmod(original_mode)

        # Now read should succeed
        result = await handler.execute(create_read_envelope(str(test_file)))
        assert result.result["status"] == "success"
        assert result.result["payload"]["content"] == "locked content"


# ============================================================================
# TestCircuitBreakerIntegrationWithFileOps
# ============================================================================


class TestCircuitBreakerIntegrationWithFileOps:
    """Test circuit breaker integration with various file operations."""

    @pytest.mark.asyncio
    async def test_write_failures_affect_circuit_breaker(
        self, handler_low_threshold: HandlerFileSystem, temp_dir: Path
    ) -> None:
        """Verify write failures contribute to circuit breaker failure count.

        Different operation types should all contribute to the same circuit breaker.
        """
        handler = handler_low_threshold

        # Try to write to non-existent nested directory without create_dirs
        for i in range(3):
            nested_path = temp_dir / f"nonexistent_{i}" / f"deep_{i}" / "file.txt"
            with pytest.raises(InfraConnectionError):
                await handler.execute(
                    create_write_envelope(
                        str(nested_path), "content", create_dirs=False
                    )
                )

        # Circuit should now be open
        assert handler._circuit_breaker_open is True

    @pytest.mark.asyncio
    async def test_mixed_operation_failures_accumulate(
        self, handler_low_threshold: HandlerFileSystem, temp_dir: Path
    ) -> None:
        """Verify mixed operation failures accumulate in circuit breaker.

        Failures from different operations (read, list, delete) should
        all contribute to the same failure counter.
        """
        handler = handler_low_threshold

        # Mix of different failing operations
        non_existent_file = temp_dir / "no_such_file.txt"
        non_existent_dir = temp_dir / "no_such_dir"

        # Failure 1: read non-existent file
        with pytest.raises(InfraConnectionError):
            await handler.execute(create_read_envelope(str(non_existent_file)))

        # Failure 2: list non-existent directory
        with pytest.raises(InfraConnectionError):
            await handler.execute(create_list_envelope(str(non_existent_dir)))

        # Failure 3: delete non-existent file (missing_ok=False)
        with pytest.raises(InfraConnectionError):
            await handler.execute(create_delete_envelope(str(non_existent_file)))

        # Circuit should now be open after 3 failures
        assert handler._circuit_breaker_open is True

        # Any operation should now fail with InfraUnavailableError
        valid_file = temp_dir / "valid.txt"
        valid_file.write_text("content")
        with pytest.raises(InfraUnavailableError):
            await handler.execute(create_read_envelope(str(valid_file)))
