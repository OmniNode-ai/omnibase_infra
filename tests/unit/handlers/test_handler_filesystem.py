# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
# ruff: noqa: S108
# S108 disabled: /tmp paths are intentional for test fixtures
# mypy: disable-error-code="index, operator, arg-type"
"""Unit tests for HandlerFileSystem.

Comprehensive test suite covering initialization, file operations,
directory operations, security validation, error handling, and lifecycle management.

Test Classes:
    - TestHandlerFileSystemInitialization: Handler initialization and configuration
    - TestHandlerFileSystemReadFile: File reading operations
    - TestHandlerFileSystemWriteFile: File writing operations
    - TestHandlerFileSystemListDirectory: Directory listing operations
    - TestHandlerFileSystemEnsureDirectory: Directory creation operations
    - TestHandlerFileSystemDeleteFile: File deletion operations
    - TestHandlerFileSystemDescribe: Handler metadata and introspection
    - TestHandlerFileSystemSecurityValidation: Security and path validation
    - TestHandlerFileSystemLifecycle: Handler lifecycle management
    - TestHandlerFileSystemCorrelationId: Correlation ID handling
"""

from __future__ import annotations

import base64
import tempfile
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumInfraTransportType,
)
from omnibase_infra.errors import (
    InfraConnectionError,
    InfraUnavailableError,
    ProtocolConfigurationError,
    RuntimeHostError,
)
from tests.helpers import DeterministicIdGenerator

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def temp_dir() -> Path:
    """Create a temporary directory for tests.

    Yields:
        Path: Temporary directory path that exists for the duration of the test.
               The path is resolved to canonical form (on macOS, /var -> /private/var)
               to ensure consistency with handler symlink validation.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Resolve to canonical path for macOS compatibility
        # /var/folders/... -> /private/var/folders/...
        yield Path(tmpdir).resolve()


@pytest.fixture
def handler():
    """Create HandlerFileSystem fixture.

    Returns:
        HandlerFileSystem: A new, uninitialized handler instance.
    """
    # Import here to allow tests to run even if handler not yet implemented
    from omnibase_infra.handlers.handler_filesystem import HandlerFileSystem

    return HandlerFileSystem()


@pytest.fixture
async def initialized_handler(handler, temp_dir: Path):
    """Create and initialize HandlerFileSystem with temp_dir as allowed path.

    Args:
        handler: The HandlerFileSystem fixture
        temp_dir: Temporary directory fixture (already resolved to canonical form)

    Yields:
        HandlerFileSystem: An initialized handler with temp_dir allowed.
    """
    # temp_dir fixture already resolves to canonical path
    await handler.initialize(
        {
            "allowed_paths": [str(temp_dir)],
            "max_read_size": 1024 * 1024,  # 1 MB for tests
            "max_write_size": 1024 * 1024,  # 1 MB for tests
        }
    )
    yield handler
    await handler.shutdown()


# ============================================================================
# TestHandlerFileSystemInitialization
# ============================================================================


class TestHandlerFileSystemInitialization:
    """Test suite for HandlerFileSystem initialization."""

    @pytest.fixture
    def handler(self):
        """Create HandlerFileSystem fixture."""
        from omnibase_infra.handlers.handler_filesystem import HandlerFileSystem

        return HandlerFileSystem()

    def test_handler_init_default_state(self, handler) -> None:
        """Test handler initializes in uninitialized state."""
        assert handler._initialized is False

    def test_handler_type_returns_infra_handler(self, handler) -> None:
        """Test handler_type property returns EnumHandlerType.INFRA_HANDLER."""
        assert handler.handler_type == EnumHandlerType.INFRA_HANDLER

    def test_handler_category_returns_effect(self, handler) -> None:
        """Test handler_category property returns EnumHandlerTypeCategory.EFFECT.

        The handler_category property identifies the behavioral classification.
        EFFECT indicates this handler performs side-effecting I/O operations
        (filesystem read/write).
        """
        assert handler.handler_category == EnumHandlerTypeCategory.EFFECT

    def test_transport_type_returns_filesystem(self, handler) -> None:
        """Test transport_type property returns EnumInfraTransportType.FILESYSTEM."""
        assert handler.transport_type == EnumInfraTransportType.FILESYSTEM

    @pytest.mark.asyncio
    async def test_initialize_with_empty_config_raises_error(self, handler) -> None:
        """Test handler raises error with empty config (requires allowed_paths)."""
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            await handler.initialize({})

        assert "allowed_paths" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_initialize_with_allowed_paths_config(
        self, handler, temp_dir: Path
    ) -> None:
        """Test handler initializes with allowed_paths configuration."""
        config: dict[str, object] = {
            "allowed_paths": [str(temp_dir), "/tmp/other"],
        }
        await handler.initialize(config)

        assert handler._initialized is True
        # _allowed_paths is a list of Path objects resolved to canonical form
        # On macOS, /var is a symlink to /private/var, so we must resolve
        # the temp_dir before comparing
        assert temp_dir.resolve() in handler._allowed_paths

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_initialize_with_size_limit_config(
        self, handler, temp_dir: Path
    ) -> None:
        """Test handler initializes with size limit configuration."""
        config: dict[str, object] = {
            "allowed_paths": [str(temp_dir)],
            "max_read_size": 1024,  # 1 KB
            "max_write_size": 2048,  # 2 KB
        }
        await handler.initialize(config)

        assert handler._initialized is True
        assert handler._max_read_size == 1024
        assert handler._max_write_size == 2048

        await handler.shutdown()


# ============================================================================
# TestHandlerFileSystemReadFile
# ============================================================================


class TestHandlerFileSystemReadFile:
    """Test suite for file reading operations."""

    @pytest.mark.asyncio
    async def test_read_file_returns_content(
        self, initialized_handler, temp_dir: Path
    ) -> None:
        """Test read_file returns file content."""
        # Arrange
        test_file = temp_dir / "test.txt"
        test_file.write_text("Hello, World!")

        envelope: dict[str, object] = {
            "operation": "filesystem.read_file",
            "payload": {"path": str(test_file)},
            "correlation_id": str(uuid4()),
        }

        # Act
        result = await initialized_handler.execute(envelope)

        # Assert
        assert result.result["status"] == "success"
        assert result.result["payload"]["content"] == "Hello, World!"

    @pytest.mark.asyncio
    async def test_read_file_with_binary_mode(
        self, initialized_handler, temp_dir: Path
    ) -> None:
        """Test read_file returns binary content when mode is binary."""
        # Arrange
        test_file = temp_dir / "binary.bin"
        binary_content = b"\x00\x01\x02\x03\x04"
        test_file.write_bytes(binary_content)

        envelope: dict[str, object] = {
            "operation": "filesystem.read_file",
            "payload": {"path": str(test_file), "binary": True},
            "correlation_id": str(uuid4()),
        }

        # Act
        result = await initialized_handler.execute(envelope)

        # Assert
        assert result.result["status"] == "success"
        # Binary content may be returned as base64 or raw bytes depending on impl
        assert result.result["payload"]["content"] is not None

    @pytest.mark.asyncio
    async def test_read_file_with_custom_encoding(
        self, initialized_handler, temp_dir: Path
    ) -> None:
        """Test read_file with custom encoding."""
        # Arrange
        test_file = temp_dir / "utf16.txt"
        test_content = "Hello UTF-16"
        test_file.write_text(test_content, encoding="utf-16")

        envelope: dict[str, object] = {
            "operation": "filesystem.read_file",
            "payload": {"path": str(test_file), "encoding": "utf-16"},
            "correlation_id": str(uuid4()),
        }

        # Act
        result = await initialized_handler.execute(envelope)

        # Assert
        assert result.result["status"] == "success"
        assert result.result["payload"]["content"] == test_content

    @pytest.mark.asyncio
    async def test_read_file_raises_error_for_path_outside_whitelist(
        self, initialized_handler
    ) -> None:
        """Test read_file raises error for path outside allowed paths."""
        envelope: dict[str, object] = {
            "operation": "filesystem.read_file",
            "payload": {"path": "/etc/passwd"},  # Outside allowed paths
            "correlation_id": str(uuid4()),
        }

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            await initialized_handler.execute(envelope)

        error_msg = str(exc_info.value).lower()
        assert "path" in error_msg or "allowed" in error_msg or "whitelist" in error_msg

    @pytest.mark.asyncio
    async def test_read_file_raises_error_for_file_too_large(
        self, handler, temp_dir: Path
    ) -> None:
        """Test read_file raises error for file exceeding size limit."""
        # Initialize with small size limit
        await handler.initialize(
            {
                "allowed_paths": [str(temp_dir)],
                "max_read_size": 10,  # 10 bytes
            }
        )

        # Create file larger than limit
        test_file = temp_dir / "large.txt"
        test_file.write_text("This content exceeds the 10 byte limit")

        envelope: dict[str, object] = {
            "operation": "filesystem.read_file",
            "payload": {"path": str(test_file)},
            "correlation_id": str(uuid4()),
        }

        with pytest.raises(InfraUnavailableError) as exc_info:
            await handler.execute(envelope)

        assert (
            "exceeds" in str(exc_info.value).lower()
            or "limit" in str(exc_info.value).lower()
        )

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_read_file_raises_error_for_symlink_escaping_allowed_paths(
        self, initialized_handler, temp_dir: Path
    ) -> None:
        """Test read_file raises error for symlink escaping allowed paths."""
        # Create a symlink pointing outside allowed paths
        symlink_path = temp_dir / "escape_link"
        try:
            symlink_path.symlink_to("/etc/passwd")
        except OSError:
            pytest.skip("Cannot create symlinks on this system")

        envelope: dict[str, object] = {
            "operation": "filesystem.read_file",
            "payload": {"path": str(symlink_path)},
            "correlation_id": str(uuid4()),
        }

        with pytest.raises(
            (ProtocolConfigurationError, InfraConnectionError)
        ) as exc_info:
            await initialized_handler.execute(envelope)

        error_msg = str(exc_info.value).lower()
        assert (
            "symlink" in error_msg
            or "path" in error_msg
            or "escape" in error_msg
            or "allowed" in error_msg
        )

    @pytest.mark.asyncio
    async def test_read_file_raises_error_for_non_existent_file(
        self, initialized_handler, temp_dir: Path
    ) -> None:
        """Test read_file raises error for non-existent file."""
        non_existent = temp_dir / "does_not_exist.txt"

        envelope: dict[str, object] = {
            "operation": "filesystem.read_file",
            "payload": {"path": str(non_existent)},
            "correlation_id": str(uuid4()),
        }

        with pytest.raises(InfraConnectionError) as exc_info:
            await initialized_handler.execute(envelope)

        error_msg = str(exc_info.value).lower()
        assert (
            "not found" in error_msg or "exist" in error_msg or "no such" in error_msg
        )


# ============================================================================
# TestHandlerFileSystemWriteFile
# ============================================================================


class TestHandlerFileSystemWriteFile:
    """Test suite for file writing operations."""

    @pytest.mark.asyncio
    async def test_write_file_creates_file(
        self, initialized_handler, temp_dir: Path
    ) -> None:
        """Test write_file creates file with content."""
        test_file = temp_dir / "new_file.txt"
        content = "New file content"

        envelope: dict[str, object] = {
            "operation": "filesystem.write_file",
            "payload": {"path": str(test_file), "content": content},
            "correlation_id": str(uuid4()),
        }

        # Act
        result = await initialized_handler.execute(envelope)

        # Assert
        assert result.result["status"] == "success"
        assert test_file.exists()
        assert test_file.read_text() == content

    @pytest.mark.asyncio
    async def test_write_file_with_binary_content(
        self, initialized_handler, temp_dir: Path
    ) -> None:
        """Test write_file with binary content using base64 encoding."""
        test_file = temp_dir / "binary_out.bin"
        # Original binary content
        binary_content = b"\x00\x01\x02\x03"
        # Encode as base64 string for the handler
        base64_content = base64.b64encode(binary_content).decode("ascii")

        envelope: dict[str, object] = {
            "operation": "filesystem.write_file",
            "payload": {
                "path": str(test_file),
                "content": base64_content,  # Base64-encoded string
                "binary": True,  # Binary mode - handler decodes base64
            },
            "correlation_id": str(uuid4()),
        }

        # Act
        result = await initialized_handler.execute(envelope)

        # Assert
        assert result.result["status"] == "success"
        assert test_file.exists()
        # Verify the file contains the original binary content
        assert test_file.read_bytes() == binary_content

    @pytest.mark.asyncio
    async def test_write_file_with_create_dirs_true(
        self, initialized_handler, temp_dir: Path
    ) -> None:
        """Test write_file with create_dirs=True creates parent directories."""
        # Path with non-existent parent directories
        test_file = temp_dir / "subdir1" / "subdir2" / "nested_file.txt"
        content = "Nested content"

        envelope: dict[str, object] = {
            "operation": "filesystem.write_file",
            "payload": {
                "path": str(test_file),
                "content": content,
                "create_dirs": True,
            },
            "correlation_id": str(uuid4()),
        }

        # Act
        result = await initialized_handler.execute(envelope)

        # Assert
        assert result.result["status"] == "success"
        assert test_file.exists()
        assert test_file.read_text() == content

    @pytest.mark.asyncio
    async def test_write_file_raises_error_for_path_outside_whitelist(
        self, initialized_handler
    ) -> None:
        """Test write_file raises error for path outside allowed paths."""
        envelope: dict[str, object] = {
            "operation": "filesystem.write_file",
            "payload": {"path": "/etc/malicious.txt", "content": "bad"},
            "correlation_id": str(uuid4()),
        }

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            await initialized_handler.execute(envelope)

        error_msg = str(exc_info.value).lower()
        assert "path" in error_msg or "allowed" in error_msg or "whitelist" in error_msg

    @pytest.mark.asyncio
    async def test_write_file_raises_error_for_content_too_large(
        self, handler, temp_dir: Path
    ) -> None:
        """Test write_file raises error for content exceeding size limit."""
        # Initialize with small size limit
        await handler.initialize(
            {
                "allowed_paths": [str(temp_dir)],
                "max_write_size": 10,  # 10 bytes
            }
        )

        test_file = temp_dir / "large_write.txt"
        large_content = "This content exceeds the 10 byte limit"

        envelope: dict[str, object] = {
            "operation": "filesystem.write_file",
            "payload": {"path": str(test_file), "content": large_content},
            "correlation_id": str(uuid4()),
        }

        with pytest.raises(InfraUnavailableError) as exc_info:
            await handler.execute(envelope)

        assert (
            "exceeds" in str(exc_info.value).lower()
            or "limit" in str(exc_info.value).lower()
        )

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_write_file_raises_error_for_symlink_escaping_allowed_paths(
        self, initialized_handler, temp_dir: Path
    ) -> None:
        """Test write_file raises error for symlink escaping allowed paths."""
        # Create a symlink pointing to a directory outside allowed paths
        symlink_dir = temp_dir / "escape_dir"
        try:
            symlink_dir.symlink_to("/tmp")
        except OSError:
            pytest.skip("Cannot create symlinks on this system")

        target_file = symlink_dir / "escaped_file.txt"

        envelope: dict[str, object] = {
            "operation": "filesystem.write_file",
            "payload": {"path": str(target_file), "content": "bad"},
            "correlation_id": str(uuid4()),
        }

        with pytest.raises(
            (ProtocolConfigurationError, InfraConnectionError)
        ) as exc_info:
            await initialized_handler.execute(envelope)

        error_msg = str(exc_info.value).lower()
        assert (
            "symlink" in error_msg
            or "path" in error_msg
            or "escape" in error_msg
            or "allowed" in error_msg
        )

    @pytest.mark.asyncio
    async def test_write_file_rejects_symlink_via_o_nofollow(
        self, initialized_handler, temp_dir: Path
    ) -> None:
        """Test write_file rejects writing to symlink using O_NOFOLLOW.

        This tests the TOCTOU race condition fix. The handler uses O_NOFOLLOW
        flag when opening files, which atomically rejects symlinks at open time
        rather than checking is_symlink() first (which could be raced).

        The symlink points to a valid file within allowed paths, but the write
        is still rejected because we don't follow symlinks during write operations.
        """
        # Create a real file within allowed paths
        real_file = temp_dir / "real_file.txt"
        real_file.write_text("original content")

        # Create a symlink pointing to the real file (within allowed paths)
        symlink_path = temp_dir / "symlink_to_real"
        try:
            symlink_path.symlink_to(real_file)
        except OSError:
            pytest.skip("Cannot create symlinks on this system")

        envelope: dict[str, object] = {
            "operation": "filesystem.write_file",
            "payload": {"path": str(symlink_path), "content": "new content"},
            "correlation_id": str(uuid4()),
        }

        # Should reject the write because the path is a symlink
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            await initialized_handler.execute(envelope)

        error_msg = str(exc_info.value).lower()
        assert "symlink" in error_msg

        # Verify the original file was not modified
        assert real_file.read_text() == "original content"

    @pytest.mark.asyncio
    async def test_write_file_binary_rejects_symlink_via_o_nofollow(
        self, initialized_handler, temp_dir: Path
    ) -> None:
        """Test write_file in binary mode also rejects symlinks using O_NOFOLLOW."""
        # Create a real file within allowed paths
        real_file = temp_dir / "real_binary.bin"
        original_content = b"\x00\x01\x02\x03"
        real_file.write_bytes(original_content)

        # Create a symlink pointing to the real file (within allowed paths)
        symlink_path = temp_dir / "symlink_to_binary"
        try:
            symlink_path.symlink_to(real_file)
        except OSError:
            pytest.skip("Cannot create symlinks on this system")

        # Base64-encoded new content
        import base64

        new_content_b64 = base64.b64encode(b"\xff\xfe\xfd").decode("ascii")

        envelope: dict[str, object] = {
            "operation": "filesystem.write_file",
            "payload": {
                "path": str(symlink_path),
                "content": new_content_b64,
                "binary": True,
            },
            "correlation_id": str(uuid4()),
        }

        # Should reject the write because the path is a symlink
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            await initialized_handler.execute(envelope)

        error_msg = str(exc_info.value).lower()
        assert "symlink" in error_msg

        # Verify the original file was not modified
        assert real_file.read_bytes() == original_content


# ============================================================================
# TestHandlerFileSystemListDirectory
# ============================================================================


class TestHandlerFileSystemListDirectory:
    """Test suite for directory listing operations."""

    @pytest.mark.asyncio
    async def test_list_directory_returns_entries(
        self, initialized_handler, temp_dir: Path
    ) -> None:
        """Test list_directory returns directory entries."""
        # Create test files
        (temp_dir / "file1.txt").write_text("content1")
        (temp_dir / "file2.txt").write_text("content2")
        (temp_dir / "subdir").mkdir()

        envelope: dict[str, object] = {
            "operation": "filesystem.list_directory",
            "payload": {"path": str(temp_dir)},
            "correlation_id": str(uuid4()),
        }

        # Act
        result = await initialized_handler.execute(envelope)

        # Assert
        assert result.result["status"] == "success"
        entries = result.result["payload"]["entries"]
        entry_names = [e["name"] for e in entries]
        assert "file1.txt" in entry_names
        assert "file2.txt" in entry_names
        assert "subdir" in entry_names

    @pytest.mark.asyncio
    async def test_list_directory_with_recursive_true(
        self, initialized_handler, temp_dir: Path
    ) -> None:
        """Test list_directory with recursive=True returns nested entries."""
        # Create nested structure
        (temp_dir / "file1.txt").write_text("content1")
        subdir = temp_dir / "subdir"
        subdir.mkdir()
        (subdir / "nested_file.txt").write_text("nested content")

        envelope: dict[str, object] = {
            "operation": "filesystem.list_directory",
            "payload": {"path": str(temp_dir), "recursive": True},
            "correlation_id": str(uuid4()),
        }

        # Act
        result = await initialized_handler.execute(envelope)

        # Assert
        assert result.result["status"] == "success"
        entries = result.result["payload"]["entries"]
        # Should include nested file
        paths = [e.get("path", e.get("name", "")) for e in entries]
        assert any("nested_file.txt" in p for p in paths)

    @pytest.mark.asyncio
    async def test_list_directory_with_glob_pattern(
        self, initialized_handler, temp_dir: Path
    ) -> None:
        """Test list_directory with glob pattern filters results."""
        # Create test files
        (temp_dir / "file1.txt").write_text("content1")
        (temp_dir / "file2.py").write_text("content2")
        (temp_dir / "file3.txt").write_text("content3")

        envelope: dict[str, object] = {
            "operation": "filesystem.list_directory",
            "payload": {"path": str(temp_dir), "pattern": "*.txt"},
            "correlation_id": str(uuid4()),
        }

        # Act
        result = await initialized_handler.execute(envelope)

        # Assert
        assert result.result["status"] == "success"
        entries = result.result["payload"]["entries"]
        entry_names = [e["name"] for e in entries]
        assert "file1.txt" in entry_names
        assert "file3.txt" in entry_names
        assert "file2.py" not in entry_names

    @pytest.mark.asyncio
    async def test_list_directory_raises_error_for_path_outside_whitelist(
        self, initialized_handler
    ) -> None:
        """Test list_directory raises error for path outside allowed paths."""
        envelope: dict[str, object] = {
            "operation": "filesystem.list_directory",
            "payload": {"path": "/etc"},
            "correlation_id": str(uuid4()),
        }

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            await initialized_handler.execute(envelope)

        error_msg = str(exc_info.value).lower()
        assert "path" in error_msg or "allowed" in error_msg or "whitelist" in error_msg

    @pytest.mark.asyncio
    async def test_list_directory_raises_error_for_non_existent_directory(
        self, initialized_handler, temp_dir: Path
    ) -> None:
        """Test list_directory raises error for non-existent directory."""
        non_existent = temp_dir / "does_not_exist"

        envelope: dict[str, object] = {
            "operation": "filesystem.list_directory",
            "payload": {"path": str(non_existent)},
            "correlation_id": str(uuid4()),
        }

        with pytest.raises(InfraConnectionError) as exc_info:
            await initialized_handler.execute(envelope)

        error_msg = str(exc_info.value).lower()
        assert (
            "not found" in error_msg or "exist" in error_msg or "no such" in error_msg
        )


# ============================================================================
# TestHandlerFileSystemEnsureDirectory
# ============================================================================


class TestHandlerFileSystemEnsureDirectory:
    """Test suite for directory creation operations."""

    @pytest.mark.asyncio
    async def test_ensure_directory_creates_directory(
        self, initialized_handler, temp_dir: Path
    ) -> None:
        """Test ensure_directory creates a new directory."""
        new_dir = temp_dir / "new_directory"

        envelope: dict[str, object] = {
            "operation": "filesystem.ensure_directory",
            "payload": {"path": str(new_dir)},
            "correlation_id": str(uuid4()),
        }

        # Act
        result = await initialized_handler.execute(envelope)

        # Assert
        assert result.result["status"] == "success"
        assert new_dir.exists()
        assert new_dir.is_dir()

    @pytest.mark.asyncio
    async def test_ensure_directory_with_exist_ok_true(
        self, initialized_handler, temp_dir: Path
    ) -> None:
        """Test ensure_directory with exist_ok=True on existing directory."""
        existing_dir = temp_dir / "existing"
        existing_dir.mkdir()

        envelope: dict[str, object] = {
            "operation": "filesystem.ensure_directory",
            "payload": {"path": str(existing_dir), "exist_ok": True},
            "correlation_id": str(uuid4()),
        }

        # Act - should not raise
        result = await initialized_handler.execute(envelope)

        # Assert
        assert result.result["status"] == "success"
        assert result.result["payload"]["already_existed"] is True

    @pytest.mark.asyncio
    async def test_ensure_directory_with_exist_ok_false_raises_for_existing(
        self, initialized_handler, temp_dir: Path
    ) -> None:
        """Test ensure_directory with exist_ok=False raises for existing directory."""
        existing_dir = temp_dir / "already_exists"
        existing_dir.mkdir()

        envelope: dict[str, object] = {
            "operation": "filesystem.ensure_directory",
            "payload": {"path": str(existing_dir), "exist_ok": False},
            "correlation_id": str(uuid4()),
        }

        # Act - should raise error for existing directory
        with pytest.raises(InfraConnectionError) as exc_info:
            await initialized_handler.execute(envelope)

        # Assert
        assert "exists" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_ensure_directory_raises_error_for_path_outside_whitelist(
        self, initialized_handler
    ) -> None:
        """Test ensure_directory raises error for path outside allowed paths."""
        envelope: dict[str, object] = {
            "operation": "filesystem.ensure_directory",
            "payload": {"path": "/etc/malicious_dir"},
            "correlation_id": str(uuid4()),
        }

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            await initialized_handler.execute(envelope)

        error_msg = str(exc_info.value).lower()
        assert "path" in error_msg or "allowed" in error_msg or "whitelist" in error_msg

    @pytest.mark.asyncio
    async def test_ensure_directory_reports_already_existed_correctly(
        self, initialized_handler, temp_dir: Path
    ) -> None:
        """Test ensure_directory reports already_existed status correctly."""
        # Test with new directory
        new_dir = temp_dir / "brand_new"
        envelope_new: dict[str, object] = {
            "operation": "filesystem.ensure_directory",
            "payload": {"path": str(new_dir), "exist_ok": True},
            "correlation_id": str(uuid4()),
        }

        result_new = await initialized_handler.execute(envelope_new)
        assert result_new.result["payload"]["already_existed"] is False

        # Test with existing directory
        envelope_existing: dict[str, object] = {
            "operation": "filesystem.ensure_directory",
            "payload": {"path": str(new_dir), "exist_ok": True},
            "correlation_id": str(uuid4()),
        }

        result_existing = await initialized_handler.execute(envelope_existing)
        assert result_existing.result["payload"]["already_existed"] is True


# ============================================================================
# TestHandlerFileSystemDeleteFile
# ============================================================================


class TestHandlerFileSystemDeleteFile:
    """Test suite for file deletion operations."""

    @pytest.mark.asyncio
    async def test_delete_file_removes_file(
        self, initialized_handler, temp_dir: Path
    ) -> None:
        """Test delete_file removes existing file."""
        # Create file to delete
        test_file = temp_dir / "to_delete.txt"
        test_file.write_text("delete me")

        envelope: dict[str, object] = {
            "operation": "filesystem.delete_file",
            "payload": {"path": str(test_file)},
            "correlation_id": str(uuid4()),
        }

        # Act
        result = await initialized_handler.execute(envelope)

        # Assert
        assert result.result["status"] == "success"
        assert not test_file.exists()

    @pytest.mark.asyncio
    async def test_delete_file_with_missing_ok_true_for_non_existent(
        self, initialized_handler, temp_dir: Path
    ) -> None:
        """Test delete_file with missing_ok=True for non-existent file."""
        non_existent = temp_dir / "does_not_exist.txt"

        envelope: dict[str, object] = {
            "operation": "filesystem.delete_file",
            "payload": {"path": str(non_existent), "missing_ok": True},
            "correlation_id": str(uuid4()),
        }

        # Act - should not raise
        result = await initialized_handler.execute(envelope)

        # Assert
        assert result.result["status"] == "success"

    @pytest.mark.asyncio
    async def test_delete_file_raises_error_for_path_outside_whitelist(
        self, initialized_handler
    ) -> None:
        """Test delete_file raises error for path outside allowed paths."""
        envelope: dict[str, object] = {
            "operation": "filesystem.delete_file",
            "payload": {"path": "/etc/passwd"},
            "correlation_id": str(uuid4()),
        }

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            await initialized_handler.execute(envelope)

        error_msg = str(exc_info.value).lower()
        assert "path" in error_msg or "allowed" in error_msg or "whitelist" in error_msg

    @pytest.mark.asyncio
    async def test_delete_file_raises_error_for_missing_file_without_missing_ok(
        self, initialized_handler, temp_dir: Path
    ) -> None:
        """Test delete_file raises error for missing file without missing_ok."""
        non_existent = temp_dir / "does_not_exist.txt"

        envelope: dict[str, object] = {
            "operation": "filesystem.delete_file",
            "payload": {"path": str(non_existent)},  # missing_ok defaults to False
            "correlation_id": str(uuid4()),
        }

        with pytest.raises(InfraConnectionError) as exc_info:
            await initialized_handler.execute(envelope)

        error_msg = str(exc_info.value).lower()
        assert (
            "not found" in error_msg or "exist" in error_msg or "no such" in error_msg
        )


# ============================================================================
# TestHandlerFileSystemDescribe
# ============================================================================


class TestHandlerFileSystemDescribe:
    """Test suite for describe operations and three-dimensional handler type system."""

    @pytest.fixture
    def handler(self):
        """Create HandlerFileSystem fixture."""
        from omnibase_infra.handlers.handler_filesystem import HandlerFileSystem

        return HandlerFileSystem()

    def test_describe_returns_expected_metadata(self, handler) -> None:
        """Test describe returns expected metadata fields."""
        description = handler.describe()

        # Three-dimensional handler type system
        assert "handler_type" in description
        assert "handler_category" in description
        assert "transport_type" in description

        # Standard metadata
        assert "initialized" in description
        assert "version" in description

    def test_describe_includes_all_supported_operations(self, handler) -> None:
        """Test describe includes all 5 supported operations."""
        description = handler.describe()

        assert "supported_operations" in description
        operations = description["supported_operations"]

        # All 5 intents should be supported
        expected_ops = {
            "filesystem.read_file",
            "filesystem.write_file",
            "filesystem.list_directory",
            "filesystem.ensure_directory",
            "filesystem.delete_file",
        }
        assert set(operations) == expected_ops

    def test_describe_includes_security_configuration(self, handler) -> None:
        """Test describe includes security configuration details."""
        description = handler.describe()

        # Should include security-related configuration
        assert "max_read_size" in description or "size_limits" in description
        assert "max_write_size" in description or "size_limits" in description

    def test_describe_returns_handler_type_infra_handler(self, handler) -> None:
        """Test describe returns handler_type as infra_handler."""
        description = handler.describe()
        assert description["handler_type"] == EnumHandlerType.INFRA_HANDLER.value

    def test_describe_returns_handler_category_effect(self, handler) -> None:
        """Test describe returns handler_category as effect."""
        description = handler.describe()
        assert description["handler_category"] == EnumHandlerTypeCategory.EFFECT.value

    def test_describe_returns_transport_type_filesystem(self, handler) -> None:
        """Test describe returns transport_type as filesystem."""
        description = handler.describe()
        assert description["transport_type"] == EnumInfraTransportType.FILESYSTEM.value


# ============================================================================
# TestHandlerFileSystemSecurityValidation
# ============================================================================


class TestHandlerFileSystemSecurityValidation:
    """Test suite for security validation features."""

    @pytest.mark.asyncio
    async def test_path_traversal_attack_prevention_double_dots(
        self, initialized_handler, temp_dir: Path
    ) -> None:
        """Test path traversal attack prevention with ../ sequences."""
        # Attempt to escape using ../
        malicious_path = str(temp_dir / ".." / ".." / "etc" / "passwd")

        envelope: dict[str, object] = {
            "operation": "filesystem.read_file",
            "payload": {"path": malicious_path},
            "correlation_id": str(uuid4()),
        }

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            await initialized_handler.execute(envelope)

        error_msg = str(exc_info.value).lower()
        assert (
            "path" in error_msg
            or "traversal" in error_msg
            or "escape" in error_msg
            or "allowed" in error_msg
        )

    @pytest.mark.asyncio
    async def test_symlink_escape_prevention(
        self, initialized_handler, temp_dir: Path
    ) -> None:
        """Test symlink escape prevention."""
        # Create a symlink pointing outside allowed paths
        escape_link = temp_dir / "escape"
        try:
            escape_link.symlink_to("/")
        except OSError:
            pytest.skip("Cannot create symlinks on this system")

        escape_target = escape_link / "etc" / "passwd"

        envelope: dict[str, object] = {
            "operation": "filesystem.read_file",
            "payload": {"path": str(escape_target)},
            "correlation_id": str(uuid4()),
        }

        with pytest.raises(
            (ProtocolConfigurationError, InfraConnectionError)
        ) as exc_info:
            await initialized_handler.execute(envelope)

        error_msg = str(exc_info.value).lower()
        assert (
            "symlink" in error_msg
            or "path" in error_msg
            or "allowed" in error_msg
            or "escape" in error_msg
        )

    @pytest.mark.asyncio
    async def test_absolute_path_validation(
        self, initialized_handler, temp_dir: Path
    ) -> None:
        """Test absolute path validation."""
        # Absolute path outside allowed paths
        envelope: dict[str, object] = {
            "operation": "filesystem.read_file",
            "payload": {"path": "/etc/hostname"},
            "correlation_id": str(uuid4()),
        }

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            await initialized_handler.execute(envelope)

        error_msg = str(exc_info.value).lower()
        assert "path" in error_msg or "allowed" in error_msg or "whitelist" in error_msg

    @pytest.mark.asyncio
    async def test_normalized_path_checking(
        self, initialized_handler, temp_dir: Path
    ) -> None:
        """Test that paths with .. that stay within allowed dirs are handled correctly."""
        # Create a subdir so we can use .. to navigate within allowed paths
        subdir = temp_dir / "subdir"
        subdir.mkdir(parents=True, exist_ok=True)

        # Create a test file in temp_dir
        test_file = temp_dir / "test.txt"
        test_file.write_text("test")

        # Use .. to go up and then back down - this should work since we stay in allowed path
        complex_path = str(subdir / ".." / "test.txt")

        envelope: dict[str, object] = {
            "operation": "filesystem.read_file",
            "payload": {"path": complex_path},
            "correlation_id": str(uuid4()),
        }

        # This should work because the resolved path is still within temp_dir
        result = await initialized_handler.execute(envelope)
        assert result.result["status"] == "success"
        assert result.result["payload"]["content"] == "test"


# ============================================================================
# TestHandlerFileSystemLifecycle
# ============================================================================


class TestHandlerFileSystemLifecycle:
    """Test suite for handler lifecycle management."""

    @pytest.fixture
    def handler(self):
        """Create HandlerFileSystem fixture."""
        from omnibase_infra.handlers.handler_filesystem import HandlerFileSystem

        return HandlerFileSystem()

    @pytest.mark.asyncio
    async def test_shutdown_sets_uninitialized_state(
        self, handler, temp_dir: Path
    ) -> None:
        """Test shutdown sets handler to uninitialized state."""
        await handler.initialize({"allowed_paths": [str(temp_dir)]})
        assert handler._initialized is True

        await handler.shutdown()
        assert handler._initialized is False

    @pytest.mark.asyncio
    async def test_execute_after_shutdown_raises_error(
        self, handler, temp_dir: Path
    ) -> None:
        """Test execute after shutdown raises RuntimeHostError."""
        await handler.initialize({"allowed_paths": [str(temp_dir)]})
        await handler.shutdown()

        envelope: dict[str, object] = {
            "operation": "filesystem.read_file",
            "payload": {"path": str(temp_dir / "test.txt")},
            "correlation_id": str(uuid4()),
        }

        with pytest.raises(RuntimeHostError) as exc_info:
            await handler.execute(envelope)

        assert "not initialized" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_execute_before_initialize_raises_error(self, handler) -> None:
        """Test execute before initialize raises RuntimeHostError."""
        envelope: dict[str, object] = {
            "operation": "filesystem.read_file",
            "payload": {"path": "/tmp/test.txt"},
            "correlation_id": str(uuid4()),
        }

        with pytest.raises(RuntimeHostError) as exc_info:
            await handler.execute(envelope)

        assert "not initialized" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_multiple_shutdown_calls_safe(self, handler, temp_dir: Path) -> None:
        """Test multiple shutdown calls are safe (idempotent)."""
        await handler.initialize({"allowed_paths": [str(temp_dir)]})
        await handler.shutdown()
        await handler.shutdown()  # Second call should not raise

        assert handler._initialized is False

    @pytest.mark.asyncio
    async def test_reinitialize_after_shutdown(self, handler, temp_dir: Path) -> None:
        """Test handler can be reinitialized after shutdown."""
        await handler.initialize({"allowed_paths": [str(temp_dir)]})
        await handler.shutdown()

        assert handler._initialized is False

        await handler.initialize({"allowed_paths": [str(temp_dir)]})
        assert handler._initialized is True

        await handler.shutdown()


# ============================================================================
# TestHandlerFileSystemCorrelationId
# ============================================================================


class TestHandlerFileSystemCorrelationId:
    """Test suite for correlation ID handling."""

    @pytest.mark.asyncio
    async def test_correlation_id_from_envelope_uuid(
        self, initialized_handler, temp_dir: Path
    ) -> None:
        """Test correlation ID extracted from envelope as UUID."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("content")

        # Use deterministic ID generator for predictable testing
        id_gen = DeterministicIdGenerator(seed=100)
        correlation_id = id_gen.next_uuid()

        envelope: dict[str, object] = {
            "operation": "filesystem.read_file",
            "payload": {"path": str(test_file)},
            "correlation_id": correlation_id,
        }

        result = await initialized_handler.execute(envelope)

        # Verify correlation ID is returned
        assert result.result["correlation_id"] == str(correlation_id)

    @pytest.mark.asyncio
    async def test_correlation_id_from_envelope_string(
        self, initialized_handler, temp_dir: Path
    ) -> None:
        """Test correlation ID extracted from envelope as string."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("content")

        correlation_id = str(uuid4())

        envelope: dict[str, object] = {
            "operation": "filesystem.read_file",
            "payload": {"path": str(test_file)},
            "correlation_id": correlation_id,
        }

        result = await initialized_handler.execute(envelope)
        assert result.result["correlation_id"] == correlation_id

    @pytest.mark.asyncio
    async def test_correlation_id_generated_when_missing(
        self, initialized_handler, temp_dir: Path
    ) -> None:
        """Test correlation ID generated when not in envelope."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("content")

        envelope: dict[str, object] = {
            "operation": "filesystem.read_file",
            "payload": {"path": str(test_file)},
            # No correlation_id
        }

        result = await initialized_handler.execute(envelope)

        # Should have a generated UUID (returned as string in result)
        assert "correlation_id" in result.result
        assert isinstance(result.result["correlation_id"], str)
        UUID(result.result["correlation_id"])  # Should not raise


# ============================================================================
# TestHandlerFileSystemOperationValidation
# ============================================================================


class TestHandlerFileSystemOperationValidation:
    """Test suite for operation validation."""

    @pytest.mark.asyncio
    async def test_unsupported_operation_raises_error(
        self, initialized_handler
    ) -> None:
        """Test unsupported operation raises ProtocolConfigurationError."""
        envelope: dict[str, object] = {
            "operation": "filesystem.unsupported_operation",
            "payload": {"path": "/tmp/test"},
            "correlation_id": str(uuid4()),
        }

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            await initialized_handler.execute(envelope)

        assert (
            "not supported" in str(exc_info.value).lower()
            or "operation" in str(exc_info.value).lower()
        )

    @pytest.mark.asyncio
    async def test_missing_operation_raises_error(self, initialized_handler) -> None:
        """Test missing operation field raises ProtocolConfigurationError."""
        envelope: dict[str, object] = {
            "payload": {"path": "/tmp/test"},
            "correlation_id": str(uuid4()),
        }

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            await initialized_handler.execute(envelope)

        assert "operation" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_missing_payload_raises_error(self, initialized_handler) -> None:
        """Test missing payload field raises ProtocolConfigurationError."""
        envelope: dict[str, object] = {
            "operation": "filesystem.read_file",
            "correlation_id": str(uuid4()),
        }

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            await initialized_handler.execute(envelope)

        assert "payload" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_missing_path_in_payload_raises_error(
        self, initialized_handler
    ) -> None:
        """Test missing path in payload raises ProtocolConfigurationError."""
        envelope: dict[str, object] = {
            "operation": "filesystem.read_file",
            "payload": {},  # No path
            "correlation_id": str(uuid4()),
        }

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            await initialized_handler.execute(envelope)

        assert "path" in str(exc_info.value).lower()
