"""Unit tests for NodeCodegenStoreEffect."""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock
from uuid import uuid4

import pytest
from omnibase_core import EnumCoreErrorCode, ModelOnexError
from omnibase_core.models.core import ModelContainer

from omninode_bridge.nodes.codegen_store_effect.v1_0_0.models import ModelStorageResult
from omninode_bridge.nodes.codegen_store_effect.v1_0_0.node import (
    NodeCodegenStoreEffect,
)
from omninode_bridge.nodes.conftest import create_test_contract_effect


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_container(temp_dir):
    """Create container with mocked services."""
    container = Mock(spec=ModelContainer)
    container.config = Mock()
    container.config.get = Mock(
        side_effect=lambda k, default: (
            temp_dir if k == "codegen_output_dir" else default
        )
    )
    container.get_service = Mock(return_value=None)
    return container


@pytest.fixture
def store_effect(mock_container):
    """Create store effect with mocked dependencies."""
    return NodeCodegenStoreEffect(mock_container)


class TestNodeCodegenStoreEffect:
    """Test suite for NodeCodegenStoreEffect."""

    @pytest.mark.asyncio
    async def test_store_single_file(self, store_effect, temp_dir):
        """Test storing a single file."""
        # Arrange
        file_content = "def foo():\n    return 42"
        file_path = "test_node.py"

        storage_requests = [
            {
                "file_path": file_path,
                "content": file_content,
                "artifact_type": "node_file",
                "create_directories": True,
            }
        ]

        contract = create_test_contract_effect(
            correlation_id=uuid4(),
            input_state={"storage_requests": storage_requests},
        )

        # Act
        result = await store_effect.execute_effect(contract)

        # Assert
        assert isinstance(result, ModelStorageResult)
        assert result.success
        assert result.artifacts_stored == 1
        assert len(result.stored_files) == 1
        assert result.total_bytes_written == len(file_content.encode("utf-8"))
        assert result.storage_time_ms < 1000

        # Verify file exists
        stored_path = Path(temp_dir) / file_path
        assert stored_path.exists()
        assert stored_path.read_text() == file_content

    @pytest.mark.asyncio
    async def test_store_multiple_files(self, store_effect, temp_dir):
        """Test storing multiple files."""
        # Arrange
        storage_requests = [
            {
                "file_path": "node.py",
                "content": "def foo(): pass",
                "artifact_type": "node_file",
            },
            {
                "file_path": "test_node.py",
                "content": "def test_foo(): pass",
                "artifact_type": "test_file",
            },
            {
                "file_path": "models.py",
                "content": "class ModelFoo: pass",
                "artifact_type": "model_file",
            },
        ]

        contract = create_test_contract_effect(
            correlation_id=uuid4(),
            input_state={"storage_requests": storage_requests},
        )

        # Act
        result = await store_effect.execute_effect(contract)

        # Assert
        assert result.success
        assert result.artifacts_stored == 3
        assert len(result.stored_files) == 3

        # Verify all files exist
        for req in storage_requests:
            file_path = Path(temp_dir) / req["file_path"]
            assert file_path.exists()
            assert file_path.read_text() == req["content"]

    @pytest.mark.asyncio
    async def test_create_nested_directories(self, store_effect, temp_dir):
        """Test creating nested directories."""
        # Arrange
        file_path = "deep/nested/directory/structure/node.py"
        file_content = "def foo(): pass"

        storage_requests = [
            {
                "file_path": file_path,
                "content": file_content,
                "create_directories": True,
            }
        ]

        contract = create_test_contract_effect(
            correlation_id=uuid4(),
            input_state={"storage_requests": storage_requests},
        )

        # Act
        result = await store_effect.execute_effect(contract)

        # Assert
        assert result.success
        stored_path = Path(temp_dir) / file_path
        assert stored_path.exists()
        assert stored_path.parent.exists()

    @pytest.mark.asyncio
    async def test_overwrite_existing_file(self, store_effect, temp_dir):
        """Test overwriting an existing file."""
        # Arrange
        file_path = "node.py"
        original_content = "original content"
        new_content = "new content"

        # Create original file
        original_file = Path(temp_dir) / file_path
        original_file.write_text(original_content)

        storage_requests = [
            {
                "file_path": file_path,
                "content": new_content,
                "overwrite": True,
            }
        ]

        contract = create_test_contract_effect(
            correlation_id=uuid4(),
            input_state={"storage_requests": storage_requests},
        )

        # Act
        result = await store_effect.execute_effect(contract)

        # Assert
        assert result.success
        assert original_file.read_text() == new_content

    @pytest.mark.asyncio
    async def test_no_overwrite_existing_file(self, store_effect, temp_dir):
        """Test not overwriting when overwrite=False."""
        # Arrange
        file_path = "node.py"
        original_content = "original content"
        new_content = "new content"

        # Create original file
        original_file = Path(temp_dir) / file_path
        original_file.write_text(original_content)

        storage_requests = [
            {
                "file_path": file_path,
                "content": new_content,
                "overwrite": False,
            }
        ]

        contract = create_test_contract_effect(
            correlation_id=uuid4(),
            input_state={"storage_requests": storage_requests},
        )

        # Act
        result = await store_effect.execute_effect(contract)

        # Assert
        assert not result.success
        assert len(result.storage_errors) == 1
        assert "overwrite=False" in result.storage_errors[0]
        assert original_file.read_text() == original_content

    @pytest.mark.asyncio
    async def test_file_permissions(self, store_effect, temp_dir):
        """Test setting file permissions."""
        # Arrange
        file_path = "node.py"
        file_content = "def foo(): pass"

        storage_requests = [
            {
                "file_path": file_path,
                "content": file_content,
                "file_permissions": "0755",
            }
        ]

        contract = create_test_contract_effect(
            correlation_id=uuid4(),
            input_state={"storage_requests": storage_requests},
        )

        # Act
        result = await store_effect.execute_effect(contract)

        # Assert
        assert result.success
        stored_path = Path(temp_dir) / file_path

        # Check permissions (on Unix-like systems)
        if os.name != "nt":  # Skip on Windows
            file_stat = stored_path.stat()
            file_mode = oct(file_stat.st_mode)[-3:]
            assert file_mode == "755"

    @pytest.mark.asyncio
    async def test_missing_storage_requests(self, store_effect):
        """Test error when storage_requests is missing."""
        # Arrange
        contract = create_test_contract_effect(
            correlation_id=uuid4(),
            input_state={},
        )

        # Act & Assert
        with pytest.raises(ModelOnexError) as exc:
            await store_effect.execute_effect(contract)

        assert exc.value.error_code == EnumCoreErrorCode.VALIDATION_ERROR
        assert "storage_requests" in exc.value.message

    @pytest.mark.asyncio
    async def test_absolute_file_path(self, store_effect, temp_dir):
        """Test storing to absolute file path."""
        # Arrange
        absolute_path = Path(temp_dir) / "absolute" / "path" / "node.py"
        file_content = "def foo(): pass"

        storage_requests = [
            {
                "file_path": str(absolute_path),
                "content": file_content,
                "create_directories": True,
            }
        ]

        contract = create_test_contract_effect(
            correlation_id=uuid4(),
            input_state={"storage_requests": storage_requests},
        )

        # Act
        result = await store_effect.execute_effect(contract)

        # Assert
        assert result.success
        assert absolute_path.exists()
        assert absolute_path.read_text() == file_content

    @pytest.mark.asyncio
    async def test_bytes_written_tracking(self, store_effect, temp_dir):
        """Test that bytes written are correctly tracked."""
        # Arrange
        contents = [
            "short",
            "medium length content",
            "very long content with lots of characters and words",
        ]

        storage_requests = [
            {
                "file_path": f"file_{i}.py",
                "content": content,
            }
            for i, content in enumerate(contents)
        ]

        contract = create_test_contract_effect(
            correlation_id=uuid4(),
            input_state={"storage_requests": storage_requests},
        )

        # Act
        result = await store_effect.execute_effect(contract)

        # Assert
        expected_bytes = sum(len(c.encode("utf-8")) for c in contents)
        assert result.total_bytes_written == expected_bytes

    @pytest.mark.asyncio
    async def test_get_metrics(self, store_effect, temp_dir):
        """Test metrics collection."""
        # Arrange
        storage_requests = [
            {
                "file_path": "node.py",
                "content": "def foo(): pass",
            }
        ]

        contract = create_test_contract_effect(
            correlation_id=uuid4(),
            input_state={"storage_requests": storage_requests},
        )

        # Act
        await store_effect.execute_effect(contract)
        metrics = store_effect.get_metrics()

        # Assert
        assert metrics["total_storage_operations"] == 1
        assert metrics["successful_storage_operations"] == 1
        assert metrics["failed_storage_operations"] == 0
        assert metrics["success_rate"] == 1.0
        assert metrics["total_bytes_written"] > 0

    @pytest.mark.asyncio
    async def test_base_directory_override(self, store_effect, temp_dir):
        """Test overriding base directory."""
        # Arrange
        custom_dir = Path(temp_dir) / "custom"
        custom_dir.mkdir()

        storage_requests = [
            {
                "file_path": "node.py",
                "content": "def foo(): pass",
            }
        ]

        contract = create_test_contract_effect(
            correlation_id=uuid4(),
            input_state={
                "storage_requests": storage_requests,
                "base_directory": str(custom_dir),
            },
        )

        # Act
        result = await store_effect.execute_effect(contract)

        # Assert
        assert result.success
        custom_file = custom_dir / "node.py"
        assert custom_file.exists()


# Performance benchmark tests
class TestStoreEffectPerformance:
    """Performance tests for storage operations."""

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_large_file_storage(self, store_effect, temp_dir):
        """Benchmark storage of large files."""
        # Generate large content (1MB)
        large_content = "# " + "x" * (1024 * 1024)

        storage_requests = [
            {
                "file_path": "large_file.py",
                "content": large_content,
            }
        ]

        contract = create_test_contract_effect(
            correlation_id=uuid4(),
            input_state={"storage_requests": storage_requests},
        )

        # Act
        result = await store_effect.execute_effect(contract)

        # Assert
        assert result.success
        assert result.storage_time_ms < 5000  # <5s for 1MB file

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_many_files_storage(self, store_effect, temp_dir):
        """Benchmark storage of many files."""
        # Create 100 small files
        storage_requests = [
            {
                "file_path": f"file_{i}.py",
                "content": f"def foo_{i}(): pass",
            }
            for i in range(100)
        ]

        contract = create_test_contract_effect(
            correlation_id=uuid4(),
            input_state={"storage_requests": storage_requests},
        )

        # Act
        result = await store_effect.execute_effect(contract)

        # Assert
        assert result.success
        assert result.artifacts_stored == 100
        assert result.storage_time_ms < 2000  # <2s for 100 small files
