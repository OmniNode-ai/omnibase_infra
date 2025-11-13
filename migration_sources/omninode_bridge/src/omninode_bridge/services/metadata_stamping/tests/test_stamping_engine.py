"""Tests for the core stamping engine."""

import uuid

import pytest

from omninode_bridge.services.metadata_stamping.engine.stamping_engine import (
    StampingEngine,
)


class TestStampingEngine:
    """Test suite for the stamping engine."""

    @pytest.fixture
    async def engine(self):
        """Create stamping engine instance for testing."""
        engine = StampingEngine()
        yield engine
        await engine.cleanup()

    @pytest.mark.asyncio
    async def test_lightweight_stamp_creation(self, engine):
        """Test lightweight stamp creation with various inputs."""
        content_hash = "a" * 64  # Simulate BLAKE3 hash
        metadata = {"uid": str(uuid.uuid4()), "author": "test_user", "version": "1.0"}

        stamp = await engine.create_lightweight_stamp(content_hash, metadata)

        assert stamp.startswith("# ONEX:")
        assert f"uid={metadata['uid']}" in stamp
        assert f"hash={content_hash}" in stamp
        assert "author=test_user" in stamp
        assert "v=1.0" in stamp

    @pytest.mark.asyncio
    async def test_rich_stamp_creation(self, engine):
        """Test rich metadata stamp creation."""
        content_hash = "b" * 64  # Simulate BLAKE3 hash
        metadata = {
            "uid": str(uuid.uuid4()),
            "file_path": "/test/file.txt",
            "file_size": 1024,
            "content_type": "text/plain",
        }

        stamp = await engine.create_rich_stamp(content_hash, metadata)

        assert "<!-- ONEX_METADATA_START -->" in stamp
        assert "<!-- ONEX_METADATA_END -->" in stamp
        assert f"uid: {metadata['uid']}" in stamp
        assert f"hash: {content_hash}" in stamp
        assert "file_path: /test/file.txt" in stamp
        assert "algorithm: BLAKE3" in stamp

    @pytest.mark.asyncio
    async def test_stamp_content_lightweight(self, engine):
        """Test stamping content with lightweight stamp."""
        content = "This is test content for stamping."
        metadata = {"author": "test"}

        result = await engine.stamp_content(
            content=content,
            file_path="/test.txt",
            stamp_type="lightweight",
            metadata=metadata,
        )

        assert "stamped_content" in result
        assert "stamp" in result
        assert "content_hash" in result
        assert result["stamp_type"] == "lightweight"
        assert result["stamped_content"].startswith("# ONEX:")
        assert content in result["stamped_content"]

    @pytest.mark.asyncio
    async def test_stamp_content_rich(self, engine):
        """Test stamping content with rich stamp."""
        content = "This is test content for rich stamping."
        metadata = {"project": "test_project"}

        result = await engine.stamp_content(
            content=content, file_path="/test.md", stamp_type="rich", metadata=metadata
        )

        assert result["stamp_type"] == "rich"
        assert "<!-- ONEX_METADATA_START -->" in result["stamped_content"]
        assert "<!-- ONEX_METADATA_END -->" in result["stamped_content"]
        assert content in result["stamped_content"]

    @pytest.mark.asyncio
    async def test_stamp_validation(self, engine):
        """Test stamp validation functionality."""
        # Create stamped content
        content = "Test content for validation."
        stamp_result = await engine.stamp_content(
            content=content, stamp_type="lightweight"
        )

        # Validate the stamped content
        validation_result = await engine.validate_stamp(
            content=stamp_result["stamped_content"]
        )

        assert validation_result["valid"] is True
        assert validation_result["stamps_found"] == 1
        assert validation_result["current_hash"] == stamp_result["content_hash"]

    @pytest.mark.asyncio
    async def test_stamp_validation_with_tampering(self, engine):
        """Test stamp validation detects tampering."""
        # Create stamped content
        content = "Original content."
        stamp_result = await engine.stamp_content(
            content=content, stamp_type="lightweight"
        )

        # Tamper with the content (but keep the stamp)
        tampered_content = stamp_result["stamped_content"].replace(
            "Original", "Tampered"
        )

        # Validate should detect tampering
        validation_result = await engine.validate_stamp(content=tampered_content)

        assert validation_result["valid"] is False
        assert validation_result["stamps_found"] == 1

    @pytest.mark.asyncio
    async def test_extract_stamps(self, engine):
        """Test stamp extraction from content."""
        # Create content with both lightweight and rich stamps
        content = """# ONEX:uid=test-uid,hash=testhash123,ts=2024-01-01T00:00:00
Some content here

<!-- ONEX_METADATA_START -->
<!-- ONEX Metadata Stamp
  uid: test-uid-2
  hash: testhash456
  timestamp: 2024-01-01T00:00:00
-->
<!-- ONEX_METADATA_END -->
"""

        stamps = await engine.extract_stamps(content)

        assert len(stamps) == 2
        assert stamps[0]["type"] == "lightweight"
        assert stamps[0]["uid"] == "test-uid"
        assert stamps[0]["hash"] == "testhash123"
        assert stamps[1]["type"] == "rich"

    @pytest.mark.asyncio
    async def test_remove_existing_stamps(self, engine):
        """Test removal of existing stamps from content."""
        stamped_content = """# ONEX:uid=test,hash=hash123,ts=2024-01-01
This is the actual content.

<!-- ONEX_METADATA_START -->
<!-- metadata here -->
<!-- ONEX_METADATA_END -->

More content here."""

        clean_content = await engine._remove_existing_stamps(stamped_content)

        assert "# ONEX:" not in clean_content
        assert "<!-- ONEX_METADATA_START -->" not in clean_content
        assert "<!-- ONEX_METADATA_END -->" not in clean_content
        assert "This is the actual content." in clean_content
        assert "More content here." in clean_content

    @pytest.mark.asyncio
    async def test_performance_metrics(self, engine):
        """Test that performance metrics are collected."""
        content = "Test content" * 100  # Make it a bit larger

        result = await engine.stamp_content(content=content, stamp_type="lightweight")

        assert "execution_time_ms" in result
        assert "performance_grade" in result
        assert result["execution_time_ms"] > 0
        assert result["performance_grade"] in ["A", "B", "C"]
