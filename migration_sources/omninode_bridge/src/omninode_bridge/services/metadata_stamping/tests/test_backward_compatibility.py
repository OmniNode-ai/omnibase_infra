"""Test backward compatibility for MetadataStampingService.

This test module ensures that existing code continues to work
with the new compliance fields while maintaining the same API surface.
"""

from datetime import datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from ..compatibility import (
    BackwardCompatibleClient,
    create_stamp_legacy,
    ensure_compliance_fields,
    get_stamp_legacy,
    strip_compliance_fields,
)
from ..database.client import MetadataStampingPostgresClient
from ..models.database import LegacyMetadataStampRecord, MetadataStampRecord


class TestBackwardCompatibility:
    """Test backward compatibility functionality."""

    @pytest.fixture
    def mock_client(self):
        """Create mock database client."""
        return AsyncMock(spec=MetadataStampingPostgresClient)

    @pytest.fixture
    def compat_client(self, mock_client):
        """Create backward compatible client."""
        return BackwardCompatibleClient(mock_client)

    @pytest.fixture
    def sample_legacy_stamp_data(self):
        """Sample legacy stamp data without compliance fields."""
        return {
            "file_hash": "abc123",
            "file_path": "/test/file.txt",
            "file_size": 1024,
            "content_type": "text/plain",
            "stamp_data": {"type": "test", "data": "sample"},
            "protocol_version": "1.0",
        }

    @pytest.fixture
    def sample_full_stamp_record(self):
        """Sample full stamp record with compliance fields."""
        return MetadataStampRecord(
            id=uuid4(),
            file_hash="abc123",
            file_path="/test/file.txt",
            file_size=1024,
            content_type="text/plain",
            stamp_data={"type": "test", "data": "sample"},
            protocol_version="1.0",
            intelligence_data={"ai_insights": "sample"},
            version=1,
            op_id=uuid4(),
            namespace="omninode.services.metadata",
            metadata_version="0.1",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

    async def test_create_metadata_stamp_legacy(
        self, compat_client, mock_client, sample_legacy_stamp_data
    ):
        """Test legacy stamp creation interface."""
        # Mock the underlying client response
        mock_client.create_metadata_stamp.return_value = {
            "id": "stamp-123",
            "file_hash": "abc123",
            "op_id": "op-456",
            "namespace": "omninode.services.metadata",
            "version": 1,
            "metadata_version": "0.1",
            "created_at": datetime.now(),
        }

        # Call legacy interface
        result = await compat_client.create_metadata_stamp_legacy(
            file_hash=sample_legacy_stamp_data["file_hash"],
            file_path=sample_legacy_stamp_data["file_path"],
            file_size=sample_legacy_stamp_data["file_size"],
            content_type=sample_legacy_stamp_data["content_type"],
            stamp_data=sample_legacy_stamp_data["stamp_data"],
            protocol_version=sample_legacy_stamp_data["protocol_version"],
        )

        # Verify underlying client was called with compliance fields
        mock_client.create_metadata_stamp.assert_called_once()
        call_kwargs = mock_client.create_metadata_stamp.call_args.kwargs

        assert call_kwargs["file_hash"] == "abc123"
        assert call_kwargs["intelligence_data"] == {}
        assert call_kwargs["version"] == 1
        assert call_kwargs["namespace"] == "omninode.services.metadata"
        assert call_kwargs["metadata_version"] == "0.1"
        assert "op_id" in call_kwargs

        # Verify legacy response format (no compliance fields)
        assert "id" in result
        assert "file_hash" in result
        assert "created_at" in result
        assert "op_id" not in result  # Compliance field should be filtered
        assert "namespace" not in result  # Compliance field should be filtered

    async def test_get_metadata_stamp_legacy(
        self, compat_client, mock_client, sample_full_stamp_record
    ):
        """Test legacy stamp retrieval interface."""
        # Mock the underlying client response (full record)
        mock_client.get_metadata_stamp.return_value = (
            sample_full_stamp_record.model_dump()
        )

        # Call legacy interface
        result = await compat_client.get_metadata_stamp_legacy("abc123")

        # Verify underlying client was called
        mock_client.get_metadata_stamp.assert_called_once_with("abc123")

        # Verify legacy response format (no compliance fields)
        assert "id" in result
        assert "file_hash" in result
        assert "stamp_data" in result
        assert "intelligence_data" not in result  # Compliance field should be filtered
        assert "op_id" not in result  # Compliance field should be filtered
        assert "namespace" not in result  # Compliance field should be filtered

    async def test_batch_insert_stamps_legacy(
        self, compat_client, mock_client, sample_legacy_stamp_data
    ):
        """Test legacy batch insert interface."""
        # Mock the underlying client response
        mock_client.batch_insert_stamps.return_value = ["stamp-1", "stamp-2"]

        # Call legacy interface with list of legacy stamp data
        stamps_data = [sample_legacy_stamp_data, sample_legacy_stamp_data.copy()]
        result = await compat_client.batch_insert_stamps_legacy(stamps_data)

        # Verify underlying client was called with enhanced data
        mock_client.batch_insert_stamps.assert_called_once()
        call_args = mock_client.batch_insert_stamps.call_args.args[0]

        # Check that compliance fields were added
        for stamp in call_args:
            assert "intelligence_data" in stamp
            assert "version" in stamp
            assert "op_id" in stamp
            assert "namespace" in stamp
            assert "metadata_version" in stamp

        # Verify return value is unchanged
        assert result == ["stamp-1", "stamp-2"]

    def test_ensure_compliance_fields(self, sample_legacy_stamp_data):
        """Test compliance fields utility function."""
        enhanced_data = ensure_compliance_fields(sample_legacy_stamp_data)

        # Verify original fields are preserved
        for key, value in sample_legacy_stamp_data.items():
            assert enhanced_data[key] == value

        # Verify compliance fields are added
        assert "intelligence_data" in enhanced_data
        assert "version" in enhanced_data
        assert "op_id" in enhanced_data
        assert "namespace" in enhanced_data
        assert "metadata_version" in enhanced_data

    def test_strip_compliance_fields(self, sample_full_stamp_record):
        """Test compliance fields stripping utility function."""
        full_data = sample_full_stamp_record.model_dump()
        stripped_data = strip_compliance_fields(full_data)

        # Verify compliance fields are removed
        assert "intelligence_data" not in stripped_data
        assert "version" not in stripped_data
        assert "op_id" not in stripped_data
        assert "namespace" not in stripped_data
        assert "metadata_version" not in stripped_data

        # Verify core fields are preserved
        assert "id" in stripped_data
        assert "file_hash" in stripped_data
        assert "stamp_data" in stripped_data

    def test_legacy_metadata_stamp_record_conversion(self, sample_full_stamp_record):
        """Test legacy record model conversion."""
        legacy_record = LegacyMetadataStampRecord.from_full_record(
            sample_full_stamp_record
        )

        # Verify core fields are preserved
        assert legacy_record.id == sample_full_stamp_record.id
        assert legacy_record.file_hash == sample_full_stamp_record.file_hash
        assert legacy_record.stamp_data == sample_full_stamp_record.stamp_data

        # Verify legacy model doesn't have compliance fields
        legacy_dict = legacy_record.model_dump()
        assert "intelligence_data" not in legacy_dict
        assert "op_id" not in legacy_dict
        assert "namespace" not in legacy_dict

    async def test_create_stamp_legacy_function(
        self, mock_client, sample_legacy_stamp_data
    ):
        """Test legacy stamp creation function interface."""
        # Mock the underlying client response
        mock_client.create_metadata_stamp.return_value = {
            "id": "stamp-123",
            "file_hash": "abc123",
            "op_id": "op-456",
            "namespace": "omninode.services.metadata",
            "version": 1,
            "metadata_version": "0.1",
            "created_at": datetime.now(),
        }

        # Call legacy function
        result = await create_stamp_legacy(
            mock_client,
            file_hash=sample_legacy_stamp_data["file_hash"],
            file_path=sample_legacy_stamp_data["file_path"],
            file_size=sample_legacy_stamp_data["file_size"],
            content_type=sample_legacy_stamp_data["content_type"],
            stamp_data=sample_legacy_stamp_data["stamp_data"],
            protocol_version=sample_legacy_stamp_data["protocol_version"],
        )

        # Verify call was made and response is in legacy format
        mock_client.create_metadata_stamp.assert_called_once()
        assert "id" in result
        assert "op_id" not in result  # Should be filtered for legacy compatibility

    async def test_get_stamp_legacy_function(
        self, mock_client, sample_full_stamp_record
    ):
        """Test legacy stamp retrieval function interface."""
        # Mock the underlying client response
        mock_client.get_metadata_stamp.return_value = (
            sample_full_stamp_record.model_dump()
        )

        # Call legacy function
        result = await get_stamp_legacy(mock_client, "abc123")

        # Verify call was made and response is in legacy format
        mock_client.get_metadata_stamp.assert_called_once_with("abc123")
        assert "id" in result
        assert (
            "intelligence_data" not in result
        )  # Should be filtered for legacy compatibility
