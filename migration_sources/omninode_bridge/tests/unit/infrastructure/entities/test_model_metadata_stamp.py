#!/usr/bin/env python3
"""Unit tests for ModelMetadataStamp.

Tests cover:
- Model instantiation with all fields
- Model validation
- Default values
- JSON serialization/deserialization
- Field validation
- Model configuration
"""

import json
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from omninode_bridge.infrastructure.entities.model_metadata_stamp import (
    ModelMetadataStamp,
)


class TestModelMetadataStamp:
    """Test suite for ModelMetadataStamp."""

    def test_model_instantiation_with_all_fields(self):
        """Test model instantiation with all required fields."""
        id = uuid4()
        workflow_id = uuid4()
        file_hash = "a" * 64  # 64 hex characters
        stamp_data = {"stamp_type": "inline", "metadata": {"key": "value"}}
        namespace = "production"
        created_at = datetime.now(UTC)

        stamp = ModelMetadataStamp(
            id=id,
            workflow_id=workflow_id,
            file_hash=file_hash,
            stamp_data=stamp_data,
            namespace=namespace,
            created_at=created_at,
        )

        assert stamp.id == id
        assert stamp.workflow_id == workflow_id
        assert stamp.file_hash == file_hash
        assert stamp.stamp_data == stamp_data
        assert stamp.namespace == namespace
        assert stamp.created_at == created_at

    def test_model_instantiation_with_required_fields_only(self):
        """Test model instantiation with only required fields."""
        file_hash = "a" * 64  # 64 hex characters
        stamp_data = {"stamp_type": "inline"}
        namespace = "production"

        stamp = ModelMetadataStamp(
            file_hash=file_hash, stamp_data=stamp_data, namespace=namespace
        )

        assert stamp.id is None  # Default value
        assert stamp.workflow_id is None  # Default value
        assert stamp.file_hash == file_hash
        assert stamp.stamp_data == stamp_data
        assert stamp.namespace == namespace
        assert stamp.created_at is None  # Default value

    def test_model_validation_missing_required_fields(self):
        """Test model validation with missing required fields."""
        # Missing file_hash, stamp_data, namespace

        with pytest.raises(ValidationError) as exc_info:
            ModelMetadataStamp()

        errors = exc_info.value.errors()
        assert len(errors) == 3
        assert any("file_hash" in str(error) for error in errors)
        assert any("stamp_data" in str(error) for error in errors)
        assert any("namespace" in str(error) for error in errors)

    def test_model_validation_invalid_file_hash(self):
        """Test model validation with invalid file_hash."""
        stamp_data = {"stamp_type": "inline"}
        namespace = "production"

        # Too short file_hash
        with pytest.raises(ValidationError) as exc_info:
            ModelMetadataStamp(
                file_hash="a" * 63,  # 63 characters, min is 64
                stamp_data=stamp_data,
                namespace=namespace,
            )

        assert "file_hash" in str(exc_info.value)

        # Too long file_hash
        with pytest.raises(ValidationError) as exc_info:
            ModelMetadataStamp(
                file_hash="a" * 65,  # 65 characters, max is 64
                stamp_data=stamp_data,
                namespace=namespace,
            )

        assert "file_hash" in str(exc_info.value)

        # Invalid characters in file_hash
        with pytest.raises(ValidationError) as exc_info:
            ModelMetadataStamp(
                file_hash="g" * 64,  # 'g' is not a valid hex character
                stamp_data=stamp_data,
                namespace=namespace,
            )

        assert "file_hash" in str(exc_info.value)

    def test_model_validation_valid_file_hash(self):
        """Test model validation with valid file_hash values."""
        stamp_data = {"stamp_type": "inline"}
        namespace = "production"

        # Valid hex characters
        valid_hashes = [
            "a" * 64,  # All lowercase 'a'
            "f" * 64,  # All lowercase 'f'
            "0" * 64,  # All '0'
            "9" * 64,  # All '9'
            "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",  # Mixed valid hex characters (64 chars)
        ]

        for file_hash in valid_hashes:
            stamp = ModelMetadataStamp(
                file_hash=file_hash, stamp_data=stamp_data, namespace=namespace
            )
            assert stamp.file_hash == file_hash

    def test_model_validation_invalid_namespace(self):
        """Test model validation with invalid namespace."""
        file_hash = "a" * 64
        stamp_data = {"stamp_type": "inline"}

        # Empty namespace
        with pytest.raises(ValidationError) as exc_info:
            ModelMetadataStamp(file_hash=file_hash, stamp_data=stamp_data, namespace="")

        assert "namespace" in str(exc_info.value)

        # Too long namespace
        with pytest.raises(ValidationError) as exc_info:
            ModelMetadataStamp(
                file_hash=file_hash,
                stamp_data=stamp_data,
                namespace="a" * 256,  # 256 characters, max is 255
            )

        assert "namespace" in str(exc_info.value)

    def test_model_serialization(self):
        """Test model serialization to JSON."""
        file_hash = "a" * 64
        stamp_data = {"stamp_type": "inline", "metadata": {"key": "value"}}
        namespace = "production"

        stamp = ModelMetadataStamp(
            file_hash=file_hash, stamp_data=stamp_data, namespace=namespace
        )

        # Test model_dump
        data = stamp.model_dump()
        assert data["file_hash"] == file_hash
        assert data["stamp_data"] == stamp_data
        assert data["namespace"] == namespace
        assert data["id"] is None
        assert data["workflow_id"] is None
        assert data["created_at"] is None

        # Test model_dump_json
        json_str = stamp.model_dump_json()
        parsed_data = json.loads(json_str)
        assert parsed_data["file_hash"] == file_hash
        assert parsed_data["stamp_data"] == stamp_data
        assert parsed_data["namespace"] == namespace

    def test_model_deserialization(self):
        """Test model deserialization from JSON."""
        id = uuid4()
        workflow_id = uuid4()
        file_hash = "a" * 64
        stamp_data = {"stamp_type": "inline", "metadata": {"key": "value"}}
        namespace = "production"
        created_at = datetime.now(UTC)

        data = {
            "id": str(id),
            "workflow_id": str(workflow_id),
            "file_hash": file_hash,
            "stamp_data": stamp_data,
            "namespace": namespace,
            "created_at": created_at.isoformat(),
        }

        # Test model_validate
        stamp = ModelMetadataStamp.model_validate(data)
        assert stamp.id == id
        assert stamp.workflow_id == workflow_id
        assert stamp.file_hash == file_hash
        assert stamp.stamp_data == stamp_data
        assert stamp.namespace == namespace
        assert stamp.created_at == created_at

    def test_model_roundtrip(self):
        """Test model serialization and deserialization roundtrip."""
        file_hash = "a" * 64
        stamp_data = {"stamp_type": "inline", "metadata": {"key": "value"}}
        namespace = "production"

        original = ModelMetadataStamp(
            file_hash=file_hash, stamp_data=stamp_data, namespace=namespace
        )

        # Serialize and deserialize
        data = original.model_dump()
        restored = ModelMetadataStamp.model_validate(data)

        # Verify all fields match
        assert restored.file_hash == original.file_hash
        assert restored.stamp_data == original.stamp_data
        assert restored.namespace == original.namespace

    def test_model_equality(self):
        """Test model equality comparison."""
        file_hash = "a" * 64
        stamp_data = {"stamp_type": "inline"}
        namespace = "production"

        stamp1 = ModelMetadataStamp(
            file_hash=file_hash, stamp_data=stamp_data, namespace=namespace
        )

        stamp2 = ModelMetadataStamp(
            file_hash=file_hash, stamp_data=stamp_data, namespace=namespace
        )

        stamp3 = ModelMetadataStamp(
            file_hash="b" * 64,  # Different hash
            stamp_data=stamp_data,
            namespace=namespace,
        )

        # Test equality
        assert stamp1.model_dump() == stamp2.model_dump()
        assert stamp1.model_dump() != stamp3.model_dump()

    def test_model_copy(self):
        """Test model copying."""
        file_hash = "a" * 64
        stamp_data = {"stamp_type": "inline", "metadata": {"key": "value"}}
        namespace = "production"

        original = ModelMetadataStamp(
            file_hash=file_hash, stamp_data=stamp_data, namespace=namespace
        )

        # Test model_copy
        copied = original.model_copy()
        assert copied.file_hash == original.file_hash
        assert copied.stamp_data == original.stamp_data
        assert copied.namespace == original.namespace
        assert copied is not original  # Different instances

    def test_model_update(self):
        """Test model field updates."""
        file_hash = "a" * 64
        stamp_data = {"stamp_type": "inline"}
        namespace = "production"

        stamp = ModelMetadataStamp(
            file_hash=file_hash, stamp_data=stamp_data, namespace=namespace
        )

        # Update fields
        updated = stamp.model_copy(
            update={
                "stamp_data": {"stamp_type": "external", "source": "api"},
                "namespace": "staging",
            }
        )

        assert updated.stamp_data == {"stamp_type": "external", "source": "api"}
        assert updated.namespace == "staging"
        # Original should be unchanged
        assert stamp.stamp_data == stamp_data
        assert stamp.namespace == namespace

    def test_model_stamp_data(self):
        """Test stamp_data field."""
        file_hash = "a" * 64
        namespace = "production"

        # Test with different stamp_data structures
        stamp_data_variants = [
            {"stamp_type": "inline"},
            {"stamp_type": "external", "source": "api"},
            {"stamp_type": "inline", "metadata": {"key": "value", "count": 42}},
            {"stamp_type": "external", "source": "file", "path": "/path/to/file"},
            {"stamp_type": "inline", "metadata": {"nested": {"key": "value"}}},
        ]

        for stamp_data in stamp_data_variants:
            stamp = ModelMetadataStamp(
                file_hash=file_hash, stamp_data=stamp_data, namespace=namespace
            )

            assert stamp.stamp_data == stamp_data

    def test_model_jsonb_validation(self):
        """Test JSONB field validation."""
        file_hash = "a" * 64
        stamp_data = {"stamp_type": "inline", "test": "value"}
        namespace = "production"

        # This should not raise an error if validation is working correctly
        stamp = ModelMetadataStamp(
            file_hash=file_hash, stamp_data=stamp_data, namespace=namespace
        )

        # The _validate_jsonb_fields method should be called during initialization
        # If it's working correctly, no exception should be raised

    def test_model_workflow_id(self):
        """Test workflow_id field."""
        file_hash = "a" * 64
        stamp_data = {"stamp_type": "inline"}
        namespace = "production"

        # Test with workflow_id
        workflow_id = uuid4()
        stamp = ModelMetadataStamp(
            file_hash=file_hash,
            stamp_data=stamp_data,
            namespace=namespace,
            workflow_id=workflow_id,
        )

        assert stamp.workflow_id == workflow_id

        # Test without workflow_id (default None)
        stamp = ModelMetadataStamp(
            file_hash=file_hash, stamp_data=stamp_data, namespace=namespace
        )

        assert stamp.workflow_id is None

    def test_model_file_hash_pattern(self):
        """Test file_hash pattern validation."""
        stamp_data = {"stamp_type": "inline"}
        namespace = "production"

        # Test valid hex patterns
        valid_patterns = [
            "0123456789abcdef" * 4,  # All valid hex characters
            "fedcba9876543210" * 4,  # Reverse order
            "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",  # Mixed pattern (64 chars)
        ]

        for file_hash in valid_patterns:
            stamp = ModelMetadataStamp(
                file_hash=file_hash, stamp_data=stamp_data, namespace=namespace
            )

            assert stamp.file_hash == file_hash

        # Test invalid patterns
        invalid_patterns = [
            "g" * 64,  # Invalid hex character
            "A" * 64,  # Uppercase not allowed by pattern
            " " * 64,  # Spaces not allowed
            "0123456789abcde",  # Too short
            "0123456789abcdef0",  # Too long
        ]

        for file_hash in invalid_patterns:
            with pytest.raises(ValidationError):
                ModelMetadataStamp(
                    file_hash=file_hash, stamp_data=stamp_data, namespace=namespace
                )
