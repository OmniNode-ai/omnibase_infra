#!/usr/bin/env python3
"""Unit tests for ModelActionDedup.

Tests cover:
- Model instantiation with all fields
- Model validation (UUID, result_hash, temporal constraints)
- Default values and factory methods
- JSON serialization/deserialization
- Field validation (hash format, expiration logic)
- Model configuration
- TTL-based factory method
"""

import hashlib
import json
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError

from omninode_bridge.infrastructure.entities.model_action_dedup import ModelActionDedup


class TestModelActionDedup:
    """Test suite for ModelActionDedup."""

    def test_model_instantiation_with_all_fields(self):
        """Test model instantiation with all required fields."""
        workflow_key = "workflow-123"
        action_id = uuid4()
        result = {"status": "completed", "items": 100}
        result_hash = hashlib.sha256(
            json.dumps(result, sort_keys=True).encode()
        ).hexdigest()
        processed_at = datetime.now(UTC)
        expires_at = processed_at + timedelta(hours=6)

        dedup = ModelActionDedup(
            workflow_key=workflow_key,
            action_id=action_id,
            result_hash=result_hash,
            processed_at=processed_at,
            expires_at=expires_at,
        )

        assert dedup.workflow_key == workflow_key
        assert dedup.action_id == action_id
        assert dedup.result_hash == result_hash.lower()
        assert dedup.processed_at == processed_at
        assert dedup.expires_at == expires_at

    def test_model_validation_missing_required_fields(self):
        """Test model validation with missing required fields."""
        # Missing all fields
        with pytest.raises(ValidationError) as exc_info:
            ModelActionDedup()

        errors = exc_info.value.errors()
        assert (
            len(errors) >= 4
        )  # At least workflow_key, action_id, result_hash, expires_at
        field_names = {error["loc"][0] for error in errors}
        assert "workflow_key" in field_names
        assert "action_id" in field_names
        assert "result_hash" in field_names

    def test_model_validation_invalid_workflow_key(self):
        """Test model validation with invalid workflow_key."""
        action_id = uuid4()
        result_hash = hashlib.sha256(b"test").hexdigest()
        processed_at = datetime.now(UTC)
        expires_at = processed_at + timedelta(hours=6)

        # Empty workflow_key
        with pytest.raises(ValidationError) as exc_info:
            ModelActionDedup(
                workflow_key="",
                action_id=action_id,
                result_hash=result_hash,
                processed_at=processed_at,
                expires_at=expires_at,
            )

        assert "workflow_key" in str(exc_info.value)

    def test_model_validation_invalid_action_id(self):
        """Test model validation with invalid action_id."""
        workflow_key = "workflow-123"
        result_hash = hashlib.sha256(b"test").hexdigest()
        processed_at = datetime.now(UTC)
        expires_at = processed_at + timedelta(hours=6)

        # Invalid UUID string
        with pytest.raises(ValidationError) as exc_info:
            ModelActionDedup(
                workflow_key=workflow_key,
                action_id="not-a-uuid",
                result_hash=result_hash,
                processed_at=processed_at,
                expires_at=expires_at,
            )

        assert "action_id" in str(exc_info.value)

    def test_model_validation_invalid_result_hash(self):
        """Test model validation with invalid result_hash."""
        workflow_key = "workflow-123"
        action_id = uuid4()
        processed_at = datetime.now(UTC)
        expires_at = processed_at + timedelta(hours=6)

        # Too short hash
        with pytest.raises(ValidationError) as exc_info:
            ModelActionDedup(
                workflow_key=workflow_key,
                action_id=action_id,
                result_hash="short",
                processed_at=processed_at,
                expires_at=expires_at,
            )

        assert "result_hash" in str(exc_info.value)

        # Too long hash
        with pytest.raises(ValidationError) as exc_info:
            ModelActionDedup(
                workflow_key=workflow_key,
                action_id=action_id,
                result_hash="a" * 65,
                processed_at=processed_at,
                expires_at=expires_at,
            )

        assert "result_hash" in str(exc_info.value)

        # Invalid hex characters
        with pytest.raises(ValidationError) as exc_info:
            ModelActionDedup(
                workflow_key=workflow_key,
                action_id=action_id,
                result_hash="g" * 64,  # 'g' is not a valid hex character
                processed_at=processed_at,
                expires_at=expires_at,
            )

        assert "result_hash" in str(exc_info.value)

    def test_model_validation_result_hash_normalization(self):
        """Test that result_hash is normalized to lowercase."""
        workflow_key = "workflow-123"
        action_id = uuid4()
        result_hash_upper = "ABCDEF" + "0" * 58  # 64 chars total
        processed_at = datetime.now(UTC)
        expires_at = processed_at + timedelta(hours=6)

        dedup = ModelActionDedup(
            workflow_key=workflow_key,
            action_id=action_id,
            result_hash=result_hash_upper,
            processed_at=processed_at,
            expires_at=expires_at,
        )

        # Should be normalized to lowercase
        assert dedup.result_hash == result_hash_upper.lower()

    def test_model_validation_expires_at_constraint(self):
        """Test that expires_at must be after processed_at."""
        workflow_key = "workflow-123"
        action_id = uuid4()
        result_hash = hashlib.sha256(b"test").hexdigest()
        processed_at = datetime.now(UTC)

        # expires_at before processed_at (invalid)
        with pytest.raises(ValidationError) as exc_info:
            ModelActionDedup(
                workflow_key=workflow_key,
                action_id=action_id,
                result_hash=result_hash,
                processed_at=processed_at,
                expires_at=processed_at - timedelta(hours=1),
            )

        assert "expires_at" in str(exc_info.value)

        # expires_at equal to processed_at (invalid)
        with pytest.raises(ValidationError) as exc_info:
            ModelActionDedup(
                workflow_key=workflow_key,
                action_id=action_id,
                result_hash=result_hash,
                processed_at=processed_at,
                expires_at=processed_at,
            )

        assert "expires_at" in str(exc_info.value)

        # expires_at after processed_at (valid)
        dedup = ModelActionDedup(
            workflow_key=workflow_key,
            action_id=action_id,
            result_hash=result_hash,
            processed_at=processed_at,
            expires_at=processed_at + timedelta(hours=6),
        )

        assert dedup.expires_at > dedup.processed_at

    def test_model_serialization(self):
        """Test model serialization to JSON."""
        workflow_key = "workflow-123"
        action_id = uuid4()
        result_hash = hashlib.sha256(b"test").hexdigest()
        processed_at = datetime.now(UTC)
        expires_at = processed_at + timedelta(hours=6)

        dedup = ModelActionDedup(
            workflow_key=workflow_key,
            action_id=action_id,
            result_hash=result_hash,
            processed_at=processed_at,
            expires_at=expires_at,
        )

        # Test model_dump
        data = dedup.model_dump()
        assert data["workflow_key"] == workflow_key
        assert data["action_id"] == action_id  # UUID not converted in model_dump
        assert data["result_hash"] == result_hash

        # Test model_dump_json
        json_str = dedup.model_dump_json()
        parsed_data = json.loads(json_str)
        assert parsed_data["workflow_key"] == workflow_key
        assert parsed_data["action_id"] == str(action_id)  # UUID converted to string
        assert parsed_data["result_hash"] == result_hash

    def test_model_deserialization(self):
        """Test model deserialization from JSON."""
        workflow_key = "workflow-123"
        action_id = uuid4()
        result_hash = hashlib.sha256(b"test").hexdigest()
        processed_at = datetime.now(UTC)
        expires_at = processed_at + timedelta(hours=6)

        data = {
            "workflow_key": workflow_key,
            "action_id": str(action_id),
            "result_hash": result_hash,
            "processed_at": processed_at.isoformat(),
            "expires_at": expires_at.isoformat(),
        }

        # Test model_validate
        dedup = ModelActionDedup.model_validate(data)
        assert dedup.workflow_key == workflow_key
        assert dedup.action_id == action_id
        assert dedup.result_hash == result_hash

    def test_model_roundtrip(self):
        """Test model serialization and deserialization roundtrip."""
        workflow_key = "workflow-123"
        action_id = uuid4()
        result_hash = hashlib.sha256(b"test").hexdigest()
        processed_at = datetime.now(UTC)
        expires_at = processed_at + timedelta(hours=6)

        original = ModelActionDedup(
            workflow_key=workflow_key,
            action_id=action_id,
            result_hash=result_hash,
            processed_at=processed_at,
            expires_at=expires_at,
        )

        # Serialize and deserialize
        data = original.model_dump()
        restored = ModelActionDedup.model_validate(data)

        # Verify all fields match
        assert restored.workflow_key == original.workflow_key
        assert restored.action_id == original.action_id
        assert restored.result_hash == original.result_hash
        assert restored.processed_at == original.processed_at
        assert restored.expires_at == original.expires_at

    def test_model_equality(self):
        """Test model equality comparison."""
        workflow_key = "workflow-123"
        action_id = uuid4()
        result_hash = hashlib.sha256(b"test").hexdigest()
        processed_at = datetime.now(UTC)
        expires_at = processed_at + timedelta(hours=6)

        dedup1 = ModelActionDedup(
            workflow_key=workflow_key,
            action_id=action_id,
            result_hash=result_hash,
            processed_at=processed_at,
            expires_at=expires_at,
        )

        dedup2 = ModelActionDedup(
            workflow_key=workflow_key,
            action_id=action_id,
            result_hash=result_hash,
            processed_at=processed_at,
            expires_at=expires_at,
        )

        dedup3 = ModelActionDedup(
            workflow_key=workflow_key,
            action_id=uuid4(),  # Different action_id
            result_hash=result_hash,
            processed_at=processed_at,
            expires_at=expires_at,
        )

        # Test equality
        assert dedup1.model_dump() == dedup2.model_dump()
        assert dedup1.model_dump() != dedup3.model_dump()

    def test_model_copy(self):
        """Test model copying."""
        workflow_key = "workflow-123"
        action_id = uuid4()
        result_hash = hashlib.sha256(b"test").hexdigest()
        processed_at = datetime.now(UTC)
        expires_at = processed_at + timedelta(hours=6)

        original = ModelActionDedup(
            workflow_key=workflow_key,
            action_id=action_id,
            result_hash=result_hash,
            processed_at=processed_at,
            expires_at=expires_at,
        )

        # Test model_copy
        copied = original.model_copy()
        assert copied.workflow_key == original.workflow_key
        assert copied.action_id == original.action_id
        assert copied.result_hash == original.result_hash
        assert copied.processed_at == original.processed_at
        assert copied.expires_at == original.expires_at
        assert copied is not original  # Different instances

    def test_create_with_ttl_default(self):
        """Test factory method with default TTL."""
        workflow_key = "workflow-123"
        action_id = uuid4()
        result_hash = hashlib.sha256(b"test").hexdigest()

        dedup = ModelActionDedup.create_with_ttl(
            workflow_key=workflow_key,
            action_id=action_id,
            result_hash=result_hash,
        )

        assert dedup.workflow_key == workflow_key
        assert dedup.action_id == action_id
        assert dedup.result_hash == result_hash
        assert dedup.processed_at is not None
        assert dedup.expires_at is not None

        # Default TTL is 6 hours
        ttl = dedup.expires_at - dedup.processed_at
        assert ttl == timedelta(hours=6)

    def test_create_with_ttl_custom(self):
        """Test factory method with custom TTL."""
        workflow_key = "workflow-123"
        action_id = uuid4()
        result_hash = hashlib.sha256(b"test").hexdigest()
        custom_ttl_hours = 12

        dedup = ModelActionDedup.create_with_ttl(
            workflow_key=workflow_key,
            action_id=action_id,
            result_hash=result_hash,
            ttl_hours=custom_ttl_hours,
        )

        # Custom TTL is 12 hours
        ttl = dedup.expires_at - dedup.processed_at
        assert ttl == timedelta(hours=custom_ttl_hours)

    def test_create_with_ttl_custom_processed_at(self):
        """Test factory method with custom processed_at timestamp."""
        workflow_key = "workflow-123"
        action_id = uuid4()
        result_hash = hashlib.sha256(b"test").hexdigest()
        custom_processed_at = datetime(2025, 10, 21, 12, 0, 0, tzinfo=UTC)

        dedup = ModelActionDedup.create_with_ttl(
            workflow_key=workflow_key,
            action_id=action_id,
            result_hash=result_hash,
            ttl_hours=6,
            processed_at=custom_processed_at,
        )

        assert dedup.processed_at == custom_processed_at
        expected_expires_at = custom_processed_at + timedelta(hours=6)
        assert dedup.expires_at == expected_expires_at

    def test_hash_computation_consistency(self):
        """Test that hash computation is consistent and correct."""
        result = {"status": "completed", "items": 100, "namespace": "test"}

        # Compute hash multiple times
        hash1 = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
        hash2 = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()

        assert hash1 == hash2

        # Create dedup record with hash
        dedup = ModelActionDedup.create_with_ttl(
            workflow_key="workflow-123",
            action_id=uuid4(),
            result_hash=hash1,
        )

        assert dedup.result_hash == hash1.lower()

    def test_composite_primary_key_uniqueness(self):
        """Test that composite primary key (workflow_key, action_id) is unique."""
        workflow_key = "workflow-123"
        action_id = uuid4()
        result_hash = hashlib.sha256(b"test").hexdigest()

        dedup1 = ModelActionDedup.create_with_ttl(
            workflow_key=workflow_key,
            action_id=action_id,
            result_hash=result_hash,
        )

        dedup2 = ModelActionDedup.create_with_ttl(
            workflow_key=workflow_key,
            action_id=action_id,
            result_hash=result_hash,
        )

        # Same composite key
        assert dedup1.workflow_key == dedup2.workflow_key
        assert dedup1.action_id == dedup2.action_id

        # Different action_id, same workflow_key (should be allowed)
        dedup3 = ModelActionDedup.create_with_ttl(
            workflow_key=workflow_key,
            action_id=uuid4(),  # Different action_id
            result_hash=result_hash,
        )

        assert dedup3.workflow_key == dedup1.workflow_key
        assert dedup3.action_id != dedup1.action_id

    def test_ttl_expiration_logic(self):
        """Test TTL expiration logic."""
        workflow_key = "workflow-123"
        action_id = uuid4()
        result_hash = hashlib.sha256(b"test").hexdigest()
        processed_at = datetime.now(UTC)

        # Create with 1-hour TTL
        dedup = ModelActionDedup.create_with_ttl(
            workflow_key=workflow_key,
            action_id=action_id,
            result_hash=result_hash,
            ttl_hours=1,
            processed_at=processed_at,
        )

        # Check expiration
        now = datetime.now(UTC)
        future = processed_at + timedelta(hours=2)

        # Should not be expired now
        assert dedup.expires_at > now

        # Should be expired in 2 hours
        assert dedup.expires_at < future
