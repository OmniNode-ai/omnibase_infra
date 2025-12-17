# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ModelNodeRegistration.

Tests validate:
- Required field instantiation
- Optional field handling
- Mutable model (can update fields)
- JSON serialization/deserialization roundtrip
- Default values for optional fields
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from omnibase_infra.models.registration import ModelNodeRegistration


class TestModelNodeRegistrationBasicInstantiation:
    """Tests for basic model instantiation."""

    def test_valid_instantiation_required_fields(self) -> None:
        """Test creating registration with only required fields."""
        test_node_id = uuid4()
        now = datetime.now(UTC)
        registration = ModelNodeRegistration(
            node_id=test_node_id,
            node_type="effect",
            registered_at=now,
            updated_at=now,
        )
        assert registration.node_id == test_node_id
        assert registration.node_type == "effect"
        assert registration.node_version == "1.0.0"  # Default
        assert registration.capabilities == {}  # Default
        assert registration.endpoints == {}  # Default
        assert registration.metadata == {}  # Default
        assert registration.health_endpoint is None  # Default
        assert registration.last_heartbeat is None  # Default
        assert registration.registered_at == now
        assert registration.updated_at == now

    def test_valid_instantiation_all_fields(self) -> None:
        """Test creating registration with all fields populated."""
        test_node_id = uuid4()
        now = datetime.now(UTC)
        heartbeat_time = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)
        registration = ModelNodeRegistration(
            node_id=test_node_id,
            node_type="effect",
            node_version="2.1.0",
            capabilities={"postgres": True, "read": True, "write": True},
            endpoints={
                "health": "http://localhost:8080/health",
                "metrics": "http://localhost:8080/metrics",
            },
            metadata={"environment": "production", "region": "us-west-2"},
            health_endpoint="http://localhost:8080/health",
            last_heartbeat=heartbeat_time,
            registered_at=now,
            updated_at=now,
        )
        assert registration.node_id == test_node_id
        assert registration.node_type == "effect"
        assert registration.node_version == "2.1.0"
        assert registration.capabilities == {
            "postgres": True,
            "read": True,
            "write": True,
        }
        assert registration.endpoints["health"] == "http://localhost:8080/health"
        assert registration.endpoints["metrics"] == "http://localhost:8080/metrics"
        assert registration.metadata["environment"] == "production"
        assert str(registration.health_endpoint) == "http://localhost:8080/health"
        assert registration.last_heartbeat == heartbeat_time
        assert registration.registered_at == now
        assert registration.updated_at == now


class TestModelNodeRegistrationMutability:
    """Tests for mutable model (can update fields)."""

    def test_mutable_model_can_update_node_version(self) -> None:
        """Test that node_version can be modified after creation."""
        test_node_id = uuid4()
        now = datetime.now(UTC)
        registration = ModelNodeRegistration(
            node_id=test_node_id,
            node_type="effect",
            registered_at=now,
            updated_at=now,
        )
        registration.node_version = "2.0.0"
        assert registration.node_version == "2.0.0"

    def test_mutable_model_can_update_capabilities(self) -> None:
        """Test that capabilities dict can be reassigned."""
        test_node_id = uuid4()
        now = datetime.now(UTC)
        registration = ModelNodeRegistration(
            node_id=test_node_id,
            node_type="effect",
            capabilities={"original": True},
            registered_at=now,
            updated_at=now,
        )
        registration.capabilities = {"modified": True, "new_capability": True}
        assert registration.capabilities == {"modified": True, "new_capability": True}

    def test_mutable_model_can_update_endpoints(self) -> None:
        """Test that endpoints dict can be reassigned."""
        test_node_id = uuid4()
        now = datetime.now(UTC)
        registration = ModelNodeRegistration(
            node_id=test_node_id,
            node_type="effect",
            endpoints={"health": "http://old/health"},
            registered_at=now,
            updated_at=now,
        )
        registration.endpoints = {
            "health": "http://new/health",
            "api": "http://new/api",
        }
        assert registration.endpoints["health"] == "http://new/health"
        assert registration.endpoints["api"] == "http://new/api"

    def test_mutable_model_can_update_metadata(self) -> None:
        """Test that metadata dict can be reassigned."""
        test_node_id = uuid4()
        now = datetime.now(UTC)
        registration = ModelNodeRegistration(
            node_id=test_node_id,
            node_type="effect",
            metadata={"key": "old_value"},
            registered_at=now,
            updated_at=now,
        )
        registration.metadata = {"key": "new_value", "extra": "data"}
        assert registration.metadata == {"key": "new_value", "extra": "data"}

    def test_mutable_model_can_update_health_endpoint(self) -> None:
        """Test that health_endpoint can be modified."""
        test_node_id = uuid4()
        now = datetime.now(UTC)
        registration = ModelNodeRegistration(
            node_id=test_node_id,
            node_type="effect",
            registered_at=now,
            updated_at=now,
        )
        assert registration.health_endpoint is None
        registration.health_endpoint = "http://localhost:8080/health"
        assert registration.health_endpoint == "http://localhost:8080/health"

    def test_mutable_model_can_update_last_heartbeat(self) -> None:
        """Test that last_heartbeat can be modified."""
        test_node_id = uuid4()
        now = datetime.now(UTC)
        registration = ModelNodeRegistration(
            node_id=test_node_id,
            node_type="effect",
            registered_at=now,
            updated_at=now,
        )
        assert registration.last_heartbeat is None
        new_heartbeat = datetime.now(UTC)
        registration.last_heartbeat = new_heartbeat
        assert registration.last_heartbeat == new_heartbeat

    def test_mutable_model_can_update_updated_at(self) -> None:
        """Test that updated_at can be modified."""
        test_node_id = uuid4()
        initial_time = datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC)
        registration = ModelNodeRegistration(
            node_id=test_node_id,
            node_type="effect",
            registered_at=initial_time,
            updated_at=initial_time,
        )
        new_time = datetime.now(UTC)
        registration.updated_at = new_time
        assert registration.updated_at == new_time

    def test_mutable_model_can_update_node_id(self) -> None:
        """Test that node_id can be modified (though unusual)."""
        test_node_id = uuid4()
        new_node_id = uuid4()
        now = datetime.now(UTC)
        registration = ModelNodeRegistration(
            node_id=test_node_id,
            node_type="effect",
            registered_at=now,
            updated_at=now,
        )
        registration.node_id = new_node_id
        assert registration.node_id == new_node_id

    def test_mutable_model_can_update_node_type(self) -> None:
        """Test that node_type can be modified (though unusual)."""
        test_node_id = uuid4()
        now = datetime.now(UTC)
        registration = ModelNodeRegistration(
            node_id=test_node_id,
            node_type="effect",
            registered_at=now,
            updated_at=now,
        )
        registration.node_type = "compute"
        assert registration.node_type == "compute"


class TestModelNodeRegistrationDefaultValues:
    """Tests for default values."""

    def test_default_node_version(self) -> None:
        """Test that node_version defaults to '1.0.0'."""
        test_node_id = uuid4()
        now = datetime.now(UTC)
        registration = ModelNodeRegistration(
            node_id=test_node_id,
            node_type="effect",
            registered_at=now,
            updated_at=now,
        )
        assert registration.node_version == "1.0.0"

    def test_default_capabilities_empty_dict(self) -> None:
        """Test that capabilities defaults to empty dict."""
        test_node_id = uuid4()
        now = datetime.now(UTC)
        registration = ModelNodeRegistration(
            node_id=test_node_id,
            node_type="effect",
            registered_at=now,
            updated_at=now,
        )
        assert registration.capabilities == {}
        assert isinstance(registration.capabilities, dict)

    def test_default_endpoints_empty_dict(self) -> None:
        """Test that endpoints defaults to empty dict."""
        test_node_id = uuid4()
        now = datetime.now(UTC)
        registration = ModelNodeRegistration(
            node_id=test_node_id,
            node_type="effect",
            registered_at=now,
            updated_at=now,
        )
        assert registration.endpoints == {}
        assert isinstance(registration.endpoints, dict)

    def test_default_metadata_empty_dict(self) -> None:
        """Test that metadata defaults to empty dict."""
        test_node_id = uuid4()
        now = datetime.now(UTC)
        registration = ModelNodeRegistration(
            node_id=test_node_id,
            node_type="effect",
            registered_at=now,
            updated_at=now,
        )
        assert registration.metadata == {}
        assert isinstance(registration.metadata, dict)

    def test_default_health_endpoint_none(self) -> None:
        """Test that health_endpoint defaults to None."""
        test_node_id = uuid4()
        now = datetime.now(UTC)
        registration = ModelNodeRegistration(
            node_id=test_node_id,
            node_type="effect",
            registered_at=now,
            updated_at=now,
        )
        assert registration.health_endpoint is None

    def test_default_last_heartbeat_none(self) -> None:
        """Test that last_heartbeat defaults to None."""
        test_node_id = uuid4()
        now = datetime.now(UTC)
        registration = ModelNodeRegistration(
            node_id=test_node_id,
            node_type="effect",
            registered_at=now,
            updated_at=now,
        )
        assert registration.last_heartbeat is None

    def test_default_dicts_are_independent(self) -> None:
        """Test that default dicts are independent between instances."""
        test_node_id1 = uuid4()
        test_node_id2 = uuid4()
        now = datetime.now(UTC)
        reg1 = ModelNodeRegistration(
            node_id=test_node_id1,
            node_type="effect",
            registered_at=now,
            updated_at=now,
        )
        reg2 = ModelNodeRegistration(
            node_id=test_node_id2,
            node_type="compute",
            registered_at=now,
            updated_at=now,
        )
        # Modify reg1's capabilities
        reg1.capabilities["test"] = True
        # reg2's capabilities should be unaffected
        assert "test" not in reg2.capabilities


class TestModelNodeRegistrationSerialization:
    """Tests for JSON serialization and deserialization."""

    def test_json_serialization_roundtrip_minimal(self) -> None:
        """Test JSON serialization and deserialization with minimal fields."""
        test_node_id = uuid4()
        now = datetime.now(UTC)
        registration = ModelNodeRegistration(
            node_id=test_node_id,
            node_type="reducer",
            registered_at=now,
            updated_at=now,
        )
        json_str = registration.model_dump_json()
        restored = ModelNodeRegistration.model_validate_json(json_str)
        assert restored.node_id == registration.node_id
        assert restored.node_type == registration.node_type
        assert restored.node_version == registration.node_version
        assert restored.capabilities == registration.capabilities
        assert restored.endpoints == registration.endpoints

    def test_json_serialization_roundtrip_full(self) -> None:
        """Test JSON serialization and deserialization with all fields."""
        test_node_id = uuid4()
        now = datetime.now(UTC)
        heartbeat_time = datetime(2025, 6, 15, 10, 30, 0, tzinfo=UTC)
        registration = ModelNodeRegistration(
            node_id=test_node_id,
            node_type="effect",
            node_version="3.0.0",
            capabilities={"database": True, "transactions": True},
            endpoints={"health": "http://localhost:8080/health"},
            metadata={"cluster": "primary", "priority": 1},
            health_endpoint="http://localhost:8080/health",
            last_heartbeat=heartbeat_time,
            registered_at=now,
            updated_at=now,
        )
        json_str = registration.model_dump_json()
        restored = ModelNodeRegistration.model_validate_json(json_str)

        assert restored.node_id == registration.node_id
        assert restored.node_type == registration.node_type
        assert restored.node_version == registration.node_version
        assert restored.capabilities == registration.capabilities
        assert restored.endpoints == registration.endpoints
        assert restored.metadata == registration.metadata
        assert restored.health_endpoint == registration.health_endpoint
        assert restored.last_heartbeat == registration.last_heartbeat
        # Timestamps should match within reasonable precision
        assert (
            abs((restored.registered_at - registration.registered_at).total_seconds())
            < 1
        )
        assert abs((restored.updated_at - registration.updated_at).total_seconds()) < 1

    def test_model_dump_dict(self) -> None:
        """Test model_dump produces correct dict structure."""
        test_node_id = uuid4()
        now = datetime.now(UTC)
        registration = ModelNodeRegistration(
            node_id=test_node_id,
            node_type="effect",
            node_version="1.5.0",
            capabilities={"feature": True},
            registered_at=now,
            updated_at=now,
        )
        data = registration.model_dump()
        assert isinstance(data, dict)
        assert data["node_id"] == test_node_id
        assert data["node_type"] == "effect"
        assert data["node_version"] == "1.5.0"
        assert data["capabilities"] == {"feature": True}

    def test_model_dump_mode_json(self) -> None:
        """Test model_dump with mode='json' for JSON-compatible output."""
        test_node_id = uuid4()
        now = datetime.now(UTC)
        registration = ModelNodeRegistration(
            node_id=test_node_id,
            node_type="compute",
            registered_at=now,
            updated_at=now,
        )
        data = registration.model_dump(mode="json")
        # UUID should be serialized as string in JSON mode
        assert data["node_id"] == str(test_node_id)
        # Datetime should be serialized as ISO string
        assert isinstance(data["registered_at"], str)
        assert isinstance(data["updated_at"], str)


class TestModelNodeRegistrationRequiredFields:
    """Tests for required field validation."""

    def test_missing_node_id_raises_validation_error(self) -> None:
        """Test that missing node_id raises ValidationError."""
        now = datetime.now(UTC)
        with pytest.raises(ValidationError) as exc_info:
            ModelNodeRegistration(
                node_type="effect",
                registered_at=now,
                updated_at=now,
            )  # type: ignore[call-arg]
        assert "node_id" in str(exc_info.value)

    def test_missing_node_type_raises_validation_error(self) -> None:
        """Test that missing node_type raises ValidationError."""
        test_node_id = uuid4()
        now = datetime.now(UTC)
        with pytest.raises(ValidationError) as exc_info:
            ModelNodeRegistration(
                node_id=test_node_id,
                registered_at=now,
                updated_at=now,
            )  # type: ignore[call-arg]
        assert "node_type" in str(exc_info.value)

    def test_missing_registered_at_raises_validation_error(self) -> None:
        """Test that missing registered_at raises ValidationError."""
        test_node_id = uuid4()
        now = datetime.now(UTC)
        with pytest.raises(ValidationError) as exc_info:
            ModelNodeRegistration(
                node_id=test_node_id,
                node_type="effect",
                updated_at=now,
            )  # type: ignore[call-arg]
        assert "registered_at" in str(exc_info.value)

    def test_missing_updated_at_raises_validation_error(self) -> None:
        """Test that missing updated_at raises ValidationError."""
        test_node_id = uuid4()
        now = datetime.now(UTC)
        with pytest.raises(ValidationError) as exc_info:
            ModelNodeRegistration(
                node_id=test_node_id,
                node_type="effect",
                registered_at=now,
            )  # type: ignore[call-arg]
        assert "updated_at" in str(exc_info.value)


class TestModelNodeRegistrationEdgeCases:
    """Tests for edge cases and special values."""

    def test_invalid_node_id_empty_string_raises_error(self) -> None:
        """Test that empty string is not allowed for node_id (UUID type)."""
        now = datetime.now(UTC)
        with pytest.raises(ValidationError):
            ModelNodeRegistration(
                node_id="",  # type: ignore[arg-type]
                node_type="effect",
                registered_at=now,
                updated_at=now,
            )

    def test_empty_string_node_type_rejected(self) -> None:
        """Test that empty string is rejected for node_type.

        ModelNodeRegistration uses strict Literal validation matching
        ModelNodeIntrospectionEvent. Only "effect", "compute", "reducer",
        and "orchestrator" are valid node types.
        """
        test_node_id = uuid4()
        now = datetime.now(UTC)
        with pytest.raises(ValidationError) as exc_info:
            ModelNodeRegistration(
                node_id=test_node_id,
                node_type="",
                registered_at=now,
                updated_at=now,
            )
        assert "node_type" in str(exc_info.value)
        assert "literal_error" in str(exc_info.value)

    def test_complex_capabilities_dict(self) -> None:
        """Test capabilities with complex nested values."""
        test_node_id = uuid4()
        now = datetime.now(UTC)
        complex_capabilities: dict[str, Any] = {
            "database": True,
            "max_connections": 100,
            "supported_operations": ["read", "write", "delete"],
            "config": {"pool_size": 10, "timeout": 30},
        }
        registration = ModelNodeRegistration(
            node_id=test_node_id,
            node_type="effect",
            capabilities=complex_capabilities,
            registered_at=now,
            updated_at=now,
        )
        assert registration.capabilities["database"] is True
        assert registration.capabilities["max_connections"] == 100
        assert registration.capabilities["supported_operations"] == [
            "read",
            "write",
            "delete",
        ]
        assert registration.capabilities["config"]["pool_size"] == 10

    def test_complex_metadata_dict(self) -> None:
        """Test metadata with complex nested values."""
        test_node_id = uuid4()
        now = datetime.now(UTC)
        complex_metadata: dict[str, Any] = {
            "environment": "production",
            "tags": ["primary", "critical"],
            "config": {"replicas": 3, "region": "us-west-2"},
        }
        registration = ModelNodeRegistration(
            node_id=test_node_id,
            node_type="effect",
            metadata=complex_metadata,
            registered_at=now,
            updated_at=now,
        )
        assert registration.metadata["environment"] == "production"
        assert registration.metadata["tags"] == ["primary", "critical"]
        assert registration.metadata["config"]["replicas"] == 3

    def test_unicode_in_node_type_rejected(self) -> None:
        """Test that Unicode node_type is rejected.

        ModelNodeRegistration uses strict Literal validation matching
        ModelNodeIntrospectionEvent. Only the four canonical ONEX node types
        are allowed. This test verifies unicode metadata is still allowed
        while node_type is strictly validated.
        """
        test_node_id = uuid4()
        now = datetime.now(UTC)
        # Unicode node_type should be rejected
        with pytest.raises(ValidationError) as exc_info:
            ModelNodeRegistration(
                node_id=test_node_id,
                node_type="效果节点",
                node_version="1.0.0",
                metadata={"description": "Узел обработки"},
                registered_at=now,
                updated_at=now,
            )
        assert "node_type" in str(exc_info.value)
        assert "literal_error" in str(exc_info.value)

    def test_unicode_in_metadata_allowed(self) -> None:
        """Test that Unicode characters are allowed in metadata fields."""
        test_node_id = uuid4()
        now = datetime.now(UTC)
        registration = ModelNodeRegistration(
            node_id=test_node_id,
            node_type="effect",  # Valid canonical node type
            node_version="1.0.0",
            metadata={"description": "Узел обработки", "名前": "効果ノード"},
            registered_at=now,
            updated_at=now,
        )
        assert registration.node_id == test_node_id
        assert registration.node_type == "effect"
        assert registration.metadata["description"] == "Узел обработки"
        assert registration.metadata["名前"] == "効果ノード"

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are forbidden by model config."""
        test_node_id = uuid4()
        now = datetime.now(UTC)
        with pytest.raises(ValidationError) as exc_info:
            ModelNodeRegistration(
                node_id=test_node_id,
                node_type="effect",
                registered_at=now,
                updated_at=now,
                extra_field="not_allowed",  # type: ignore[call-arg]
            )
        assert "extra_field" in str(exc_info.value)

    def test_long_version_string(self) -> None:
        """Test that long version strings are allowed."""
        test_node_id = uuid4()
        now = datetime.now(UTC)
        long_version = "1.0.0-alpha.1.20250115.build123456+metadata.abcdef"
        registration = ModelNodeRegistration(
            node_id=test_node_id,
            node_type="effect",
            node_version=long_version,
            registered_at=now,
            updated_at=now,
        )
        assert registration.node_version == long_version

    def test_long_health_endpoint_url(self) -> None:
        """Test that long health endpoint URLs are allowed."""
        test_node_id = uuid4()
        now = datetime.now(UTC)
        long_url = "http://subdomain.example.com:8080/api/v1/health/deep/check?timeout=30&include=all"
        registration = ModelNodeRegistration(
            node_id=test_node_id,
            node_type="effect",
            health_endpoint=long_url,
            registered_at=now,
            updated_at=now,
        )
        assert str(registration.health_endpoint) == long_url


class TestModelNodeRegistrationTimestampHandling:
    """Tests for timestamp field handling."""

    def test_registered_at_and_updated_at_can_differ(self) -> None:
        """Test that registered_at and updated_at can have different values."""
        test_node_id = uuid4()
        registered = datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC)
        updated = datetime(2025, 6, 15, 12, 0, 0, tzinfo=UTC)
        registration = ModelNodeRegistration(
            node_id=test_node_id,
            node_type="effect",
            registered_at=registered,
            updated_at=updated,
        )
        assert registration.registered_at == registered
        assert registration.updated_at == updated
        assert registration.updated_at > registration.registered_at

    def test_last_heartbeat_can_be_different_from_other_timestamps(self) -> None:
        """Test that last_heartbeat can have different value from other timestamps."""
        test_node_id = uuid4()
        registered = datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC)
        updated = datetime(2025, 1, 2, 0, 0, 0, tzinfo=UTC)
        heartbeat = datetime(2025, 1, 3, 0, 0, 0, tzinfo=UTC)
        registration = ModelNodeRegistration(
            node_id=test_node_id,
            node_type="effect",
            registered_at=registered,
            updated_at=updated,
            last_heartbeat=heartbeat,
        )
        assert registration.registered_at == registered
        assert registration.updated_at == updated
        assert registration.last_heartbeat == heartbeat

    def test_timestamps_can_be_same(self) -> None:
        """Test that all timestamps can have the same value."""
        test_node_id = uuid4()
        same_time = datetime.now(UTC)
        registration = ModelNodeRegistration(
            node_id=test_node_id,
            node_type="effect",
            registered_at=same_time,
            updated_at=same_time,
            last_heartbeat=same_time,
        )
        assert registration.registered_at == same_time
        assert registration.updated_at == same_time
        assert registration.last_heartbeat == same_time


class TestModelNodeRegistrationFromAttributes:
    """Tests for from_attributes configuration (ORM mode)."""

    def test_from_dict_like_object(self) -> None:
        """Test creating model from dict-like object."""
        test_node_id = uuid4()
        now = datetime.now(UTC)

        class DictLike:
            def __init__(self, node_id: UUID) -> None:
                self.node_id = node_id
                self.node_type = "compute"
                self.node_version = "1.0.0"
                self.capabilities: dict[str, bool] = {}
                self.endpoints: dict[str, str] = {}
                self.metadata: dict[str, str] = {}
                self.health_endpoint = None
                self.last_heartbeat = None
                self.registered_at = now
                self.updated_at = now

        obj = DictLike(test_node_id)
        registration = ModelNodeRegistration.model_validate(obj)
        assert registration.node_id == test_node_id
        assert registration.node_type == "compute"
        assert registration.registered_at == now
        assert registration.updated_at == now


class TestModelNodeRegistrationMutableDicts:
    """Tests for mutable dict field behavior."""

    def test_can_mutate_capabilities_in_place(self) -> None:
        """Test that capabilities dict can be mutated in place."""
        test_node_id = uuid4()
        now = datetime.now(UTC)
        registration = ModelNodeRegistration(
            node_id=test_node_id,
            node_type="effect",
            capabilities={"original": True},
            registered_at=now,
            updated_at=now,
        )
        registration.capabilities["new_key"] = "new_value"
        assert registration.capabilities["original"] is True
        assert registration.capabilities["new_key"] == "new_value"

    def test_can_mutate_endpoints_in_place(self) -> None:
        """Test that endpoints dict can be mutated in place."""
        test_node_id = uuid4()
        now = datetime.now(UTC)
        registration = ModelNodeRegistration(
            node_id=test_node_id,
            node_type="effect",
            endpoints={"health": "http://localhost/health"},
            registered_at=now,
            updated_at=now,
        )
        registration.endpoints["metrics"] = "http://localhost/metrics"
        assert registration.endpoints["health"] == "http://localhost/health"
        assert registration.endpoints["metrics"] == "http://localhost/metrics"

    def test_can_mutate_metadata_in_place(self) -> None:
        """Test that metadata dict can be mutated in place."""
        test_node_id = uuid4()
        now = datetime.now(UTC)
        registration = ModelNodeRegistration(
            node_id=test_node_id,
            node_type="effect",
            metadata={"key1": "value1"},
            registered_at=now,
            updated_at=now,
        )
        registration.metadata["key2"] = "value2"
        del registration.metadata["key1"]
        assert "key1" not in registration.metadata
        assert registration.metadata["key2"] == "value2"

    def test_can_clear_dicts(self) -> None:
        """Test that dict fields can be cleared."""
        test_node_id = uuid4()
        now = datetime.now(UTC)
        registration = ModelNodeRegistration(
            node_id=test_node_id,
            node_type="effect",
            capabilities={"key": "value"},
            endpoints={"health": "http://localhost/health"},
            metadata={"meta": "data"},
            registered_at=now,
            updated_at=now,
        )
        registration.capabilities.clear()
        registration.endpoints.clear()
        registration.metadata.clear()
        assert registration.capabilities == {}
        assert registration.endpoints == {}
        assert registration.metadata == {}


class TestModelNodeRegistrationEquality:
    """Tests for model equality comparison."""

    def test_equal_registrations_are_equal(self) -> None:
        """Test that two registrations with same values are equal."""
        test_node_id = uuid4()
        now = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        reg1 = ModelNodeRegistration(
            node_id=test_node_id,
            node_type="effect",
            registered_at=now,
            updated_at=now,
        )
        reg2 = ModelNodeRegistration(
            node_id=test_node_id,
            node_type="effect",
            registered_at=now,
            updated_at=now,
        )
        assert reg1 == reg2

    def test_different_node_id_not_equal(self) -> None:
        """Test that registrations with different node_id are not equal."""
        now = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        reg1 = ModelNodeRegistration(
            node_id=uuid4(),
            node_type="effect",
            registered_at=now,
            updated_at=now,
        )
        reg2 = ModelNodeRegistration(
            node_id=uuid4(),
            node_type="effect",
            registered_at=now,
            updated_at=now,
        )
        assert reg1 != reg2

    def test_different_node_type_not_equal(self) -> None:
        """Test that registrations with different node_type are not equal."""
        test_node_id = uuid4()
        now = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        reg1 = ModelNodeRegistration(
            node_id=test_node_id,
            node_type="effect",
            registered_at=now,
            updated_at=now,
        )
        reg2 = ModelNodeRegistration(
            node_id=test_node_id,
            node_type="compute",
            registered_at=now,
            updated_at=now,
        )
        assert reg1 != reg2

    def test_not_equal_to_non_model(self) -> None:
        """Test that registration is not equal to non-model objects."""
        test_node_id = uuid4()
        now = datetime.now(UTC)
        registration = ModelNodeRegistration(
            node_id=test_node_id,
            node_type="effect",
            registered_at=now,
            updated_at=now,
        )
        assert registration != "not a model"
        assert registration != 42
        assert registration != None


class TestModelNodeRegistrationStringRepresentation:
    """Tests for model string representation."""

    def test_str_contains_node_id(self) -> None:
        """Test that __str__ contains field information."""
        test_node_id = uuid4()
        now = datetime.now(UTC)
        registration = ModelNodeRegistration(
            node_id=test_node_id,
            node_type="effect",
            registered_at=now,
            updated_at=now,
        )
        str_repr = str(registration)
        # Pydantic models include field values in string representation
        assert "node_id" in str_repr or str(test_node_id) in str_repr

    def test_repr_is_valid(self) -> None:
        """Test that __repr__ produces valid representation."""
        test_node_id = uuid4()
        now = datetime.now(UTC)
        registration = ModelNodeRegistration(
            node_id=test_node_id,
            node_type="effect",
            registered_at=now,
            updated_at=now,
        )
        repr_str = repr(registration)
        assert isinstance(repr_str, str)
        assert len(repr_str) > 0

    def test_str_and_repr_contain_node_type(self) -> None:
        """Test that string representations contain node_type."""
        test_node_id = uuid4()
        now = datetime.now(UTC)
        registration = ModelNodeRegistration(
            node_id=test_node_id,
            node_type="orchestrator",
            registered_at=now,
            updated_at=now,
        )
        str_repr = str(registration)
        repr_str = repr(registration)
        # At least one should contain the node_type value
        assert "orchestrator" in str_repr or "orchestrator" in repr_str


class TestModelNodeRegistrationCopying:
    """Tests for model copying behavior."""

    def test_model_copy_creates_new_instance(self) -> None:
        """Test that model_copy creates a new instance."""
        test_node_id = uuid4()
        now = datetime.now(UTC)
        registration = ModelNodeRegistration(
            node_id=test_node_id,
            node_type="effect",
            registered_at=now,
            updated_at=now,
        )
        copied = registration.model_copy()
        assert copied is not registration
        assert copied == registration

    def test_model_copy_with_update(self) -> None:
        """Test that model_copy can update fields."""
        test_node_id = uuid4()
        new_node_id = uuid4()
        now = datetime.now(UTC)
        registration = ModelNodeRegistration(
            node_id=test_node_id,
            node_type="effect",
            registered_at=now,
            updated_at=now,
        )
        copied = registration.model_copy(update={"node_id": new_node_id})
        assert copied.node_id == new_node_id
        assert copied.node_type == registration.node_type
        # Original is unchanged
        assert registration.node_id == test_node_id

    def test_model_copy_deep(self) -> None:
        """Test that deep copy creates independent nested objects."""
        test_node_id = uuid4()
        now = datetime.now(UTC)
        registration = ModelNodeRegistration(
            node_id=test_node_id,
            node_type="effect",
            capabilities={"key": "value"},
            registered_at=now,
            updated_at=now,
        )
        copied = registration.model_copy(deep=True)
        # Both should have same values
        assert copied.capabilities == registration.capabilities
        # But dict should be independent (deep copy)
        assert copied.capabilities is not registration.capabilities
        # Modifying copy should not affect original
        copied.capabilities["new_key"] = "new_value"
        assert "new_key" not in registration.capabilities

    def test_model_copy_shallow_shares_dict_reference(self) -> None:
        """Test that shallow copy shares nested dict references."""
        test_node_id = uuid4()
        now = datetime.now(UTC)
        registration = ModelNodeRegistration(
            node_id=test_node_id,
            node_type="effect",
            capabilities={"key": "value"},
            registered_at=now,
            updated_at=now,
        )
        copied = registration.model_copy(deep=False)
        # Shallow copy shares the same dict reference
        assert copied.capabilities is registration.capabilities


class TestModelNodeRegistrationHashing:
    """Tests for model hashing behavior (mutable model)."""

    def test_mutable_model_not_hashable_by_default(self) -> None:
        """Test that mutable model is not hashable by default.

        Note: Mutable Pydantic models are not hashable because their
        fields can change, which would break hash consistency.
        """
        test_node_id = uuid4()
        now = datetime.now(UTC)
        registration = ModelNodeRegistration(
            node_id=test_node_id,
            node_type="effect",
            registered_at=now,
            updated_at=now,
        )
        # Mutable models should raise TypeError when hashed
        with pytest.raises(TypeError):
            hash(registration)


class TestModelNodeRegistrationSemverValidation:
    """Tests for semantic version validation on node_version field."""

    def test_valid_semver_basic(self) -> None:
        """Test that basic semver strings are accepted."""
        test_node_id = uuid4()
        now = datetime.now(UTC)
        valid_versions = ["0.0.0", "1.0.0", "2.1.3", "10.20.30", "999.999.999"]
        for version in valid_versions:
            registration = ModelNodeRegistration(
                node_id=test_node_id,
                node_type="effect",
                node_version=version,
                registered_at=now,
                updated_at=now,
            )
            assert registration.node_version == version

    def test_valid_semver_with_prerelease(self) -> None:
        """Test that semver with prerelease identifiers are accepted."""
        test_node_id = uuid4()
        now = datetime.now(UTC)
        valid_versions = [
            "1.0.0-alpha",
            "1.0.0-alpha.1",
            "1.0.0-beta",
            "1.0.0-beta.2",
            "1.0.0-rc.1",
            "1.0.0-0.3.7",
            "1.0.0-x.7.z.92",
        ]
        for version in valid_versions:
            registration = ModelNodeRegistration(
                node_id=test_node_id,
                node_type="effect",
                node_version=version,
                registered_at=now,
                updated_at=now,
            )
            assert registration.node_version == version

    def test_valid_semver_with_build_metadata(self) -> None:
        """Test that semver with build metadata are accepted."""
        test_node_id = uuid4()
        now = datetime.now(UTC)
        valid_versions = [
            "1.0.0+build.123",
            "1.0.0+20130313144700",
            "1.0.0+exp.sha.5114f85",
        ]
        for version in valid_versions:
            registration = ModelNodeRegistration(
                node_id=test_node_id,
                node_type="effect",
                node_version=version,
                registered_at=now,
                updated_at=now,
            )
            assert registration.node_version == version

    def test_valid_semver_with_prerelease_and_build(self) -> None:
        """Test that semver with both prerelease and build metadata are accepted."""
        test_node_id = uuid4()
        now = datetime.now(UTC)
        valid_versions = [
            "1.0.0-alpha+001",
            "1.0.0-alpha.1+build.123",
            "1.0.0-beta.2+exp.sha.5114f85",
        ]
        for version in valid_versions:
            registration = ModelNodeRegistration(
                node_id=test_node_id,
                node_type="effect",
                node_version=version,
                registered_at=now,
                updated_at=now,
            )
            assert registration.node_version == version

    def test_invalid_semver_missing_patch(self) -> None:
        """Test that version missing patch number is rejected."""
        test_node_id = uuid4()
        now = datetime.now(UTC)
        with pytest.raises(ValidationError) as exc_info:
            ModelNodeRegistration(
                node_id=test_node_id,
                node_type="effect",
                node_version="1.0",
                registered_at=now,
                updated_at=now,
            )
        assert "Invalid semantic version" in str(exc_info.value)

    def test_invalid_semver_with_v_prefix(self) -> None:
        """Test that version with 'v' prefix is rejected."""
        test_node_id = uuid4()
        now = datetime.now(UTC)
        with pytest.raises(ValidationError) as exc_info:
            ModelNodeRegistration(
                node_id=test_node_id,
                node_type="effect",
                node_version="v1.0.0",
                registered_at=now,
                updated_at=now,
            )
        assert "Invalid semantic version" in str(exc_info.value)

    def test_invalid_semver_four_parts(self) -> None:
        """Test that version with four parts is rejected."""
        test_node_id = uuid4()
        now = datetime.now(UTC)
        with pytest.raises(ValidationError) as exc_info:
            ModelNodeRegistration(
                node_id=test_node_id,
                node_type="effect",
                node_version="1.0.0.0",
                registered_at=now,
                updated_at=now,
            )
        assert "Invalid semantic version" in str(exc_info.value)

    def test_invalid_semver_arbitrary_string(self) -> None:
        """Test that arbitrary strings are rejected."""
        test_node_id = uuid4()
        now = datetime.now(UTC)
        invalid_versions = ["invalid", "latest", "stable", "dev", ""]
        for version in invalid_versions:
            with pytest.raises(ValidationError) as exc_info:
                ModelNodeRegistration(
                    node_id=test_node_id,
                    node_type="effect",
                    node_version=version,
                    registered_at=now,
                    updated_at=now,
                )
            assert "Invalid semantic version" in str(exc_info.value)

    def test_invalid_semver_non_numeric_parts(self) -> None:
        """Test that versions with non-numeric parts are rejected."""
        test_node_id = uuid4()
        now = datetime.now(UTC)
        invalid_versions = ["a.b.c", "1.x.0", "1.0.x"]
        for version in invalid_versions:
            with pytest.raises(ValidationError) as exc_info:
                ModelNodeRegistration(
                    node_id=test_node_id,
                    node_type="effect",
                    node_version=version,
                    registered_at=now,
                    updated_at=now,
                )
            assert "Invalid semantic version" in str(exc_info.value)

    def test_semver_error_message_format(self) -> None:
        """Test that validation error contains helpful message."""
        test_node_id = uuid4()
        now = datetime.now(UTC)
        with pytest.raises(ValidationError) as exc_info:
            ModelNodeRegistration(
                node_id=test_node_id,
                node_type="effect",
                node_version="invalid",
                registered_at=now,
                updated_at=now,
            )
        error_message = str(exc_info.value)
        assert "Invalid semantic version 'invalid'" in error_message
        assert "MAJOR.MINOR.PATCH" in error_message


class TestModelNodeRegistrationHealthEndpointValidation:
    """Tests for health_endpoint URL validation using HttpUrl."""

    def test_valid_http_url(self) -> None:
        """Test that valid HTTP URLs are accepted."""
        test_node_id = uuid4()
        now = datetime.now(UTC)
        registration = ModelNodeRegistration(
            node_id=test_node_id,
            node_type="effect",
            health_endpoint="http://localhost:8080/health",
            registered_at=now,
            updated_at=now,
        )
        assert str(registration.health_endpoint) == "http://localhost:8080/health"

    def test_valid_https_url(self) -> None:
        """Test that valid HTTPS URLs are accepted."""
        test_node_id = uuid4()
        now = datetime.now(UTC)
        registration = ModelNodeRegistration(
            node_id=test_node_id,
            node_type="effect",
            health_endpoint="https://api.example.com/health",
            registered_at=now,
            updated_at=now,
        )
        assert str(registration.health_endpoint) == "https://api.example.com/health"

    def test_valid_url_with_path_and_query(self) -> None:
        """Test that URLs with paths and query parameters are accepted."""
        test_node_id = uuid4()
        now = datetime.now(UTC)
        registration = ModelNodeRegistration(
            node_id=test_node_id,
            node_type="effect",
            health_endpoint="http://localhost:8080/api/v1/health?timeout=30",
            registered_at=now,
            updated_at=now,
        )
        assert (
            str(registration.health_endpoint)
            == "http://localhost:8080/api/v1/health?timeout=30"
        )

    def test_none_health_endpoint_allowed(self) -> None:
        """Test that None is allowed for health_endpoint."""
        test_node_id = uuid4()
        now = datetime.now(UTC)
        registration = ModelNodeRegistration(
            node_id=test_node_id,
            node_type="effect",
            health_endpoint=None,
            registered_at=now,
            updated_at=now,
        )
        assert registration.health_endpoint is None

    def test_invalid_url_missing_scheme(self) -> None:
        """Test that URLs without scheme are rejected."""
        test_node_id = uuid4()
        now = datetime.now(UTC)
        with pytest.raises(ValidationError) as exc_info:
            ModelNodeRegistration(
                node_id=test_node_id,
                node_type="effect",
                health_endpoint="localhost:8080/health",
                registered_at=now,
                updated_at=now,
            )
        assert "health_endpoint" in str(exc_info.value)

    def test_invalid_url_file_scheme(self) -> None:
        """Test that non-HTTP schemes like file:// are rejected."""
        test_node_id = uuid4()
        now = datetime.now(UTC)
        with pytest.raises(ValidationError) as exc_info:
            ModelNodeRegistration(
                node_id=test_node_id,
                node_type="effect",
                health_endpoint="file:///etc/passwd",
                registered_at=now,
                updated_at=now,
            )
        assert "health_endpoint" in str(exc_info.value)

    def test_invalid_url_plain_string(self) -> None:
        """Test that plain strings are rejected."""
        test_node_id = uuid4()
        now = datetime.now(UTC)
        with pytest.raises(ValidationError) as exc_info:
            ModelNodeRegistration(
                node_id=test_node_id,
                node_type="effect",
                health_endpoint="not-a-url",
                registered_at=now,
                updated_at=now,
            )
        assert "health_endpoint" in str(exc_info.value)

    def test_invalid_url_empty_string(self) -> None:
        """Test that empty strings are rejected."""
        test_node_id = uuid4()
        now = datetime.now(UTC)
        with pytest.raises(ValidationError) as exc_info:
            ModelNodeRegistration(
                node_id=test_node_id,
                node_type="effect",
                health_endpoint="",
                registered_at=now,
                updated_at=now,
            )
        assert "health_endpoint" in str(exc_info.value)

    def test_invalid_url_relative_path(self) -> None:
        """Test that relative paths are rejected."""
        test_node_id = uuid4()
        now = datetime.now(UTC)
        with pytest.raises(ValidationError) as exc_info:
            ModelNodeRegistration(
                node_id=test_node_id,
                node_type="effect",
                health_endpoint="/health",
                registered_at=now,
                updated_at=now,
            )
        assert "health_endpoint" in str(exc_info.value)

    def test_health_endpoint_serialization_roundtrip(self) -> None:
        """Test that health_endpoint survives JSON serialization roundtrip."""
        test_node_id = uuid4()
        now = datetime.now(UTC)
        registration = ModelNodeRegistration(
            node_id=test_node_id,
            node_type="effect",
            health_endpoint="https://api.example.com/health",
            registered_at=now,
            updated_at=now,
        )
        json_str = registration.model_dump_json()
        restored = ModelNodeRegistration.model_validate_json(json_str)
        assert str(restored.health_endpoint) == str(registration.health_endpoint)
