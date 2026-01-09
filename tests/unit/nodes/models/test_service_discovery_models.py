# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for Service Discovery Effect Node Models.

This module validates the models used by NodeServiceDiscoveryEffect:
- EnumHealthStatus: Health status enumeration with helper properties
- ModelServiceInfo: Discovered service information
- ModelDiscoveryQuery: Query parameters for discovery

Test Coverage:
    - Model construction and validation
    - Enum values and helper properties
    - Immutability (frozen=True)
    - Default values
    - Validation constraints (min_length, ge, le, etc.)
    - Serialization and round-trip

Related:
    - OMN-1131: Capability-oriented node architecture
    - NodeServiceDiscoveryEffect: Effect node using these models
    - PR #119: Test coverage for capability-oriented nodes
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from omnibase_infra.nodes.node_service_discovery_effect.models import (
    EnumHealthStatus,
    ModelDiscoveryQuery,
    ModelServiceInfo,
)

# =============================================================================
# EnumHealthStatus Tests
# =============================================================================


class TestEnumHealthStatusValues:
    """Tests for EnumHealthStatus enum values."""

    def test_healthy_value(self) -> None:
        """HEALTHY enum has correct value."""
        assert EnumHealthStatus.HEALTHY.value == "healthy"

    def test_unhealthy_value(self) -> None:
        """UNHEALTHY enum has correct value."""
        assert EnumHealthStatus.UNHEALTHY.value == "unhealthy"

    def test_unknown_value(self) -> None:
        """UNKNOWN enum has correct value."""
        assert EnumHealthStatus.UNKNOWN.value == "unknown"

    def test_enum_is_string_subclass(self) -> None:
        """EnumHealthStatus extends str for easy serialization."""
        assert isinstance(EnumHealthStatus.HEALTHY, str)
        # Enum value is the string, not str() which returns full enum name
        assert EnumHealthStatus.HEALTHY.value == "healthy"
        # Enum can be compared directly to string
        assert EnumHealthStatus.HEALTHY == "healthy"


class TestEnumHealthStatusProperties:
    """Tests for EnumHealthStatus helper properties."""

    def test_is_healthy_for_healthy(self) -> None:
        """is_healthy returns True for HEALTHY."""
        assert EnumHealthStatus.HEALTHY.is_healthy is True

    def test_is_healthy_for_unhealthy(self) -> None:
        """is_healthy returns False for UNHEALTHY."""
        assert EnumHealthStatus.UNHEALTHY.is_healthy is False

    def test_is_healthy_for_unknown(self) -> None:
        """is_healthy returns False for UNKNOWN."""
        assert EnumHealthStatus.UNKNOWN.is_healthy is False

    def test_is_unhealthy_for_unhealthy(self) -> None:
        """is_unhealthy returns True for UNHEALTHY."""
        assert EnumHealthStatus.UNHEALTHY.is_unhealthy is True

    def test_is_unhealthy_for_healthy(self) -> None:
        """is_unhealthy returns False for HEALTHY."""
        assert EnumHealthStatus.HEALTHY.is_unhealthy is False

    def test_is_unhealthy_for_unknown(self) -> None:
        """is_unhealthy returns False for UNKNOWN."""
        assert EnumHealthStatus.UNKNOWN.is_unhealthy is False

    def test_is_unknown_for_unknown(self) -> None:
        """is_unknown returns True for UNKNOWN."""
        assert EnumHealthStatus.UNKNOWN.is_unknown is True

    def test_is_unknown_for_healthy(self) -> None:
        """is_unknown returns False for HEALTHY."""
        assert EnumHealthStatus.HEALTHY.is_unknown is False

    def test_is_unknown_for_unhealthy(self) -> None:
        """is_unknown returns False for UNHEALTHY."""
        assert EnumHealthStatus.UNHEALTHY.is_unknown is False


class TestEnumHealthStatusMutualExclusion:
    """Tests for EnumHealthStatus property mutual exclusion."""

    def test_healthy_properties_mutually_exclusive(self) -> None:
        """HEALTHY has exactly one True property."""
        status = EnumHealthStatus.HEALTHY
        assert status.is_healthy is True
        assert status.is_unhealthy is False
        assert status.is_unknown is False

    def test_unhealthy_properties_mutually_exclusive(self) -> None:
        """UNHEALTHY has exactly one True property."""
        status = EnumHealthStatus.UNHEALTHY
        assert status.is_healthy is False
        assert status.is_unhealthy is True
        assert status.is_unknown is False

    def test_unknown_properties_mutually_exclusive(self) -> None:
        """UNKNOWN has exactly one True property."""
        status = EnumHealthStatus.UNKNOWN
        assert status.is_healthy is False
        assert status.is_unhealthy is False
        assert status.is_unknown is True


# =============================================================================
# ModelServiceInfo Tests
# =============================================================================


class TestModelServiceInfoConstruction:
    """Tests for ModelServiceInfo construction and validation."""

    def test_create_minimal_service_info(self) -> None:
        """Create service info with only required fields."""
        service_id = uuid4()
        info = ModelServiceInfo(
            service_id=service_id,
            service_name="user-service",
        )

        assert info.service_id == service_id
        assert info.service_name == "user-service"
        assert info.address is None
        assert info.port is None
        assert info.tags == ()
        assert info.health_status == EnumHealthStatus.UNKNOWN
        assert info.metadata == {}

    def test_create_full_service_info(self) -> None:
        """Create service info with all fields."""
        service_id = uuid4()
        info = ModelServiceInfo(
            service_id=service_id,
            service_name="api-gateway",
            address="10.0.0.5",
            port=8080,
            tags=("api", "v2", "production"),
            health_status=EnumHealthStatus.HEALTHY,
            metadata={"version": "2.1.0", "region": "us-east"},
        )

        assert info.service_id == service_id
        assert info.service_name == "api-gateway"
        assert info.address == "10.0.0.5"
        assert info.port == 8080
        assert info.tags == ("api", "v2", "production")
        assert info.health_status == EnumHealthStatus.HEALTHY
        assert info.metadata == {"version": "2.1.0", "region": "us-east"}


class TestModelServiceInfoValidation:
    """Tests for ModelServiceInfo validation constraints."""

    def test_service_name_cannot_be_empty(self) -> None:
        """service_name must have at least 1 character."""
        with pytest.raises(ValidationError) as exc_info:
            ModelServiceInfo(
                service_id=uuid4(),
                service_name="",  # Empty
            )

        assert "service_name" in str(exc_info.value)

    def test_port_minimum_is_one(self) -> None:
        """port must be at least 1."""
        with pytest.raises(ValidationError) as exc_info:
            ModelServiceInfo(
                service_id=uuid4(),
                service_name="test-service",
                port=0,  # Invalid
            )

        assert "port" in str(exc_info.value)

    def test_port_maximum_is_65535(self) -> None:
        """port must be at most 65535."""
        with pytest.raises(ValidationError) as exc_info:
            ModelServiceInfo(
                service_id=uuid4(),
                service_name="test-service",
                port=65536,  # Invalid
            )

        assert "port" in str(exc_info.value)

    def test_port_at_boundaries(self) -> None:
        """port accepts boundary values."""
        info_min = ModelServiceInfo(
            service_id=uuid4(),
            service_name="test-service",
            port=1,
        )
        info_max = ModelServiceInfo(
            service_id=uuid4(),
            service_name="test-service",
            port=65535,
        )

        assert info_min.port == 1
        assert info_max.port == 65535


class TestModelServiceInfoImmutability:
    """Tests for ModelServiceInfo immutability."""

    def test_model_is_frozen(self) -> None:
        """ModelServiceInfo is immutable (frozen)."""
        info = ModelServiceInfo(
            service_id=uuid4(),
            service_name="test-service",
        )

        with pytest.raises((TypeError, ValidationError)):
            info.service_name = "new-name"  # type: ignore[misc]

    def test_extra_fields_forbidden(self) -> None:
        """Extra fields are not allowed."""
        with pytest.raises(ValidationError):
            ModelServiceInfo(
                service_id=uuid4(),
                service_name="test-service",
                unknown_field="value",  # type: ignore[call-arg]
            )


# =============================================================================
# ModelDiscoveryQuery Tests
# =============================================================================


class TestModelDiscoveryQueryConstruction:
    """Tests for ModelDiscoveryQuery construction and validation."""

    def test_create_empty_query(self) -> None:
        """Create query with all defaults (match all services)."""
        query = ModelDiscoveryQuery()

        assert query.service_name is None
        assert query.tags is None
        assert query.health_filter is None
        # correlation_id is auto-generated
        assert query.correlation_id is not None

    def test_create_query_by_service_name(self) -> None:
        """Create query filtering by service name."""
        query = ModelDiscoveryQuery(service_name="user-service")

        assert query.service_name == "user-service"

    def test_create_query_by_tags(self) -> None:
        """Create query filtering by tags."""
        query = ModelDiscoveryQuery(tags=("api", "v2"))

        assert query.tags == ("api", "v2")

    def test_create_query_by_health_filter(self) -> None:
        """Create query filtering by health status."""
        query = ModelDiscoveryQuery(health_filter=EnumHealthStatus.HEALTHY)

        assert query.health_filter == EnumHealthStatus.HEALTHY

    def test_create_combined_query(self) -> None:
        """Create query with multiple filters."""
        query = ModelDiscoveryQuery(
            service_name="api-gateway",
            tags=("production",),
            health_filter=EnumHealthStatus.HEALTHY,
        )

        assert query.service_name == "api-gateway"
        assert query.tags == ("production",)
        assert query.health_filter == EnumHealthStatus.HEALTHY

    def test_correlation_id_auto_generated(self) -> None:
        """correlation_id is automatically generated if not provided."""
        query1 = ModelDiscoveryQuery()
        query2 = ModelDiscoveryQuery()

        # Both have correlation IDs
        assert query1.correlation_id is not None
        assert query2.correlation_id is not None

        # They should be different (unique)
        assert query1.correlation_id != query2.correlation_id

    def test_correlation_id_can_be_provided(self) -> None:
        """correlation_id can be explicitly provided."""
        explicit_id = uuid4()
        query = ModelDiscoveryQuery(correlation_id=explicit_id)

        assert query.correlation_id == explicit_id


class TestModelDiscoveryQueryImmutability:
    """Tests for ModelDiscoveryQuery immutability."""

    def test_model_is_frozen(self) -> None:
        """ModelDiscoveryQuery is immutable (frozen)."""
        query = ModelDiscoveryQuery()

        with pytest.raises((TypeError, ValidationError)):
            query.service_name = "new-service"  # type: ignore[misc]

    def test_extra_fields_forbidden(self) -> None:
        """Extra fields are not allowed."""
        with pytest.raises(ValidationError):
            ModelDiscoveryQuery(
                unknown_field="value",  # type: ignore[call-arg]
            )


# =============================================================================
# Model Serialization Tests
# =============================================================================


class TestServiceDiscoveryModelSerialization:
    """Tests for service discovery model serialization."""

    def test_service_info_to_dict(self) -> None:
        """ModelServiceInfo serializes to dict correctly."""
        service_id = uuid4()
        info = ModelServiceInfo(
            service_id=service_id,
            service_name="test-service",
            address="192.168.1.100",
            port=8080,
            tags=("api",),
            health_status=EnumHealthStatus.HEALTHY,
        )

        data = info.model_dump()

        assert data["service_id"] == service_id
        assert data["service_name"] == "test-service"
        assert data["address"] == "192.168.1.100"
        assert data["port"] == 8080
        assert data["tags"] == ("api",)
        assert data["health_status"] == EnumHealthStatus.HEALTHY

    def test_discovery_query_to_dict(self) -> None:
        """ModelDiscoveryQuery serializes to dict correctly."""
        correlation_id = uuid4()
        query = ModelDiscoveryQuery(
            service_name="user-service",
            tags=("production", "api"),
            health_filter=EnumHealthStatus.HEALTHY,
            correlation_id=correlation_id,
        )

        data = query.model_dump()

        assert data["service_name"] == "user-service"
        assert data["tags"] == ("production", "api")
        assert data["health_filter"] == EnumHealthStatus.HEALTHY
        assert data["correlation_id"] == correlation_id

    def test_service_info_json_round_trip(self) -> None:
        """ModelServiceInfo survives JSON round-trip."""
        info = ModelServiceInfo(
            service_id=uuid4(),
            service_name="api-gateway",
            address="10.0.0.5",
            port=443,
            tags=("secure", "api"),
            health_status=EnumHealthStatus.HEALTHY,
            metadata={"version": "3.0.0"},
        )

        json_str = info.model_dump_json()
        restored = ModelServiceInfo.model_validate_json(json_str)

        assert restored.service_id == info.service_id
        assert restored.service_name == info.service_name
        assert restored.address == info.address
        assert restored.port == info.port
        assert restored.tags == info.tags
        assert restored.health_status == info.health_status
        assert restored.metadata == info.metadata

    def test_discovery_query_json_round_trip(self) -> None:
        """ModelDiscoveryQuery survives JSON round-trip."""
        query = ModelDiscoveryQuery(
            service_name="order-service",
            health_filter=EnumHealthStatus.HEALTHY,
        )

        json_str = query.model_dump_json()
        restored = ModelDiscoveryQuery.model_validate_json(json_str)

        assert restored.service_name == query.service_name
        assert restored.health_filter == query.health_filter
        assert restored.correlation_id == query.correlation_id

    def test_enum_serializes_as_string(self) -> None:
        """EnumHealthStatus serializes as string value."""
        info = ModelServiceInfo(
            service_id=uuid4(),
            service_name="test",
            health_status=EnumHealthStatus.HEALTHY,
        )

        json_str = info.model_dump_json()
        assert '"healthy"' in json_str


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestServiceDiscoveryModelEdgeCases:
    """Edge case tests for service discovery models."""

    def test_service_info_with_empty_tags_tuple(self) -> None:
        """Empty tags tuple is valid."""
        info = ModelServiceInfo(
            service_id=uuid4(),
            service_name="test",
            tags=(),
        )

        assert info.tags == ()

    def test_service_info_with_single_tag(self) -> None:
        """Single tag in tuple is valid."""
        info = ModelServiceInfo(
            service_id=uuid4(),
            service_name="test",
            tags=("single-tag",),
        )

        assert info.tags == ("single-tag",)

    def test_service_info_with_many_tags(self) -> None:
        """Many tags are accepted."""
        tags = tuple(f"tag-{i}" for i in range(100))
        info = ModelServiceInfo(
            service_id=uuid4(),
            service_name="test",
            tags=tags,
        )

        assert len(info.tags) == 100

    def test_service_info_with_empty_metadata(self) -> None:
        """Empty metadata dict is valid."""
        info = ModelServiceInfo(
            service_id=uuid4(),
            service_name="test",
            metadata={},
        )

        assert info.metadata == {}

    def test_discovery_query_all_none_filters(self) -> None:
        """Query with all None filters matches all services."""
        query = ModelDiscoveryQuery()

        assert query.service_name is None
        assert query.tags is None
        assert query.health_filter is None


__all__: list[str] = [
    "TestEnumHealthStatusValues",
    "TestEnumHealthStatusProperties",
    "TestEnumHealthStatusMutualExclusion",
    "TestModelServiceInfoConstruction",
    "TestModelServiceInfoValidation",
    "TestModelServiceInfoImmutability",
    "TestModelDiscoveryQueryConstruction",
    "TestModelDiscoveryQueryImmutability",
    "TestServiceDiscoveryModelSerialization",
    "TestServiceDiscoveryModelEdgeCases",
]
