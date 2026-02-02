# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for contract and topic projection models.

Tests validate:
- ModelContractProjection: Contract projection for registry queries
- ModelTopicProjection: Topic projection for routing discovery
- ModelPersistenceResult: Persistence operation result

Related Tickets:
    - OMN-1845: Create ProjectionReaderContract for contract/topic queries
    - OMN-1653: Contract registry state materialization
    - OMN-1709: Topic orphan handling documentation
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from omnibase_infra.enums import EnumPostgresErrorCode
from omnibase_infra.models.projection.model_contract_projection import (
    ModelContractProjection,
)
from omnibase_infra.models.projection.model_topic_projection import (
    ModelTopicProjection,
)
from omnibase_infra.nodes.node_contract_persistence_effect.models.model_persistence_result import (
    ModelPersistenceResult,
)

# =============================================================================
# ModelContractProjection Tests
# =============================================================================


class TestModelContractProjectionRequiredFields:
    """Tests for ModelContractProjection required field validation."""

    def test_contract_projection_required_fields(self) -> None:
        """Test instantiation with all required fields."""
        now = datetime.now(UTC)
        proj = ModelContractProjection(
            contract_id="node-registry-effect:1.0.0",
            node_name="node-registry-effect",
            version_major=1,
            version_minor=0,
            version_patch=0,
            contract_hash="abc123def456",
            contract_yaml="name: node-registry-effect\nversion: 1.0.0",
            registered_at=now,
            last_seen_at=now,
        )

        assert proj.contract_id == "node-registry-effect:1.0.0"
        assert proj.node_name == "node-registry-effect"
        assert proj.version_major == 1
        assert proj.version_minor == 0
        assert proj.version_patch == 0
        assert proj.contract_hash == "abc123def456"
        assert proj.contract_yaml == "name: node-registry-effect\nversion: 1.0.0"
        assert proj.registered_at == now
        assert proj.last_seen_at == now
        # Check defaults
        assert proj.is_active is True
        assert proj.deregistered_at is None
        assert proj.last_event_topic is None
        assert proj.last_event_partition is None
        assert proj.last_event_offset is None
        assert proj.created_at is None
        assert proj.updated_at is None

    def test_contract_projection_missing_contract_id_raises(self) -> None:
        """Test that missing contract_id raises ValidationError."""
        now = datetime.now(UTC)
        with pytest.raises(ValidationError) as exc_info:
            ModelContractProjection(
                node_name="node-registry-effect",
                version_major=1,
                version_minor=0,
                version_patch=0,
                contract_hash="abc123def456",
                contract_yaml="name: node-registry-effect",
                registered_at=now,
                last_seen_at=now,
            )  # type: ignore[call-arg]
        assert "contract_id" in str(exc_info.value)

    def test_contract_projection_empty_contract_id_raises(self) -> None:
        """Test that empty contract_id raises ValidationError."""
        now = datetime.now(UTC)
        with pytest.raises(ValidationError) as exc_info:
            ModelContractProjection(
                contract_id="",
                node_name="node-registry-effect",
                version_major=1,
                version_minor=0,
                version_patch=0,
                contract_hash="abc123def456",
                contract_yaml="name: node-registry-effect",
                registered_at=now,
                last_seen_at=now,
            )
        assert "contract_id" in str(exc_info.value)

    def test_contract_projection_negative_version_raises(self) -> None:
        """Test that negative version components raise ValidationError."""
        now = datetime.now(UTC)
        with pytest.raises(ValidationError) as exc_info:
            ModelContractProjection(
                contract_id="test:1.0.0",
                node_name="test",
                version_major=-1,
                version_minor=0,
                version_patch=0,
                contract_hash="abc123",
                contract_yaml="name: test",
                registered_at=now,
                last_seen_at=now,
            )
        assert "version_major" in str(exc_info.value)


class TestModelContractProjectionVersionString:
    """Tests for version_string computed property."""

    def test_contract_projection_version_string_property(self) -> None:
        """Test version_string computed property returns correct format."""
        now = datetime.now(UTC)
        proj = ModelContractProjection(
            contract_id="test:1.2.3",
            node_name="test",
            version_major=1,
            version_minor=2,
            version_patch=3,
            contract_hash="abc123",
            contract_yaml="name: test",
            registered_at=now,
            last_seen_at=now,
        )
        assert proj.version_string == "1.2.3"

    def test_contract_projection_version_string_zero_versions(self) -> None:
        """Test version_string with all zeros."""
        now = datetime.now(UTC)
        proj = ModelContractProjection(
            contract_id="test:0.0.0",
            node_name="test",
            version_major=0,
            version_minor=0,
            version_patch=0,
            contract_hash="abc123",
            contract_yaml="name: test",
            registered_at=now,
            last_seen_at=now,
        )
        assert proj.version_string == "0.0.0"

    def test_contract_projection_version_string_large_versions(self) -> None:
        """Test version_string with large version numbers."""
        now = datetime.now(UTC)
        proj = ModelContractProjection(
            contract_id="test:100.200.300",
            node_name="test",
            version_major=100,
            version_minor=200,
            version_patch=300,
            contract_hash="abc123",
            contract_yaml="name: test",
            registered_at=now,
            last_seen_at=now,
        )
        assert proj.version_string == "100.200.300"


class TestModelContractProjectionFrozen:
    """Tests for model immutability (frozen=True)."""

    def test_contract_projection_frozen(self) -> None:
        """Test that model is immutable - assignment raises ValidationError."""
        now = datetime.now(UTC)
        proj = ModelContractProjection(
            contract_id="test:1.0.0",
            node_name="test",
            version_major=1,
            version_minor=0,
            version_patch=0,
            contract_hash="abc123",
            contract_yaml="name: test",
            registered_at=now,
            last_seen_at=now,
        )

        with pytest.raises(ValidationError):
            proj.contract_id = "new-id:2.0.0"  # type: ignore[misc]

        with pytest.raises(ValidationError):
            proj.version_major = 2  # type: ignore[misc]

        with pytest.raises(ValidationError):
            proj.is_active = False  # type: ignore[misc]


class TestModelContractProjectionExtraForbid:
    """Tests for extra fields being forbidden."""

    def test_contract_projection_extra_forbid(self) -> None:
        """Test that extra fields raise ValidationError."""
        now = datetime.now(UTC)
        with pytest.raises(ValidationError) as exc_info:
            ModelContractProjection(
                contract_id="test:1.0.0",
                node_name="test",
                version_major=1,
                version_minor=0,
                version_patch=0,
                contract_hash="abc123",
                contract_yaml="name: test",
                registered_at=now,
                last_seen_at=now,
                unknown_field="value",  # type: ignore[call-arg]
            )
        assert "extra" in str(exc_info.value).lower() or "unknown_field" in str(
            exc_info.value
        )


class TestModelContractProjectionKafkaTracking:
    """Tests for Kafka position tracking fields."""

    def test_contract_projection_with_kafka_metadata(self) -> None:
        """Test contract projection with full Kafka metadata."""
        now = datetime.now(UTC)
        proj = ModelContractProjection(
            contract_id="test:1.0.0",
            node_name="test",
            version_major=1,
            version_minor=0,
            version_patch=0,
            contract_hash="abc123",
            contract_yaml="name: test",
            registered_at=now,
            last_seen_at=now,
            last_event_topic="onex.evt.contract.registered.v1",
            last_event_partition=0,
            last_event_offset=12345,
        )

        assert proj.last_event_topic == "onex.evt.contract.registered.v1"
        assert proj.last_event_partition == 0
        assert proj.last_event_offset == 12345


class TestModelContractProjectionFromAttributes:
    """Tests for from_attributes=True config."""

    def test_contract_projection_from_attributes(self) -> None:
        """Test that from_attributes works with class instances."""
        now = datetime.now(UTC)

        class MockRow:
            """Mock database row object."""

            def __init__(self) -> None:
                self.contract_id = "test:1.0.0"
                self.node_name = "test"
                self.version_major = 1
                self.version_minor = 0
                self.version_patch = 0
                self.contract_hash = "abc123"
                self.contract_yaml = "name: test"
                self.registered_at = now
                self.deregistered_at = None
                self.last_seen_at = now
                self.is_active = True
                self.last_event_topic = None
                self.last_event_partition = None
                self.last_event_offset = None
                self.created_at = now
                self.updated_at = now

        row = MockRow()
        proj = ModelContractProjection.model_validate(row)
        assert proj.contract_id == "test:1.0.0"
        assert proj.node_name == "test"
        assert proj.version_string == "1.0.0"


# =============================================================================
# ModelTopicProjection Tests
# =============================================================================


class TestModelTopicProjectionRequiredFields:
    """Tests for ModelTopicProjection required field validation."""

    def test_topic_projection_required_fields(self) -> None:
        """Test instantiation with required fields."""
        now = datetime.now(UTC)
        proj = ModelTopicProjection(
            topic_suffix="onex.evt.platform.contract-registered.v1",
            direction="publish",
            first_seen_at=now,
            last_seen_at=now,
        )

        assert proj.topic_suffix == "onex.evt.platform.contract-registered.v1"
        assert proj.direction == "publish"
        assert proj.first_seen_at == now
        assert proj.last_seen_at == now
        # Check defaults
        assert proj.contract_ids == []
        assert proj.is_active is True
        assert proj.created_at is None
        assert proj.updated_at is None

    def test_topic_projection_with_contracts(self) -> None:
        """Test topic projection with contract references."""
        now = datetime.now(UTC)
        proj = ModelTopicProjection(
            topic_suffix="onex.evt.platform.contract-registered.v1",
            direction="publish",
            contract_ids=["node-a:1.0.0", "node-b:2.0.0"],
            first_seen_at=now,
            last_seen_at=now,
        )

        assert proj.contract_ids == ["node-a:1.0.0", "node-b:2.0.0"]

    def test_topic_projection_missing_topic_suffix_raises(self) -> None:
        """Test that missing topic_suffix raises ValidationError."""
        now = datetime.now(UTC)
        with pytest.raises(ValidationError) as exc_info:
            ModelTopicProjection(
                direction="publish",
                first_seen_at=now,
                last_seen_at=now,
            )  # type: ignore[call-arg]
        assert "topic_suffix" in str(exc_info.value)

    def test_topic_projection_empty_topic_suffix_raises(self) -> None:
        """Test that empty topic_suffix raises ValidationError."""
        now = datetime.now(UTC)
        with pytest.raises(ValidationError) as exc_info:
            ModelTopicProjection(
                topic_suffix="",
                direction="publish",
                first_seen_at=now,
                last_seen_at=now,
            )
        assert "topic_suffix" in str(exc_info.value)


class TestModelTopicProjectionContractCount:
    """Tests for contract_count computed property."""

    def test_topic_projection_contract_count_property(self) -> None:
        """Test contract_count returns correct count."""
        now = datetime.now(UTC)
        proj = ModelTopicProjection(
            topic_suffix="test.topic.v1",
            direction="subscribe",
            contract_ids=["a:1.0.0", "b:1.0.0", "c:1.0.0"],
            first_seen_at=now,
            last_seen_at=now,
        )
        assert proj.contract_count == 3

    def test_topic_projection_contract_count_empty(self) -> None:
        """Test contract_count returns 0 for empty list."""
        now = datetime.now(UTC)
        proj = ModelTopicProjection(
            topic_suffix="test.topic.v1",
            direction="publish",
            contract_ids=[],
            first_seen_at=now,
            last_seen_at=now,
        )
        assert proj.contract_count == 0

    def test_topic_projection_contract_count_single(self) -> None:
        """Test contract_count returns 1 for single contract."""
        now = datetime.now(UTC)
        proj = ModelTopicProjection(
            topic_suffix="test.topic.v1",
            direction="publish",
            contract_ids=["node:1.0.0"],
            first_seen_at=now,
            last_seen_at=now,
        )
        assert proj.contract_count == 1


class TestModelTopicProjectionIsOrphaned:
    """Tests for is_orphaned computed property."""

    def test_topic_projection_is_orphaned_property(self) -> None:
        """Test is_orphaned returns True when no contracts and inactive."""
        now = datetime.now(UTC)
        proj = ModelTopicProjection(
            topic_suffix="test.topic.v1",
            direction="publish",
            contract_ids=[],
            is_active=False,
            first_seen_at=now,
            last_seen_at=now,
        )
        assert proj.is_orphaned is True

    def test_topic_projection_not_orphaned_with_contracts(self) -> None:
        """Test is_orphaned returns False when contracts exist."""
        now = datetime.now(UTC)
        proj = ModelTopicProjection(
            topic_suffix="test.topic.v1",
            direction="publish",
            contract_ids=["node:1.0.0"],
            is_active=False,
            first_seen_at=now,
            last_seen_at=now,
        )
        assert proj.is_orphaned is False

    def test_topic_projection_not_orphaned_when_active(self) -> None:
        """Test is_orphaned returns False when still active."""
        now = datetime.now(UTC)
        proj = ModelTopicProjection(
            topic_suffix="test.topic.v1",
            direction="publish",
            contract_ids=[],
            is_active=True,
            first_seen_at=now,
            last_seen_at=now,
        )
        assert proj.is_orphaned is False

    def test_topic_projection_not_orphaned_active_with_contracts(self) -> None:
        """Test is_orphaned returns False when active with contracts."""
        now = datetime.now(UTC)
        proj = ModelTopicProjection(
            topic_suffix="test.topic.v1",
            direction="subscribe",
            contract_ids=["a:1.0.0"],
            is_active=True,
            first_seen_at=now,
            last_seen_at=now,
        )
        assert proj.is_orphaned is False


class TestModelTopicProjectionDirectionLiteral:
    """Tests for direction literal type validation."""

    def test_topic_projection_direction_literal(self) -> None:
        """Test that only 'publish' or 'subscribe' are allowed."""
        now = datetime.now(UTC)

        # Valid: publish
        proj_publish = ModelTopicProjection(
            topic_suffix="test.topic.v1",
            direction="publish",
            first_seen_at=now,
            last_seen_at=now,
        )
        assert proj_publish.direction == "publish"

        # Valid: subscribe
        proj_subscribe = ModelTopicProjection(
            topic_suffix="test.topic.v1",
            direction="subscribe",
            first_seen_at=now,
            last_seen_at=now,
        )
        assert proj_subscribe.direction == "subscribe"

    def test_topic_projection_invalid_direction_raises(self) -> None:
        """Test that invalid direction values raise ValidationError."""
        now = datetime.now(UTC)
        with pytest.raises(ValidationError) as exc_info:
            ModelTopicProjection(
                topic_suffix="test.topic.v1",
                direction="invalid",  # type: ignore[arg-type]
                first_seen_at=now,
                last_seen_at=now,
            )
        assert "direction" in str(exc_info.value)


class TestModelTopicProjectionFrozen:
    """Tests for model immutability (frozen=True)."""

    def test_topic_projection_frozen(self) -> None:
        """Test that model is immutable."""
        now = datetime.now(UTC)
        proj = ModelTopicProjection(
            topic_suffix="test.topic.v1",
            direction="publish",
            first_seen_at=now,
            last_seen_at=now,
        )

        with pytest.raises(ValidationError):
            proj.topic_suffix = "new.topic.v1"  # type: ignore[misc]

        with pytest.raises(ValidationError):
            proj.direction = "subscribe"  # type: ignore[misc]


class TestModelTopicProjectionExtraForbid:
    """Tests for extra fields being forbidden."""

    def test_topic_projection_extra_forbid(self) -> None:
        """Test that extra fields raise ValidationError."""
        now = datetime.now(UTC)
        with pytest.raises(ValidationError):
            ModelTopicProjection(
                topic_suffix="test.topic.v1",
                direction="publish",
                first_seen_at=now,
                last_seen_at=now,
                unknown_field="value",  # type: ignore[call-arg]
            )


class TestModelTopicProjectionFromAttributes:
    """Tests for from_attributes=True config."""

    def test_topic_projection_from_attributes(self) -> None:
        """Test that from_attributes works with class instances."""
        now = datetime.now(UTC)

        class MockRow:
            """Mock database row object."""

            def __init__(self) -> None:
                self.topic_suffix = "test.topic.v1"
                self.direction = "subscribe"
                self.contract_ids = ["node:1.0.0"]
                self.first_seen_at = now
                self.last_seen_at = now
                self.is_active = True
                self.created_at = now
                self.updated_at = now

        row = MockRow()
        proj = ModelTopicProjection.model_validate(row)
        assert proj.topic_suffix == "test.topic.v1"
        assert proj.direction == "subscribe"
        assert proj.contract_count == 1


# =============================================================================
# ModelPersistenceResult Tests
# =============================================================================


class TestModelPersistenceResultSuccess:
    """Tests for successful persistence results."""

    def test_persistence_result_success(self) -> None:
        """Test creating a success result."""
        correlation_id = uuid4()
        result = ModelPersistenceResult(
            success=True,
            duration_ms=15.5,
            correlation_id=correlation_id,
            rows_affected=1,
        )

        assert result.success is True
        assert result.error is None
        assert result.error_code is None
        assert result.duration_ms == 15.5
        assert result.correlation_id == correlation_id
        assert result.rows_affected == 1

    def test_persistence_result_success_with_multiple_rows(self) -> None:
        """Test success result with multiple rows affected."""
        result = ModelPersistenceResult(
            success=True,
            duration_ms=25.0,
            rows_affected=5,
        )

        assert result.success is True
        assert result.rows_affected == 5


class TestModelPersistenceResultFailure:
    """Tests for failed persistence results."""

    def test_persistence_result_failure_with_error(self) -> None:
        """Test creating a failure result with error and error_code."""
        correlation_id = uuid4()
        result = ModelPersistenceResult(
            success=False,
            error="Connection refused to database",
            error_code=EnumPostgresErrorCode.CONNECTION_ERROR,
            duration_ms=1.5,
            correlation_id=correlation_id,
            rows_affected=0,
        )

        assert result.success is False
        assert result.error == "Connection refused to database"
        assert result.error_code == EnumPostgresErrorCode.CONNECTION_ERROR
        assert result.duration_ms == 1.5
        assert result.correlation_id == correlation_id
        assert result.rows_affected == 0

    def test_persistence_result_failure_without_error_code(self) -> None:
        """Test failure result with error message but no error_code."""
        result = ModelPersistenceResult(
            success=False,
            error="Unknown error occurred",
            duration_ms=0.5,
        )

        assert result.success is False
        assert result.error == "Unknown error occurred"
        assert result.error_code is None


class TestModelPersistenceResultTimestamp:
    """Tests for timestamp field behavior."""

    def test_persistence_result_timestamp_utc(self) -> None:
        """Test that default timestamp has UTC timezone."""
        result = ModelPersistenceResult(success=True)

        # Verify timestamp exists and has UTC timezone
        assert result.timestamp is not None
        assert result.timestamp.tzinfo is not None
        # Check that it's UTC (or equivalent offset)
        assert result.timestamp.utcoffset() == timedelta(0)

    def test_persistence_result_timestamp_custom(self) -> None:
        """Test that custom timestamp is preserved."""
        custom_time = datetime(2025, 1, 15, 10, 30, 0, tzinfo=UTC)
        result = ModelPersistenceResult(
            success=True,
            timestamp=custom_time,
        )

        assert result.timestamp == custom_time

    def test_persistence_result_timestamp_recent(self) -> None:
        """Test that auto-generated timestamp is recent."""
        before = datetime.now(UTC)
        result = ModelPersistenceResult(success=True)
        after = datetime.now(UTC)

        # Timestamp should be between before and after
        assert before <= result.timestamp <= after


class TestModelPersistenceResultFromAttributes:
    """Tests for from_attributes=True config."""

    def test_persistence_result_from_attributes(self) -> None:
        """Test that from_attributes works with class instances."""
        correlation_id = uuid4()
        timestamp = datetime.now(UTC)

        class MockResult:
            """Mock result object."""

            def __init__(self) -> None:
                self.success = True
                self.error = None
                self.error_code = None
                self.duration_ms = 10.0
                self.correlation_id = correlation_id
                self.rows_affected = 2
                self.timestamp = timestamp

        mock = MockResult()
        result = ModelPersistenceResult.model_validate(mock)

        assert result.success is True
        assert result.duration_ms == 10.0
        assert result.correlation_id == correlation_id
        assert result.rows_affected == 2
        assert result.timestamp == timestamp


class TestModelPersistenceResultFrozen:
    """Tests for model immutability (frozen=True)."""

    def test_persistence_result_frozen(self) -> None:
        """Test that model is immutable."""
        result = ModelPersistenceResult(success=True)

        with pytest.raises(ValidationError):
            result.success = False  # type: ignore[misc]

        with pytest.raises(ValidationError):
            result.rows_affected = 10  # type: ignore[misc]


class TestModelPersistenceResultExtraForbid:
    """Tests for extra fields being forbidden."""

    def test_persistence_result_extra_forbid(self) -> None:
        """Test that extra fields raise ValidationError."""
        with pytest.raises(ValidationError):
            ModelPersistenceResult(
                success=True,
                unknown_field="value",  # type: ignore[call-arg]
            )


class TestModelPersistenceResultDefaults:
    """Tests for default field values."""

    def test_persistence_result_defaults(self) -> None:
        """Test default values are applied correctly."""
        result = ModelPersistenceResult(success=True)

        assert result.error is None
        assert result.error_code is None
        assert result.duration_ms == 0.0
        assert result.correlation_id is None
        assert result.rows_affected == 0
        # timestamp has default_factory so should be set
        assert result.timestamp is not None
