# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for agent_actions observability models.

This module tests model validation behavior:
    - Envelope strict validation (extra="forbid")
    - Payload models allow extras (extra="allow")
    - Type validation (UUID, datetime, dict[str, object])
    - Required vs optional field enforcement

Related Tickets:
    - OMN-1743: Migrate agent_actions_consumer to omnibase_infra
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from omnibase_infra.services.observability.agent_actions.models import (
    ModelAgentAction,
    ModelDetectionFailure,
    ModelExecutionLog,
    ModelObservabilityEnvelope,
    ModelPerformanceMetric,
    ModelRoutingDecision,
    ModelTransformationEvent,
)

# =============================================================================
# Envelope Strict Validation Tests
# =============================================================================


class TestModelObservabilityEnvelopeStrict:
    """Test that ModelObservabilityEnvelope has strict validation (extra='forbid')."""

    def test_envelope_rejects_extra_fields(self) -> None:
        """Envelope should reject unknown fields with ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelObservabilityEnvelope(
                event_id=uuid4(),
                event_time=datetime.now(UTC),
                producer_id="test-producer",
                schema_version="1.0.0",
                unknown_field="should_fail",  # type: ignore[call-arg]
            )

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "extra_forbidden"
        assert "unknown_field" in str(errors[0]["loc"])

    def test_envelope_rejects_multiple_extra_fields(self) -> None:
        """Envelope should reject all unknown fields."""
        with pytest.raises(ValidationError) as exc_info:
            ModelObservabilityEnvelope(
                event_id=uuid4(),
                event_time=datetime.now(UTC),
                producer_id="test-producer",
                schema_version="1.0.0",
                extra1="value1",  # type: ignore[call-arg]
                extra2="value2",  # type: ignore[call-arg]
            )

        errors = exc_info.value.errors()
        # Multiple extra fields should each produce an error
        assert len(errors) >= 1
        error_types = {e["type"] for e in errors}
        assert "extra_forbidden" in error_types

    def test_envelope_required_fields_enforced(self) -> None:
        """Envelope should require all mandatory fields."""
        with pytest.raises(ValidationError) as exc_info:
            ModelObservabilityEnvelope()  # type: ignore[call-arg]

        errors = exc_info.value.errors()
        error_locs = {e["loc"][0] for e in errors}
        assert "event_id" in error_locs
        assert "event_time" in error_locs
        assert "producer_id" in error_locs
        assert "schema_version" in error_locs

    def test_envelope_optional_correlation_id(self) -> None:
        """Envelope should allow correlation_id to be omitted."""
        envelope = ModelObservabilityEnvelope(
            event_id=uuid4(),
            event_time=datetime.now(UTC),
            producer_id="test-producer",
            schema_version="1.0.0",
        )
        assert envelope.correlation_id is None

    def test_envelope_accepts_valid_correlation_id(self) -> None:
        """Envelope should accept a valid UUID correlation_id."""
        cid = uuid4()
        envelope = ModelObservabilityEnvelope(
            event_id=uuid4(),
            event_time=datetime.now(UTC),
            producer_id="test-producer",
            schema_version="1.0.0",
            correlation_id=cid,
        )
        assert envelope.correlation_id == cid

    def test_envelope_is_frozen(self) -> None:
        """Envelope should be immutable after creation."""
        envelope = ModelObservabilityEnvelope(
            event_id=uuid4(),
            event_time=datetime.now(UTC),
            producer_id="test-producer",
            schema_version="1.0.0",
        )

        with pytest.raises(ValidationError):
            envelope.producer_id = "new-producer"  # type: ignore[misc]


# =============================================================================
# Payload Models Allow Extras Tests
# =============================================================================


class TestModelAgentActionExtrasAllowed:
    """Test that ModelAgentAction allows extra fields (extra='allow')."""

    def test_agent_action_accepts_extra_fields(self) -> None:
        """Agent action should accept and preserve extra fields."""
        action = ModelAgentAction(  # type: ignore[call-arg]
            id=uuid4(),
            correlation_id=uuid4(),
            agent_name="test-agent",
            action_type="tool_call",
            action_name="Read",
            created_at=datetime.now(UTC),
            custom_field="extra_value",  # Extra field - should be allowed
            another_extra=123,  # Another extra field
        )

        # Extra fields should be accessible via model_extra
        assert action.model_extra is not None
        assert action.model_extra.get("custom_field") == "extra_value"
        assert action.model_extra.get("another_extra") == 123

    def test_agent_action_required_fields_enforced(self) -> None:
        """Agent action should still enforce required fields."""
        with pytest.raises(ValidationError) as exc_info:
            ModelAgentAction()  # type: ignore[call-arg]

        errors = exc_info.value.errors()
        error_locs = {e["loc"][0] for e in errors}
        assert "id" in error_locs
        assert "correlation_id" in error_locs
        assert "agent_name" in error_locs
        assert "action_type" in error_locs
        assert "action_name" in error_locs
        assert "created_at" in error_locs

    def test_agent_action_optional_fields_work(self) -> None:
        """Agent action optional fields should default to None."""
        action = ModelAgentAction(
            id=uuid4(),
            correlation_id=uuid4(),
            agent_name="test-agent",
            action_type="tool_call",
            action_name="Read",
            created_at=datetime.now(UTC),
        )

        assert action.status is None
        assert action.duration_ms is None
        assert action.result is None
        assert action.error_message is None
        assert action.metadata is None
        assert action.raw_payload is None


class TestModelRoutingDecisionExtrasAllowed:
    """Test that ModelRoutingDecision allows extra fields."""

    def test_routing_decision_accepts_extra_fields(self) -> None:
        """Routing decision should accept and preserve extra fields."""
        decision = ModelRoutingDecision(  # type: ignore[call-arg]
            id=uuid4(),
            correlation_id=uuid4(),
            selected_agent="api-architect",
            confidence_score=0.95,
            created_at=datetime.now(UTC),
            custom_routing_field="allowed",
        )

        assert decision.model_extra is not None
        assert decision.model_extra.get("custom_routing_field") == "allowed"

    def test_routing_decision_required_fields_enforced(self) -> None:
        """Routing decision should enforce required fields."""
        with pytest.raises(ValidationError) as exc_info:
            ModelRoutingDecision()  # type: ignore[call-arg]

        errors = exc_info.value.errors()
        error_locs = {e["loc"][0] for e in errors}
        assert "id" in error_locs
        assert "correlation_id" in error_locs
        assert "selected_agent" in error_locs
        assert "confidence_score" in error_locs
        assert "created_at" in error_locs


class TestModelTransformationEventExtrasAllowed:
    """Test that ModelTransformationEvent allows extra fields."""

    def test_transformation_event_accepts_extra_fields(self) -> None:
        """Transformation event should accept and preserve extra fields."""
        event = ModelTransformationEvent(  # type: ignore[call-arg]
            id=uuid4(),
            correlation_id=uuid4(),
            source_agent="polymorphic-agent",
            target_agent="api-architect",
            created_at=datetime.now(UTC),
            extra_transform_data={"key": "value"},
        )

        assert event.model_extra is not None
        assert event.model_extra.get("extra_transform_data") == {"key": "value"}


class TestModelPerformanceMetricExtrasAllowed:
    """Test that ModelPerformanceMetric allows extra fields."""

    def test_performance_metric_accepts_extra_fields(self) -> None:
        """Performance metric should accept and preserve extra fields."""
        metric = ModelPerformanceMetric(  # type: ignore[call-arg]
            id=uuid4(),
            metric_name="routing_latency_ms",
            metric_value=45.2,
            created_at=datetime.now(UTC),
            extra_metric_tag="custom_tag",
        )

        assert metric.model_extra is not None
        assert metric.model_extra.get("extra_metric_tag") == "custom_tag"


class TestModelDetectionFailureExtrasAllowed:
    """Test that ModelDetectionFailure allows extra fields."""

    def test_detection_failure_accepts_extra_fields(self) -> None:
        """Detection failure should accept and preserve extra fields."""
        failure = ModelDetectionFailure(  # type: ignore[call-arg]
            correlation_id=uuid4(),
            failure_reason="No matching pattern",
            created_at=datetime.now(UTC),
            debug_info="extra debugging data",
        )

        assert failure.model_extra is not None
        assert failure.model_extra.get("debug_info") == "extra debugging data"


class TestModelExecutionLogExtrasAllowed:
    """Test that ModelExecutionLog allows extra fields."""

    def test_execution_log_accepts_extra_fields(self) -> None:
        """Execution log should accept and preserve extra fields."""
        log = ModelExecutionLog(  # type: ignore[call-arg]
            execution_id=uuid4(),
            correlation_id=uuid4(),
            agent_name="testing",
            status="completed",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            custom_log_field=42,
        )

        assert log.model_extra is not None
        assert log.model_extra.get("custom_log_field") == 42


# =============================================================================
# Type Validation Tests
# =============================================================================


class TestUUIDValidation:
    """Test UUID field validation across models."""

    def test_uuid_accepts_valid_uuid(self) -> None:
        """UUID fields should accept valid UUID objects."""
        uid = uuid4()
        action = ModelAgentAction(
            id=uid,
            correlation_id=uuid4(),
            agent_name="test-agent",
            action_type="tool_call",
            action_name="Read",
            created_at=datetime.now(UTC),
        )
        assert action.id == uid

    def test_uuid_accepts_string_uuid(self) -> None:
        """UUID fields should accept valid UUID strings."""
        uid_str = str(uuid4())
        action = ModelAgentAction(
            id=uid_str,  # type: ignore[arg-type]
            correlation_id=uuid4(),
            agent_name="test-agent",
            action_type="tool_call",
            action_name="Read",
            created_at=datetime.now(UTC),
        )
        assert str(action.id) == uid_str

    def test_uuid_rejects_invalid_string(self) -> None:
        """UUID fields should reject invalid UUID strings."""
        with pytest.raises(ValidationError) as exc_info:
            ModelAgentAction(
                id="not-a-uuid",  # type: ignore[arg-type]
                correlation_id=uuid4(),
                agent_name="test-agent",
                action_type="tool_call",
                action_name="Read",
                created_at=datetime.now(UTC),
            )

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("id",) for e in errors)


class TestDatetimeValidation:
    """Test datetime field validation across models."""

    def test_datetime_accepts_utc_datetime(self) -> None:
        """Datetime fields should accept UTC datetime objects."""
        now = datetime.now(UTC)
        action = ModelAgentAction(
            id=uuid4(),
            correlation_id=uuid4(),
            agent_name="test-agent",
            action_type="tool_call",
            action_name="Read",
            created_at=now,
        )
        assert action.created_at == now

    def test_datetime_accepts_iso_string(self) -> None:
        """Datetime fields should accept valid ISO format strings."""
        now = datetime.now(UTC)
        iso_str = now.isoformat()
        action = ModelAgentAction(
            id=uuid4(),
            correlation_id=uuid4(),
            agent_name="test-agent",
            action_type="tool_call",
            action_name="Read",
            created_at=iso_str,  # type: ignore[arg-type]
        )
        assert action.created_at is not None

    def test_datetime_rejects_invalid_string(self) -> None:
        """Datetime fields should reject invalid datetime strings."""
        with pytest.raises(ValidationError) as exc_info:
            ModelAgentAction(
                id=uuid4(),
                correlation_id=uuid4(),
                agent_name="test-agent",
                action_type="tool_call",
                action_name="Read",
                created_at="not-a-datetime",  # type: ignore[arg-type]
            )

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("created_at",) for e in errors)


class TestRawPayloadValidation:
    """Test raw_payload field validation (dict[str, object])."""

    def test_raw_payload_accepts_dict(self) -> None:
        """raw_payload should accept dict[str, object]."""
        payload = {"key": "value", "number": 123, "nested": {"a": 1}}
        action = ModelAgentAction(
            id=uuid4(),
            correlation_id=uuid4(),
            agent_name="test-agent",
            action_type="tool_call",
            action_name="Read",
            created_at=datetime.now(UTC),
            raw_payload=payload,
        )
        assert action.raw_payload == payload

    def test_raw_payload_accepts_none(self) -> None:
        """raw_payload should accept None."""
        action = ModelAgentAction(
            id=uuid4(),
            correlation_id=uuid4(),
            agent_name="test-agent",
            action_type="tool_call",
            action_name="Read",
            created_at=datetime.now(UTC),
            raw_payload=None,
        )
        assert action.raw_payload is None

    def test_raw_payload_accepts_complex_nested_dict(self) -> None:
        """raw_payload should accept deeply nested structures."""
        payload = {
            "level1": {
                "level2": {
                    "level3": [1, 2, {"deep": "value"}],
                },
            },
            "array": [1, "two", 3.0, True, None],
        }
        action = ModelAgentAction(
            id=uuid4(),
            correlation_id=uuid4(),
            agent_name="test-agent",
            action_type="tool_call",
            action_name="Read",
            created_at=datetime.now(UTC),
            raw_payload=payload,
        )
        assert action.raw_payload == payload


class TestMetadataValidation:
    """Test metadata field validation (dict[str, object])."""

    def test_metadata_accepts_dict(self) -> None:
        """metadata should accept dict[str, object]."""
        metadata = {"tool": "Read", "file": "/path/to/file.py"}
        action = ModelAgentAction(
            id=uuid4(),
            correlation_id=uuid4(),
            agent_name="test-agent",
            action_type="tool_call",
            action_name="Read",
            created_at=datetime.now(UTC),
            metadata=metadata,
        )
        assert action.metadata == metadata


# =============================================================================
# Model-Specific Validation Tests
# =============================================================================


class TestModelAgentActionSpecific:
    """Model-specific tests for ModelAgentAction."""

    def test_agent_action_with_all_optional_fields(self) -> None:
        """Agent action should work with all optional fields populated."""
        now = datetime.now(UTC)
        action = ModelAgentAction(
            id=uuid4(),
            correlation_id=uuid4(),
            agent_name="test-agent",
            action_type="tool_call",
            action_name="Bash",
            created_at=now,
            status="completed",
            duration_ms=1500,
            result="Success",
            error_message=None,
            metadata={"command": "ls -la"},
            raw_payload={"full": "payload"},
        )

        assert action.status == "completed"
        assert action.duration_ms == 1500
        assert action.result == "Success"
        assert action.metadata == {"command": "ls -la"}


class TestModelRoutingDecisionSpecific:
    """Model-specific tests for ModelRoutingDecision."""

    def test_routing_decision_confidence_score_float(self) -> None:
        """Confidence score should accept float values."""
        decision = ModelRoutingDecision(
            id=uuid4(),
            correlation_id=uuid4(),
            selected_agent="api-architect",
            confidence_score=0.875,
            created_at=datetime.now(UTC),
        )
        assert decision.confidence_score == 0.875

    def test_routing_decision_alternatives_list(self) -> None:
        """Alternatives should accept list of strings."""
        decision = ModelRoutingDecision(
            id=uuid4(),
            correlation_id=uuid4(),
            selected_agent="api-architect",
            confidence_score=0.95,
            created_at=datetime.now(UTC),
            alternatives=["testing", "debug", "code-reviewer"],
        )
        assert decision.alternatives == ["testing", "debug", "code-reviewer"]


class TestModelExecutionLogSpecific:
    """Model-specific tests for ModelExecutionLog."""

    def test_execution_log_requires_both_timestamps(self) -> None:
        """Execution log should require both created_at and updated_at."""
        with pytest.raises(ValidationError) as exc_info:
            ModelExecutionLog(
                execution_id=uuid4(),
                correlation_id=uuid4(),
                agent_name="testing",
                status="running",
                created_at=datetime.now(UTC),
                # missing updated_at
            )  # type: ignore[call-arg]

        errors = exc_info.value.errors()
        error_locs = {e["loc"][0] for e in errors}
        assert "updated_at" in error_locs

    def test_execution_log_lifecycle_tracking_fields(self) -> None:
        """Execution log should support lifecycle tracking fields."""
        now = datetime.now(UTC)
        log = ModelExecutionLog(
            execution_id=uuid4(),
            correlation_id=uuid4(),
            agent_name="testing",
            status="completed",
            created_at=now,
            updated_at=now,
            started_at=now,
            completed_at=now,
            duration_ms=5000,
            exit_code=0,
        )

        assert log.started_at == now
        assert log.completed_at == now
        assert log.duration_ms == 5000
        assert log.exit_code == 0


class TestModelDetectionFailureSpecific:
    """Model-specific tests for ModelDetectionFailure."""

    def test_detection_failure_correlation_as_idempotency_key(self) -> None:
        """Detection failure uses correlation_id as idempotency key (not separate id)."""
        cid = uuid4()
        failure = ModelDetectionFailure(
            correlation_id=cid,
            failure_reason="No pattern matched",
            created_at=datetime.now(UTC),
        )
        # No 'id' field - correlation_id serves as the key
        assert failure.correlation_id == cid

    def test_detection_failure_attempted_patterns(self) -> None:
        """Detection failure should accept list of attempted patterns."""
        failure = ModelDetectionFailure(
            correlation_id=uuid4(),
            failure_reason="Low confidence scores",
            created_at=datetime.now(UTC),
            attempted_patterns=["code-review", "testing", "infrastructure"],
        )
        assert failure.attempted_patterns == [
            "code-review",
            "testing",
            "infrastructure",
        ]


__all__ = [
    "TestModelObservabilityEnvelopeStrict",
    "TestModelAgentActionExtrasAllowed",
    "TestModelRoutingDecisionExtrasAllowed",
    "TestModelTransformationEventExtrasAllowed",
    "TestModelPerformanceMetricExtrasAllowed",
    "TestModelDetectionFailureExtrasAllowed",
    "TestModelExecutionLogExtrasAllowed",
    "TestUUIDValidation",
    "TestDatetimeValidation",
    "TestRawPayloadValidation",
    "TestMetadataValidation",
    "TestModelAgentActionSpecific",
    "TestModelRoutingDecisionSpecific",
    "TestModelExecutionLogSpecific",
    "TestModelDetectionFailureSpecific",
]
