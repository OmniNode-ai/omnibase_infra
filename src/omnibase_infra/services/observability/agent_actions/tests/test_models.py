# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for agent_actions observability models.

This module tests model validation behavior:
    - All models use strict validation (extra="forbid", frozen=True)
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
    ModelAgentStatusEvent,
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
# Payload Models Strict Validation Tests
# =============================================================================


class TestModelAgentActionStrict:
    """Test that ModelAgentAction has strict validation (extra='forbid', frozen=True)."""

    def test_agent_action_rejects_extra_fields(self) -> None:
        """Agent action should reject unknown fields with ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelAgentAction(  # type: ignore[call-arg]
                id=uuid4(),
                correlation_id=uuid4(),
                agent_name="test-agent",
                action_type="tool_call",
                action_name="Read",
                created_at=datetime.now(UTC),
                custom_field="should_fail",
            )

        errors = exc_info.value.errors()
        error_types = {e["type"] for e in errors}
        assert "extra_forbidden" in error_types

    def test_agent_action_is_frozen(self) -> None:
        """Agent action should be immutable after creation."""
        action = ModelAgentAction(
            id=uuid4(),
            correlation_id=uuid4(),
            agent_name="test-agent",
            action_type="tool_call",
            action_name="Read",
            created_at=datetime.now(UTC),
        )

        with pytest.raises(ValidationError):
            action.agent_name = "new-agent"  # type: ignore[misc]

    def test_agent_action_required_fields_enforced(self) -> None:
        """Agent action should enforce required fields."""
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


class TestModelRoutingDecisionStrict:
    """Test that ModelRoutingDecision has strict validation."""

    def test_routing_decision_rejects_extra_fields(self) -> None:
        """Routing decision should reject unknown fields with ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelRoutingDecision(  # type: ignore[call-arg]
                id=uuid4(),
                correlation_id=uuid4(),
                selected_agent="api-architect",
                confidence_score=0.95,
                created_at=datetime.now(UTC),
                custom_routing_field="should_fail",
            )

        errors = exc_info.value.errors()
        error_types = {e["type"] for e in errors}
        assert "extra_forbidden" in error_types

    def test_routing_decision_is_frozen(self) -> None:
        """Routing decision should be immutable after creation."""
        decision = ModelRoutingDecision(
            id=uuid4(),
            correlation_id=uuid4(),
            selected_agent="api-architect",
            confidence_score=0.95,
            created_at=datetime.now(UTC),
        )

        with pytest.raises(ValidationError):
            decision.selected_agent = "new-agent"  # type: ignore[misc]

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


class TestModelTransformationEventStrict:
    """Test that ModelTransformationEvent has strict validation."""

    def test_transformation_event_rejects_extra_fields(self) -> None:
        """Transformation event should reject unknown fields."""
        with pytest.raises(ValidationError) as exc_info:
            ModelTransformationEvent(  # type: ignore[call-arg]
                id=uuid4(),
                correlation_id=uuid4(),
                source_agent="polymorphic-agent",
                target_agent="api-architect",
                created_at=datetime.now(UTC),
                extra_transform_data={"key": "value"},
            )

        errors = exc_info.value.errors()
        error_types = {e["type"] for e in errors}
        assert "extra_forbidden" in error_types

    def test_transformation_event_is_frozen(self) -> None:
        """Transformation event should be immutable after creation."""
        event = ModelTransformationEvent(
            id=uuid4(),
            correlation_id=uuid4(),
            source_agent="polymorphic-agent",
            target_agent="api-architect",
            created_at=datetime.now(UTC),
        )

        with pytest.raises(ValidationError):
            event.source_agent = "new-agent"  # type: ignore[misc]


class TestModelPerformanceMetricStrict:
    """Test that ModelPerformanceMetric has strict validation."""

    def test_performance_metric_rejects_extra_fields(self) -> None:
        """Performance metric should reject unknown fields."""
        with pytest.raises(ValidationError) as exc_info:
            ModelPerformanceMetric(  # type: ignore[call-arg]
                id=uuid4(),
                metric_name="routing_latency_ms",
                metric_value=45.2,
                created_at=datetime.now(UTC),
                extra_metric_tag="should_fail",
            )

        errors = exc_info.value.errors()
        error_types = {e["type"] for e in errors}
        assert "extra_forbidden" in error_types

    def test_performance_metric_is_frozen(self) -> None:
        """Performance metric should be immutable after creation."""
        metric = ModelPerformanceMetric(
            id=uuid4(),
            metric_name="routing_latency_ms",
            metric_value=45.2,
            created_at=datetime.now(UTC),
        )

        with pytest.raises(ValidationError):
            metric.metric_name = "new_metric"  # type: ignore[misc]


class TestModelDetectionFailureStrict:
    """Test that ModelDetectionFailure has strict validation."""

    def test_detection_failure_rejects_extra_fields(self) -> None:
        """Detection failure should reject unknown fields."""
        with pytest.raises(ValidationError) as exc_info:
            ModelDetectionFailure(  # type: ignore[call-arg]
                correlation_id=uuid4(),
                failure_reason="No matching pattern",
                created_at=datetime.now(UTC),
                debug_info="should_fail",
            )

        errors = exc_info.value.errors()
        error_types = {e["type"] for e in errors}
        assert "extra_forbidden" in error_types

    def test_detection_failure_is_frozen(self) -> None:
        """Detection failure should be immutable after creation."""
        failure = ModelDetectionFailure(
            correlation_id=uuid4(),
            failure_reason="No matching pattern",
            created_at=datetime.now(UTC),
        )

        with pytest.raises(ValidationError):
            failure.failure_reason = "new reason"  # type: ignore[misc]


class TestModelExecutionLogStrict:
    """Test that ModelExecutionLog has strict validation."""

    def test_execution_log_rejects_extra_fields(self) -> None:
        """Execution log should reject unknown fields."""
        with pytest.raises(ValidationError) as exc_info:
            ModelExecutionLog(  # type: ignore[call-arg]
                execution_id=uuid4(),
                correlation_id=uuid4(),
                agent_name="testing",
                status="completed",
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
                custom_log_field=42,
            )

        errors = exc_info.value.errors()
        error_types = {e["type"] for e in errors}
        assert "extra_forbidden" in error_types

    def test_execution_log_is_frozen(self) -> None:
        """Execution log should be immutable after creation."""
        log = ModelExecutionLog(
            execution_id=uuid4(),
            correlation_id=uuid4(),
            agent_name="testing",
            status="completed",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        with pytest.raises(ValidationError):
            log.status = "failed"  # type: ignore[misc]


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

    def test_routing_decision_alternatives_tuple(self) -> None:
        """Alternatives should accept and store as tuple of strings."""
        decision = ModelRoutingDecision(
            id=uuid4(),
            correlation_id=uuid4(),
            selected_agent="api-architect",
            confidence_score=0.95,
            created_at=datetime.now(UTC),
            alternatives=("testing", "debug", "code-reviewer"),
        )
        assert decision.alternatives == ("testing", "debug", "code-reviewer")

    def test_routing_decision_rejects_confidence_score_above_one(self) -> None:
        """Confidence score above 1.0 should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ModelRoutingDecision(
                id=uuid4(),
                correlation_id=uuid4(),
                selected_agent="test-agent",
                confidence_score=1.5,  # Invalid - above 1.0
                created_at=datetime.now(UTC),
            )

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("confidence_score",) for e in errors)
        assert any(e["type"] == "less_than_equal" for e in errors)

    def test_routing_decision_rejects_confidence_score_below_zero(self) -> None:
        """Confidence score below 0.0 should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ModelRoutingDecision(
                id=uuid4(),
                correlation_id=uuid4(),
                selected_agent="test-agent",
                confidence_score=-0.1,  # Invalid - below 0.0
                created_at=datetime.now(UTC),
            )

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("confidence_score",) for e in errors)
        assert any(e["type"] == "greater_than_equal" for e in errors)

    def test_routing_decision_accepts_boundary_confidence_scores(self) -> None:
        """Confidence score at boundaries (0.0 and 1.0) should be accepted."""
        # Test lower boundary
        decision_zero = ModelRoutingDecision(
            id=uuid4(),
            correlation_id=uuid4(),
            selected_agent="test-agent",
            confidence_score=0.0,  # Valid - exactly 0.0
            created_at=datetime.now(UTC),
        )
        assert decision_zero.confidence_score == 0.0

        # Test upper boundary
        decision_one = ModelRoutingDecision(
            id=uuid4(),
            correlation_id=uuid4(),
            selected_agent="test-agent",
            confidence_score=1.0,  # Valid - exactly 1.0
            created_at=datetime.now(UTC),
        )
        assert decision_one.confidence_score == 1.0


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
        """Detection failure should accept and store as tuple of attempted patterns."""
        failure = ModelDetectionFailure(
            correlation_id=uuid4(),
            failure_reason="Low confidence scores",
            created_at=datetime.now(UTC),
            attempted_patterns=("code-review", "testing", "infrastructure"),
        )
        assert failure.attempted_patterns == (
            "code-review",
            "testing",
            "infrastructure",
        )


# =============================================================================
# Agent Status Event Tests (OMN-1849)
# =============================================================================


class TestModelAgentStatusEventStrict:
    """Test that ModelAgentStatusEvent has strict validation (extra='forbid', frozen=True)."""

    def test_agent_status_event_rejects_extra_fields(self) -> None:
        """Agent status event should reject unknown fields with ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelAgentStatusEvent(  # type: ignore[call-arg]
                correlation_id=uuid4(),
                agent_name="test-agent",
                session_id="session-123",
                state="working",
                message="Processing request",
                created_at=datetime.now(UTC),
                custom_field="should_fail",
            )

        errors = exc_info.value.errors()
        error_types = {e["type"] for e in errors}
        assert "extra_forbidden" in error_types

    def test_agent_status_event_is_frozen(self) -> None:
        """Agent status event should be immutable after creation."""
        event = ModelAgentStatusEvent(
            correlation_id=uuid4(),
            agent_name="test-agent",
            session_id="session-123",
            state="working",
            message="Processing request",
            created_at=datetime.now(UTC),
        )

        with pytest.raises(ValidationError):
            event.state = "idle"  # type: ignore[misc]

    def test_agent_status_event_required_fields_enforced(self) -> None:
        """Agent status event should enforce required fields."""
        with pytest.raises(ValidationError) as exc_info:
            ModelAgentStatusEvent()  # type: ignore[call-arg]

        errors = exc_info.value.errors()
        error_locs = {e["loc"][0] for e in errors}
        assert "correlation_id" in error_locs
        assert "agent_name" in error_locs
        assert "session_id" in error_locs
        assert "state" in error_locs
        assert "message" in error_locs

    def test_agent_status_event_optional_fields_default(self) -> None:
        """Agent status event optional fields should default correctly."""
        event = ModelAgentStatusEvent(
            correlation_id=uuid4(),
            agent_name="test-agent",
            session_id="session-123",
            state="idle",
            message="Waiting for input",
        )

        assert event.progress is None
        assert event.current_phase is None
        assert event.current_task is None
        assert event.blocking_reason is None
        assert event.metadata is None
        assert event.status_schema_version == 1

    def test_agent_status_event_id_auto_generated(self) -> None:
        """Agent status event should auto-generate id if not provided."""
        event = ModelAgentStatusEvent(
            correlation_id=uuid4(),
            agent_name="test-agent",
            session_id="session-123",
            state="working",
            message="Processing",
        )

        assert event.id is not None

    def test_agent_status_event_created_at_auto_generated(self) -> None:
        """Agent status event should auto-generate created_at if not provided."""
        event = ModelAgentStatusEvent(
            correlation_id=uuid4(),
            agent_name="test-agent",
            session_id="session-123",
            state="working",
            message="Processing",
        )

        assert event.created_at is not None


class TestModelAgentStatusEventSpecific:
    """Model-specific tests for ModelAgentStatusEvent."""

    def test_agent_status_event_with_all_optional_fields(self) -> None:
        """Agent status event should work with all optional fields populated."""
        now = datetime.now(UTC)
        event = ModelAgentStatusEvent(
            id=uuid4(),
            correlation_id=uuid4(),
            agent_name="test-agent",
            session_id="session-123",
            state="working",
            status_schema_version=1,
            message="Reviewing code",
            progress=0.75,
            current_phase="analysis",
            current_task="Checking imports",
            blocking_reason=None,
            created_at=now,
            metadata={"file": "/test/path.py"},
        )

        assert event.state == "working"
        assert event.progress == 0.75
        assert event.current_phase == "analysis"
        assert event.current_task == "Checking imports"
        assert event.metadata == {"file": "/test/path.py"}

    def test_agent_status_event_progress_boundary_zero(self) -> None:
        """Progress at 0.0 should be accepted."""
        event = ModelAgentStatusEvent(
            correlation_id=uuid4(),
            agent_name="test-agent",
            session_id="session-123",
            state="working",
            message="Starting",
            progress=0.0,
        )
        assert event.progress == 0.0

    def test_agent_status_event_progress_boundary_one(self) -> None:
        """Progress at 1.0 should be accepted."""
        event = ModelAgentStatusEvent(
            correlation_id=uuid4(),
            agent_name="test-agent",
            session_id="session-123",
            state="working",
            message="Complete",
            progress=1.0,
        )
        assert event.progress == 1.0

    def test_agent_status_event_rejects_progress_above_one(self) -> None:
        """Progress above 1.0 should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ModelAgentStatusEvent(
                correlation_id=uuid4(),
                agent_name="test-agent",
                session_id="session-123",
                state="working",
                message="Invalid progress",
                progress=1.5,
            )

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("progress",) for e in errors)

    def test_agent_status_event_rejects_progress_below_zero(self) -> None:
        """Progress below 0.0 should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ModelAgentStatusEvent(
                correlation_id=uuid4(),
                agent_name="test-agent",
                session_id="session-123",
                state="working",
                message="Invalid progress",
                progress=-0.1,
            )

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("progress",) for e in errors)

    def test_agent_status_event_blocked_state_with_reason(self) -> None:
        """Blocked agent should include blocking_reason."""
        event = ModelAgentStatusEvent(
            correlation_id=uuid4(),
            agent_name="test-agent",
            session_id="session-123",
            state="blocked",
            message="Waiting for dependency",
            blocking_reason="Dependency OMN-1847 not yet merged",
        )

        assert event.state == "blocked"
        assert event.blocking_reason == "Dependency OMN-1847 not yet merged"

    def test_agent_status_event_str_representation(self) -> None:
        """String representation should include key fields."""
        event = ModelAgentStatusEvent(
            correlation_id=uuid4(),
            agent_name="test-agent",
            session_id="session-123",
            state="working",
            message="Processing",
            progress=0.5,
            current_phase="analysis",
        )

        result = str(event)
        assert "test-agent" in result
        assert "working" in result
        assert "0.5" in result
        assert "analysis" in result


# =============================================================================
# Project Context Field Tests (OMN-2057)
# =============================================================================


class TestModelAgentActionProjectContext:
    """Test project context fields on ModelAgentAction (OMN-2057)."""

    def test_agent_action_project_context_defaults_to_none(self) -> None:
        """Project context fields should default to None when not provided."""
        action = ModelAgentAction(
            id=uuid4(),
            correlation_id=uuid4(),
            agent_name="test-agent",
            action_type="tool_call",
            action_name="Read",
            created_at=datetime.now(UTC),
        )

        assert action.project_path is None
        assert action.project_name is None
        assert action.working_directory is None

    def test_agent_action_project_context_accepts_values(self) -> None:
        """Project context fields should accept valid string values."""
        action = ModelAgentAction(
            id=uuid4(),
            correlation_id=uuid4(),
            agent_name="test-agent",
            action_type="tool_call",
            action_name="Read",
            created_at=datetime.now(UTC),
            project_path="/home/user/projects/omnibase_infra",
            project_name="omnibase_infra",
            working_directory="/home/user/projects/omnibase_infra/src",
        )

        assert action.project_path == "/home/user/projects/omnibase_infra"
        assert action.project_name == "omnibase_infra"
        assert action.working_directory == "/home/user/projects/omnibase_infra/src"


class TestModelRoutingDecisionProjectContext:
    """Test project context fields on ModelRoutingDecision (OMN-2057)."""

    def test_routing_decision_project_context_defaults_to_none(self) -> None:
        """Project context fields should default to None when not provided."""
        decision = ModelRoutingDecision(
            id=uuid4(),
            correlation_id=uuid4(),
            selected_agent="api-architect",
            confidence_score=0.95,
            created_at=datetime.now(UTC),
        )

        assert decision.project_path is None
        assert decision.project_name is None
        assert decision.claude_session_id is None

    def test_routing_decision_project_context_accepts_values(self) -> None:
        """Project context fields should accept valid string values."""
        decision = ModelRoutingDecision(
            id=uuid4(),
            correlation_id=uuid4(),
            selected_agent="api-architect",
            confidence_score=0.95,
            created_at=datetime.now(UTC),
            project_path="/home/user/projects/omnibase_infra",
            project_name="omnibase_infra",
            claude_session_id="session-abc-123",
        )

        assert decision.project_path == "/home/user/projects/omnibase_infra"
        assert decision.project_name == "omnibase_infra"
        assert decision.claude_session_id == "session-abc-123"


class TestModelTransformationEventProjectContext:
    """Test project context fields on ModelTransformationEvent (OMN-2057)."""

    def test_transformation_event_project_context_defaults_to_none(self) -> None:
        """Project context fields should default to None when not provided."""
        event = ModelTransformationEvent(
            id=uuid4(),
            correlation_id=uuid4(),
            source_agent="polymorphic-agent",
            target_agent="api-architect",
            created_at=datetime.now(UTC),
        )

        assert event.project_path is None
        assert event.project_name is None
        assert event.claude_session_id is None

    def test_transformation_event_project_context_accepts_values(self) -> None:
        """Project context fields should accept valid string values."""
        event = ModelTransformationEvent(
            id=uuid4(),
            correlation_id=uuid4(),
            source_agent="polymorphic-agent",
            target_agent="api-architect",
            created_at=datetime.now(UTC),
            project_path="/home/user/projects/omnibase_infra",
            project_name="omnibase_infra",
            claude_session_id="session-def-456",
        )

        assert event.project_path == "/home/user/projects/omnibase_infra"
        assert event.project_name == "omnibase_infra"
        assert event.claude_session_id == "session-def-456"


class TestModelDetectionFailureProjectContext:
    """Test project context fields on ModelDetectionFailure (OMN-2057)."""

    def test_detection_failure_project_context_defaults_to_none(self) -> None:
        """Project context fields should default to None when not provided."""
        failure = ModelDetectionFailure(
            correlation_id=uuid4(),
            failure_reason="No matching pattern",
            created_at=datetime.now(UTC),
        )

        assert failure.project_path is None
        assert failure.project_name is None
        assert failure.claude_session_id is None

    def test_detection_failure_project_context_accepts_values(self) -> None:
        """Project context fields should accept valid string values."""
        failure = ModelDetectionFailure(
            correlation_id=uuid4(),
            failure_reason="No matching pattern",
            created_at=datetime.now(UTC),
            project_path="/home/user/projects/omnibase_infra",
            project_name="omnibase_infra",
            claude_session_id="session-ghi-789",
        )

        assert failure.project_path == "/home/user/projects/omnibase_infra"
        assert failure.project_name == "omnibase_infra"
        assert failure.claude_session_id == "session-ghi-789"


class TestModelExecutionLogProjectContext:
    """Test project context fields on ModelExecutionLog (OMN-2057)."""

    def test_execution_log_project_context_defaults_to_none(self) -> None:
        """Project context fields should default to None when not provided."""
        log = ModelExecutionLog(
            execution_id=uuid4(),
            correlation_id=uuid4(),
            agent_name="testing",
            status="completed",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        assert log.project_path is None
        assert log.project_name is None
        assert log.claude_session_id is None
        assert log.terminal_id is None

    def test_execution_log_project_context_accepts_values(self) -> None:
        """Project context fields should accept valid string values."""
        log = ModelExecutionLog(
            execution_id=uuid4(),
            correlation_id=uuid4(),
            agent_name="testing",
            status="completed",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            project_path="/home/user/projects/omnibase_infra",
            project_name="omnibase_infra",
            claude_session_id="session-jkl-012",
            terminal_id="/dev/ttys003",
        )

        assert log.project_path == "/home/user/projects/omnibase_infra"
        assert log.project_name == "omnibase_infra"
        assert log.claude_session_id == "session-jkl-012"
        assert log.terminal_id == "/dev/ttys003"


__all__ = [
    "TestModelObservabilityEnvelopeStrict",
    "TestModelAgentActionStrict",
    "TestModelAgentStatusEventStrict",
    "TestModelAgentStatusEventSpecific",
    "TestModelRoutingDecisionStrict",
    "TestModelTransformationEventStrict",
    "TestModelPerformanceMetricStrict",
    "TestModelDetectionFailureStrict",
    "TestModelExecutionLogStrict",
    "TestUUIDValidation",
    "TestDatetimeValidation",
    "TestRawPayloadValidation",
    "TestMetadataValidation",
    "TestModelAgentActionSpecific",
    "TestModelRoutingDecisionSpecific",
    "TestModelExecutionLogSpecific",
    "TestModelDetectionFailureSpecific",
    "TestModelAgentActionProjectContext",
    "TestModelRoutingDecisionProjectContext",
    "TestModelTransformationEventProjectContext",
    "TestModelDetectionFailureProjectContext",
    "TestModelExecutionLogProjectContext",
]
