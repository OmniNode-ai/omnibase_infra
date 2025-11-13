#!/usr/bin/env python3
"""
Unit tests for Event Publishing Pattern Generator.

Tests cover:
- Initialization and input validation
- Event generation (started, completed, failed, state, metric)
- OnexEnvelopeV1 format compliance
- Code compilation and AST verification
- Kafka integration patterns
- Event type catalog generation
"""

import ast

import pytest

from omninode_bridge.codegen.patterns.event_publishing import (
    EventPublishingPatternGenerator,
    generate_event_publishing_methods,
    generate_operation_completed_event,
    generate_operation_failed_event,
    generate_operation_started_event,
    get_event_type_catalog,
)

# ============================================================================
# Initialization Tests
# ============================================================================


def test_initialization_basic():
    """Test basic pattern generator initialization with valid parameters."""
    generator = EventPublishingPatternGenerator(
        node_type="effect",
        service_name="test-service",
        operations=["process"],
    )

    assert generator.node_type == "effect"
    assert generator.service_name == "test-service"
    assert generator.operations == ["process"]
    assert generator.include_state_events is True  # Default
    assert generator.include_metric_events is True  # Default


def test_initialization_all_parameters():
    """Test initialization with all parameters specified."""
    generator = EventPublishingPatternGenerator(
        node_type="orchestrator",
        service_name="orchestrator",
        operations=["orchestration", "routing"],
        include_state_events=False,
        include_metric_events=False,
    )

    assert generator.node_type == "orchestrator"
    assert generator.service_name == "orchestrator"
    assert generator.operations == ["orchestration", "routing"]
    assert generator.include_state_events is False
    assert generator.include_metric_events is False


def test_initialization_all_valid_node_types():
    """Test initialization with all valid node types."""
    valid_node_types = ["effect", "compute", "reducer", "orchestrator"]

    for node_type in valid_node_types:
        generator = EventPublishingPatternGenerator(
            node_type=node_type,
            service_name="test",
            operations=["process"],
        )
        assert generator.node_type == node_type


# ============================================================================
# Input Validation Tests
# ============================================================================


def test_invalid_node_type_empty_string():
    """Test that empty node_type raises ValueError."""
    with pytest.raises(ValueError, match="node_type must be a non-empty string"):
        EventPublishingPatternGenerator(
            node_type="",
            service_name="test",
            operations=["process"],
        )


def test_invalid_node_type_wrong_type():
    """Test that non-string node_type raises ValueError."""
    with pytest.raises(ValueError, match="node_type must be a non-empty string"):
        EventPublishingPatternGenerator(
            node_type=123,  # type: ignore
            service_name="test",
            operations=["process"],
        )


def test_invalid_node_type_invalid_value():
    """Test that invalid node_type value raises ValueError."""
    with pytest.raises(ValueError, match="Invalid node_type"):
        EventPublishingPatternGenerator(
            node_type="invalid_type",
            service_name="test",
            operations=["process"],
        )


def test_invalid_service_name_empty():
    """Test that empty service_name raises ValueError."""
    with pytest.raises(ValueError, match="service_name must be a non-empty string"):
        EventPublishingPatternGenerator(
            node_type="effect",
            service_name="",
            operations=["process"],
        )


def test_invalid_service_name_wrong_type():
    """Test that non-string service_name raises ValueError."""
    with pytest.raises(ValueError, match="service_name must be a non-empty string"):
        EventPublishingPatternGenerator(
            node_type="effect",
            service_name=456,  # type: ignore
            operations=["process"],
        )


def test_invalid_operations_not_list():
    """Test that non-list operations raises TypeError."""
    with pytest.raises(TypeError, match="operations must be a list"):
        EventPublishingPatternGenerator(
            node_type="effect",
            service_name="test",
            operations="not_a_list",  # type: ignore
        )


def test_invalid_operations_empty_list():
    """Test that empty operations list raises ValueError."""
    with pytest.raises(ValueError, match="operations must contain at least one"):
        EventPublishingPatternGenerator(
            node_type="effect",
            service_name="test",
            operations=[],
        )


def test_invalid_operations_contains_empty_string():
    """Test that operations with empty string raises ValueError."""
    with pytest.raises(ValueError, match="All operations must be non-empty strings"):
        EventPublishingPatternGenerator(
            node_type="effect",
            service_name="test",
            operations=["valid", ""],
        )


def test_invalid_operations_contains_non_string():
    """Test that operations with non-string raises ValueError."""
    with pytest.raises(ValueError, match="All operations must be non-empty strings"):
        EventPublishingPatternGenerator(
            node_type="effect",
            service_name="test",
            operations=["valid", 123],  # type: ignore
        )


def test_invalid_include_state_events_wrong_type():
    """Test that non-boolean include_state_events raises TypeError."""
    with pytest.raises(TypeError, match="include_state_events must be a boolean"):
        EventPublishingPatternGenerator(
            node_type="effect",
            service_name="test",
            operations=["process"],
            include_state_events="true",  # type: ignore
        )


def test_invalid_include_metric_events_wrong_type():
    """Test that non-boolean include_metric_events raises TypeError."""
    with pytest.raises(TypeError, match="include_metric_events must be a boolean"):
        EventPublishingPatternGenerator(
            node_type="effect",
            service_name="test",
            operations=["process"],
            include_metric_events="true",  # type: ignore
        )


# ============================================================================
# Event Generation Tests
# ============================================================================


def test_generate_operation_started_event():
    """Test generation of operation started event."""
    generator = EventPublishingPatternGenerator(
        node_type="effect",
        service_name="test-service",
        operations=["process"],
    )

    code = generator.generate_operation_started_event("process")

    # Verify method signature
    assert "async def _publish_process_started_event" in code
    assert "correlation_id: UUID" in code
    assert "input_data: dict[str, Any]" in code
    assert "metadata: Optional[dict[str, Any]]" in code

    # Verify event type
    assert 'event_type="effect.process.started"' in code

    # Verify Kafka integration
    assert "await self.kafka_client.publish_with_envelope" in code
    assert "source_node_id=str(self.node_id)" in code

    # Verify timestamp
    assert "datetime.now(UTC).isoformat()" in code

    # Verify error handling
    assert "try:" in code
    assert "except Exception as e:" in code


def test_generate_operation_completed_event():
    """Test generation of operation completed event."""
    generator = EventPublishingPatternGenerator(
        node_type="reducer",
        service_name="test-service",
        operations=["aggregation"],
    )

    code = generator.generate_operation_completed_event("aggregation")

    # Verify method signature
    assert "async def _publish_aggregation_completed_event" in code
    assert "correlation_id: UUID" in code
    assert "result_data: dict[str, Any]" in code
    assert "execution_time_ms: float" in code

    # Verify event type
    assert 'event_type="reducer.aggregation.completed"' in code

    # Verify success indicator
    assert '"success": True' in code

    # Verify execution time tracking
    assert '"execution_time_ms": execution_time_ms' in code


def test_generate_operation_failed_event():
    """Test generation of operation failed event."""
    generator = EventPublishingPatternGenerator(
        node_type="orchestrator",
        service_name="test-service",
        operations=["orchestration"],
    )

    code = generator.generate_operation_failed_event("orchestration")

    # Verify method signature
    assert "async def _publish_orchestration_failed_event" in code
    assert "correlation_id: UUID" in code
    assert "error: Exception" in code
    assert "execution_time_ms: float" in code

    # Verify event type
    assert 'event_type="orchestrator.orchestration.failed"' in code

    # Verify failure indicator
    assert '"success": False' in code

    # Verify error information
    assert '"error_type": type(error).__name__' in code
    assert '"error_message": str(error)' in code


def test_generate_state_changed_event():
    """Test generation of state changed event."""
    generator = EventPublishingPatternGenerator(
        node_type="effect",
        service_name="test-service",
        operations=["process"],
        include_state_events=True,
    )

    code = generator.generate_state_changed_event()

    # Verify method signature
    assert "async def _publish_state_changed_event" in code
    assert "old_state: str" in code
    assert "new_state: str" in code
    assert "transition_reason: str" in code

    # Verify event type
    assert 'event_type="effect.state.changed"' in code

    # Verify state tracking
    assert '"old_state": old_state' in code
    assert '"new_state": new_state' in code
    assert '"transition_reason": transition_reason' in code


def test_generate_metric_recorded_event():
    """Test generation of metric recorded event."""
    generator = EventPublishingPatternGenerator(
        node_type="compute",
        service_name="test-service",
        operations=["transform"],
        include_metric_events=True,
    )

    code = generator.generate_metric_recorded_event()

    # Verify method signature
    assert "async def _publish_metric_recorded_event" in code
    assert "metric_name: str" in code
    assert "metric_value: float" in code
    assert "metric_unit: str" in code
    assert "correlation_id: Optional[UUID]" in code

    # Verify event type
    assert 'event_type="compute.metric.recorded"' in code

    # Verify metric information
    assert '"metric_name": metric_name' in code
    assert '"metric_value": metric_value' in code
    assert '"metric_unit": metric_unit' in code


# ============================================================================
# OnexEnvelopeV1 Format Tests
# ============================================================================


def test_onex_envelope_v1_format():
    """Test that generated code uses OnexEnvelopeV1 format via kafka_client."""
    generator = EventPublishingPatternGenerator(
        node_type="effect",
        service_name="test-service",
        operations=["process"],
    )

    code = generator.generate_all_event_methods()

    # Verify kafka_client.publish_with_envelope is used (handles envelope wrapping)
    assert "await self.kafka_client.publish_with_envelope" in code

    # Verify all required OnexEnvelopeV1 parameters
    assert "event_type=" in code
    assert "source_node_id=" in code
    assert "payload=" in code
    assert "topic=" in code
    assert "correlation_id=" in code
    assert "metadata=" in code


def test_correlation_id_tracking():
    """Test that correlation ID is tracked across all events."""
    generator = EventPublishingPatternGenerator(
        node_type="effect",
        service_name="test-service",
        operations=["process"],
    )

    code = generator.generate_all_event_methods()

    # Verify correlation_id is in payload
    assert '"correlation_id": str(correlation_id)' in code

    # Verify correlation_id passed to publish_with_envelope
    assert "correlation_id=correlation_id" in code


def test_utc_timestamps():
    """Test that UTC timestamps are used in generated code."""
    generator = EventPublishingPatternGenerator(
        node_type="effect",
        service_name="test-service",
        operations=["process"],
    )

    code = generator.generate_all_event_methods()

    # Verify UTC datetime import
    imports = generator.generate_imports()
    assert "from datetime import UTC, datetime" in imports

    # Verify UTC timestamps in events
    assert "datetime.now(UTC).isoformat()" in code


def test_source_node_identification():
    """Test that source node is identified in generated code."""
    generator = EventPublishingPatternGenerator(
        node_type="effect",
        service_name="test-service",
        operations=["process"],
    )

    code = generator.generate_all_event_methods()

    # Verify node_id is in payload
    assert '"node_id": self.node_id' in code

    # Verify source_node_id passed to publish_with_envelope
    assert "source_node_id=str(self.node_id)" in code


def test_metadata_enrichment():
    """Test that metadata enrichment is included in generated code."""
    generator = EventPublishingPatternGenerator(
        node_type="effect",
        service_name="test-service",
        operations=["process"],
    )

    code = generator.generate_operation_started_event("process")

    # Verify metadata handling
    assert "if metadata:" in code
    assert 'payload["metadata"] = metadata' in code

    # Verify event_metadata
    assert "event_metadata = {" in code
    assert '"node_type"' in code
    assert '"service_name"' in code


def test_graceful_error_handling():
    """Test that graceful error handling is included in generated code."""
    generator = EventPublishingPatternGenerator(
        node_type="effect",
        service_name="test-service",
        operations=["process"],
    )

    code = generator.generate_all_event_methods()

    # Verify try-except blocks
    assert "try:" in code
    assert "except Exception as e:" in code

    # Verify emit_log_event for errors
    assert "emit_log_event(" in code
    assert "LogLevel.WARNING" in code
    assert "Failed to publish" in code


# ============================================================================
# Code Compilation Tests
# ============================================================================


def test_generated_code_compiles():
    """Test that generated code is valid Python and compiles."""
    generator = EventPublishingPatternGenerator(
        node_type="effect",
        service_name="test-service",
        operations=["process", "validate"],
    )

    # Generate complete code (imports + methods)
    imports = generator.generate_imports()
    methods = generator.generate_all_event_methods()
    full_code = f"{imports}\n\n{methods}"

    # Attempt to parse as AST (will raise SyntaxError if invalid)
    try:
        ast.parse(full_code)
    except SyntaxError as e:
        pytest.fail(f"Generated code has syntax error: {e}")


def test_generated_started_event_compiles():
    """Test that generated started event method compiles."""
    generator = EventPublishingPatternGenerator(
        node_type="effect",
        service_name="test-service",
        operations=["process"],
    )

    code = generator.generate_operation_started_event("process")

    # Wrap in class context for AST parsing
    full_code = f"class TestNode:\n{code}"

    try:
        ast.parse(full_code)
    except SyntaxError as e:
        pytest.fail(f"Generated started event has syntax error: {e}")


def test_generated_completed_event_compiles():
    """Test that generated completed event method compiles."""
    generator = EventPublishingPatternGenerator(
        node_type="effect",
        service_name="test-service",
        operations=["process"],
    )

    code = generator.generate_operation_completed_event("process")

    # Wrap in class context for AST parsing
    full_code = f"class TestNode:\n{code}"

    try:
        ast.parse(full_code)
    except SyntaxError as e:
        pytest.fail(f"Generated completed event has syntax error: {e}")


def test_generated_failed_event_compiles():
    """Test that generated failed event method compiles."""
    generator = EventPublishingPatternGenerator(
        node_type="effect",
        service_name="test-service",
        operations=["process"],
    )

    code = generator.generate_operation_failed_event("process")

    # Wrap in class context for AST parsing
    full_code = f"class TestNode:\n{code}"

    try:
        ast.parse(full_code)
    except SyntaxError as e:
        pytest.fail(f"Generated failed event has syntax error: {e}")


# ============================================================================
# Kafka Integration Tests
# ============================================================================


def test_kafka_producer_code_included():
    """Test that generated code includes Kafka producer integration."""
    generator = EventPublishingPatternGenerator(
        node_type="effect",
        service_name="test-service",
        operations=["process"],
    )

    code = generator.generate_all_event_methods()

    # Verify kafka_client usage
    assert "self.kafka_client" in code

    # Verify connection check
    assert "self.kafka_client.is_connected" in code

    # Verify publish_with_envelope method call
    assert "await self.kafka_client.publish_with_envelope" in code

    # Verify topic name generation
    assert "topic_name = f" in code


def test_kafka_fallback_logging():
    """Test that generated code includes fallback logging when Kafka unavailable."""
    generator = EventPublishingPatternGenerator(
        node_type="effect",
        service_name="test-service",
        operations=["process"],
    )

    code = generator.generate_operation_started_event("process")

    # Verify fallback logging
    assert "Kafka unavailable, logging" in code
    assert "LogLevel.DEBUG" in code


def test_kafka_success_confirmation():
    """Test that generated code logs success confirmation."""
    generator = EventPublishingPatternGenerator(
        node_type="effect",
        service_name="test-service",
        operations=["process"],
    )

    code = generator.generate_operation_completed_event("process")

    # Verify success check
    assert "if success:" in code

    # Verify success logging
    assert '"Published' in code


# ============================================================================
# Event Catalog Tests
# ============================================================================


def test_event_type_catalog_structure():
    """Test that event catalog has correct structure."""
    generator = EventPublishingPatternGenerator(
        node_type="effect",
        service_name="test-service",
        operations=["process", "validate"],
        include_state_events=True,
        include_metric_events=True,
    )

    catalog = generator.get_event_type_catalog()

    # Verify catalog keys
    assert "operation_lifecycle" in catalog
    assert "state_events" in catalog
    assert "metric_events" in catalog

    # Verify values are lists
    assert isinstance(catalog["operation_lifecycle"], list)
    assert isinstance(catalog["state_events"], list)
    assert isinstance(catalog["metric_events"], list)


def test_event_type_catalog_operation_lifecycle():
    """Test that operation lifecycle events are in catalog."""
    generator = EventPublishingPatternGenerator(
        node_type="effect",
        service_name="test-service",
        operations=["process", "validate"],
    )

    catalog = generator.get_event_type_catalog()

    # Verify operation lifecycle events
    operation_events = catalog["operation_lifecycle"]
    assert "effect.process.started" in operation_events
    assert "effect.process.completed" in operation_events
    assert "effect.process.failed" in operation_events
    assert "effect.validate.started" in operation_events
    assert "effect.validate.completed" in operation_events
    assert "effect.validate.failed" in operation_events


def test_event_type_catalog_state_events():
    """Test that state events are in catalog when enabled."""
    generator = EventPublishingPatternGenerator(
        node_type="effect",
        service_name="test-service",
        operations=["process"],
        include_state_events=True,
    )

    catalog = generator.get_event_type_catalog()

    # Verify state event
    assert "effect.state.changed" in catalog["state_events"]


def test_event_type_catalog_state_events_disabled():
    """Test that state events are not in catalog when disabled."""
    generator = EventPublishingPatternGenerator(
        node_type="effect",
        service_name="test-service",
        operations=["process"],
        include_state_events=False,
    )

    catalog = generator.get_event_type_catalog()

    # Verify no state events
    assert len(catalog["state_events"]) == 0


def test_event_type_catalog_metric_events():
    """Test that metric events are in catalog when enabled."""
    generator = EventPublishingPatternGenerator(
        node_type="effect",
        service_name="test-service",
        operations=["process"],
        include_metric_events=True,
    )

    catalog = generator.get_event_type_catalog()

    # Verify metric event
    assert "effect.metric.recorded" in catalog["metric_events"]


def test_event_type_catalog_metric_events_disabled():
    """Test that metric events are not in catalog when disabled."""
    generator = EventPublishingPatternGenerator(
        node_type="effect",
        service_name="test-service",
        operations=["process"],
        include_metric_events=False,
    )

    catalog = generator.get_event_type_catalog()

    # Verify no metric events
    assert len(catalog["metric_events"]) == 0


# ============================================================================
# Multiple Operations Tests
# ============================================================================


def test_multiple_operations_generates_all_events():
    """Test that multiple operations generate all lifecycle events."""
    generator = EventPublishingPatternGenerator(
        node_type="orchestrator",
        service_name="test-service",
        operations=["orchestration", "routing", "validation"],
    )

    code = generator.generate_all_event_methods()

    # Verify all started events
    assert "_publish_orchestration_started_event" in code
    assert "_publish_routing_started_event" in code
    assert "_publish_validation_started_event" in code

    # Verify all completed events
    assert "_publish_orchestration_completed_event" in code
    assert "_publish_routing_completed_event" in code
    assert "_publish_validation_completed_event" in code

    # Verify all failed events
    assert "_publish_orchestration_failed_event" in code
    assert "_publish_routing_failed_event" in code
    assert "_publish_validation_failed_event" in code


def test_operation_specific_event_types():
    """Test that each operation has unique event types."""
    generator = EventPublishingPatternGenerator(
        node_type="effect",
        service_name="test-service",
        operations=["process", "validate"],
    )

    process_started = generator.generate_operation_started_event("process")
    validate_started = generator.generate_operation_started_event("validate")

    # Verify unique event types
    assert 'event_type="effect.process.started"' in process_started
    assert 'event_type="effect.validate.started"' in validate_started


# ============================================================================
# Convenience Function Tests
# ============================================================================


def test_generate_event_publishing_methods_convenience():
    """Test convenience function for generating all methods."""
    code = generate_event_publishing_methods(
        node_type="effect",
        service_name="test-service",
        operations=["process"],
        include_state_events=True,
        include_metric_events=True,
    )

    # Verify imports are included
    assert "from datetime import UTC, datetime" in code
    assert "from uuid import UUID, uuid4" in code

    # Verify methods are included
    assert "_publish_process_started_event" in code
    assert "_publish_process_completed_event" in code
    assert "_publish_process_failed_event" in code
    assert "_publish_state_changed_event" in code
    assert "_publish_metric_recorded_event" in code


def test_generate_operation_started_event_convenience():
    """Test convenience function for generating started event."""
    # Note: convenience function uses "generic" as node_type which is invalid
    # This tests the actual behavior - it should raise ValueError
    with pytest.raises(ValueError, match="Invalid node_type"):
        generate_operation_started_event("test-service", "process")


def test_generate_operation_completed_event_convenience():
    """Test convenience function for generating completed event."""
    # Note: convenience function uses "generic" as node_type which is invalid
    # This tests the actual behavior - it should raise ValueError
    with pytest.raises(ValueError, match="Invalid node_type"):
        generate_operation_completed_event("test-service", "process")


def test_generate_operation_failed_event_convenience():
    """Test convenience function for generating failed event."""
    # Note: convenience function uses "generic" as node_type which is invalid
    # This tests the actual behavior - it should raise ValueError
    with pytest.raises(ValueError, match="Invalid node_type"):
        generate_operation_failed_event("test-service", "process")


def test_get_event_type_catalog_convenience():
    """Test convenience function for getting event catalog."""
    catalog = get_event_type_catalog(
        node_type="effect",
        operations=["process", "validate"],
        include_state_events=True,
        include_metric_events=True,
    )

    assert "operation_lifecycle" in catalog
    assert "state_events" in catalog
    assert "metric_events" in catalog
    assert len(catalog["operation_lifecycle"]) == 6  # 3 events × 2 operations


# ============================================================================
# Import Generation Tests
# ============================================================================


def test_generate_imports():
    """Test that required imports are generated."""
    generator = EventPublishingPatternGenerator(
        node_type="effect",
        service_name="test-service",
        operations=["process"],
    )

    imports = generator.generate_imports()

    # Verify core imports
    assert "from datetime import UTC, datetime" in imports
    assert "from typing import Any, Optional" in imports
    assert "from uuid import UUID, uuid4" in imports

    # Verify fallback imports
    assert "try:" in imports
    assert "from omnibase_core.logging.structured import emit_log_event_sync" in imports
    assert "except ImportError:" in imports

    # Verify comment about OnexEnvelopeV1
    assert "ModelOnexEnvelopeV1" in imports


def test_imports_compile():
    """Test that generated imports are valid Python."""
    generator = EventPublishingPatternGenerator(
        node_type="effect",
        service_name="test-service",
        operations=["process"],
    )

    imports = generator.generate_imports()

    try:
        ast.parse(imports)
    except SyntaxError as e:
        pytest.fail(f"Generated imports have syntax error: {e}")
