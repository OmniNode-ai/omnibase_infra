#!/usr/bin/env python3
"""
Event Publishing Pattern Generator - OnexEnvelopeV1 Compliant.

Generates production-ready event publishing code for all node operation lifecycle
events, state changes, and metrics. All generated code uses ModelOnexEnvelopeV1
format with proper correlation tracking, timestamps, and error handling.

Phase 2 Workstream 3: Reduces manual completion from 50% → 10%

ONEX v2.0 Compliance:
- ModelOnexEnvelopeV1 format for all events
- Correlation ID tracking across operations
- UTC timestamps (datetime.now(UTC))
- Source node identification
- Graceful error handling
- Metadata enrichment

Event Categories:
1. Operation Lifecycle: {operation}.started, {operation}.completed, {operation}.failed
2. State Events: {node}.state.changed
3. Metric Events: {node}.metric.recorded
"""

from typing import Any


class EventPublishingPatternGenerator:
    """
    Generator for OnexEnvelopeV1-compliant event publishing code.

    Produces complete event publishing methods with:
    - Proper imports (ModelOnexEnvelopeV1, datetime, UUID, emit_log_event)
    - Correlation ID tracking
    - UTC timestamps
    - Source node tracking
    - Metadata enrichment
    - Graceful error handling
    """

    def __init__(
        self,
        node_type: str,
        service_name: str,
        operations: list[str],
        include_state_events: bool = True,
        include_metric_events: bool = True,
    ):
        """
        Initialize event pattern generator.

        Args:
            node_type: Node type (e.g., "orchestrator", "reducer", "effect")
            service_name: Service name for event topics
            operations: List of operations to generate events for
            include_state_events: Generate state change events
            include_metric_events: Generate metric recording events
        """
        # Input validation
        if not node_type or not isinstance(node_type, str):
            raise ValueError(
                f"node_type must be a non-empty string, got: {node_type!r}. "
                f"Valid options: 'effect', 'compute', 'reducer', 'orchestrator'"
            )

        VALID_NODE_TYPES = {"effect", "compute", "reducer", "orchestrator"}
        if node_type.lower() not in VALID_NODE_TYPES:
            raise ValueError(
                f"Invalid node_type: {node_type!r}. "
                f"Valid options: {', '.join(sorted(VALID_NODE_TYPES))}"
            )

        if not service_name or not isinstance(service_name, str):
            raise ValueError(
                f"service_name must be a non-empty string, got: {service_name!r}. "
                f"Valid examples: 'orchestrator', 'metadata_stamping', 'bridge_reducer'"
            )

        if not isinstance(operations, list):
            raise TypeError(
                f"operations must be a list, got: {type(operations).__name__}. "
                f"Example: ['orchestration', 'routing', 'validation']"
            )

        if not operations:
            raise ValueError(
                "operations must contain at least one operation, got empty list. "
                "Example: ['orchestration', 'routing', 'validation']"
            )

        for op in operations:
            if not isinstance(op, str) or not op:
                raise ValueError(
                    f"All operations must be non-empty strings, got invalid operation: {op!r}. "
                    f"Valid examples: 'orchestration', 'aggregation', 'query'"
                )

        if not isinstance(include_state_events, bool):
            raise TypeError(
                f"include_state_events must be a boolean, got: {type(include_state_events).__name__}"
            )

        if not isinstance(include_metric_events, bool):
            raise TypeError(
                f"include_metric_events must be a boolean, got: {type(include_metric_events).__name__}"
            )

        self.node_type = node_type
        self.service_name = service_name
        self.operations = operations
        self.include_state_events = include_state_events
        self.include_metric_events = include_metric_events

    def generate_imports(self) -> str:
        """Generate required imports for event publishing."""
        return '''"""Required imports for event publishing."""
from datetime import UTC, datetime
from typing import Any, Optional
from uuid import UUID, uuid4

try:
    from omnibase_core.logging.structured import emit_log_event_sync as emit_log_event
    from omnibase_core.enums.enum_log_level import EnumLogLevel as LogLevel
except ImportError:
    # Fallback for testing
    from enum import Enum

    class LogLevel(str, Enum):
        DEBUG = "DEBUG"
        INFO = "INFO"
        WARNING = "WARNING"
        ERROR = "ERROR"

    def emit_log_event(level: LogLevel, message: str, metadata: dict[str, Any]) -> None:
        print(f"[{level.value}] {message}: {metadata}")

# Import ModelOnexEnvelopeV1 (will be available via kafka_client)
# The kafka_client.publish_with_envelope method handles envelope creation
'''

    def generate_all_event_methods(self) -> str:
        """Generate all event publishing methods for the node."""
        methods = []

        # Generate operation lifecycle events
        for operation in self.operations:
            methods.append(self.generate_operation_started_event(operation))
            methods.append(self.generate_operation_completed_event(operation))
            methods.append(self.generate_operation_failed_event(operation))

        # Generate state change event
        if self.include_state_events:
            methods.append(self.generate_state_changed_event())

        # Generate metric recording event
        if self.include_metric_events:
            methods.append(self.generate_metric_recorded_event())

        return "\n\n".join(methods)

    def generate_operation_started_event(self, operation: str) -> str:
        """
        Generate {operation}.started event publisher.

        Args:
            operation: Operation name (e.g., "orchestration", "aggregation")

        Returns:
            Complete async method for publishing operation started event
        """
        method_name = f"_publish_{operation}_started_event"
        event_type = f"{self.node_type}.{operation}.started"

        return f'''    async def {method_name}(
        self,
        correlation_id: UUID,
        input_data: dict[str, Any],
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Publish {operation} started event.

        Args:
            correlation_id: Correlation ID for tracking related events
            input_data: Input data for the operation
            metadata: Optional additional metadata
        """
        try:
            # Prepare event payload
            payload = {{
                "operation": "{operation}",
                "correlation_id": str(correlation_id),
                "node_id": self.node_id,
                "started_at": datetime.now(UTC).isoformat(),
                "input_summary": {{k: type(v).__name__ for k, v in input_data.items()}},
            }}

            # Add optional metadata
            if metadata:
                payload["metadata"] = metadata

            # Publish event using kafka_client (handles OnexEnvelopeV1 wrapping)
            if self.kafka_client and self.kafka_client.is_connected:
                topic_name = f"{{self.default_namespace}}.{self.service_name}.{event_type}.v1"

                event_metadata = {{
                    "node_type": "{self.node_type}",
                    "service_name": "{self.service_name}",
                    "operation": "{operation}",
                    "lifecycle_stage": "started",
                }}

                success = await self.kafka_client.publish_with_envelope(
                    event_type="{event_type}",
                    source_node_id=str(self.node_id),
                    payload=payload,
                    topic=topic_name,
                    correlation_id=correlation_id,
                    metadata=event_metadata,
                )

                if success:
                    emit_log_event(
                        LogLevel.DEBUG,
                        f"Published {operation} started event",
                        {{
                            "correlation_id": str(correlation_id),
                            "event_type": "{event_type}",
                            "topic": topic_name,
                        }},
                    )
            else:
                emit_log_event(
                    LogLevel.DEBUG,
                    f"Kafka unavailable, logging {operation} started event",
                    {{"correlation_id": str(correlation_id), "payload": payload}},
                )

        except Exception as e:
            emit_log_event(
                LogLevel.WARNING,
                f"Failed to publish {operation} started event: {{e}}",
                {{"correlation_id": str(correlation_id), "error": str(e)}},
            )'''

    def generate_operation_completed_event(self, operation: str) -> str:
        """
        Generate {operation}.completed event publisher.

        Args:
            operation: Operation name (e.g., "orchestration", "aggregation")

        Returns:
            Complete async method for publishing operation completed event
        """
        method_name = f"_publish_{operation}_completed_event"
        event_type = f"{self.node_type}.{operation}.completed"

        return f'''    async def {method_name}(
        self,
        correlation_id: UUID,
        result_data: dict[str, Any],
        execution_time_ms: float,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Publish {operation} completed event.

        Args:
            correlation_id: Correlation ID for tracking related events
            result_data: Result data from the operation
            execution_time_ms: Operation execution time in milliseconds
            metadata: Optional additional metadata
        """
        try:
            # Prepare event payload
            payload = {{
                "operation": "{operation}",
                "correlation_id": str(correlation_id),
                "node_id": self.node_id,
                "completed_at": datetime.now(UTC).isoformat(),
                "execution_time_ms": execution_time_ms,
                "result_summary": {{k: type(v).__name__ for k, v in result_data.items()}},
                "success": True,
            }}

            # Add optional metadata
            if metadata:
                payload["metadata"] = metadata

            # Publish event using kafka_client (handles OnexEnvelopeV1 wrapping)
            if self.kafka_client and self.kafka_client.is_connected:
                topic_name = f"{{self.default_namespace}}.{self.service_name}.{event_type}.v1"

                event_metadata = {{
                    "node_type": "{self.node_type}",
                    "service_name": "{self.service_name}",
                    "operation": "{operation}",
                    "lifecycle_stage": "completed",
                    "execution_time_ms": execution_time_ms,
                }}

                success = await self.kafka_client.publish_with_envelope(
                    event_type="{event_type}",
                    source_node_id=str(self.node_id),
                    payload=payload,
                    topic=topic_name,
                    correlation_id=correlation_id,
                    metadata=event_metadata,
                )

                if success:
                    emit_log_event(
                        LogLevel.INFO,
                        f"Published {operation} completed event",
                        {{
                            "correlation_id": str(correlation_id),
                            "event_type": "{event_type}",
                            "topic": topic_name,
                            "execution_time_ms": execution_time_ms,
                        }},
                    )
            else:
                emit_log_event(
                    LogLevel.DEBUG,
                    f"Kafka unavailable, logging {operation} completed event",
                    {{"correlation_id": str(correlation_id), "payload": payload}},
                )

        except Exception as e:
            emit_log_event(
                LogLevel.WARNING,
                f"Failed to publish {operation} completed event: {{e}}",
                {{"correlation_id": str(correlation_id), "error": str(e)}},
            )'''

    def generate_operation_failed_event(self, operation: str) -> str:
        """
        Generate {operation}.failed event publisher.

        Args:
            operation: Operation name (e.g., "orchestration", "aggregation")

        Returns:
            Complete async method for publishing operation failed event
        """
        method_name = f"_publish_{operation}_failed_event"
        event_type = f"{self.node_type}.{operation}.failed"

        return f'''    async def {method_name}(
        self,
        correlation_id: UUID,
        error: Exception,
        execution_time_ms: float,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Publish {operation} failed event.

        Args:
            correlation_id: Correlation ID for tracking related events
            error: Exception that caused the failure
            execution_time_ms: Operation execution time before failure
            metadata: Optional additional metadata
        """
        try:
            # Prepare event payload
            payload = {{
                "operation": "{operation}",
                "correlation_id": str(correlation_id),
                "node_id": self.node_id,
                "failed_at": datetime.now(UTC).isoformat(),
                "execution_time_ms": execution_time_ms,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "success": False,
            }}

            # Add optional metadata
            if metadata:
                payload["metadata"] = metadata

            # Publish event using kafka_client (handles OnexEnvelopeV1 wrapping)
            if self.kafka_client and self.kafka_client.is_connected:
                topic_name = f"{{self.default_namespace}}.{self.service_name}.{event_type}.v1"

                event_metadata = {{
                    "node_type": "{self.node_type}",
                    "service_name": "{self.service_name}",
                    "operation": "{operation}",
                    "lifecycle_stage": "failed",
                    "error_type": type(error).__name__,
                    "execution_time_ms": execution_time_ms,
                }}

                success = await self.kafka_client.publish_with_envelope(
                    event_type="{event_type}",
                    source_node_id=str(self.node_id),
                    payload=payload,
                    topic=topic_name,
                    correlation_id=correlation_id,
                    metadata=event_metadata,
                )

                if success:
                    emit_log_event(
                        LogLevel.ERROR,
                        f"Published {operation} failed event",
                        {{
                            "correlation_id": str(correlation_id),
                            "event_type": "{event_type}",
                            "topic": topic_name,
                            "error": str(error),
                        }},
                    )
            else:
                emit_log_event(
                    LogLevel.DEBUG,
                    f"Kafka unavailable, logging {operation} failed event",
                    {{"correlation_id": str(correlation_id), "payload": payload}},
                )

        except Exception as e:
            emit_log_event(
                LogLevel.WARNING,
                f"Failed to publish {operation} failed event: {{e}}",
                {{"correlation_id": str(correlation_id), "error": str(e)}},
            )'''

    def generate_state_changed_event(self) -> str:
        """Generate {node}.state.changed event publisher."""
        method_name = "_publish_state_changed_event"
        event_type = f"{self.node_type}.state.changed"

        return f'''    async def {method_name}(
        self,
        correlation_id: UUID,
        old_state: str,
        new_state: str,
        transition_reason: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Publish state changed event.

        Args:
            correlation_id: Correlation ID for tracking related events
            old_state: Previous state
            new_state: New state after transition
            transition_reason: Reason for state transition
            metadata: Optional additional metadata
        """
        try:
            # Prepare event payload
            payload = {{
                "node_id": self.node_id,
                "node_type": "{self.node_type}",
                "correlation_id": str(correlation_id),
                "changed_at": datetime.now(UTC).isoformat(),
                "old_state": old_state,
                "new_state": new_state,
                "transition_reason": transition_reason,
            }}

            # Add optional metadata
            if metadata:
                payload["metadata"] = metadata

            # Publish event using kafka_client (handles OnexEnvelopeV1 wrapping)
            if self.kafka_client and self.kafka_client.is_connected:
                topic_name = f"{{self.default_namespace}}.{self.service_name}.{event_type}.v1"

                event_metadata = {{
                    "node_type": "{self.node_type}",
                    "service_name": "{self.service_name}",
                    "event_category": "state_transition",
                    "old_state": old_state,
                    "new_state": new_state,
                }}

                success = await self.kafka_client.publish_with_envelope(
                    event_type="{event_type}",
                    source_node_id=str(self.node_id),
                    payload=payload,
                    topic=topic_name,
                    correlation_id=correlation_id,
                    metadata=event_metadata,
                )

                if success:
                    emit_log_event(
                        LogLevel.INFO,
                        f"Published state changed event: {{old_state}} → {{new_state}}",
                        {{
                            "correlation_id": str(correlation_id),
                            "event_type": "{event_type}",
                            "topic": topic_name,
                            "transition": f"{{old_state}} → {{new_state}}",
                        }},
                    )
            else:
                emit_log_event(
                    LogLevel.DEBUG,
                    f"Kafka unavailable, logging state changed event",
                    {{"correlation_id": str(correlation_id), "payload": payload}},
                )

        except Exception as e:
            emit_log_event(
                LogLevel.WARNING,
                f"Failed to publish state changed event: {{e}}",
                {{"correlation_id": str(correlation_id), "error": str(e)}},
            )'''

    def generate_metric_recorded_event(self) -> str:
        """Generate {node}.metric.recorded event publisher."""
        method_name = "_publish_metric_recorded_event"
        event_type = f"{self.node_type}.metric.recorded"

        return f'''    async def {method_name}(
        self,
        metric_name: str,
        metric_value: float,
        metric_unit: str,
        correlation_id: Optional[UUID] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Publish metric recorded event.

        Args:
            metric_name: Name of the metric
            metric_value: Metric value
            metric_unit: Unit of measurement
            correlation_id: Optional correlation ID
            metadata: Optional additional metadata
        """
        try:
            # Prepare event payload
            payload = {{
                "node_id": self.node_id,
                "node_type": "{self.node_type}",
                "recorded_at": datetime.now(UTC).isoformat(),
                "metric_name": metric_name,
                "metric_value": metric_value,
                "metric_unit": metric_unit,
            }}

            # Add correlation ID if provided
            if correlation_id:
                payload["correlation_id"] = str(correlation_id)

            # Add optional metadata
            if metadata:
                payload["metadata"] = metadata

            # Publish event using kafka_client (handles OnexEnvelopeV1 wrapping)
            if self.kafka_client and self.kafka_client.is_connected:
                topic_name = f"{{self.default_namespace}}.{self.service_name}.{event_type}.v1"

                event_metadata = {{
                    "node_type": "{self.node_type}",
                    "service_name": "{self.service_name}",
                    "event_category": "metric",
                    "metric_name": metric_name,
                    "metric_unit": metric_unit,
                }}

                success = await self.kafka_client.publish_with_envelope(
                    event_type="{event_type}",
                    source_node_id=str(self.node_id),
                    payload=payload,
                    topic=topic_name,
                    correlation_id=correlation_id,
                    metadata=event_metadata,
                )

                if success:
                    emit_log_event(
                        LogLevel.DEBUG,
                        f"Published metric recorded event: {{metric_name}}={{metric_value}}{{metric_unit}}",
                        {{
                            "correlation_id": str(correlation_id) if correlation_id else None,
                            "event_type": "{event_type}",
                            "topic": topic_name,
                            "metric": f"{{metric_name}}={{metric_value}}{{metric_unit}}",
                        }},
                    )
            else:
                emit_log_event(
                    LogLevel.DEBUG,
                    f"Kafka unavailable, logging metric recorded event",
                    {{"metric_name": metric_name, "payload": payload}},
                )

        except Exception as e:
            emit_log_event(
                LogLevel.WARNING,
                f"Failed to publish metric recorded event: {{e}}",
                {{"metric_name": metric_name, "error": str(e)}},
            )'''

    def get_event_type_catalog(self) -> dict[str, list[str]]:
        """
        Get catalog of all event types that will be generated.

        Returns:
            Dictionary mapping event categories to event type lists
        """
        catalog: dict[str, list[str]] = {
            "operation_lifecycle": [],
            "state_events": [],
            "metric_events": [],
        }

        # Operation lifecycle events
        for operation in self.operations:
            catalog["operation_lifecycle"].extend(
                [
                    f"{self.node_type}.{operation}.started",
                    f"{self.node_type}.{operation}.completed",
                    f"{self.node_type}.{operation}.failed",
                ]
            )

        # State events
        if self.include_state_events:
            catalog["state_events"].append(f"{self.node_type}.state.changed")

        # Metric events
        if self.include_metric_events:
            catalog["metric_events"].append(f"{self.node_type}.metric.recorded")

        return catalog


# Convenience Functions for Direct Generation


def generate_event_publishing_methods(
    node_type: str,
    service_name: str,
    operations: list[str],
    include_state_events: bool = True,
    include_metric_events: bool = True,
) -> str:
    """
    Generate all event publishing methods for a node.

    Args:
        node_type: Node type (e.g., "orchestrator", "reducer")
        service_name: Service name for event topics
        operations: List of operations to generate events for
        include_state_events: Generate state change events
        include_metric_events: Generate metric recording events

    Returns:
        Complete Python code with all event publishing methods

    Example:
        >>> code = generate_event_publishing_methods(
        ...     node_type="orchestrator",
        ...     service_name="orchestrator",
        ...     operations=["orchestration", "routing"],
        ... )
        >>> print(code)  # Outputs complete event publishing methods
    """
    generator = EventPublishingPatternGenerator(
        node_type=node_type,
        service_name=service_name,
        operations=operations,
        include_state_events=include_state_events,
        include_metric_events=include_metric_events,
    )

    return f"{generator.generate_imports()}\n\n{generator.generate_all_event_methods()}"


def generate_operation_started_event(service_name: str, operation: str) -> str:
    """
    Generate operation started event publisher.

    Args:
        service_name: Service name for event topics
        operation: Operation name

    Returns:
        Complete async method for publishing operation started event
    """
    generator = EventPublishingPatternGenerator(
        node_type="generic",
        service_name=service_name,
        operations=[operation],
        include_state_events=False,
        include_metric_events=False,
    )
    return generator.generate_operation_started_event(operation)


def generate_operation_completed_event(service_name: str, operation: str) -> str:
    """
    Generate operation completed event publisher.

    Args:
        service_name: Service name for event topics
        operation: Operation name

    Returns:
        Complete async method for publishing operation completed event
    """
    generator = EventPublishingPatternGenerator(
        node_type="generic",
        service_name=service_name,
        operations=[operation],
        include_state_events=False,
        include_metric_events=False,
    )
    return generator.generate_operation_completed_event(operation)


def generate_operation_failed_event(service_name: str, operation: str) -> str:
    """
    Generate operation failed event publisher.

    Args:
        service_name: Service name for event topics
        operation: Operation name

    Returns:
        Complete async method for publishing operation failed event
    """
    generator = EventPublishingPatternGenerator(
        node_type="generic",
        service_name=service_name,
        operations=[operation],
        include_state_events=False,
        include_metric_events=False,
    )
    return generator.generate_operation_failed_event(operation)


def get_event_type_catalog(
    node_type: str,
    operations: list[str],
    include_state_events: bool = True,
    include_metric_events: bool = True,
) -> dict[str, list[str]]:
    """
    Get catalog of all event types for a node.

    Args:
        node_type: Node type
        operations: List of operations
        include_state_events: Include state events
        include_metric_events: Include metric events

    Returns:
        Dictionary mapping event categories to event type lists
    """
    generator = EventPublishingPatternGenerator(
        node_type=node_type,
        service_name=node_type,  # Use node_type as default service_name
        operations=operations,
        include_state_events=include_state_events,
        include_metric_events=include_metric_events,
    )
    return generator.get_event_type_catalog()


# Example Usage and Documentation


def example_orchestrator_events() -> str:
    """
    Example: Generate all event publishing methods for orchestrator node.

    Returns:
        Complete Python code with event publishing methods
    """
    return generate_event_publishing_methods(
        node_type="orchestrator",
        service_name="orchestrator",
        operations=["orchestration", "routing", "intelligence_query"],
        include_state_events=True,
        include_metric_events=True,
    )


def example_reducer_events() -> str:
    """
    Example: Generate all event publishing methods for reducer node.

    Returns:
        Complete Python code with event publishing methods
    """
    return generate_event_publishing_methods(
        node_type="reducer",
        service_name="reducer",
        operations=["aggregation", "state_snapshot"],
        include_state_events=True,
        include_metric_events=True,
    )


def example_event_catalog() -> dict[str, Any]:
    """
    Example: Get event type catalog for orchestrator.

    Returns:
        Event type catalog with all event names
    """
    return get_event_type_catalog(
        node_type="orchestrator",
        operations=["orchestration", "routing", "intelligence_query"],
        include_state_events=True,
        include_metric_events=True,
    )


if __name__ == "__main__":
    # Generate example orchestrator events
    print("=== Orchestrator Event Publishing Methods ===")
    print(example_orchestrator_events())
    print("\n\n")

    # Generate example reducer events
    print("=== Reducer Event Publishing Methods ===")
    print(example_reducer_events())
    print("\n\n")

    # Show event catalog
    print("=== Event Type Catalog ===")
    import json

    catalog = example_event_catalog()
    print(json.dumps(catalog, indent=2))
