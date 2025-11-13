#!/usr/bin/env python3
"""
Helper utilities for creating ModelOnexEnvelopeV1-wrapped introspection events.

Provides factory functions and examples for wrapping introspection event models
in ModelOnexEnvelopeV1 format for Kafka publishing.

ONEX v2.0 Compliance:
- ModelOnexEnvelopeV1 integration (canonical envelope model)
- Kafka topic routing
- Event correlation and causation tracking
"""

from datetime import UTC, datetime
from typing import Optional
from uuid import UUID, uuid4

from omninode_bridge.nodes.registry.v1_0_0.models import ModelOnexEnvelopeV1

from .enum_workflow_event import EnumWorkflowEvent
from .model_node_heartbeat_event import EnumNodeHealthStatus, ModelNodeHeartbeatEvent
from .model_node_introspection_event import ModelNodeIntrospectionEvent
from .model_registry_request_event import (
    EnumIntrospectionReason,
    ModelRegistryRequestEvent,
)

# Factory Functions


def create_node_introspection_envelope(
    introspection_data: ModelNodeIntrospectionEvent,
    source_instance: Optional[str] = None,
    environment: str = "development",
    correlation_id: Optional[UUID] = None,
    causation_id: Optional[str] = None,
    network_id: Optional[str] = None,
    deployment_id: Optional[str] = None,
    epoch: Optional[int] = None,
) -> ModelOnexEnvelopeV1:
    """
    Create ModelOnexEnvelopeV1 wrapper for NODE_INTROSPECTION event.

    Args:
        introspection_data: Node introspection event data
        source_instance: Optional source instance ID
        environment: Environment name
        correlation_id: Optional correlation ID
        causation_id: Optional causation event ID
        network_id: Optional network identifier for multi-network support
        deployment_id: Optional deployment instance identifier
        epoch: Optional deployment epoch for blue-green deployments

    Returns:
        ModelOnexEnvelopeV1 with introspection payload

    Example:
        >>> introspection = ModelNodeIntrospectionEvent.create(
        ...     node_id="node-123",
        ...     node_type="orchestrator",
        ...     capabilities={"hash_generation": {"algorithm": "BLAKE3"}},
        ...     endpoints={"health": "http://localhost:8053/health"}
        ... )
        >>> envelope = create_node_introspection_envelope(
        ...     introspection_data=introspection,
        ...     source_instance="orchestrator-001",
        ...     environment="production",
        ...     network_id="omninode-network-1",
        ...     deployment_id="prod-us-west-2-001",
        ...     epoch=1
        ... )
        >>> topic = envelope.to_kafka_topic()
        >>> # "prod.omninode_bridge.onex.evt.node-introspection.v1"
        >>> # Note: Topic format uses standardized ONEX convention: "prod.omninode_bridge.onex.evt.node-introspection.v1"
    """
    # Build metadata with network topology fields (Phase 1a MVP)
    metadata = {
        "node_type": introspection_data.node_type,
        "capabilities_count": len(introspection_data.capabilities),
        "endpoints_count": len(introspection_data.endpoints),
    }

    # Add network topology metadata if provided
    if network_id is not None:
        metadata["network_id"] = network_id
    if deployment_id is not None:
        metadata["deployment_id"] = deployment_id
    if epoch is not None:
        metadata["epoch"] = epoch

    return ModelOnexEnvelopeV1(
        event_type=EnumWorkflowEvent.NODE_INTROSPECTION.value,
        source_node_id=str(introspection_data.node_id),
        source_instance=source_instance,
        environment=environment,
        correlation_id=correlation_id,
        causation_id=causation_id,
        partition_key=str(introspection_data.node_id),  # Partition by node_id
        payload=introspection_data.to_dict(),
        metadata=metadata,
    )


def create_registry_request_envelope(
    request_data: ModelRegistryRequestEvent,
    source_instance: Optional[str] = None,
    environment: str = "development",
    correlation_id: Optional[UUID] = None,
) -> ModelOnexEnvelopeV1:
    """
    Create ModelOnexEnvelopeV1 wrapper for REGISTRY_REQUEST_INTROSPECTION event.

    Args:
        request_data: Registry request event data
        source_instance: Optional source instance ID
        environment: Environment name
        correlation_id: Optional correlation ID

    Returns:
        ModelOnexEnvelopeV1 with registry request payload

    Example:
        >>> request = ModelRegistryRequestEvent.create(
        ...     registry_id="registry-001",
        ...     reason=EnumIntrospectionReason.STARTUP,
        ...     response_timeout_ms=5000
        ... )
        >>> envelope = create_registry_request_envelope(
        ...     request_data=request,
        ...     source_instance="registry-001",
        ...     environment="production"
        ... )
        >>> topic = envelope.to_kafka_topic()
        >>> # "prod.omninode_bridge.onex.evt.registry-request-introspection.v1"
        >>> # Note: Topic format uses standardized ONEX convention: "prod.omninode_bridge.onex.evt.registry-request-introspection.v1"
    """
    return ModelOnexEnvelopeV1(
        event_type=EnumWorkflowEvent.REGISTRY_REQUEST_INTROSPECTION.value,
        source_node_id=str(request_data.registry_id),
        source_instance=source_instance,
        environment=environment,
        correlation_id=correlation_id,
        partition_key=str(request_data.registry_id),  # Partition by registry_id
        payload=request_data.to_dict(),
        metadata={
            "reason": request_data.reason.value,
            "target_node_types": request_data.target_node_types,
            "response_timeout_ms": request_data.response_timeout_ms,
        },
    )


def create_heartbeat_envelope(
    heartbeat_data: ModelNodeHeartbeatEvent,
    source_instance: Optional[str] = None,
    environment: str = "development",
    correlation_id: Optional[UUID] = None,
) -> ModelOnexEnvelopeV1:
    """
    Create ModelOnexEnvelopeV1 wrapper for NODE_HEARTBEAT event.

    Args:
        heartbeat_data: Node heartbeat event data
        source_instance: Optional source instance ID
        environment: Environment name
        correlation_id: Optional correlation ID

    Returns:
        ModelOnexEnvelopeV1 with heartbeat payload

    Example:
        >>> heartbeat = ModelNodeHeartbeatEvent.create(
        ...     node_id="node-123",
        ...     node_type="orchestrator",
        ...     health_status=EnumNodeHealthStatus.HEALTHY,
        ...     uptime_seconds=3600,
        ...     last_activity_timestamp=datetime.now(UTC),
        ...     active_operations=5
        ... )
        >>> envelope = create_heartbeat_envelope(
        ...     heartbeat_data=heartbeat,
        ...     source_instance="orchestrator-001",
        ...     environment="production"
        ... )
        >>> topic = envelope.to_kafka_topic()
        >>> # "prod.omninode_bridge.onex.evt.node-heartbeat.v1"
        >>> # Note: Topic format uses standardized ONEX convention: "prod.omninode_bridge.onex.evt.node-heartbeat.v1"
    """
    return ModelOnexEnvelopeV1(
        event_type=EnumWorkflowEvent.NODE_HEARTBEAT.value,
        source_node_id=str(heartbeat_data.node_id),
        source_instance=source_instance,
        environment=environment,
        correlation_id=correlation_id,
        partition_key=str(heartbeat_data.node_id),  # Partition by node_id
        payload=heartbeat_data.to_dict(),
        metadata={
            "node_type": heartbeat_data.node_type,
            "health_status": heartbeat_data.health_status.value,
            "uptime_seconds": heartbeat_data.uptime_seconds,
            "active_operations": heartbeat_data.active_operations,
        },
    )


# Usage Examples


def example_node_introspection_workflow():
    """
    Example: Complete workflow for node introspection broadcasting.

    This example demonstrates:
    1. Creating introspection data
    2. Wrapping in OnexEnvelopeV1
    3. Publishing to Kafka (pseudocode)
    """
    # Step 1: Create introspection data
    introspection = ModelNodeIntrospectionEvent.create(
        node_id="orchestrator-001",
        node_type="orchestrator",
        capabilities={
            "hash_generation": {"algorithm": "BLAKE3", "max_file_size_mb": 100},
            "metadata_stamping": {"formats": ["markdown", "html"], "batch_size": 50},
            "workflow_orchestration": {
                "max_concurrent": 100,
                "timeout_ms": 5000,
            },
        },
        endpoints={
            "health": "http://orchestrator-001:8053/health",
            "api": "http://orchestrator-001:8053/api/v1",
            "metrics": "http://orchestrator-001:9090/metrics",
        },
        metadata={
            "version": "1.0.0",
            "environment": "production",
            "region": "us-west-2",
            "resource_limits": {"cpu_cores": 4, "memory_gb": 16},
        },
    )

    # Step 2: Wrap in OnexEnvelopeV1
    envelope = create_node_introspection_envelope(
        introspection_data=introspection,
        source_instance="orchestrator-001",
        environment="production",
        correlation_id=uuid4(),
    )

    # Step 3: Publish to Kafka (pseudocode)
    topic = envelope.to_kafka_topic()
    key = envelope.get_kafka_key()
    # kafka_producer.send(topic=topic, key=key, value=envelope.model_dump_json())

    return envelope


def example_registry_request_workflow():
    """
    Example: Registry requesting introspection from all nodes.

    This example demonstrates:
    1. Registry creates request event
    2. Wraps in OnexEnvelopeV1
    3. Broadcasts to all nodes
    """
    # Step 1: Create registry request
    request = ModelRegistryRequestEvent.create(
        registry_id="registry-001",
        reason=EnumIntrospectionReason.STARTUP,
        target_node_types=None,  # Request from all node types
        response_timeout_ms=5000,
        metadata={
            "registry_version": "1.0.0",
            "environment": "production",
            "expected_node_count": 10,
        },
    )

    # Step 2: Wrap in OnexEnvelopeV1
    correlation_id = uuid4()
    envelope = create_registry_request_envelope(
        request_data=request,
        source_instance="registry-001",
        environment="production",
        correlation_id=correlation_id,
    )

    # Step 3: Broadcast to all nodes
    topic = envelope.to_kafka_topic()
    # kafka_producer.send(topic=topic, value=envelope.model_dump_json())

    # Nodes respond with NODE_INTROSPECTION events using same correlation_id
    return envelope, correlation_id


def example_heartbeat_workflow():
    """
    Example: Periodic heartbeat broadcasting.

    This example demonstrates:
    1. Node collects health metrics
    2. Creates heartbeat event
    3. Broadcasts periodically
    """
    # Step 1: Create heartbeat with current metrics
    heartbeat = ModelNodeHeartbeatEvent.create(
        node_id="orchestrator-001",
        node_type="orchestrator",
        health_status=EnumNodeHealthStatus.HEALTHY,
        uptime_seconds=3600,
        last_activity_timestamp=datetime.now(UTC),
        active_operations=5,
        resource_usage={
            "cpu_percent": 45.2,
            "memory_mb": 512,
            "memory_percent": 32.0,
            "disk_usage_percent": 68.5,
        },
        metadata={
            "version": "1.0.0",
            "error_count_last_minute": 0,
            "request_count_last_minute": 150,
        },
    )

    # Step 2: Wrap in OnexEnvelopeV1
    envelope = create_heartbeat_envelope(
        heartbeat_data=heartbeat,
        source_instance="orchestrator-001",
        environment="production",
    )

    # Step 3: Publish to Kafka (typically on 30s interval)
    topic = envelope.to_kafka_topic()
    key = envelope.get_kafka_key()
    # kafka_producer.send(topic=topic, key=key, value=envelope.model_dump_json())

    return envelope
