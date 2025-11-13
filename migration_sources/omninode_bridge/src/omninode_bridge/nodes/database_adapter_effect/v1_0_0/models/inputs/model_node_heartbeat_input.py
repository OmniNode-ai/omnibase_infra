#!/usr/bin/env python3
"""
ModelNodeHeartbeatInput - Node Heartbeat Update Input.

Input model for updating node heartbeat timestamps and health status
in PostgreSQL. Supports periodic health checks and node lifecycle
management for the bridge node registry.

ONEX v2.0 Compliance:
- Suffix-based naming: ModelNodeHeartbeatInput
- Node identity tracking
- Health status validation
- Comprehensive field validation with Pydantic v2
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ModelNodeHeartbeatInput(BaseModel):
    """
    Input model for node heartbeat database UPDATE operations.

    This model updates the heartbeat timestamp and health status for
    registered nodes in the node_registrations table. Periodic heartbeats
    enable detection of failed or stale nodes and facilitate automatic
    cleanup and failover.

    Database Table: node_registrations
    Primary Key: node_id (string)
    Indexes: health_status, last_heartbeat

    Health Statuses:
    - HEALTHY: Node operating normally
    - DEGRADED: Node experiencing issues but operational
    - UNHEALTHY: Node failing health checks
    - UNKNOWN: Health status cannot be determined
    - OFFLINE: Node explicitly offline or shutdown

    Event Sources:
    - NODE_HEARTBEAT: Published periodically by all bridge nodes

    Heartbeat Intervals:
    - Normal operation: Every 30 seconds
    - Degraded: Every 10 seconds (more frequent monitoring)
    - Shutdown: Final heartbeat with OFFLINE status

    Staleness Detection:
    - If last_heartbeat > 60 seconds ago: Node considered stale
    - If last_heartbeat > 120 seconds ago: Node marked for cleanup

    Example (Healthy Heartbeat):
        >>> from datetime import datetime
        >>> heartbeat = ModelNodeHeartbeatInput(
        ...     node_id="orchestrator-01",
        ...     health_status="HEALTHY",
        ...     metadata={
        ...         "version": "1.0.0",
        ...         "uptime_seconds": 3600,
        ...         "memory_usage_mb": 256,
        ...         "cpu_usage_percent": 15.5,
        ...         "active_workflows": 42,
        ...         "events_processed": 1000
        ...     }
        ... )

    Example (Degraded Node):
        >>> degraded_heartbeat = ModelNodeHeartbeatInput(
        ...     node_id="reducer-02",
        ...     health_status="DEGRADED",
        ...     metadata={
        ...         "version": "1.0.0",
        ...         "uptime_seconds": 7200,
        ...         "memory_usage_mb": 480,  # High memory usage
        ...         "cpu_usage_percent": 85.0,  # High CPU usage
        ...         "active_aggregations": 5,
        ...         "backlog_items": 500,  # Growing backlog
        ...         "warning": "High resource utilization"
        ...     }
        ... )

    Example (Shutdown Heartbeat):
        >>> shutdown_heartbeat = ModelNodeHeartbeatInput(
        ...     node_id="registry-01",
        ...     health_status="OFFLINE",
        ...     metadata={
        ...         "version": "1.0.0",
        ...         "shutdown_reason": "Planned maintenance",
        ...         "final_uptime_seconds": 86400,
        ...         "total_registrations": 150
        ...     }
        ... )
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        frozen=False,
        extra="forbid",
    )

    # === Node Identity ===
    node_id: str = Field(
        ...,
        description="""
        Unique identifier for the node.

        This ID must match the primary key in the node_registrations table.
        Typically follows the format: <node_type>-<instance_id>

        Examples:
        - "orchestrator-01"
        - "reducer-02"
        - "registry-main"
        - "metadata-stamping-service-01"
        """,
        min_length=1,
        max_length=255,
    )

    # === Health Status ===
    health_status: str = Field(
        ...,
        description="""
        Current health status of the node.

        Valid values:
        - "HEALTHY": Node operating normally, all systems functional
        - "DEGRADED": Node experiencing issues but still operational
        - "UNHEALTHY": Node failing health checks, may be non-functional
        - "UNKNOWN": Health status cannot be determined
        - "OFFLINE": Node explicitly offline or shutdown

        Status Transitions:
        - HEALTHY → DEGRADED: Performance degradation detected
        - DEGRADED → UNHEALTHY: Critical issues detected
        - DEGRADED → HEALTHY: Issues resolved
        - Any → OFFLINE: Node shutdown
        - OFFLINE → HEALTHY: Node restarted
        """,
        min_length=1,
        max_length=50,
    )

    # === Heartbeat Metadata ===
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="""
        Extended heartbeat metadata and diagnostics.

        Example (Comprehensive Metrics):
        {
            "version": "1.0.0",
            "uptime_seconds": 3600,
            "memory_usage_mb": 256,
            "cpu_usage_percent": 15.5,
            "disk_usage_percent": 45.0,
            "network_throughput_mbps": 100,

            // Node-specific metrics
            "active_workflows": 42,         // Orchestrator
            "events_processed": 1000,
            "active_aggregations": 5,       // Reducer
            "aggregation_backlog": 50,
            "registered_nodes": 10,         // Registry
            "stale_nodes": 1,

            // Performance metrics
            "avg_request_latency_ms": 10,
            "p95_latency_ms": 25,
            "p99_latency_ms": 50,
            "error_rate": 0.001,

            // Dependencies
            "kafka_connected": true,
            "postgres_connected": true,
            "onextree_available": true,

            // Warnings/Issues
            "warnings": [
                "High memory usage: 480MB/512MB",
                "Growing backlog: 500 items"
            ],
            "degradation_reason": "Database connection pool saturation"
        }
        """,
    )

    # === Temporal Tracking ===
    last_heartbeat: datetime = Field(
        default_factory=datetime.utcnow,
        description="""
        Timestamp of this heartbeat update.

        Automatically set to current UTC time.
        Used for staleness detection and node lifecycle management.

        Staleness Rules:
        - Fresh: last_heartbeat within 60 seconds
        - Stale: last_heartbeat between 60-120 seconds
        - Dead: last_heartbeat > 120 seconds (eligible for cleanup)
        """,
    )
