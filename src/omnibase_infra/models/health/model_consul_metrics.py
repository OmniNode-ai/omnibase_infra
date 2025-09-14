"""Consul Metrics Model.

Strongly-typed model for Consul service discovery health metrics.
Replaces Dict[str, Any] usage to maintain ONEX compliance.
"""

from pydantic import BaseModel, Field
from typing import Optional


class ModelConsulMetrics(BaseModel):
    """Model for Consul service discovery health metrics."""

    # Service registry metrics
    registered_services: int = Field(
        ge=0,
        description="Number of registered services"
    )

    healthy_services: int = Field(
        ge=0,
        description="Number of services reporting healthy status"
    )

    unhealthy_services: int = Field(
        ge=0,
        description="Number of services reporting unhealthy status"
    )

    service_health_check_success_rate: float = Field(
        ge=0.0,
        le=100.0,
        description="Service health check success rate percentage"
    )

    # Key-Value store metrics
    kv_operations_per_second: float = Field(
        ge=0.0,
        description="Key-value operations per second"
    )

    kv_read_latency_ms: float = Field(
        ge=0.0,
        description="Average key-value read latency in milliseconds"
    )

    kv_write_latency_ms: float = Field(
        ge=0.0,
        description="Average key-value write latency in milliseconds"
    )

    kv_store_size_mb: float = Field(
        ge=0.0,
        description="Key-value store size in megabytes"
    )

    # Cluster metrics
    cluster_nodes: int = Field(
        ge=1,
        description="Number of nodes in Consul cluster"
    )

    leader_elected: bool = Field(
        description="Whether cluster has an elected leader"
    )

    raft_commits_per_second: float = Field(
        ge=0.0,
        description="Raft log commits per second"
    )

    raft_log_size_mb: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Raft log size in megabytes"
    )

    # Connection metrics
    client_connections: int = Field(
        ge=0,
        description="Number of active client connections"
    )

    api_request_rate: float = Field(
        ge=0.0,
        description="API requests per second"
    )

    api_error_rate: float = Field(
        ge=0.0,
        le=100.0,
        description="API error rate percentage"
    )

    # Performance metrics
    dns_queries_per_second: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="DNS queries handled per second"
    )

    catalog_operations_per_second: float = Field(
        ge=0.0,
        description="Service catalog operations per second"
    )

    # Resource utilization
    memory_usage_mb: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Consul agent memory usage in megabytes"
    )

    cpu_usage_percent: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Consul agent CPU usage percentage"
    )

    # Health check metrics
    health_checks_total: int = Field(
        ge=0,
        description="Total number of health checks registered"
    )

    health_checks_passing: int = Field(
        ge=0,
        description="Number of health checks currently passing"
    )

    health_checks_failing: int = Field(
        ge=0,
        description="Number of health checks currently failing"
    )

    avg_health_check_duration_ms: float = Field(
        ge=0.0,
        description="Average health check execution time in milliseconds"
    )

    # Network metrics
    gossip_messages_per_second: float = Field(
        ge=0.0,
        description="Gossip protocol messages per second"
    )

    network_latency_ms: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Average network latency between nodes in milliseconds"
    )