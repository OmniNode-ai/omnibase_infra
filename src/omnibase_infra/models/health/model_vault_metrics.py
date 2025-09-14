"""Vault Metrics Model.

Strongly-typed model for Vault secret management health metrics.
Replaces Dict[str, Any] usage to maintain ONEX compliance.
"""

from pydantic import BaseModel, Field
from typing import Optional


class ModelVaultMetrics(BaseModel):
    """Model for Vault secret management health metrics."""

    # Authentication metrics
    active_tokens: int = Field(
        ge=0,
        description="Number of active authentication tokens"
    )

    token_lookups_per_second: float = Field(
        ge=0.0,
        description="Token lookup operations per second"
    )

    authentication_success_rate: float = Field(
        ge=0.0,
        le=100.0,
        description="Authentication success rate percentage"
    )

    token_renewals_per_second: float = Field(
        ge=0.0,
        description="Token renewal operations per second"
    )

    # Secret engine metrics
    secret_engines_mounted: int = Field(
        ge=0,
        description="Number of mounted secret engines"
    )

    secrets_read_per_second: float = Field(
        ge=0.0,
        description="Secret read operations per second"
    )

    secrets_written_per_second: float = Field(
        ge=0.0,
        description="Secret write operations per second"
    )

    kv_operations_per_second: float = Field(
        ge=0.0,
        description="Key-value secret operations per second"
    )

    # Performance metrics
    avg_secret_read_latency_ms: float = Field(
        ge=0.0,
        description="Average secret read latency in milliseconds"
    )

    avg_secret_write_latency_ms: float = Field(
        ge=0.0,
        description="Average secret write latency in milliseconds"
    )

    policy_evaluations_per_second: float = Field(
        ge=0.0,
        description="Policy evaluations per second"
    )

    # Storage metrics
    storage_operations_per_second: float = Field(
        ge=0.0,
        description="Backend storage operations per second"
    )

    storage_size_mb: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Backend storage size in megabytes"
    )

    # HA and clustering metrics
    is_leader: bool = Field(
        description="Whether this Vault node is the cluster leader"
    )

    cluster_nodes: int = Field(
        ge=1,
        description="Number of nodes in Vault cluster"
    )

    unsealed_nodes: int = Field(
        ge=0,
        description="Number of unsealed nodes in cluster"
    )

    # Error and audit metrics
    operation_errors_per_second: float = Field(
        ge=0.0,
        description="Operation errors per second"
    )

    audit_log_failures: int = Field(
        ge=0,
        description="Number of audit log write failures"
    )

    seal_status_checks: int = Field(
        ge=0,
        description="Number of seal status checks performed"
    )

    # Resource utilization
    memory_usage_mb: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Vault process memory usage in megabytes"
    )

    cpu_usage_percent: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Vault process CPU usage percentage"
    )

    # Certificate metrics (if using PKI engine)
    certificates_issued_total: Optional[int] = Field(
        default=None,
        ge=0,
        description="Total number of certificates issued"
    )

    certificates_revoked_total: Optional[int] = Field(
        default=None,
        ge=0,
        description="Total number of certificates revoked"
    )

    # Transit engine metrics (if enabled)
    encryption_operations_per_second: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Encryption operations per second"
    )

    decryption_operations_per_second: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Decryption operations per second"
    )

    # Connection metrics
    client_connections: int = Field(
        ge=0,
        description="Number of active client connections"
    )

    api_requests_per_second: float = Field(
        ge=0.0,
        description="API requests per second"
    )

    api_response_time_ms: float = Field(
        ge=0.0,
        description="Average API response time in milliseconds"
    )