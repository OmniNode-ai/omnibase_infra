"""Tracing Configuration Model.

Shared model for distributed tracing configuration settings.
Used across tracing infrastructure for consistent setup.
"""

from pydantic import BaseModel, Field


class ModelTracingConfig(BaseModel):
    """Model for distributed tracing configuration."""

    environment: str = Field(
        description="Target environment for tracing configuration",
    )

    service_name: str = Field(
        default="omnibase_infrastructure",
        description="Service name for tracing identification",
    )

    service_version: str = Field(
        default="1.0.0",
        description="Service version for tracing identification",
    )

    otlp_endpoint: str = Field(
        default="http://localhost:4317",
        description="OpenTelemetry Protocol (OTLP) endpoint",
    )

    otlp_headers: dict[str, str] | None = Field(
        default=None,
        description="OTLP headers for authentication and configuration",
    )

    trace_sample_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Trace sampling rate (0.0 to 1.0)",
    )

    enable_db_instrumentation: bool = Field(
        default=True,
        description="Enable database instrumentation",
    )

    enable_kafka_instrumentation: bool = Field(
        default=True,
        description="Enable Kafka instrumentation",
    )

    enable_audit_integration: bool = Field(
        default=True,
        description="Enable integration with audit logging",
    )

    batch_timeout_ms: int = Field(
        default=5000,
        gt=0,
        description="Batch span processor timeout in milliseconds",
    )

    max_export_batch_size: int = Field(
        default=512,
        gt=0,
        description="Maximum batch size for span export",
    )

    max_queue_size: int = Field(
        default=2048,
        gt=0,
        description="Maximum queue size for spans",
    )
