"""Configuration management for Distributed Tracing Compute Node.

Provides ONEX-compliant configuration injection patterns for OpenTelemetry tracing setup.
Validates external dependencies and enforces secure configuration practices.
"""

import os

from omnibase_core.core.errors.onex_error import CoreErrorCode, OnexError
from pydantic import BaseModel, Field, HttpUrl, validator


class TracingConfig(BaseModel):
    """
    Configuration model for distributed tracing with OpenTelemetry.

    Provides secure endpoint validation and environment-specific configuration
    following ONEX dependency injection patterns.
    """

    otel_exporter_otlp_endpoint: HttpUrl = Field(
        default="http://localhost:4317",
        description="The OTLP endpoint for exporting traces. Must be a valid HTTP/HTTPS URL.",
    )

    trace_sample_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Trace sampling rate between 0.0 (no traces) and 1.0 (all traces).",
    )

    service_name: str = Field(
        default="omnibase_infrastructure",
        min_length=1,
        description="OpenTelemetry service name for trace identification.",
    )

    service_version: str = Field(
        default="1.0.0",
        min_length=1,
        description="Service version for trace metadata.",
    )

    environment: str = Field(
        default="development",
        min_length=1,
        description="Deployment environment (development, staging, production).",
    )

    @validator("otel_exporter_otlp_endpoint")
    def validate_otlp_endpoint(cls, v):
        """Validate OTLP endpoint URL for security and format compliance."""
        if not v:
            raise ValueError("OTLP endpoint URL cannot be empty")

        # Ensure only HTTP/HTTPS schemes are allowed
        allowed_schemes = {"http", "https"}
        if v.scheme not in allowed_schemes:
            raise ValueError(f"OTLP endpoint must use HTTP or HTTPS scheme, got: {v.scheme}")

        # Validate network location is present
        if not v.host:
            raise ValueError("OTLP endpoint must include a valid hostname")

        return v

    @validator("environment")
    def validate_environment(cls, v):
        """Validate environment name against known deployment environments."""
        allowed_environments = {"development", "dev", "staging", "stage", "production", "prod", "test", "testing"}
        if v.lower() not in allowed_environments:
            # Log warning but don't fail - allow custom environments
            pass
        return v.lower()


def load_tracing_config() -> TracingConfig:
    """
    Load and validate tracing configuration from environment variables.

    Follows ONEX configuration injection pattern by centralizing environment
    variable access and validation at application startup.

    Returns:
        TracingConfig: Validated configuration object

    Raises:
        OnexError: If configuration validation fails
    """
    try:
        config_data = {}

        # Load OTLP endpoint with validation
        if endpoint := os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
            config_data["otel_exporter_otlp_endpoint"] = endpoint

        # Load trace sample rate with validation
        if sample_rate := os.getenv("OTEL_TRACE_SAMPLE_RATE"):
            try:
                config_data["trace_sample_rate"] = float(sample_rate)
            except ValueError:
                raise ValueError(f"Invalid trace sample rate: {sample_rate}. Must be a float between 0.0 and 1.0")

        # Load service information
        if service_name := os.getenv("OTEL_SERVICE_NAME"):
            config_data["service_name"] = service_name

        if service_version := os.getenv("OTEL_SERVICE_VERSION"):
            config_data["service_version"] = service_version

        # Detect environment from multiple possible variables
        env_vars = ["ENVIRONMENT", "ENV", "DEPLOYMENT_ENV", "NODE_ENV", "OMNIBASE_ENV"]
        for var in env_vars:
            if env_value := os.getenv(var):
                config_data["environment"] = env_value.lower()
                break

        # Create and validate configuration
        return TracingConfig(**config_data)

    except Exception as e:
        raise OnexError(
            message=f"Failed to load tracing configuration: {e!s}",
            error_code=CoreErrorCode.CONFIGURATION_ERROR,
        ) from e


def create_test_tracing_config(
    endpoint: str | None = None,
    sample_rate: float | None = None,
    environment: str | None = None,
) -> TracingConfig:
    """
    Create a test configuration for unit testing.

    Args:
        endpoint: Override OTLP endpoint
        sample_rate: Override trace sample rate
        environment: Override environment

    Returns:
        TracingConfig: Test configuration object
    """
    config_data = {}

    if endpoint:
        config_data["otel_exporter_otlp_endpoint"] = endpoint
    if sample_rate is not None:
        config_data["trace_sample_rate"] = sample_rate
    if environment:
        config_data["environment"] = environment

    return TracingConfig(**config_data)
