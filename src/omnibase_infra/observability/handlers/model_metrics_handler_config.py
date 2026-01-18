# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Configuration model for the Prometheus metrics handler.

This module defines the configuration schema for HandlerMetricsPrometheus,
including HTTP server settings and optional push gateway configuration.

Security Notes:
    - DEFAULT BIND ADDRESS IS LOCALHOST (127.0.0.1) for security
    - To expose externally, explicitly set host="0.0.0.0" AND deploy behind
      a reverse proxy with authentication, TLS, and rate limiting
    - See HandlerMetricsPrometheus docstring for full security model

Configuration Options:
    - host: Bind address for the metrics HTTP server (default: "127.0.0.1")
    - port: Port number for the metrics endpoint (default: 9090)
    - path: URL path for the metrics endpoint (default: "/metrics")
    - push_gateway_url: Optional URL for Prometheus Pushgateway (for short-lived jobs)
    - enable_server: Whether to start the HTTP server (default: True)
    - max_request_size_bytes: Maximum allowed request body size (default: 1MB)
    - request_timeout_seconds: Timeout for request processing (default: 30.0s)

Usage:
    >>> from omnibase_infra.observability.handlers import ModelMetricsHandlerConfig
    >>>
    >>> # Default configuration (localhost only - secure)
    >>> config = ModelMetricsHandlerConfig()
    >>> assert config.host == "127.0.0.1"  # Secure default
    >>>
    >>> # Expose on all interfaces (REQUIRES reverse proxy protection)
    >>> config = ModelMetricsHandlerConfig(
    ...     host="0.0.0.0",  # WARNING: Only use with reverse proxy
    ...     port=9091,
    ...     path="/custom_metrics",
    ... )
    >>>
    >>> # Push mode configuration (for short-lived jobs)
    >>> config = ModelMetricsHandlerConfig(
    ...     enable_server=False,
    ...     push_gateway_url="http://pushgateway:9091",
    ... )
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelMetricsHandlerConfig(BaseModel):
    """Configuration model for Prometheus metrics handler.

    This model defines the configuration for the HTTP metrics endpoint
    and optional push gateway integration.

    Security:
        The default bind address is "127.0.0.1" (localhost only) for security.
        This prevents accidental exposure of metrics to untrusted networks.

        To expose metrics externally, you MUST:
        1. Explicitly set host="0.0.0.0" in configuration
        2. Deploy behind a reverse proxy (nginx/traefik/envoy) with:
           - TLS termination (HTTPS)
           - IP allowlisting for Prometheus scrapers
           - Rate limiting
           - Authentication (mTLS or bearer tokens recommended)

    Attributes:
        host: Bind address for the HTTP server. Default: "127.0.0.1" (localhost).
            Use "0.0.0.0" ONLY with reverse proxy protection.
        port: TCP port number for the metrics endpoint. Standard Prometheus
            exporters typically use ports in the 9xxx range. Default: 9090.
        path: URL path where metrics are exposed. Must start with "/".
            Default: "/metrics".
        push_gateway_url: Optional URL for Prometheus Pushgateway. When set,
            metrics can be pushed instead of scraped. Useful for short-lived
            batch jobs that may not live long enough to be scraped.
            Format: "http://host:port" or "https://host:port".
        enable_server: Whether to start the HTTP server for metric scraping.
            Set to False when using push mode only. Default: True.
        job_name: Job name for Pushgateway metrics. Only used when pushing
            to Pushgateway. Default: "onex_metrics".
        push_interval_seconds: Interval between metric pushes to Pushgateway.
            Only used when push_gateway_url is set. Default: 10.0.
        shutdown_timeout_seconds: Maximum time to wait for graceful server
            shutdown. Default: 5.0.
        max_request_size_bytes: Maximum allowed request body size in bytes.
            Requests exceeding this limit are rejected with 413 status.
            Default: 1MB (1048576 bytes).
        request_timeout_seconds: Timeout for processing individual requests.
            Requests exceeding this timeout are terminated with 503 status.
            Default: 30.0 seconds.

    Example:
        >>> # Secure default (localhost only)
        >>> config = ModelMetricsHandlerConfig()
        >>> assert config.host == "127.0.0.1"
        >>> assert config.max_request_size_bytes == 1048576
        >>>
        >>> # External exposure (REQUIRES reverse proxy)
        >>> config = ModelMetricsHandlerConfig(
        ...     host="0.0.0.0",  # WARNING: Use only with reverse proxy
        ...     port=9090,
        ...     path="/metrics",
        ... )
    """

    host: str = Field(
        default="127.0.0.1",
        description=(
            "Bind address for the HTTP metrics server. "
            "Default is localhost (127.0.0.1) for security. "
            "Use 0.0.0.0 only with reverse proxy protection."
        ),
    )
    port: int = Field(
        default=9090,
        ge=1,
        le=65535,
        description="Port number for the metrics endpoint",
    )
    path: str = Field(
        default="/metrics",
        pattern=r"^/.*",
        description="URL path for the metrics endpoint",
    )
    push_gateway_url: str | None = Field(
        default=None,
        description="Optional URL for Prometheus Pushgateway",
    )
    enable_server: bool = Field(
        default=True,
        description="Whether to start the HTTP server for metric scraping",
    )
    job_name: str = Field(
        default="onex_metrics",
        description="Job name for Pushgateway metrics",
    )
    push_interval_seconds: float = Field(
        default=10.0,
        gt=0.0,
        description="Interval between metric pushes to Pushgateway",
    )
    shutdown_timeout_seconds: float = Field(
        default=5.0,
        gt=0.0,
        description="Maximum time to wait for graceful server shutdown",
    )
    max_request_size_bytes: int = Field(
        default=1048576,  # 1MB
        ge=1024,  # Minimum 1KB
        le=104857600,  # Maximum 100MB
        description=(
            "Maximum allowed request body size in bytes. "
            "Requests exceeding this limit are rejected with 413 status. "
            "Default: 1MB (1048576 bytes)."
        ),
    )
    request_timeout_seconds: float = Field(
        default=30.0,
        gt=0.0,
        le=300.0,  # Maximum 5 minutes
        description=(
            "Timeout for processing individual HTTP requests. "
            "Requests exceeding this timeout are terminated with 503 status. "
            "Default: 30.0 seconds."
        ),
    )

    model_config = ConfigDict(frozen=True, extra="forbid")


__all__: list[str] = ["ModelMetricsHandlerConfig"]
