# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Configuration model for the Prometheus metrics handler.

This module defines the configuration schema for HandlerMetricsPrometheus,
including HTTP server settings and optional push gateway configuration.

Configuration Options:
    - host: Bind address for the metrics HTTP server (default: "0.0.0.0")
    - port: Port number for the metrics endpoint (default: 9090)
    - path: URL path for the metrics endpoint (default: "/metrics")
    - push_gateway_url: Optional URL for Prometheus Pushgateway (for short-lived jobs)
    - enable_server: Whether to start the HTTP server (default: True)

Usage:
    >>> from omnibase_infra.observability.handlers import ModelMetricsHandlerConfig
    >>>
    >>> # Default configuration
    >>> config = ModelMetricsHandlerConfig()
    >>>
    >>> # Custom configuration
    >>> config = ModelMetricsHandlerConfig(
    ...     host="127.0.0.1",
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

    Attributes:
        host: Bind address for the HTTP server. Use "0.0.0.0" to accept
            connections from any network interface, or "127.0.0.1" for
            localhost only. Default: "0.0.0.0".
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

    Example:
        >>> config = ModelMetricsHandlerConfig(
        ...     host="0.0.0.0",
        ...     port=9090,
        ...     path="/metrics",
        ...     enable_server=True,
        ... )
        >>> assert config.host == "0.0.0.0"
        >>> assert config.port == 9090
    """

    host: str = Field(
        default="0.0.0.0",  # noqa: S104 - Binding to all interfaces is intentional
        description="Bind address for the HTTP metrics server",
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

    model_config = ConfigDict(frozen=True, extra="forbid")


__all__: list[str] = ["ModelMetricsHandlerConfig"]
