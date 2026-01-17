# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Payload model for the Prometheus metrics handler.

This module defines the payload schema for HandlerMetricsPrometheus operations,
containing operation-specific data for metrics scrape and push operations.

Usage:
    >>> from omnibase_infra.observability.handlers import ModelMetricsHandlerPayload
    >>>
    >>> # Scrape response payload
    >>> scrape_payload = ModelMetricsHandlerPayload(
    ...     operation_type="metrics.scrape",
    ...     metrics_text="# HELP up Scrape status\\nup 1",
    ...     content_type="text/plain; version=0.0.4; charset=utf-8",
    ... )
    >>>
    >>> # Push response payload
    >>> push_payload = ModelMetricsHandlerPayload(
    ...     operation_type="metrics.push",
    ...     pushed_at="2025-01-17T12:00:00Z",
    ...     push_gateway_url="http://pushgateway:9091",
    ...     job_name="onex_metrics",
    ... )
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ModelMetricsHandlerPayload(BaseModel):
    """Payload model for metrics handler operations.

    Contains the operation-specific data returned by the metrics handler.
    The structure varies based on operation_type.

    Attributes:
        operation_type: Type of operation performed. Either "metrics.scrape"
            for metric retrieval or "metrics.push" for Pushgateway operations.
        metrics_text: Prometheus metrics in text format. Only populated for
            scrape operations. Contains the raw metric exposition format.
        content_type: MIME content type for the metrics response. Standard
            Prometheus format is "text/plain; version=0.0.4; charset=utf-8".
        pushed_at: ISO timestamp when metrics were pushed to Pushgateway.
            Only populated for push operations.
        push_gateway_url: URL of the Pushgateway that received the metrics.
            Only populated for push operations.
        job_name: Job name used when pushing to Pushgateway.
            Only populated for push operations.

    Example:
        >>> # Scrape response payload
        >>> scrape_payload = ModelMetricsHandlerPayload(
        ...     operation_type="metrics.scrape",
        ...     metrics_text="# HELP up Scrape status\\nup 1",
        ...     content_type="text/plain; version=0.0.4; charset=utf-8",
        ... )
        >>>
        >>> # Push response payload
        >>> push_payload = ModelMetricsHandlerPayload(
        ...     operation_type="metrics.push",
        ...     pushed_at="2025-01-17T12:00:00Z",
        ...     push_gateway_url="http://pushgateway:9091",
        ...     job_name="onex_metrics",
        ... )
    """

    operation_type: Literal["metrics.scrape", "metrics.push"] = Field(
        description="Type of metrics operation performed",
    )
    metrics_text: str | None = Field(
        default=None,
        description="Prometheus metrics in text exposition format",
    )
    content_type: str | None = Field(
        default=None,
        description="MIME content type for the metrics response",
    )
    pushed_at: str | None = Field(
        default=None,
        description="ISO timestamp when metrics were pushed",
    )
    push_gateway_url: str | None = Field(
        default=None,
        description="URL of the Pushgateway that received metrics",
    )
    job_name: str | None = Field(
        default=None,
        description="Job name used for Pushgateway push",
    )

    model_config = {"frozen": True, "extra": "forbid"}


__all__: list[str] = [
    "ModelMetricsHandlerPayload",
]
