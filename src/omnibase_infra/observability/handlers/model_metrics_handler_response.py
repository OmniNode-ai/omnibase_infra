# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Response model for the Prometheus metrics handler.

This module defines the response schema for HandlerMetricsPrometheus operations,
wrapping the operation payload with status and correlation tracking information.

Usage:
    >>> from omnibase_infra.observability.handlers import (
    ...     ModelMetricsHandlerResponse,
    ...     ModelMetricsHandlerPayload,
    ... )
    >>> from omnibase_infra.enums import EnumResponseStatus
    >>> from uuid import uuid4
    >>>
    >>> payload = ModelMetricsHandlerPayload(
    ...     metrics_text="# HELP http_requests_total Total HTTP requests",
    ...     content_type="text/plain; version=0.0.4; charset=utf-8",
    ...     operation_type="metrics.scrape",
    ... )
    >>> response = ModelMetricsHandlerResponse(
    ...     status=EnumResponseStatus.SUCCESS,
    ...     payload=payload,
    ...     correlation_id=uuid4(),
    ... )
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field

from omnibase_infra.enums import EnumResponseStatus
from omnibase_infra.observability.handlers.model_metrics_handler_payload import (
    ModelMetricsHandlerPayload,
)


class ModelMetricsHandlerResponse(BaseModel):
    """Response model for Prometheus metrics handler operations.

    This is the canonical response type for all HandlerMetricsPrometheus
    operations. It wraps the operation payload with status and correlation
    tracking information.

    Attributes:
        status: Operation status indicating success or failure.
        payload: Operation-specific response data.
        correlation_id: Correlation ID for distributed tracing.
        error_message: Error description if status is ERROR.

    Example:
        >>> from uuid import uuid4
        >>> response = ModelMetricsHandlerResponse(
        ...     status=EnumResponseStatus.SUCCESS,
        ...     payload=ModelMetricsHandlerPayload(
        ...         operation_type="metrics.scrape",
        ...         metrics_text="# HELP up Scrape status",
        ...         content_type="text/plain; version=0.0.4; charset=utf-8",
        ...     ),
        ...     correlation_id=uuid4(),
        ... )
    """

    status: EnumResponseStatus = Field(
        description="Operation status (success or error)",
    )
    payload: ModelMetricsHandlerPayload = Field(
        description="Operation-specific response payload",
    )
    correlation_id: UUID = Field(
        description="Correlation ID for distributed tracing",
    )
    error_message: str | None = Field(
        default=None,
        description="Error description if status is ERROR",
    )

    model_config = {"frozen": True, "extra": "forbid"}


__all__: list[str] = [
    "ModelMetricsHandlerResponse",
]
