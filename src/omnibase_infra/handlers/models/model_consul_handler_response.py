# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Consul Handler Response Model.

This module provides the Pydantic model for Consul handler response envelopes
used by the ConsulHandler.

This model provides type consistency with ModelDbQueryResponse, ensuring
both handlers return strongly-typed Pydantic models with consistent
interfaces (status, payload, correlation_id).
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.handlers.models.model_consul_handler_payload import (
    ModelConsulHandlerPayload,
)


class ModelConsulHandlerResponse(BaseModel):
    """Full Consul handler response envelope.

    Provides a standardized response format for Consul operations
    with status, payload, and correlation tracking.

    This model mirrors ModelDbQueryResponse to ensure consistent
    interfaces across infrastructure handlers.

    Attributes:
        status: Operation status ("success" or "error")
        payload: Consul operation result payload containing operation-specific data
        correlation_id: UUID for request/response correlation

    Example:
        >>> from uuid import uuid4
        >>> response = ModelConsulHandlerResponse(
        ...     status="success",
        ...     payload=ModelConsulHandlerPayload(data={"registered": True}),
        ...     correlation_id=uuid4(),
        ... )
        >>> print(response.status)
        'success'
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,  # Support pytest-xdist compatibility
    )

    status: Literal["success", "error"] = Field(
        description="Operation status indicator",
    )
    payload: ModelConsulHandlerPayload = Field(
        description="Consul operation result payload",
    )
    correlation_id: UUID = Field(
        description="UUID for request/response correlation",
    )


__all__: list[str] = ["ModelConsulHandlerResponse"]
