# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Consul Handler Response Model.

This module provides the Pydantic model for Consul handler response envelopes
used by the HandlerConsul.

This model provides type consistency with ModelDbQueryResponse, ensuring
both handlers return strongly-typed Pydantic models with consistent
interfaces (status, payload, correlation_id).
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import EnumResponseStatus
from omnibase_infra.handlers.models.consul import (
    ModelConsulHandlerPayload,
)


class ModelConsulHandlerResponse(BaseModel):
    """Full Consul handler response envelope.

    Provides a standardized response format for Consul operations
    with status, payload, and correlation tracking.

    This model mirrors ModelDbQueryResponse to ensure consistent
    interfaces across infrastructure handlers.

    Attributes:
        status: Operation status (EnumResponseStatus.SUCCESS or EnumResponseStatus.ERROR)
        payload: Consul operation result payload containing operation-specific data
        correlation_id: UUID for request/response correlation

    Example:
        >>> from uuid import uuid4
        >>> from omnibase_infra.enums import EnumResponseStatus
        >>> from omnibase_infra.handlers.models.consul import ModelConsulRegisterPayload
        >>> response = ModelConsulHandlerResponse(
        ...     status=EnumResponseStatus.SUCCESS,
        ...     payload=ModelConsulHandlerPayload(
        ...         data=ModelConsulRegisterPayload(
        ...             registered=True,
        ...             name="my-service",
        ...             consul_service_id="my-service-1",
        ...         ),
        ...     ),
        ...     correlation_id=uuid4(),
        ... )
        >>> print(response.status)
        <EnumResponseStatus.SUCCESS: 'success'>
        >>> print(response.payload.data.operation_type)
        'register'
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,  # Support pytest-xdist compatibility
    )

    status: EnumResponseStatus = Field(
        description="Operation status indicator",
    )
    payload: ModelConsulHandlerPayload = Field(
        description="Consul operation result payload",
    )
    correlation_id: UUID = Field(
        description="UUID for request/response correlation",
    )

    @property
    def is_success(self) -> bool:
        """Check if the response indicates a successful operation.

        Returns:
            True if status is SUCCESS, False otherwise.
        """
        return self.status == EnumResponseStatus.SUCCESS

    @property
    def is_error(self) -> bool:
        """Check if the response indicates an error.

        Returns:
            True if status is ERROR, False otherwise.
        """
        return self.status == EnumResponseStatus.ERROR


__all__: list[str] = ["ModelConsulHandlerResponse"]
