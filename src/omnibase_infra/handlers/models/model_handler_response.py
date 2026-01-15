# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Generic Handler Response Model.

This module provides a generic handler response model that can be parameterized
with different payload types, enabling consistent response patterns across all
infrastructure handlers.

Usage Pattern:
    Each handler defines its own payload model and uses ModelHandlerResponse[PayloadType]:

    - HandlerConsul: ModelHandlerResponse[ModelConsulHandlerPayload]
    - HandlerDb: ModelHandlerResponse[ModelDbQueryPayload]
    - HandlerVault: ModelHandlerResponse[ModelVaultHandlerPayload]
    - HttpRestHandler: ModelHandlerResponse[ModelHttpHandlerPayload]

This replaces raw dict[str, JsonType] responses with strongly-typed models,
ensuring consistent access patterns (response.status, response.payload, etc.)
across all handlers.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import EnumResponseStatus


class ModelHandlerResponse[PayloadT: BaseModel](BaseModel):
    """Generic handler response envelope.

    Provides a standardized response format for all handler operations
    with status, payload, and correlation tracking.

    This generic model ensures consistent interfaces across all infrastructure
    handlers. Each handler specializes this with its own payload type.

    Type Parameters:
        PayloadT: The payload model type specific to the handler.

    Attributes:
        status: Operation status ("success" or "error")
        payload: Handler-specific result payload
        correlation_id: UUID for request/response correlation

    Example:
        >>> from uuid import uuid4
        >>> from pydantic import BaseModel
        >>> from omnibase_infra.enums import EnumResponseStatus
        >>>
        >>> class MyPayload(BaseModel):
        ...     data: str
        ...
        >>> response = ModelHandlerResponse[MyPayload](
        ...     status=EnumResponseStatus.SUCCESS,
        ...     payload=MyPayload(data="test"),
        ...     correlation_id=uuid4(),
        ... )
        >>> print(response.status)
        <EnumResponseStatus.SUCCESS: 'success'>
        >>> print(response.payload.data)
        'test'
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
    payload: PayloadT = Field(
        description="Handler-specific result payload",
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


__all__: list[str] = ["ModelHandlerResponse"]
