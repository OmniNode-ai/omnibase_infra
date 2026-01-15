# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""HTTP Handler Response Model.

This module provides the Pydantic model for HTTP handler response envelopes
used by the HttpRestHandler.

This model provides type consistency with ModelDbQueryResponse,
ModelConsulHandlerResponse, and ModelVaultHandlerResponse, ensuring
all handlers return strongly-typed Pydantic models with consistent
interfaces (status, payload, correlation_id).
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import EnumResponseStatus
from omnibase_infra.handlers.models.http import (
    ModelHttpHandlerPayload,
)


class ModelHttpHandlerResponse(BaseModel):
    """Full HTTP handler response envelope.

    Provides a standardized response format for HTTP operations
    with status, payload, and correlation tracking.

    This model mirrors ModelDbQueryResponse, ModelConsulHandlerResponse,
    and ModelVaultHandlerResponse to ensure consistent interfaces across
    infrastructure handlers.

    Attributes:
        status: Operation status (EnumResponseStatus.SUCCESS or EnumResponseStatus.ERROR)
        payload: HTTP operation result payload containing operation-specific data
        correlation_id: UUID for request/response correlation

    Example:
        >>> from uuid import uuid4
        >>> from omnibase_infra.enums import EnumResponseStatus
        >>> from omnibase_infra.handlers.models.http import (
        ...     ModelHttpGetPayload,
        ...     ModelHttpHandlerPayload,
        ... )
        >>> response = ModelHttpHandlerResponse(
        ...     status=EnumResponseStatus.SUCCESS,
        ...     payload=ModelHttpHandlerPayload(
        ...         data=ModelHttpGetPayload(
        ...             status_code=200,
        ...             headers={"content-type": "application/json"},
        ...             body={"message": "success"},
        ...         ),
        ...     ),
        ...     correlation_id=uuid4(),
        ... )
        >>> print(response.status)
        <EnumResponseStatus.SUCCESS: 'success'>
        >>> print(response.payload.data.status_code)
        200
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
    payload: ModelHttpHandlerPayload = Field(
        description="HTTP operation result payload",
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


__all__: list[str] = ["ModelHttpHandlerResponse"]
