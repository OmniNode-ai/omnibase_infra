# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Graph Handler Response Model.

This module provides the Pydantic model for Graph handler response envelopes
used by the HandlerGraph.

This model provides type consistency with ModelDbQueryResponse and
ModelConsulHandlerResponse, ensuring all handlers return strongly-typed
Pydantic models with consistent interfaces (status, payload, correlation_id).
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import EnumResponseStatus
from omnibase_infra.handlers.models.graph import (
    ModelGraphHandlerPayload,
)


class ModelGraphHandlerResponse(BaseModel):
    """Full Graph handler response envelope.

    Provides a standardized response format for Graph database (Memgraph/Neo4j)
    operations with status, payload, and correlation tracking.

    This model mirrors ModelConsulHandlerResponse and ModelDbQueryResponse
    to ensure consistent interfaces across infrastructure handlers.

    Attributes:
        status: Operation status (EnumResponseStatus.SUCCESS or EnumResponseStatus.ERROR)
        payload: Graph operation result payload containing operation-specific data
        correlation_id: UUID for request/response correlation

    Example:
        >>> from uuid import uuid4
        >>> from omnibase_infra.enums import EnumResponseStatus
        >>> from omnibase_infra.handlers.models.graph import (
        ...     ModelGraphHandlerPayload, ModelGraphQueryPayload,
        ... )
        >>> response = ModelGraphHandlerResponse(
        ...     status=EnumResponseStatus.SUCCESS,
        ...     payload=ModelGraphHandlerPayload(
        ...         data=ModelGraphQueryPayload(
        ...             cypher="MATCH (n) RETURN n",
        ...             records=[],
        ...             summary={"nodes_created": 0},
        ...         ),
        ...     ),
        ...     correlation_id=uuid4(),
        ... )
        >>> print(response.status)
        <EnumResponseStatus.SUCCESS: 'success'>
        >>> print(response.payload.data.operation_type)
        'graph.query'
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
    payload: ModelGraphHandlerPayload = Field(
        description="Graph operation result payload",
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


__all__: list[str] = ["ModelGraphHandlerResponse"]
