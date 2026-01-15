# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Database Query Response Model.

This module provides the Pydantic model for database query response envelopes
used by the HandlerDb.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import EnumResponseStatus
from omnibase_infra.handlers.models.model_db_query_payload import ModelDbQueryPayload


class ModelDbQueryResponse(BaseModel):
    """Full database query response envelope.

    Provides a standardized response format for database operations
    with status, payload, and correlation tracking.

    Attributes:
        status: Operation status (EnumResponseStatus.SUCCESS or EnumResponseStatus.ERROR)
        payload: Query result payload containing rows and count
        correlation_id: UUID for request/response correlation

    Example:
        >>> from uuid import uuid4
        >>> from omnibase_infra.enums import EnumResponseStatus
        >>> response = ModelDbQueryResponse(
        ...     status=EnumResponseStatus.SUCCESS,
        ...     payload=ModelDbQueryPayload(rows=[], row_count=0),
        ...     correlation_id=uuid4(),
        ... )
        >>> print(response.status)
        <EnumResponseStatus.SUCCESS: 'success'>
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
    payload: ModelDbQueryPayload = Field(
        description="Query result payload",
    )
    correlation_id: UUID = Field(
        description="UUID for request/response correlation",
    )


__all__: list[str] = ["ModelDbQueryResponse"]
