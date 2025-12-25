# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Vault Handler Response Model.

This module provides the Pydantic model for Vault handler response envelopes
used by the VaultHandler.

This model provides type consistency with ModelDbQueryResponse and
ModelConsulHandlerResponse, ensuring all handlers return strongly-typed
Pydantic models with consistent interfaces (status, payload, correlation_id).
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.handlers.models.vault import (
    ModelVaultHandlerPayload,
)


class ModelVaultHandlerResponse(BaseModel):
    """Full Vault handler response envelope.

    Provides a standardized response format for Vault operations
    with status, payload, and correlation tracking.

    This model mirrors ModelDbQueryResponse and ModelConsulHandlerResponse
    to ensure consistent interfaces across infrastructure handlers.

    Attributes:
        status: Operation status ("success" or "error")
        payload: Vault operation result payload containing operation-specific data
        correlation_id: UUID for request/response correlation

    Example:
        >>> from uuid import uuid4
        >>> from omnibase_infra.handlers.models.vault import (
        ...     ModelVaultSecretPayload,
        ...     ModelVaultHandlerPayload,
        ... )
        >>> response = ModelVaultHandlerResponse(
        ...     status="success",
        ...     payload=ModelVaultHandlerPayload(
        ...         data=ModelVaultSecretPayload(
        ...             data={"username": "admin"},
        ...             metadata={"version": 1},
        ...         ),
        ...     ),
        ...     correlation_id=uuid4(),
        ... )
        >>> print(response.status)
        'success'
        >>> print(response.payload.data.operation_type)
        <EnumVaultOperationType.READ_SECRET: 'read_secret'>
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
    payload: ModelVaultHandlerPayload = Field(
        description="Vault operation result payload",
    )
    correlation_id: UUID = Field(
        description="UUID for request/response correlation",
    )

    @property
    def is_success(self) -> bool:
        """Check if the response indicates a successful operation.

        Returns:
            True if status is "success", False otherwise.
        """
        return self.status == "success"

    @property
    def is_error(self) -> bool:
        """Check if the response indicates an error.

        Returns:
            True if status is "error", False otherwise.
        """
        return self.status == "error"


__all__: list[str] = ["ModelVaultHandlerResponse"]
