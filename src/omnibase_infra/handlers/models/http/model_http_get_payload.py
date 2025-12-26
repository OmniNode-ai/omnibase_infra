# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""HTTP GET Payload Model.

This module provides the Pydantic model for http.get operation results.
"""

from __future__ import annotations

from typing import Literal

from omnibase_core.types import JsonValue
from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.handlers.models.http.enum_http_operation_type import (
    EnumHttpOperationType,
)


class ModelHttpGetPayload(BaseModel):
    """Payload for http.get operation result.

    Contains the HTTP response from a GET request including status code,
    headers, and response body.

    Attributes:
        operation_type: Discriminator set to "get"
        status_code: HTTP response status code
        headers: Response headers as key-value dictionary
        body: Response body (parsed as JSON if content-type is application/json)

    Example:
        >>> payload = ModelHttpGetPayload(
        ...     status_code=200,
        ...     headers={"content-type": "application/json"},
        ...     body={"message": "success"},
        ... )
        >>> print(payload.status_code)
        200
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    operation_type: Literal[EnumHttpOperationType.GET] = Field(
        default=EnumHttpOperationType.GET,
        description="Operation type discriminator",
    )
    status_code: int = Field(
        ge=100,
        le=599,
        description="HTTP response status code",
    )
    headers: dict[str, str] = Field(
        default_factory=dict,
        description="Response headers as key-value dictionary",
    )
    body: JsonValue = Field(
        description="Response body",
    )


__all__: list[str] = ["ModelHttpGetPayload"]
