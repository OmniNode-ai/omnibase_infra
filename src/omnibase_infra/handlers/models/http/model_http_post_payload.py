# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""HTTP POST Payload Model.

This module provides the Pydantic model for http.post operation results.
"""

from __future__ import annotations

from typing import Literal

from omnibase_core.types import JsonType
from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.handlers.models.http.enum_http_operation_type import (
    EnumHttpOperationType,
)


class ModelHttpPostPayload(BaseModel):
    """Payload for http.post operation result.

    Contains the HTTP response from a POST request including status code,
    headers, and response body.

    Attributes:
        operation_type: Discriminator set to "post"
        status_code: HTTP response status code
        headers: Response headers as key-value dictionary
        body: Response body (parsed as JSON if content-type is application/json)

    Example:
        >>> payload = ModelHttpPostPayload(
        ...     status_code=201,
        ...     headers={"content-type": "application/json", "location": "/items/1"},
        ...     body={"id": 1, "created": True},
        ... )
        >>> print(payload.status_code)
        201
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    operation_type: Literal[EnumHttpOperationType.POST] = Field(
        default=EnumHttpOperationType.POST,
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
    body: JsonType = Field(
        description="Response body",
    )


__all__: list[str] = ["ModelHttpPostPayload"]
