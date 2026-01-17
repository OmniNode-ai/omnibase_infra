# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""HTTP GET Payload Model.

This module provides the Pydantic model for http.get operation results.
Uses the registry pattern for dynamic type resolution.

Related:
    - RegistryPayloadHttp: Payload type registration and lookup
    - OMN-1007: Union reduction refactoring
"""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field

from omnibase_core.types import JsonType
from omnibase_infra.handlers.models.http.enum_http_operation_type import (
    EnumHttpOperationType,
)
from omnibase_infra.handlers.models.http.model_payload_http import (
    ModelPayloadHttp,
)
from omnibase_infra.handlers.models.http.registry_payload_http import (
    RegistryPayloadHttp,
)


@RegistryPayloadHttp.register("get")
class ModelHttpGetPayload(ModelPayloadHttp):
    """Payload for http.get operation result.

    This model is registered with RegistryPayloadHttp for dynamic type resolution.
    Use @RegistryPayloadHttp.register("get") decorator pattern.

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

        # Dynamic type lookup:
        >>> from omnibase_infra.handlers.models.http import RegistryPayloadHttp
        >>> payload_cls = RegistryPayloadHttp.get_type("get")
        >>> payload_cls.__name__
        'ModelHttpGetPayload'
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
    body: JsonType = Field(
        description="Response body",
    )


__all__: list[str] = ["ModelHttpGetPayload"]
