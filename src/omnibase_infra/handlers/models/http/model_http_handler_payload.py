# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""HTTP Handler Payload Model.

This module provides the discriminated union type for HTTP operation payloads
and the wrapper model that contains them.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag

from omnibase_infra.handlers.models.http.model_http_get_payload import (
    ModelHttpGetPayload,
)
from omnibase_infra.handlers.models.http.model_http_post_payload import (
    ModelHttpPostPayload,
)


def _http_payload_discriminator(value: object) -> str:
    """Discriminator function for HttpPayload union.

    Determines which payload type to use based on the operation_type field.

    Args:
        value: The raw value being validated (dict or model instance)

    Returns:
        The discriminator tag value matching the operation_type
    """
    if isinstance(value, dict):
        operation_type = value.get("operation_type", "get")
    else:
        operation_type = getattr(value, "operation_type", "get")

    # Handle both enum values and string values
    if hasattr(operation_type, "value"):
        return str(operation_type.value)
    return str(operation_type)


# Discriminated union of all HTTP payload types
HttpPayload = Annotated[
    Annotated[ModelHttpGetPayload, Tag("get")]
    | Annotated[ModelHttpPostPayload, Tag("post")],
    Discriminator(_http_payload_discriminator),
]


class ModelHttpHandlerPayload(BaseModel):
    """Wrapper model for HTTP handler payloads.

    Contains the discriminated union of all possible HTTP operation payloads.
    This allows the HttpRestHandler to return a consistent payload wrapper while
    the actual payload type varies based on the operation.

    Attributes:
        data: The operation-specific payload (discriminated union)

    Example:
        >>> from omnibase_infra.handlers.models.http import ModelHttpGetPayload
        >>> get_payload = ModelHttpGetPayload(
        ...     status_code=200,
        ...     headers={"content-type": "application/json"},
        ...     body={"message": "success"},
        ... )
        >>> wrapper = ModelHttpHandlerPayload(data=get_payload)
        >>> print(wrapper.data.operation_type)
        <EnumHttpOperationType.GET: 'get'>
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    data: HttpPayload = Field(
        description="Operation-specific payload",
    )


__all__: list[str] = [
    "HttpPayload",
    "ModelHttpHandlerPayload",
]
