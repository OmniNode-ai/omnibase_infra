# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Consul Handler Payload Model.

This module provides the discriminated union type and wrapper model for Consul handler
response payloads.

Union Type:
    ConsulPayload: Discriminated union of all payload types using operation_type field
"""

from __future__ import annotations

from typing import Annotated, Union

from pydantic import BaseModel, ConfigDict, Discriminator, Field

from omnibase_infra.handlers.models.consul.model_consul_deregister_payload import (
    ModelConsulDeregisterPayload,
)
from omnibase_infra.handlers.models.consul.model_consul_kv_get_found_payload import (
    ModelConsulKVGetFoundPayload,
)
from omnibase_infra.handlers.models.consul.model_consul_kv_get_not_found_payload import (
    ModelConsulKVGetNotFoundPayload,
)
from omnibase_infra.handlers.models.consul.model_consul_kv_get_recurse_payload import (
    ModelConsulKVGetRecursePayload,
)
from omnibase_infra.handlers.models.consul.model_consul_kv_put_payload import (
    ModelConsulKVPutPayload,
)
from omnibase_infra.handlers.models.consul.model_consul_register_payload import (
    ModelConsulRegisterPayload,
)

# Discriminated union type for all Consul payloads
ConsulPayload = Annotated[
    Union[
        ModelConsulKVGetFoundPayload,
        ModelConsulKVGetNotFoundPayload,
        ModelConsulKVGetRecursePayload,
        ModelConsulKVPutPayload,
        ModelConsulRegisterPayload,
        ModelConsulDeregisterPayload,
    ],
    Discriminator("operation_type"),
]


class ModelConsulHandlerPayload(BaseModel):
    """Payload containing Consul operation results.

    This model wraps the strongly-typed ConsulPayload discriminated union.
    The payload type is determined by the operation_type field.

    Attributes:
        data: Strongly-typed operation result with discriminated union.

    Example:
        >>> from omnibase_infra.handlers.models import ModelConsulHandlerPayload
        >>> payload = ModelConsulHandlerPayload(
        ...     data=ModelConsulKVGetFoundPayload(
        ...         found=True,
        ...         key="mykey",
        ...         value="myvalue",
        ...         index=100,
        ...     ),
        ... )
        >>> print(payload.data.operation_type)
        'kv_get_found'
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,  # Support pytest-xdist compatibility
    )

    data: ConsulPayload = Field(
        description="Strongly-typed operation result with discriminated union",
    )


__all__: list[str] = [
    "ConsulPayload",
    "ModelConsulHandlerPayload",
]
