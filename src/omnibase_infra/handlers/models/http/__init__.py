# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""HTTP Handler Models Module.

This module exports Pydantic models for HTTP handler request/response structures.
All models are strongly typed to eliminate Any usage.

Exports:
    EnumHttpOperationType: Discriminator enum for HTTP operation types
    ModelPayloadHttp: Base model for registry-managed HTTP payloads
    RegistryPayloadHttp: Decorator-based registry for HTTP payload types
    ModelHttpGetPayload: Payload for http.get result
    ModelHttpPostPayload: Payload for http.post result
    ModelHttpHealthCheckPayload: Payload for HTTP health check result
    HttpPayload: Discriminated union of all HTTP payload types (legacy)
    ModelHttpHandlerPayload: Wrapper containing discriminated union payload
"""

from omnibase_infra.handlers.models.http.enum_http_operation_type import (
    EnumHttpOperationType,
)
from omnibase_infra.handlers.models.http.model_http_get_payload import (
    ModelHttpGetPayload,
)
from omnibase_infra.handlers.models.http.model_http_handler_payload import (
    HttpPayload,
    ModelHttpHandlerPayload,
)
from omnibase_infra.handlers.models.http.model_http_health_check_payload import (
    ModelHttpHealthCheckPayload,
)
from omnibase_infra.handlers.models.http.model_http_post_payload import (
    ModelHttpPostPayload,
)
from omnibase_infra.handlers.models.http.model_payload_http import (
    ModelPayloadHttp,
    RegistryPayloadHttp,
)

__all__: list[str] = [
    "EnumHttpOperationType",
    "ModelPayloadHttp",
    "RegistryPayloadHttp",
    "ModelHttpGetPayload",
    "ModelHttpPostPayload",
    "ModelHttpHealthCheckPayload",
    "HttpPayload",
    "ModelHttpHandlerPayload",
]
