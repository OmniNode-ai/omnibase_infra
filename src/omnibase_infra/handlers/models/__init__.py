# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler Models Module.

This module exports Pydantic models for handler request/response structures.
All models are strongly typed to eliminate Any usage.

Exports:
    ModelDbQueryPayload: Payload containing database query results
    ModelDbQueryResponse: Full database query response envelope
    ModelDbHealthResponse: Database adapter health check response
    ModelDbDescribeResponse: Database adapter metadata and capabilities
    ModelConsulHandlerPayload: Payload containing Consul operation results
    ModelConsulHandlerResponse: Full Consul handler response envelope

Consul Operation Payloads:
    EnumConsulOperationType: Discriminator enum for Consul operation types
    ModelConsulKVItem: Single KV item from recurse query
    ModelConsulKVGetFoundPayload: Payload for consul.kv_get when key is found
    ModelConsulKVGetNotFoundPayload: Payload for consul.kv_get when key not found
    ModelConsulKVGetRecursePayload: Payload for consul.kv_get with recurse=True
    ModelConsulKVPutPayload: Payload for consul.kv_put result
    ModelConsulRegisterPayload: Payload for consul.register result
    ModelConsulDeregisterPayload: Payload for consul.deregister result
    ModelConsulHealthCheckPayload: Payload for consul.health_check result
    ConsulPayload: Discriminated union of all Consul payload types
"""

from omnibase_infra.handlers.models.consul import (
    ConsulPayload,
    EnumConsulOperationType,
    ModelConsulDeregisterPayload,
    ModelConsulHandlerPayload,
    ModelConsulHealthCheckPayload,
    ModelConsulKVGetFoundPayload,
    ModelConsulKVGetNotFoundPayload,
    ModelConsulKVGetRecursePayload,
    ModelConsulKVItem,
    ModelConsulKVPutPayload,
    ModelConsulRegisterPayload,
)
from omnibase_infra.handlers.models.model_consul_handler_response import (
    ModelConsulHandlerResponse,
)
from omnibase_infra.handlers.models.model_db_describe_response import (
    ModelDbDescribeResponse,
)
from omnibase_infra.handlers.models.model_db_health_response import (
    ModelDbHealthResponse,
)
from omnibase_infra.handlers.models.model_db_query_payload import ModelDbQueryPayload
from omnibase_infra.handlers.models.model_db_query_response import ModelDbQueryResponse

__all__: list[str] = [
    # Consul payload types (discriminated union)
    "EnumConsulOperationType",
    "ModelConsulKVItem",
    "ModelConsulKVGetFoundPayload",
    "ModelConsulKVGetNotFoundPayload",
    "ModelConsulKVGetRecursePayload",
    "ModelConsulKVPutPayload",
    "ModelConsulRegisterPayload",
    "ModelConsulDeregisterPayload",
    "ModelConsulHealthCheckPayload",
    "ConsulPayload",
    # Consul wrapper models
    "ModelConsulHandlerPayload",
    "ModelConsulHandlerResponse",
    # Database models
    "ModelDbQueryPayload",
    "ModelDbQueryResponse",
    "ModelDbHealthResponse",
    "ModelDbDescribeResponse",
]
