# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler Models Module.

This module exports Pydantic models for handler request/response structures.
All models are strongly typed to eliminate Any usage.

Common Models:
    ModelRetryState: Encapsulates retry state for handler operations
    ModelOperationContext: Encapsulates operation context for handler tracking

Generic Response Model:
    ModelHandlerResponse: Generic handler response envelope (parameterized by payload type)

Database Models:
    ModelDbQueryPayload: Payload containing database query results
    ModelDbQueryResponse: Full database query response envelope
    ModelDbDescribeResponse: Database handler metadata and capabilities

Consul Models:
    ModelConsulHandlerPayload: Payload containing Consul operation results
    ModelConsulHandlerResponse: Full Consul handler response envelope
    EnumConsulOperationType: Discriminator enum for Consul operation types
    ModelConsulKVItem: Single KV item from recurse query
    ModelConsulKVGetFoundPayload: Payload for consul.kv_get when key is found
    ModelConsulKVGetNotFoundPayload: Payload for consul.kv_get when key not found
    ModelConsulKVGetRecursePayload: Payload for consul.kv_get with recurse=True
    ModelConsulKVPutPayload: Payload for consul.kv_put result
    ModelConsulRegisterPayload: Payload for consul.register result
    ModelConsulDeregisterPayload: Payload for consul.deregister result
    ConsulPayload: Discriminated union of all Consul payload types

Vault Models:
    ModelVaultHandlerPayload: Payload containing Vault operation results
    ModelVaultHandlerResponse: Full Vault handler response envelope
    EnumVaultOperationType: Discriminator enum for Vault operation types
    ModelVaultSecretPayload: Payload for vault.read_secret result
    ModelVaultWritePayload: Payload for vault.write_secret result
    ModelVaultDeletePayload: Payload for vault.delete_secret result
    ModelVaultListPayload: Payload for vault.list_secrets result
    ModelVaultRenewTokenPayload: Payload for vault.renew_token result
    VaultPayload: Discriminated union of all Vault payload types

HTTP Models:
    ModelHttpHandlerPayload: Payload containing HTTP operation results
    ModelHttpHandlerResponse: Full HTTP handler response envelope
    EnumHttpOperationType: Discriminator enum for HTTP operation types
    ModelHttpGetPayload: Payload for http.get result
    ModelHttpPostPayload: Payload for http.post result
    HttpPayload: Discriminated union of all HTTP payload types
"""

from omnibase_infra.handlers.models.consul import (
    ConsulPayload,
    EnumConsulOperationType,
    ModelConsulDeregisterPayload,
    ModelConsulHandlerPayload,
    ModelConsulKVGetFoundPayload,
    ModelConsulKVGetNotFoundPayload,
    ModelConsulKVGetRecursePayload,
    ModelConsulKVItem,
    ModelConsulKVPutPayload,
    ModelConsulRegisterPayload,
)
from omnibase_infra.handlers.models.http import (
    EnumHttpOperationType,
    HttpPayload,
    ModelHttpBodyContent,
    ModelHttpGetPayload,
    ModelHttpHandlerPayload,
    ModelHttpPostPayload,
)
from omnibase_infra.handlers.models.model_consul_handler_response import (
    ModelConsulHandlerResponse,
)
from omnibase_infra.handlers.models.model_db_describe_response import (
    ModelDbDescribeResponse,
)
from omnibase_infra.handlers.models.model_db_query_payload import ModelDbQueryPayload
from omnibase_infra.handlers.models.model_db_query_response import ModelDbQueryResponse
from omnibase_infra.handlers.models.model_handler_response import (
    ModelHandlerResponse,
)
from omnibase_infra.handlers.models.model_http_handler_response import (
    ModelHttpHandlerResponse,
)
from omnibase_infra.handlers.models.model_operation_context import (
    ModelOperationContext,
)
from omnibase_infra.handlers.models.model_retry_state import ModelRetryState
from omnibase_infra.handlers.models.model_vault_handler_response import (
    ModelVaultHandlerResponse,
)
from omnibase_infra.handlers.models.vault import (
    EnumVaultOperationType,
    ModelVaultDeletePayload,
    ModelVaultHandlerPayload,
    ModelVaultListPayload,
    ModelVaultRenewTokenPayload,
    ModelVaultSecretPayload,
    ModelVaultWritePayload,
    VaultPayload,
)

__all__: list[str] = [
    # Common models for retry and operation tracking
    "ModelRetryState",
    "ModelOperationContext",
    # Generic response model
    "ModelHandlerResponse",
    # Consul payload types (discriminated union)
    "EnumConsulOperationType",
    "ModelConsulKVItem",
    "ModelConsulKVGetFoundPayload",
    "ModelConsulKVGetNotFoundPayload",
    "ModelConsulKVGetRecursePayload",
    "ModelConsulKVPutPayload",
    "ModelConsulRegisterPayload",
    "ModelConsulDeregisterPayload",
    "ConsulPayload",
    # Consul wrapper models
    "ModelConsulHandlerPayload",
    "ModelConsulHandlerResponse",
    # Database models
    "ModelDbQueryPayload",
    "ModelDbQueryResponse",
    "ModelDbDescribeResponse",
    # Vault payload types (discriminated union)
    "EnumVaultOperationType",
    "ModelVaultSecretPayload",
    "ModelVaultWritePayload",
    "ModelVaultDeletePayload",
    "ModelVaultListPayload",
    "ModelVaultRenewTokenPayload",
    "VaultPayload",
    # Vault wrapper models
    "ModelVaultHandlerPayload",
    "ModelVaultHandlerResponse",
    # HTTP payload types (discriminated union)
    "EnumHttpOperationType",
    "ModelHttpBodyContent",
    "ModelHttpGetPayload",
    "ModelHttpPostPayload",
    "HttpPayload",
    # HTTP wrapper models
    "ModelHttpHandlerPayload",
    "ModelHttpHandlerResponse",
]
