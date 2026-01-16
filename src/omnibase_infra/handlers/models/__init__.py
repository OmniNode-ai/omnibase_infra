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

Qdrant Models:
    ModelQdrantHandlerPayload: Payload containing Qdrant operation results
    ModelQdrantHandlerResponse: Full Qdrant handler response envelope
    EnumQdrantOperationType: Discriminator enum for Qdrant operation types
    ModelQdrantSearchPayload: Payload for qdrant.search result
    ModelQdrantUpsertPayload: Payload for qdrant.upsert result
    ModelQdrantDeletePayload: Payload for qdrant.delete result
    ModelQdrantCollectionPayload: Payload for qdrant.collection operations
    QdrantPayload: Discriminated union of all Qdrant payload types

Graph Models:
    ModelGraphHandlerPayload: Payload containing Graph operation results
    ModelGraphHandlerResponse: Full Graph handler response envelope
    EnumGraphOperationType: Discriminator enum for Graph operation types
    ModelGraphQueryPayload: Payload for graph.query result
    ModelGraphExecutePayload: Payload for graph.execute result
    GraphPayload: Discriminated union of all Graph payload types

HTTP Models:
    ModelHttpHandlerPayload: Payload containing HTTP operation results
    ModelHttpHandlerResponse: Full HTTP handler response envelope
    EnumHttpOperationType: Discriminator enum for HTTP operation types
    ModelHttpGetPayload: Payload for http.get result
    ModelHttpPostPayload: Payload for http.post result
    HttpPayload: Discriminated union of all HTTP payload types

Filesystem Models:
    ModelFileSystemConfig: Configuration for HandlerFileSystem initialization
    ModelReadFilePayload: Payload for filesystem.read_file operation
    ModelReadFileResult: Result from filesystem.read_file operation
    ModelWriteFilePayload: Payload for filesystem.write_file operation
    ModelWriteFileResult: Result from filesystem.write_file operation
    ModelListDirectoryPayload: Payload for filesystem.list_directory operation
    ModelDirectoryEntry: Single entry from directory listing
    ModelListDirectoryResult: Result from filesystem.list_directory operation
    ModelEnsureDirectoryPayload: Payload for filesystem.ensure_directory operation
    ModelEnsureDirectoryResult: Result from filesystem.ensure_directory operation
    ModelDeleteFilePayload: Payload for filesystem.delete_file operation
    ModelDeleteFileResult: Result from filesystem.delete_file operation

Manifest Persistence Models:
    ModelManifestPersistenceConfig: Configuration for HandlerManifestPersistence
    ModelManifestStorePayload: Payload for manifest.store operation
    ModelManifestStoreResult: Result from manifest.store operation
    ModelManifestRetrievePayload: Payload for manifest.retrieve operation
    ModelManifestRetrieveResult: Result from manifest.retrieve operation
    ModelManifestQueryPayload: Payload for manifest.query operation
    ModelManifestQueryResult: Result from manifest.query operation
    ModelManifestMetadata: Lightweight metadata for manifest queries
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

# Filesystem models (one model per file per ONEX convention)
from omnibase_infra.handlers.models.model_filesystem_config import ModelFileSystemConfig
from omnibase_infra.handlers.models.model_filesystem_delete_payload import (
    ModelDeleteFilePayload,
)
from omnibase_infra.handlers.models.model_filesystem_delete_result import (
    ModelDeleteFileResult,
)
from omnibase_infra.handlers.models.model_filesystem_directory_entry import (
    ModelDirectoryEntry,
)
from omnibase_infra.handlers.models.model_filesystem_ensure_directory_payload import (
    ModelEnsureDirectoryPayload,
)
from omnibase_infra.handlers.models.model_filesystem_ensure_directory_result import (
    ModelEnsureDirectoryResult,
)
from omnibase_infra.handlers.models.model_filesystem_list_directory_payload import (
    ModelListDirectoryPayload,
)
from omnibase_infra.handlers.models.model_filesystem_list_directory_result import (
    ModelListDirectoryResult,
)
from omnibase_infra.handlers.models.model_filesystem_read_payload import (
    ModelReadFilePayload,
)
from omnibase_infra.handlers.models.model_filesystem_read_result import (
    ModelReadFileResult,
)
from omnibase_infra.handlers.models.model_filesystem_write_payload import (
    ModelWriteFilePayload,
)
from omnibase_infra.handlers.models.model_filesystem_write_result import (
    ModelWriteFileResult,
)
from omnibase_infra.handlers.models.model_graph_handler_response import (
    ModelGraphHandlerResponse,
)
from omnibase_infra.handlers.models.model_handler_response import (
    ModelHandlerResponse,
)
from omnibase_infra.handlers.models.model_http_handler_response import (
    ModelHttpHandlerResponse,
)

# Manifest persistence models (one model per file per ONEX convention)
from omnibase_infra.handlers.models.model_manifest_metadata import (
    ModelManifestMetadata,
)
from omnibase_infra.handlers.models.model_manifest_persistence_config import (
    ModelManifestPersistenceConfig,
)
from omnibase_infra.handlers.models.model_manifest_query_payload import (
    ModelManifestQueryPayload,
)
from omnibase_infra.handlers.models.model_manifest_query_result import (
    ModelManifestQueryResult,
)
from omnibase_infra.handlers.models.model_manifest_retrieve_payload import (
    ModelManifestRetrievePayload,
)
from omnibase_infra.handlers.models.model_manifest_retrieve_result import (
    ModelManifestRetrieveResult,
)
from omnibase_infra.handlers.models.model_manifest_store_payload import (
    ModelManifestStorePayload,
)
from omnibase_infra.handlers.models.model_manifest_store_result import (
    ModelManifestStoreResult,
)
from omnibase_infra.handlers.models.model_operation_context import (
    ModelOperationContext,
)
from omnibase_infra.handlers.models.model_qdrant_handler_response import (
    ModelQdrantHandlerResponse,
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
    "ConsulPayload",
    # Consul payload types (discriminated union)
    "EnumConsulOperationType",
    # HTTP payload types (discriminated union)
    "EnumHttpOperationType",
    # Vault payload types (discriminated union)
    "EnumVaultOperationType",
    "HttpPayload",
    "ModelConsulDeregisterPayload",
    # Consul wrapper models
    "ModelConsulHandlerPayload",
    "ModelConsulHandlerResponse",
    "ModelConsulKVGetFoundPayload",
    "ModelConsulKVGetNotFoundPayload",
    "ModelConsulKVGetRecursePayload",
    "ModelConsulKVItem",
    "ModelConsulKVPutPayload",
    "ModelConsulRegisterPayload",
    # Database models
    "ModelDbQueryPayload",
    "ModelDbQueryResponse",
    "ModelDbDescribeResponse",
    # Generic response model
    "ModelHandlerResponse",
    # Graph wrapper models
    "ModelGraphHandlerResponse",
    # HTTP models
    "ModelHttpBodyContent",
    "ModelHttpGetPayload",
    # HTTP wrapper models
    "ModelHttpHandlerPayload",
    "ModelHttpHandlerResponse",
    "ModelHttpPostPayload",
    # Common models for retry and operation tracking
    "ModelOperationContext",
    # Qdrant wrapper models
    "ModelQdrantHandlerResponse",
    "ModelRetryState",
    # Vault models
    "ModelVaultDeletePayload",
    # Vault wrapper models
    "ModelVaultHandlerPayload",
    "ModelVaultHandlerResponse",
    "ModelVaultListPayload",
    "ModelVaultRenewTokenPayload",
    "ModelVaultSecretPayload",
    "ModelVaultWritePayload",
    "VaultPayload",
    # Filesystem models
    "ModelFileSystemConfig",
    "ModelReadFilePayload",
    "ModelReadFileResult",
    "ModelWriteFilePayload",
    "ModelWriteFileResult",
    "ModelListDirectoryPayload",
    "ModelDirectoryEntry",
    "ModelListDirectoryResult",
    "ModelEnsureDirectoryPayload",
    "ModelEnsureDirectoryResult",
    "ModelDeleteFilePayload",
    "ModelDeleteFileResult",
    # Manifest persistence models
    "ModelManifestPersistenceConfig",
    "ModelManifestStorePayload",
    "ModelManifestStoreResult",
    "ModelManifestRetrievePayload",
    "ModelManifestRetrieveResult",
    "ModelManifestQueryPayload",
    "ModelManifestQueryResult",
    "ModelManifestMetadata",
]
