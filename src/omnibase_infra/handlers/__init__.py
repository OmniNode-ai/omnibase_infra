"""Handlers module for omnibase_infra.

This module provides handler implementations for various infrastructure
communication patterns including HTTP REST and database operations.

Handlers are responsible for:
- Processing incoming requests and messages
- Routing to appropriate services
- Formatting and returning responses
- Error handling and logging

Available Handlers:
- HandlerHttpRest: HTTP/REST protocol handler (MVP: GET, POST only)
- HandlerDb: PostgreSQL database handler (MVP: query, execute only)
- HandlerVault: HashiCorp Vault secret management handler (MVP: KV v2 secrets)
- HandlerConsul: HashiCorp Consul service discovery handler (MVP: KV, service registration)
- HandlerMCP: Model Context Protocol handler for AI agent tool integration
- HandlerFileSystem: Filesystem handler with path whitelisting and size limits
- HandlerManifestPersistence: Execution manifest persistence with filesystem storage
- HandlerGraph: Graph database handler (Memgraph/Neo4j via Bolt protocol)
- HandlerIntent: Intent storage and query handler wrapping HandlerGraph (demo wiring)
- HandlerQdrant: Qdrant vector database handler (MVP: create, upsert, search, delete)

Response Models:
- ModelDbQueryPayload: Database query result payload
- ModelDbQueryResponse: Database query response envelope
- ModelDbDescribeResponse: Database handler metadata
- ModelConsulHandlerPayload: Consul operation result payload
- ModelConsulHandlerResponse: Consul handler response envelope
- ModelGraphHandlerResponse: Graph handler response envelope
- ModelQdrantHandlerResponse: Qdrant handler response envelope
"""

from omnibase_infra.handlers.handler_consul import HandlerConsul
from omnibase_infra.handlers.handler_db import HandlerDb
from omnibase_infra.handlers.handler_filesystem import HandlerFileSystem
from omnibase_infra.handlers.handler_graph import HandlerGraph
from omnibase_infra.handlers.handler_http import HandlerHttpRest
from omnibase_infra.handlers.handler_intent import (  # DEMO: Temporary intent handler wiring (OMN-1515)
    HANDLER_ID_INTENT,
    HandlerIntent,
)
from omnibase_infra.handlers.handler_manifest_persistence import (
    HandlerManifestPersistence,
)
from omnibase_infra.handlers.handler_mcp import HandlerMCP
from omnibase_infra.handlers.handler_qdrant import HandlerQdrant
from omnibase_infra.handlers.handler_vault import HandlerVault
from omnibase_infra.handlers.models import (
    ModelConsulHandlerPayload,
    ModelConsulHandlerResponse,
    ModelDbDescribeResponse,
    ModelDbQueryPayload,
    ModelDbQueryResponse,
)
from omnibase_infra.handlers.models.model_graph_handler_response import (
    ModelGraphHandlerResponse,
)
from omnibase_infra.handlers.models.model_qdrant_handler_response import (
    ModelQdrantHandlerResponse,
)

__all__: list[str] = [
    "HANDLER_ID_INTENT",
    "HandlerConsul",
    "HandlerDb",
    "HandlerFileSystem",
    "HandlerGraph",
    "HandlerHttpRest",
    "HandlerIntent",
    "HandlerManifestPersistence",
    "HandlerMCP",
    "HandlerQdrant",
    "HandlerVault",
    "ModelConsulHandlerPayload",
    "ModelConsulHandlerResponse",
    "ModelDbDescribeResponse",
    "ModelDbQueryPayload",
    "ModelDbQueryResponse",
    "ModelGraphHandlerResponse",
    "ModelQdrantHandlerResponse",
]
