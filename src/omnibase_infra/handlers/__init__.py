# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Handlers module for omnibase_infra.  # ai-slop-ok: pre-existing docstring opener

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
- HandlerMCP: Model Context Protocol handler for AI agent tool integration
- HandlerFileSystem: Filesystem handler with path whitelisting and size limits
- HandlerManifestPersistence: Execution manifest persistence with filesystem storage
- HandlerGraph: Graph database handler (Memgraph/Neo4j via Bolt protocol)
- HandlerInfisical: Infisical secret management handler with caching and circuit breaker (OMN-2286)
- HandlerIntent: Intent storage and query handler wrapping HandlerGraph (demo wiring)
- HandlerQdrant: Qdrant vector database handler (MVP: create, upsert, search, delete)
- HandlerSlackWebhook: Slack webhook handler for infrastructure alerting
- HandlerGmailApi: Gmail API handler with OAuth2 token management (OMN-2729)
- HandlerTopicCatalogPostgres: PostgreSQL-backed topic catalog handler (OMN-4011)

Response Models:
- ModelDbQueryPayload: Database query result payload
- ModelDbQueryResponse: Database query response envelope
- ModelDbDescribeResponse: Database handler metadata
- ModelGraphHandlerResponse: Graph handler response envelope
- ModelQdrantHandlerResponse: Qdrant handler response envelope
"""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibase_infra.handlers.handler_db import HandlerDb
    from omnibase_infra.handlers.handler_filesystem import HandlerFileSystem
    from omnibase_infra.handlers.handler_gmail_api import HandlerGmailApi
    from omnibase_infra.handlers.handler_graph import HandlerGraph
    from omnibase_infra.handlers.handler_http import HandlerHttpRest
    from omnibase_infra.handlers.handler_infisical import (
        HANDLER_ID_INFISICAL,
        HandlerInfisical,
    )
    from omnibase_infra.handlers.handler_intent import (  # DEMO: Temporary intent handler wiring (OMN-1515)
        HANDLER_ID_INTENT,
        HandlerIntent,
    )
    from omnibase_infra.handlers.handler_manifest_persistence import (
        HandlerManifestPersistence,
    )
    from omnibase_infra.handlers.handler_mcp import HandlerMCP
    from omnibase_infra.handlers.handler_qdrant import HandlerQdrant
    from omnibase_infra.handlers.handler_slack_webhook import HandlerSlackWebhook
    from omnibase_infra.handlers.handler_topic_catalog_postgres import (
        HandlerTopicCatalogPostgres,
    )
    from omnibase_infra.handlers.models import (
        ModelDbDescribeResponse,
        ModelDbQueryPayload,
        ModelDbQueryResponse,
    )
    from omnibase_infra.handlers.models.model_gmail_message import ModelGmailMessage
    from omnibase_infra.handlers.models.model_graph_handler_response import (
        ModelGraphHandlerResponse,
    )
    from omnibase_infra.handlers.models.model_qdrant_handler_response import (
        ModelQdrantHandlerResponse,
    )

# OMN-19444: Lazy imports keep `onex <cmd> --help` fast.
_LAZY_EXPORTS: dict[str, str] = {
    "HANDLER_ID_INFISICAL": "omnibase_infra.handlers.handler_infisical",
    "HANDLER_ID_INTENT": "omnibase_infra.handlers.handler_intent",
    "HandlerDb": "omnibase_infra.handlers.handler_db",
    "HandlerFileSystem": "omnibase_infra.handlers.handler_filesystem",
    "HandlerGmailApi": "omnibase_infra.handlers.handler_gmail_api",
    "HandlerGraph": "omnibase_infra.handlers.handler_graph",
    "HandlerHttpRest": "omnibase_infra.handlers.handler_http",
    "HandlerInfisical": "omnibase_infra.handlers.handler_infisical",
    "HandlerIntent": "omnibase_infra.handlers.handler_intent",
    "HandlerMCP": "omnibase_infra.handlers.handler_mcp",
    "HandlerManifestPersistence": "omnibase_infra.handlers.handler_manifest_persistence",
    "HandlerQdrant": "omnibase_infra.handlers.handler_qdrant",
    "HandlerSlackWebhook": "omnibase_infra.handlers.handler_slack_webhook",
    "HandlerTopicCatalogPostgres": "omnibase_infra.handlers.handler_topic_catalog_postgres",
    "ModelDbDescribeResponse": "omnibase_infra.handlers.models",
    "ModelDbQueryPayload": "omnibase_infra.handlers.models",
    "ModelDbQueryResponse": "omnibase_infra.handlers.models",
    "ModelGmailMessage": "omnibase_infra.handlers.models.model_gmail_message",
    "ModelGraphHandlerResponse": "omnibase_infra.handlers.models.model_graph_handler_response",
    "ModelQdrantHandlerResponse": "omnibase_infra.handlers.models.model_qdrant_handler_response",
}

__all__: list[str] = [
    "HANDLER_ID_INFISICAL",
    "HANDLER_ID_INTENT",
    "HandlerGmailApi",
    "HandlerDb",
    "HandlerTopicCatalogPostgres",
    "HandlerFileSystem",
    "HandlerGraph",
    "HandlerHttpRest",
    "HandlerInfisical",
    "HandlerIntent",
    "HandlerManifestPersistence",
    "HandlerMCP",
    "HandlerQdrant",
    "HandlerSlackWebhook",
    "ModelDbDescribeResponse",
    "ModelDbQueryPayload",
    "ModelDbQueryResponse",
    "ModelGmailMessage",
    "ModelGraphHandlerResponse",
    "ModelQdrantHandlerResponse",
]


def __getattr__(name: str) -> object:
    if name in _LAZY_EXPORTS:
        module = importlib.import_module(_LAZY_EXPORTS[name])
        value: object = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
