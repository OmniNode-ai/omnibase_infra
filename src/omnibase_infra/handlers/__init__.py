"""Handlers module for omnibase_infra.

This module provides handler implementations for various infrastructure
communication patterns including HTTP REST and database operations.

Handlers are responsible for:
- Processing incoming requests and messages
- Routing to appropriate services
- Formatting and returning responses
- Error handling and logging

Available Handlers:
- HttpRestHandler: HTTP/REST protocol handler (MVP: GET, POST only)
- HandlerDb: PostgreSQL database handler (MVP: query, execute only)
- HandlerVault: HashiCorp Vault secret management handler (MVP: KV v2 secrets)
- HandlerConsul: HashiCorp Consul service discovery handler (MVP: KV, service registration)
- HandlerMCP: Model Context Protocol handler for AI agent tool integration

Response Models:
- ModelDbQueryPayload: Database query result payload
- ModelDbQueryResponse: Database query response envelope
- ModelDbDescribeResponse: Database handler metadata
- ModelConsulHandlerPayload: Consul operation result payload
- ModelConsulHandlerResponse: Consul handler response envelope
"""

from omnibase_infra.handlers.handler_consul import HandlerConsul
from omnibase_infra.handlers.handler_db import HandlerDb
from omnibase_infra.handlers.handler_http import HttpRestHandler
from omnibase_infra.handlers.handler_mcp import HandlerMCP
from omnibase_infra.handlers.handler_vault import HandlerVault
from omnibase_infra.handlers.models import (
    ModelConsulHandlerPayload,
    ModelConsulHandlerResponse,
    ModelDbDescribeResponse,
    ModelDbQueryPayload,
    ModelDbQueryResponse,
)

__all__: list[str] = [
    "HandlerConsul",
    "HandlerDb",
    "HandlerMCP",
    "HandlerVault",
    "HttpRestHandler",
    "ModelConsulHandlerPayload",
    "ModelConsulHandlerResponse",
    "ModelDbDescribeResponse",
    "ModelDbQueryPayload",
    "ModelDbQueryResponse",
]
