"""Handlers module for omnibase_infra.

This module provides adapter implementations for various infrastructure
communication patterns including HTTP REST and database operations.

Adapters are responsible for:
- Processing incoming requests and messages
- Routing to appropriate services
- Formatting and returning responses
- Error handling and logging

Available Adapters:
- HttpRestHandler: HTTP/REST protocol handler (MVP: GET, POST only)
- DbHandler: PostgreSQL database handler (MVP: query, execute only)
- VaultHandler: HashiCorp Vault secret management handler (MVP: KV v2 secrets)
- ConsulHandler: HashiCorp Consul service discovery handler (MVP: KV, service registration)

Response Models:
- ModelDbQueryPayload: Database query result payload
- ModelDbQueryResponse: Database query response envelope
- ModelDbHealthResponse: Database health check response
- ModelDbDescribeResponse: Database adapter metadata
- ModelConsulHandlerPayload: Consul operation result payload
- ModelConsulHandlerResponse: Consul handler response envelope
"""

from omnibase_infra.handlers.handler_consul import ConsulHandler
from omnibase_infra.handlers.handler_db import DbHandler
from omnibase_infra.handlers.handler_http import HttpRestHandler
from omnibase_infra.handlers.handler_vault import VaultHandler
from omnibase_infra.handlers.models import (
    ModelConsulHandlerPayload,
    ModelConsulHandlerResponse,
    ModelDbDescribeResponse,
    ModelDbHealthResponse,
    ModelDbQueryPayload,
    ModelDbQueryResponse,
)

__all__: list[str] = [
    "ConsulHandler",
    "DbHandler",
    "HttpRestHandler",
    "VaultHandler",
    "ModelConsulHandlerPayload",
    "ModelConsulHandlerResponse",
    "ModelDbQueryPayload",
    "ModelDbQueryResponse",
    "ModelDbHealthResponse",
    "ModelDbDescribeResponse",
]
