"""Handlers module for omnibase_infra.

This module provides adapter implementations for various infrastructure
communication patterns including HTTP REST and database operations.

Adapters are responsible for:
- Processing incoming requests and messages
- Routing to appropriate services
- Formatting and returning responses
- Error handling and logging

Available Adapters:
- HttpRestAdapter: HTTP/REST protocol adapter (MVP: GET, POST only)
- DbAdapter: PostgreSQL database adapter (MVP: query, execute only)
- VaultAdapter: HashiCorp Vault secret management adapter (MVP: KV v2 secrets)
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
from omnibase_infra.handlers.handler_db import DbAdapter
from omnibase_infra.handlers.handler_http import HttpRestAdapter
from omnibase_infra.handlers.handler_vault import VaultAdapter
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
    "DbAdapter",
    "HttpRestAdapter",
    "VaultAdapter",
    "ModelConsulHandlerPayload",
    "ModelConsulHandlerResponse",
    "ModelDbQueryPayload",
    "ModelDbQueryResponse",
    "ModelDbHealthResponse",
    "ModelDbDescribeResponse",
]
