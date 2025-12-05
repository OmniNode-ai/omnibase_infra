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
"""

from omnibase_infra.handlers.handler_db import DbAdapter
from omnibase_infra.handlers.handler_http import HttpRestAdapter

__all__: list[str] = [
    "DbAdapter",
    "HttpRestAdapter",
]
