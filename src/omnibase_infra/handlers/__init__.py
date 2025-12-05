"""Handlers module for omnibase_infra.

This module provides handler implementations for various infrastructure
communication patterns including HTTP REST and database operations.

Handlers are responsible for:
- Processing incoming requests and messages
- Routing to appropriate services
- Formatting and returning responses
- Error handling and logging

Available Handlers:
- HttpHandler: HTTP/REST protocol handler (MVP: GET, POST only)
"""

from omnibase_infra.handlers.http_handler import HttpHandler

__all__: list[str] = [
    "HttpHandler",
    # "DBHandler",  # Database operation handler (future)
]
