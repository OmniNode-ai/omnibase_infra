"""Handlers module for omnibase_infra.

This module provides handler implementations for various infrastructure
communication patterns including HTTP REST and database operations.

Handlers are responsible for:
- Processing incoming requests and messages
- Routing to appropriate services
- Formatting and returning responses
- Error handling and logging

Available Handlers:
- HandlerHttp: HTTP/REST protocol handler (MVP: GET, POST only)
"""

from omnibase_infra.handlers.handler_http import HandlerHttp

__all__: list[str] = [
    "HandlerHttp",
    # "HandlerDb",  # Database operation handler (future)
]
