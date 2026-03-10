# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""HandlerResourceManager — manages httpx async clients per handler_id.

Lives outside the ONEX graph. Infrastructure-layer lifecycle management only.
Not a singleton — callers control instantiation and lifetime.
Not registered as a handler or node.
"""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)


class HandlerResourceManager:
    """Manages runtime resources (httpx async clients) keyed by handler_id.

    The ONEX graph operates on logical state while runtime resources (open
    connections, async clients) are managed here — external to the graph.

    This enables:
    - Handlers to be stateless ONEX effect nodes with no internal client lifecycle
    - Resource pooling and reuse across node invocations without graph coupling
    - Clean shutdown: drain connections independently of node teardown

    Monolithic handlers (HandlerHttp, HandlerKafka) are preserved for MVP.
    This class is the foundation for future handler-as-nodes decomposition.
    """

    def __init__(self) -> None:
        self._clients: dict[str, httpx.AsyncClient] = {}

    async def get_or_create_client(
        self,
        handler_id: str,
        base_url: str | None = None,
    ) -> httpx.AsyncClient:
        """Return the existing client for handler_id, or create one.

        Args:
            handler_id: Unique identifier for the handler owning this client.
            base_url: Optional base URL to configure on a newly created client.
                      Ignored when an existing client is returned.

        Returns:
            An open httpx.AsyncClient bound to handler_id.
        """
        if handler_id not in self._clients:
            if base_url is not None:
                self._clients[handler_id] = httpx.AsyncClient(base_url=base_url)
            else:
                self._clients[handler_id] = httpx.AsyncClient()
            logger.debug("Created httpx client for handler_id=%s", handler_id)
        return self._clients[handler_id]

    async def release_client(self, handler_id: str) -> None:
        """Close and remove the client for handler_id, if one exists.

        Args:
            handler_id: Unique identifier for the handler whose client to release.
        """
        client = self._clients.pop(handler_id, None)
        if client is not None:
            await client.aclose()
            logger.debug("Released httpx client for handler_id=%s", handler_id)

    async def shutdown_all(self) -> None:
        """Close all open clients.

        Safe to call multiple times; subsequent calls are no-ops.
        Errors from individual aclose() calls are logged and do not interrupt
        shutdown of remaining clients.
        """
        handler_ids = list(self._clients.keys())
        closed = 0
        for handler_id in handler_ids:
            client = self._clients.pop(handler_id, None)
            if client is not None:
                try:
                    await client.aclose()
                    closed += 1
                    logger.debug("Shutdown httpx client for handler_id=%s", handler_id)
                except Exception:
                    logger.exception(
                        "Error closing httpx client for handler_id=%s", handler_id
                    )
        logger.debug(
            "HandlerResourceManager shutdown complete (%d/%d clients closed)",
            closed,
            len(handler_ids),
        )
