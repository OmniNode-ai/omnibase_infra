# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""HTTP client provider.

Creates httpx.AsyncClient instances from environment-driven configuration.

Part of OMN-1976: Contract dependency materialization.
"""

from __future__ import annotations

import logging

import httpx

from omnibase_infra.runtime.models.model_http_client_config import (
    ModelHttpClientConfig,
)

logger = logging.getLogger(__name__)


class ProviderHttpClient:
    """Creates and manages shared HTTP clients.

    Clients are created from HTTP_* environment variables and shared
    across all contracts that declare http_client dependencies.
    """

    def __init__(self, config: ModelHttpClientConfig) -> None:
        """Initialize the HTTP client provider.

        Args:
            config: HTTP client configuration (timeout, redirects).
        """
        self._config = config

    async def create(self) -> httpx.AsyncClient:
        """Create an httpx.AsyncClient.

        Returns:
            httpx.AsyncClient instance.
        """
        logger.info(
            "Creating HTTP client",
            extra={
                "timeout_seconds": self._config.timeout_seconds,
                "follow_redirects": self._config.follow_redirects,
            },
        )

        client = self.create_client(
            timeout_seconds=self._config.timeout_seconds,
            follow_redirects=self._config.follow_redirects,
        )

        logger.info("HTTP client created successfully")
        return client

    @staticmethod
    def create_client(
        *, timeout_seconds: float, follow_redirects: bool = False
    ) -> httpx.AsyncClient:
        """Materialize an owned client for synchronous transport constructors.

        Client construction performs no I/O. Callers retain responsibility for
        closing the returned client, just as for ``create``. Explicit settings
        preserve transport-specific timeouts and redirect policies.
        """
        return httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds),
            follow_redirects=follow_redirects,
        )

    @staticmethod
    async def close(resource: httpx.AsyncClient | None) -> None:
        """Close an HTTP client.

        Args:
            resource: The httpx.AsyncClient to close.
        """
        if resource is not None and hasattr(resource, "aclose"):
            await resource.aclose()
            logger.info("HTTP client closed")


__all__ = ["ProviderHttpClient"]
