# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""HTTP client provider.

Creates httpx.AsyncClient instances from environment-driven configuration.

Part of OMN-1976: Contract dependency materialization.
"""

from __future__ import annotations

import logging
from typing import Any

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
        self._config = config

    # ONEX_EXCLUDE: any_type - returns httpx.AsyncClient
    async def create(self) -> Any:
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

        client = httpx.AsyncClient(
            timeout=httpx.Timeout(self._config.timeout_seconds),
            follow_redirects=self._config.follow_redirects,
        )

        logger.info("HTTP client created successfully")
        return client

    @staticmethod
    # ONEX_EXCLUDE: any_type - resource is httpx.AsyncClient, typed as Any for provider interface
    async def close(resource: Any) -> None:
        """Close an HTTP client.

        Args:
            resource: The httpx.AsyncClient to close.
        """
        if resource is not None and hasattr(resource, "aclose"):
            await resource.aclose()
            logger.info("HTTP client closed")


__all__ = ["ProviderHttpClient"]
