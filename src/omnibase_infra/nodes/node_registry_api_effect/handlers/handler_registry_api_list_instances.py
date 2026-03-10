# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Handler for registry API operation: list_instances.

Queries Consul for live service instances. Returns an empty list with a
warning log when Consul is unavailable — never raises.

Ticket: OMN-4482
"""

from __future__ import annotations

import logging
import os
from uuid import UUID

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_registry_api_effect.models import (
    ModelRegistryApiResponse,
)

logger = logging.getLogger(__name__)

__all__ = ["HandlerRegistryApiListInstances"]

_DEFAULT_CONSUL_HOST = "localhost"
_DEFAULT_CONSUL_PORT = 8500
_DEFAULT_CONSUL_PORT_STR = str(_DEFAULT_CONSUL_PORT)


class HandlerRegistryApiListInstances:
    """Handler for operation: list_instances.

    Lists live service instances from Consul. Returns an empty list with
    a warning log when Consul is unavailable rather than propagating the
    connection error.

    Attributes:
        _consul_host: Consul agent hostname.
        _consul_port: Consul agent port.
    """

    def __init__(
        self,
        consul_host: str | None = None,
        consul_port: int | None = None,
    ) -> None:
        """Initialise the handler with Consul connection parameters.

        Args:
            consul_host: Consul agent hostname. Defaults to CONSUL_HOST env
                var or ``localhost``.
            consul_port: Consul agent port. Defaults to CONSUL_PORT env var
                or 8500.
        """
        _env_host = os.environ.get("CONSUL_HOST", _DEFAULT_CONSUL_HOST)  # ONEX_EXCLUDE
        _env_port = os.environ.get("CONSUL_PORT", _DEFAULT_CONSUL_PORT_STR)  # ONEX_EXCLUDE  # fmt: skip
        self._consul_host = consul_host or _env_host
        self._consul_port = consul_port or int(_env_port)

    @property
    def handler_type(self) -> EnumHandlerType:
        """Return the architectural role: INFRA_HANDLER."""
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Return the behavioral classification: EFFECT (external I/O)."""
        return EnumHandlerTypeCategory.EFFECT

    async def handle(self, request: object, correlation_id: UUID) -> object:
        """Handle list_instances operation.

        Attempts to query Consul for registered service instances. Returns an
        empty list with a warning if Consul is unreachable (ConnectionError or
        any other exception). Never raises.

        Args:
            request: Ignored for this operation.
            correlation_id: Distributed tracing identifier.

        Returns:
            ModelRegistryApiResponse with instance list in ``data``. Never raises.
        """
        try:
            import httpx

            consul_url = f"http://{self._consul_host}:{self._consul_port}"
            async with httpx.AsyncClient(timeout=2) as client:
                resp = await client.get(f"{consul_url}/v1/agent/services")
                resp.raise_for_status()
                services_map = resp.json()
                instances = list(services_map.values())
        except Exception as exc:
            logger.warning(
                "HandlerRegistryApiListInstances: Consul unavailable at %s:%d — %s",
                self._consul_host,
                self._consul_port,
                exc,
            )
            instances = []

        return ModelRegistryApiResponse(
            operation="list_instances",
            correlation_id=correlation_id,
            success=True,
            data={"instances": instances, "total": len(instances)},
        )
