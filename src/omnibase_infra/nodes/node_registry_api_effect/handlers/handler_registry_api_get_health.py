# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Handler for registry API operation: get_health.

Pings each infrastructure component (DB, Kafka, Qdrant) with try/except
and returns per-component status. Never raises on individual component failure.

Ticket: OMN-4482
"""

from __future__ import annotations

import logging
from uuid import UUID

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_registry_api_effect.models import (
    ModelRegistryApiResponse,
)

logger = logging.getLogger(__name__)

__all__ = ["HandlerRegistryApiGetHealth"]


class HandlerRegistryApiGetHealth:
    """Handler for operation: get_health.

    Returns per-component health status for the registry service.
    Individual component failures are caught and reported as unhealthy
    rather than propagated as exceptions.
    """

    @property
    def handler_type(self) -> EnumHandlerType:
        """Return the architectural role: INFRA_HANDLER."""
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Return the behavioral classification: EFFECT (external I/O)."""
        return EnumHandlerTypeCategory.EFFECT

    async def handle(self, request: object, correlation_id: UUID) -> object:
        """Handle get_health operation.

        Pings DB, Kafka, and Qdrant. Each component ping is wrapped in
        try/except so a single failure does not crash the health check.

        Args:
            request: Ignored for health checks.
            correlation_id: Distributed tracing identifier.

        Returns:
            ModelRegistryApiResponse with component health status in ``data``.
        """
        components: dict[str, str] = {}

        # DB health check
        try:
            # Attempt a minimal connection and close immediately.
            import os

            import asyncpg

            db_url = os.environ.get("OMNIBASE_INFRA_DB_URL", "")  # ONEX_EXCLUDE
            if db_url:
                conn = await asyncpg.connect(db_url, timeout=2)
                await conn.close()
                components["db"] = "healthy"
            else:
                components["db"] = "unconfigured"
        except Exception as exc:
            logger.warning("Health check: DB unavailable — %s", exc)
            components["db"] = "unhealthy"

        # Kafka health check
        try:
            import os

            from aiokafka.admin import AIOKafkaAdminClient

            _kafka = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")  # ONEX_EXCLUDE  # fmt: skip
            admin = AIOKafkaAdminClient(
                bootstrap_servers=_kafka, request_timeout_ms=2000
            )
            await admin.start()
            await admin.close()
            components["kafka"] = "healthy"
        except Exception as exc:
            logger.warning("Health check: Kafka unavailable — %s", exc)
            components["kafka"] = "unhealthy"

        # Qdrant health check
        try:
            import os

            import httpx

            qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")  # ONEX_EXCLUDE  # fmt: skip
            async with httpx.AsyncClient(timeout=2) as client:
                resp = await client.get(f"{qdrant_url}/healthz")
                components["qdrant"] = (
                    "healthy" if resp.status_code == 200 else "degraded"
                )
        except Exception as exc:
            logger.warning("Health check: Qdrant unavailable — %s", exc)
            components["qdrant"] = "unhealthy"

        overall = (
            "healthy"
            if all(v == "healthy" for v in components.values())
            else "degraded"
        )
        return ModelRegistryApiResponse(
            operation="get_health",
            correlation_id=correlation_id,
            success=True,
            data={"status": overall, "components": components},
        )
