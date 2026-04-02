# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler that forwards platform events to an HTTP backend.

This is an EFFECT handler -- it performs I/O (HTTP POST).

Ported from archive: omniarchon/python/src/server/nodes/handlers
(ServiceLifecycleHandler, SystemEventHandler, ToolUpdateHandler).
"""

from __future__ import annotations

import logging
import os

import httpx

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_event_forward_effect.models.model_event_forward_request import (
    ModelEventForwardRequest,
)
from omnibase_infra.nodes.node_event_forward_effect.models.model_event_forward_result import (
    ModelEventForwardResult,
)

logger = logging.getLogger(__name__)

# Category -> backend path mapping (mirrors the three archive handlers)
_CATEGORY_ENDPOINTS: dict[str, str] = {
    "lifecycle": "/api/events/service-lifecycle",
    "system": "/api/events/system-event",
    "tool": "/api/events/tool-update",
    "generic": "/api/events/generic",
}


class HandlerEventForward:
    """Forwards platform events to a configurable HTTP backend.

    Consolidates three archive handlers (ServiceLifecycleHandler,
    SystemEventHandler, ToolUpdateHandler) into a single contract-driven
    handler that routes by event category.
    """

    def __init__(self, http_client: httpx.AsyncClient | None = None) -> None:
        self._owns_client = http_client is None
        self._http_client = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def handle(
        self, request: ModelEventForwardRequest
    ) -> ModelEventForwardResult:
        """Forward an event to the HTTP backend.

        Args:
            request: The event to forward.

        Returns:
            ModelEventForwardResult with HTTP status and success flag.
        """
        backend_url = os.environ.get(
            "EVENT_FORWARD_BACKEND_URL", "http://localhost:8000"
        )
        endpoint_path = _CATEGORY_ENDPOINTS.get(
            request.category, _CATEGORY_ENDPOINTS["generic"]
        )
        full_url = f"{backend_url.rstrip('/')}{endpoint_path}"

        logger.info(
            "Forwarding event | category=%s type=%s endpoint=%s correlation_id=%s",
            request.category,
            request.event_type,
            full_url,
            request.correlation_id,
        )

        body = {
            "event_type": request.event_type,
            "correlation_id": str(request.correlation_id),
            "timestamp": request.timestamp,
            "category": request.category,
            "severity": request.severity,
            "source": request.source,
            "payload": request.payload,
            "metadata": request.metadata,
        }

        try:
            response = await self._http_client.post(full_url, json=body)
            success = response.status_code in (200, 201, 202)

            if success:
                logger.info(
                    "Event forwarded | status=%d correlation_id=%s",
                    response.status_code,
                    request.correlation_id,
                )
            else:
                logger.warning(
                    "Backend rejected event | status=%d correlation_id=%s",
                    response.status_code,
                    request.correlation_id,
                )

            return ModelEventForwardResult(
                correlation_id=request.correlation_id,
                success=success,
                http_status=response.status_code,
                endpoint=full_url,
            )
        except Exception as exc:
            logger.exception(
                "Event forward failed | endpoint=%s correlation_id=%s",
                full_url,
                request.correlation_id,
            )
            return ModelEventForwardResult(
                correlation_id=request.correlation_id,
                success=False,
                endpoint=full_url,
                error_message=str(exc)[:500],
            )

    async def close(self) -> None:
        """Release HTTP resources if owned by this handler."""
        if self._owns_client and self._http_client is not None:
            await self._http_client.aclose()
