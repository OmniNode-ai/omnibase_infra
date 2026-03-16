# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Handler that validates TCP/HTTP health of deployment services.

Services with ``health_check_path`` set use HTTP GET; others use TCP connect.
Only LOCAL-mode services are validated; DISABLED and CLOUD services are skipped.

Invariants:
    I3 — Monkeypatch discipline:
        ``import asyncio`` and ``import httpx`` at module top.
        Patch via ``monkeypatch.setattr(handler_mod, "_check_http_health", ...)``.
    I4 — Port semantics:
        TCP checks use ``asyncio.open_connection`` (port OPEN), never ``connect_ex``.

Ticket: OMN-3494
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import httpx

from omnibase_core.enums import EnumDeploymentMode
from omnibase_core.models.core.model_deployment_topology import ModelDeploymentTopology
from omnibase_core.models.dispatch import ModelHandlerOutput
from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumInfraTransportType,
)
from omnibase_infra.errors import ModelInfraErrorContext
from omnibase_infra.nodes.node_setup_validate_effect.models.model_service_health_result import (
    ModelSetupNodeHealthResult,
)
from omnibase_infra.nodes.node_setup_validate_effect.models.model_setup_validate_effect_output import (
    ModelSetupValidateEffectOutput,
)

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer

logger = logging.getLogger(__name__)


async def _check_http_health(url: str, timeout_s: float) -> tuple[bool, float]:
    """HTTP GET health check.

    Patched via: ``monkeypatch.setattr(validate_mod, "_check_http_health", ...)``

    Args:
        url: Full HTTP URL to GET.
        timeout_s: Request timeout in seconds.

    Returns:
        Tuple of (reachable: bool, response_time_ms: float).
    """
    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            response = await client.get(url)
        elapsed_ms = (time.monotonic() - start) * 1000.0
        return response.status_code < 500, elapsed_ms
    except Exception as exc:  # noqa: BLE001 — boundary: returns degraded response
        elapsed_ms = (time.monotonic() - start) * 1000.0
        logger.debug("HTTP health check failed for %s: %s", url, exc)
        return False, elapsed_ms


async def _check_tcp_health(
    host: str, port: int, timeout_s: float
) -> tuple[bool, float]:
    """TCP connect health check (I4: checks port OPEN).

    Patched via: ``monkeypatch.setattr(validate_mod, "_check_tcp_health", ...)``

    Args:
        host: Hostname or IP to connect to.
        port: TCP port to connect to.
        timeout_s: Connection timeout in seconds.

    Returns:
        Tuple of (reachable: bool, response_time_ms: float).
    """
    start = time.monotonic()
    writer: asyncio.StreamWriter | None = None
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=timeout_s,
        )
        elapsed_ms = (time.monotonic() - start) * 1000.0
        return True, elapsed_ms
    except Exception as exc:  # noqa: BLE001 — boundary: returns degraded response
        elapsed_ms = (time.monotonic() - start) * 1000.0
        logger.debug("TCP health check failed for %s:%d: %s", host, port, exc)
        return False, elapsed_ms
    finally:
        if writer is not None:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:  # noqa: BLE001 — boundary: catch-all for resilience
                pass


class HandlerServiceValidate:
    """Validates TCP/HTTP health of all LOCAL-mode deployment services.

    For each LOCAL-mode service in the topology:
    - If ``svc.local.health_check_path`` is set → HTTP GET health check
    - Otherwise → TCP connect health check

    DISABLED and CLOUD services are excluded from results.

    Attributes:
        handler_type: ``NODE_HANDLER``
        handler_category: ``EFFECT``
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the handler."""
        self._container = container
        self._initialized: bool = False

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def initialize(self, config: dict[str, object]) -> None:
        """Initialize the handler."""
        self._initialized = True
        logger.info("HandlerServiceValidate initialized")

    async def shutdown(self) -> None:
        """Shut down the handler."""
        self._initialized = False
        logger.info("HandlerServiceValidate shutdown")

    async def execute(
        self, envelope: dict[str, object]
    ) -> ModelHandlerOutput[ModelSetupValidateEffectOutput]:
        """Validate health of all LOCAL-mode services.

        Envelope keys:
            topology: ModelDeploymentTopology instance.
            correlation_id: UUID for tracing.
            timeout_seconds: int — per-service health check timeout (default 60).
        """
        correlation_id_raw = envelope.get("correlation_id")
        context = ModelInfraErrorContext.with_correlation(
            correlation_id=correlation_id_raw
            if isinstance(correlation_id_raw, UUID)
            else None,
            transport_type=EnumInfraTransportType.HTTP,
            operation="validate_services",
            target_name="service_health",
        )
        corr_id = context.correlation_id
        if corr_id is None:
            corr_id = uuid4()

        topology_raw = envelope.get("topology")
        if not isinstance(topology_raw, ModelDeploymentTopology):
            return ModelHandlerOutput.for_compute(
                input_envelope_id=uuid4(),
                correlation_id=corr_id,
                handler_id="handler-service-validate",
                result=ModelSetupValidateEffectOutput(
                    all_healthy=False,
                    results=(),
                    correlation_id=corr_id,
                    error="topology must be a ModelDeploymentTopology instance",
                ),
            )

        timeout_raw = envelope.get("timeout_seconds", 60)
        timeout_seconds = (
            int(timeout_raw) if isinstance(timeout_raw, (int, float)) else 60
        )
        timeout_s = float(timeout_seconds)

        results: list[ModelSetupNodeHealthResult] = []

        for svc_name, svc in topology_raw.services.items():
            if svc.mode != EnumDeploymentMode.LOCAL:
                # Skip DISABLED and CLOUD services
                logger.debug("Skipping service %s (mode=%s)", svc_name, svc.mode)
                continue

            local_cfg = svc.local
            if local_cfg is None:
                # Should not happen (I2 invariant), but guard defensively
                results.append(
                    ModelSetupNodeHealthResult(
                        node_label=svc_name,
                        healthy=False,
                        message="LOCAL mode service missing local config",
                        detail=None,
                        response_time_ms=0.0,
                    )
                )
                continue

            host_port = local_cfg.host_port
            health_check_path = local_cfg.health_check_path

            if health_check_path is not None:
                url = f"http://localhost:{host_port}{health_check_path}"
                logger.debug("HTTP health check for %s: %s", svc_name, url)
                healthy, response_time_ms = await _check_http_health(url, timeout_s)
                message = f"HTTP {url} {'OK' if healthy else 'UNREACHABLE'}"
            else:
                logger.debug(
                    "TCP health check for %s: localhost:%d", svc_name, host_port
                )
                healthy, response_time_ms = await _check_tcp_health(
                    "localhost", host_port, timeout_s
                )
                message = f"TCP localhost:{host_port} {'OPEN' if healthy else 'CLOSED'}"

            results.append(
                ModelSetupNodeHealthResult(
                    node_label=svc_name,
                    healthy=healthy,
                    message=message,
                    detail=None,
                    response_time_ms=response_time_ms,
                )
            )

        all_healthy = all(r.healthy for r in results) if results else True

        logger.info(
            "HandlerServiceValidate complete: all_healthy=%s, checked=%d services",
            all_healthy,
            len(results),
        )

        return ModelHandlerOutput.for_compute(
            input_envelope_id=uuid4(),
            correlation_id=corr_id,
            handler_id="handler-service-validate",
            result=ModelSetupValidateEffectOutput(
                all_healthy=all_healthy,
                results=tuple(results),
                correlation_id=corr_id,
                error=None,
            ),
        )


__all__: list[str] = [
    "HandlerServiceValidate",
    "_check_http_health",
    "_check_tcp_health",
]
