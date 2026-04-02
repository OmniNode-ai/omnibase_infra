# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler that probes model endpoints for health and latency.

This is an EFFECT handler — it performs external I/O (HTTP health checks).
"""

from __future__ import annotations

import asyncio
import logging
import time
from uuid import UUID

import httpx

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_model_health_effect.models.model_endpoint_health import (
    ModelEndpointHealth,
)
from omnibase_infra.nodes.node_model_health_effect.models.model_health_probe_target import (
    ModelHealthProbeTarget,
)
from omnibase_infra.nodes.node_model_health_effect.models.model_health_snapshot import (
    ModelHealthSnapshot,
)

logger = logging.getLogger(__name__)


class HandlerProbeHealth:
    """Probes model endpoints for health, latency, and queue depth."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def _probe_endpoint(
        self,
        target: ModelHealthProbeTarget,
        timeout_ms: int,
    ) -> ModelEndpointHealth:
        """Probe a single model endpoint."""
        if target.transport == "sdk":
            # SDK-based models (frontier APIs) — assume healthy, no latency probe
            return ModelEndpointHealth(
                model_key=target.model_key,
                healthy=True,
                latency_ms=0,
                queue_depth=0,
            )

        timeout_sec = timeout_ms / 1000.0
        start = time.monotonic()

        try:
            async with httpx.AsyncClient(timeout=timeout_sec) as client:
                resp = await client.get(f"{target.base_url}/health")
                latency_ms = int((time.monotonic() - start) * 1000)

                if resp.status_code != 200:
                    return ModelEndpointHealth(
                        model_key=target.model_key,
                        healthy=False,
                        latency_ms=latency_ms,
                        error_message=f"Health check returned {resp.status_code}",
                    )

                # Try to read queue depth from vLLM metrics endpoint
                queue_depth = 0
                try:
                    metrics_resp = await client.get(f"{target.base_url}/metrics")
                    if metrics_resp.status_code == 200:
                        for line in metrics_resp.text.splitlines():
                            if line.startswith(
                                (
                                    "vllm:num_requests_waiting",
                                    "vllm_num_requests_waiting",
                                )
                            ):
                                parts = line.split()
                                if len(parts) >= 2:
                                    queue_depth = int(float(parts[-1]))
                                    break
                except (httpx.HTTPError, ValueError):
                    pass  # Metrics endpoint optional

                return ModelEndpointHealth(
                    model_key=target.model_key,
                    healthy=True,
                    latency_ms=latency_ms,
                    queue_depth=queue_depth,
                )

        except (httpx.HTTPError, OSError) as e:
            latency_ms = int((time.monotonic() - start) * 1000)
            return ModelEndpointHealth(
                model_key=target.model_key,
                healthy=False,
                latency_ms=latency_ms,
                error_message=str(e),
            )

    async def probe_endpoints(
        self,
        targets: tuple[ModelHealthProbeTarget, ...],
        correlation_id: UUID,
        timeout_ms: int = 5000,
    ) -> ModelHealthSnapshot:
        """Probe all model endpoints in parallel.

        Args:
            targets: Model endpoints to probe.
            correlation_id: Workflow correlation ID.
            timeout_ms: Per-endpoint timeout in milliseconds.

        Returns:
            ModelHealthSnapshot with per-endpoint health results.
        """
        logger.info(
            "Probing %d model endpoints (correlation_id=%s)",
            len(targets),
            correlation_id,
        )

        tasks = [self._probe_endpoint(t, timeout_ms) for t in targets]
        results = await asyncio.gather(*tasks)

        healthy_count = sum(1 for r in results if r.healthy)
        logger.info(
            "Health probe complete: %d/%d healthy (correlation_id=%s)",
            healthy_count,
            len(results),
            correlation_id,
        )

        return ModelHealthSnapshot(
            correlation_id=correlation_id,
            endpoints=tuple(results),
            success=True,
        )
