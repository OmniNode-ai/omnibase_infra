# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Route implementations for the Contract Resolver Bridge.

Wraps NodeContractResolveCompute.resolve() exactly — no bespoke API surface.
Only routes that mirror the node contracts are provided.

Routes:
    POST /api/nodes/contract.resolve — full resolution with patches
    GET  /health                      — liveness probe

Ticket: OMN-2756
"""

from __future__ import annotations

import logging
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.models.nodes.contract_resolve import (
    ModelContractResolveInput,
    ModelContractResolveOutput,
)
from omnibase_core.nodes.node_contract_resolve_compute import NodeContractResolveCompute
from omnibase_infra.services.contract_resolver.enum_health_status import (
    EnumHealthStatus,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Health response model
# ---------------------------------------------------------------------------


class ModelHealthResponse(BaseModel):
    """Health check response body."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    status: EnumHealthStatus = Field(
        default=EnumHealthStatus.OK, description="Service health status."
    )
    service: str = Field(
        default="node_contract_resolver_bridge",
        description="Service name.",
    )
    version: str = Field(default="1.0.0", description="Service version.")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get(
    "/health",
    response_model=ModelHealthResponse,
    summary="Liveness check",
    tags=["health"],
)
async def health() -> ModelHealthResponse:
    """Return service liveness status.

    Always returns 200 when the service process is running. Does not check
    downstream dependencies (NodeContractResolveCompute is stateless/pure).

    Returns:
        :class:`ModelHealthResponse` with ``status="ok"``.
    """
    return ModelHealthResponse()


@router.post(
    "/api/nodes/contract.resolve",
    response_model=ModelContractResolveOutput,
    summary="Resolve a contract with overlay patches",
    tags=["contract"],
    status_code=status.HTTP_200_OK,
)
async def contract_resolve(
    request: Request,
    body: ModelContractResolveInput,
) -> ModelContractResolveOutput:
    """Resolve a contract by applying ordered patches onto a base profile.

    Validates the request body against ``ModelContractResolveInput`` (Pydantic
    rejects unknown fields), then calls ``NodeContractResolveCompute.resolve()``
    directly — no Kafka round-trip in the MVP path.

    After resolution, emits ``onex.contract.resolve.completed`` to Kafka
    fire-and-forget if an event bus is configured on ``app.state``.

    Args:
        request: FastAPI request — used to access ``app.state`` for the
            optional event bus.
        body: Validated ``ModelContractResolveInput``.

    Returns:
        :class:`ModelContractResolveOutput` with the resolved contract and
        canonical hash.

    Raises:
        HTTPException: 422 if the request body fails Pydantic validation
            (handled automatically by FastAPI).
        HTTPException: 500 if NodeContractResolveCompute raises an unexpected
            error.
    """
    correlation_id: UUID = uuid4()

    logger.info(
        "contract.resolve called",
        extra={
            "correlation_id": str(correlation_id),
            "base_profile": body.base_profile_ref.profile,
            "patch_count": len(body.patches),
        },
    )

    try:
        node = NodeContractResolveCompute(correlation_id=correlation_id)
        output = node.resolve(body)
    except Exception as exc:
        logger.exception(
            "contract.resolve failed",
            extra={
                "correlation_id": str(correlation_id),
                "error_type": type(exc).__name__,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Resolution failed: {type(exc).__name__}",
        ) from exc

    logger.info(
        "contract.resolve completed",
        extra={
            "correlation_id": str(correlation_id),
            "resolved_hash": output.resolved_hash[:16],
            "overlays_applied": len(output.overlay_refs),
        },
    )

    # Fire-and-forget Kafka emission (observability, non-blocking)
    _emit_completed_event(request, output, correlation_id)

    return output


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _emit_completed_event(
    request: Request,
    output: ModelContractResolveOutput,
    correlation_id: UUID,
) -> None:
    """Emit onex.contract.resolve.completed fire-and-forget.

    Silently no-ops if no event bus is configured — the HTTP response is
    never blocked by event emission.

    Args:
        request: FastAPI request to access ``app.state.event_bus``.
        output: Resolved output model.
        correlation_id: Correlation ID for the completed event.
    """
    event_bus = getattr(getattr(request, "app", None), "state", None)
    if event_bus is None:
        return
    event_bus = getattr(event_bus, "event_bus", None)
    if event_bus is None:
        return

    try:
        from omnibase_core.models.events.contract_resolve import (
            ModelContractResolveCompletedEvent,
        )

        event = ModelContractResolveCompletedEvent.create(
            run_id=correlation_id,
            resolved_hash=output.resolved_hash,
            overlays_applied_count=len(output.overlay_refs),
            overlay_refs=list(output.overlay_refs),
            resolver_build=output.resolver_build,
            duration_ms=0,  # bridge does not track per-request latency
            correlation_id=correlation_id,
        )
        # Attempt publish — protocol is synchronous in the base implementation
        if hasattr(event_bus, "publish"):
            event_bus.publish(event)
    except Exception:
        # fallback-ok: fire-and-forget, never block the HTTP response
        logger.debug(
            "Failed to emit onex.contract.resolve.completed — continuing without event",
            exc_info=True,
        )


__all__ = ["EnumHealthStatus", "ModelHealthResponse", "router"]
