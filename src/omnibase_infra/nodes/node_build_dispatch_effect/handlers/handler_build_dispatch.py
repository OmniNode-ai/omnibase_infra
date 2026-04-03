# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler that dispatches ticket-pipeline builds via delegation.

This is an EFFECT handler - performs external I/O (delegation dispatch).

Dispatch mechanism (primary): publishes a ``ModelDelegationRequest`` per ticket
to the ``onex.cmd.omnibase-infra.delegation-request.v1`` Kafka topic.  The
runtime's ``MessageDispatchEngine`` routes each message to the delegation
orchestrator FSM which handles LLM inference routing.

Fallback mechanism: when no ``publisher`` callable is injected (e.g. in tests
or when Kafka is unavailable), writes per-ticket JSON manifest files to
``$ONEX_STATE_DIR/autopilot/dispatch/`` for consumption by
``cron-buildloop.sh``.

Related:
    - OMN-7318: node_build_dispatch_effect
    - OMN-7381: Wire handler_build_dispatch to delegation orchestrator
    - OMN-5113: Autonomous Build Loop epic
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumInfraTransportType,
)
from omnibase_infra.event_bus.topic_constants import TOPIC_DELEGATION_REQUEST
from omnibase_infra.models.errors import ModelInfraErrorContext
from omnibase_infra.nodes.node_build_dispatch_effect.models.model_build_dispatch_outcome import (
    ModelBuildDispatchOutcome,
)
from omnibase_infra.nodes.node_build_dispatch_effect.models.model_build_dispatch_result import (
    ModelBuildDispatchResult,
)
from omnibase_infra.nodes.node_build_dispatch_effect.models.model_build_target import (
    ModelBuildTarget,
)
from omnibase_infra.utils.util_friction_emitter import emit_build_loop_friction

if TYPE_CHECKING:
    from omnibase_core.types import JsonType

logger = logging.getLogger(__name__)

# Event type used by the delegation dispatcher for message routing.
# Must match DispatcherDelegationRequest.message_types.
_DELEGATION_EVENT_TYPE = "omnibase-infra.delegation-request"


def _dispatch_dir() -> Path | None:
    """Resolve the dispatch manifest directory from ONEX_STATE_DIR."""
    state_dir = os.environ.get("ONEX_STATE_DIR", "")  # ONEX_EXCLUDE: runtime config
    if not state_dir:
        return None
    return Path(state_dir) / "autopilot" / "dispatch"


class HandlerBuildDispatch:
    """Dispatches ticket-pipeline builds for AUTO_BUILDABLE tickets via delegation.

    Primary path: publishes ``ModelDelegationRequest`` envelopes to Kafka for
    each ticket.  The delegation orchestrator FSM routes each request to a
    local LLM (DeepSeek-R1, Qwen3-Coder) or frontier Claude depending on
    task type and model availability.

    Fallback path: when no publisher is injected, writes per-ticket JSON
    manifests to the dispatch directory for ``cron-buildloop.sh``.

    Failures on individual tickets do not block other dispatches.
    """

    def __init__(
        self,
        publisher: Callable[..., Awaitable[bool]] | None = None,
    ) -> None:
        """Initialise with an optional event publisher.

        Args:
            publisher: An async callable with the signature::

                    async def publish(
                        event_type: str,
                        payload: JsonType,
                        correlation_id: str | None = None,
                        topic: str | None = None,
                    ) -> bool: ...

                Typically ``AdapterProtocolEventPublisherKafka.publish``.
                When ``None``, falls back to filesystem manifest writes.
        """
        self._publisher = publisher

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def handle(
        self,
        correlation_id: UUID,
        targets: tuple[ModelBuildTarget, ...],
        dry_run: bool = False,
    ) -> ModelBuildDispatchResult:
        """Dispatch builds for each target ticket.

        Args:
            correlation_id: Cycle correlation ID.
            targets: Tickets to dispatch.
            dry_run: Skip actual dispatch.

        Returns:
            ModelBuildDispatchResult with per-ticket outcomes.
        """
        logger.info(
            "Build dispatch: %d targets (correlation_id=%s, dry_run=%s, kafka=%s)",
            len(targets),
            correlation_id,
            dry_run,
            self._publisher is not None,
        )

        outcomes: list[ModelBuildDispatchOutcome] = []
        total_dispatched = 0
        total_failed = 0

        # Filesystem fallback: only needed when no publisher is injected
        dispatch_path: Path | None = None
        if self._publisher is None and targets and not dry_run:
            dispatch_path = _dispatch_dir()
            if dispatch_path is None:
                msg = "ONEX_STATE_DIR not set — cannot write dispatch manifest"
                raise RuntimeError(msg)
            dispatch_path.mkdir(parents=True, exist_ok=True)

        seen_ticket_ids: set[str] = set()
        for target in targets:
            if target.ticket_id in seen_ticket_ids:
                msg = f"Duplicate ticket_id in dispatch batch: {target.ticket_id!r}"
                raise ValueError(msg)
            seen_ticket_ids.add(target.ticket_id)

        for target in targets:
            if dry_run:
                outcomes.append(
                    ModelBuildDispatchOutcome(
                        ticket_id=target.ticket_id,
                        dispatched=True,
                        error=None,
                    )
                )
                total_dispatched += 1
                continue

            try:
                if self._publisher is not None:
                    await self._publish_delegation_request(
                        target=target,
                        correlation_id=correlation_id,
                    )
                else:
                    self._write_dispatch_manifest(
                        dispatch_path=dispatch_path,
                        target=target,
                        correlation_id=correlation_id,
                    )
                logger.info(
                    "Dispatched ticket-pipeline for %s: %s",
                    target.ticket_id,
                    target.title,
                )
                outcomes.append(
                    ModelBuildDispatchOutcome(
                        ticket_id=target.ticket_id,
                        dispatched=True,
                        error=None,
                    )
                )
                total_dispatched += 1
            except Exception as exc:  # noqa: BLE001 — boundary: catch-all converts dispatch failure to outcome record
                transport = (
                    EnumInfraTransportType.KAFKA
                    if self._publisher is not None
                    else EnumInfraTransportType.FILESYSTEM
                )
                operation = (
                    "delegation_request_publish"
                    if self._publisher is not None
                    else "dispatch_manifest_write"
                )
                error_ctx = ModelInfraErrorContext.with_correlation(
                    transport_type=transport,
                    operation=operation,
                    target_name=target.ticket_id,
                    correlation_id=correlation_id,
                    original_error_type=type(exc).__name__,
                )
                logger.warning(
                    "Failed to dispatch %s: %s (correlation_id=%s)",
                    target.ticket_id,
                    exc,
                    error_ctx.correlation_id,
                )
                emitted = emit_build_loop_friction(
                    phase="BUILDING",
                    correlation_id=correlation_id,
                    severity="high",
                    description=f"Failed to dispatch ticket-pipeline for {target.ticket_id}",
                    error_message=str(exc),
                )
                if not emitted:
                    logger.warning(
                        "emit_build_loop_friction returned False for %s — telemetry may be lost",
                        target.ticket_id,
                    )
                outcomes.append(
                    ModelBuildDispatchOutcome(
                        ticket_id=target.ticket_id,
                        dispatched=False,
                        error=str(exc),
                    )
                )
                total_failed += 1

        logger.info(
            "Build dispatch complete: %d dispatched, %d failed",
            total_dispatched,
            total_failed,
        )

        return ModelBuildDispatchResult(
            correlation_id=correlation_id,
            outcomes=tuple(outcomes),
            total_dispatched=total_dispatched,
            total_failed=total_failed,
        )

    # ------------------------------------------------------------------
    # Primary dispatch: Kafka delegation request
    # ------------------------------------------------------------------

    async def _publish_delegation_request(
        self,
        *,
        target: ModelBuildTarget,
        correlation_id: UUID,
    ) -> None:
        """Publish a delegation request for a single ticket to Kafka.

        Builds a ``ModelDelegationRequest``-shaped payload and publishes it
        to the delegation-request topic with the correct ``event_type`` so
        that ``DispatcherDelegationRequest`` routes it to the delegation
        orchestrator FSM.

        Raises:
            RuntimeError: If the publisher returns ``False`` (publish failed).
        """
        now = datetime.now(tz=UTC)
        payload: JsonType = {
            "prompt": f"Run ticket-pipeline for {target.ticket_id}",
            "task_type": "research",
            "source_session_id": None,
            "source_file_path": None,
            "correlation_id": str(correlation_id),
            "max_tokens": 4096,
            "emitted_at": now.isoformat(),
        }

        assert self._publisher is not None  # guarded by caller
        success = await self._publisher(
            event_type=_DELEGATION_EVENT_TYPE,
            payload=payload,
            correlation_id=str(correlation_id),
            topic=TOPIC_DELEGATION_REQUEST,
        )
        if not success:
            msg = (
                f"Publisher returned False for {target.ticket_id} — "
                "delegation request was not delivered"
            )
            raise RuntimeError(msg)

    # ------------------------------------------------------------------
    # Fallback dispatch: filesystem manifest
    # ------------------------------------------------------------------

    def _write_dispatch_manifest(
        self,
        *,
        dispatch_path: Path | None,
        target: ModelBuildTarget,
        correlation_id: UUID,
    ) -> None:
        """Write a dispatch manifest JSON for a single ticket.

        The manifest contains everything a downstream runner needs to spawn
        ``claude -p "Run ticket-pipeline for {ticket_id}"``.

        Raises:
            RuntimeError: If ONEX_STATE_DIR is not set.
        """
        if dispatch_path is None:
            msg = "ONEX_STATE_DIR not set — cannot write dispatch manifest"
            raise RuntimeError(msg)

        ticket_id = target.ticket_id
        if not (ticket_id.isascii() and ticket_id.replace("-", "").isalnum()):
            msg = f"Unsafe ticket_id for dispatch manifest: {ticket_id!r}"
            raise ValueError(msg)

        manifest = {
            "ticket_id": ticket_id,
            "title": target.title,
            "buildability": target.buildability.value,
            "correlation_id": str(correlation_id),
            "dispatched_at": datetime.now(tz=UTC).isoformat(),
            "status": "pending",
            "command": f'claude -p "Run ticket-pipeline for {ticket_id}"',
        }

        manifest_path = dispatch_path / f"{ticket_id}.json"
        temp_path = manifest_path.with_suffix(".json.tmp")
        temp_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        temp_path.replace(manifest_path)
        logger.debug("Wrote dispatch manifest: %s", manifest_path)
