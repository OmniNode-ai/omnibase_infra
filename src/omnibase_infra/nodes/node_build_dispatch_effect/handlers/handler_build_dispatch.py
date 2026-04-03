# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler that dispatches ticket-pipeline builds via delegation.

This is an EFFECT handler - performs external I/O (delegation dispatch).

Architectural rule: effect handlers must NOT have direct event bus access.
Instead, this handler builds delegation request payloads and returns them
in the result.  The orchestrator is responsible for publishing them to Kafka.

Fallback mechanism: when the orchestrator does not have a publisher (e.g. in
tests or when Kafka is unavailable), writes per-ticket JSON manifest files to
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
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import yaml

from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumInfraTransportType,
)
from omnibase_infra.models.errors import ModelInfraErrorContext
from omnibase_infra.nodes.node_build_dispatch_effect.models.model_build_dispatch_outcome import (
    ModelBuildDispatchOutcome,
)
from omnibase_infra.nodes.node_build_dispatch_effect.models.model_build_dispatch_result import (
    ModelBuildDispatchResult,
    ModelDelegationPayload,
)
from omnibase_infra.nodes.node_build_dispatch_effect.models.model_build_target import (
    ModelBuildTarget,
)
from omnibase_infra.utils.util_friction_emitter import emit_build_loop_friction

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resolve delegation topic from contract.yaml (single source of truth)
# ---------------------------------------------------------------------------
_CONTRACT_PATH = Path(__file__).resolve().parent.parent / "contract.yaml"
_DELEGATION_TOPIC_SUFFIX = "delegation-request"


def _load_delegation_topic() -> str:
    """Load the delegation-request publish topic from contract.yaml.

    Raises:
        RuntimeError: If contract.yaml is missing or does not declare a
            publish topic containing 'delegation-request'.
    """
    if not _CONTRACT_PATH.exists():
        msg = f"contract.yaml not found at {_CONTRACT_PATH}"
        raise RuntimeError(msg)

    with open(_CONTRACT_PATH) as fh:
        data = yaml.safe_load(fh) or {}

    event_bus = data.get("event_bus", {}) or {}
    publish_topics: list[str] = event_bus.get("publish_topics", []) or []

    for topic in publish_topics:
        if _DELEGATION_TOPIC_SUFFIX in topic:
            return topic

    msg = (
        f"contract.yaml at {_CONTRACT_PATH} does not declare a "
        f"publish topic containing {_DELEGATION_TOPIC_SUFFIX!r}"
    )
    raise RuntimeError(msg)


_TOPIC_DELEGATION_REQUEST: str = _load_delegation_topic()

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

    Primary path: builds ``ModelDelegationPayload`` objects for each ticket
    and returns them in the result.  The orchestrator publishes these to
    Kafka (architectural rule: only orchestrators may access the event bus).

    Fallback path: when ``use_filesystem_fallback=True``, writes per-ticket
    JSON manifests to the dispatch directory for ``cron-buildloop.sh``.

    Failures on individual tickets do not block other dispatches.
    """

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
        use_filesystem_fallback: bool = False,
    ) -> ModelBuildDispatchResult:
        """Dispatch builds for each target ticket.

        Args:
            correlation_id: Cycle correlation ID.
            targets: Tickets to dispatch.
            dry_run: Skip actual dispatch.
            use_filesystem_fallback: Write filesystem manifests instead of
                building delegation payloads (used when no Kafka publisher
                is available).

        Returns:
            ModelBuildDispatchResult with per-ticket outcomes and delegation
            payloads for the orchestrator to publish.
        """
        logger.info(
            "Build dispatch: %d targets (correlation_id=%s, dry_run=%s, fallback=%s)",
            len(targets),
            correlation_id,
            dry_run,
            use_filesystem_fallback,
        )

        outcomes: list[ModelBuildDispatchOutcome] = []
        delegation_payloads: list[ModelDelegationPayload] = []
        total_dispatched = 0
        total_failed = 0

        # Filesystem fallback: only needed when explicitly requested
        dispatch_path: Path | None = None
        if use_filesystem_fallback and targets and not dry_run:
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
                if use_filesystem_fallback:
                    self._write_dispatch_manifest(
                        dispatch_path=dispatch_path,
                        target=target,
                        correlation_id=correlation_id,
                    )
                else:
                    payload = self._build_delegation_payload(
                        target=target,
                        correlation_id=correlation_id,
                    )
                    delegation_payloads.append(payload)
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
                    EnumInfraTransportType.FILESYSTEM
                    if use_filesystem_fallback
                    else EnumInfraTransportType.KAFKA
                )
                operation = (
                    "dispatch_manifest_write"
                    if use_filesystem_fallback
                    else "delegation_payload_build"
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
            delegation_payloads=tuple(delegation_payloads),
        )

    # ------------------------------------------------------------------
    # Primary dispatch: build delegation payload (orchestrator publishes)
    # ------------------------------------------------------------------

    def _build_delegation_payload(
        self,
        *,
        target: ModelBuildTarget,
        correlation_id: UUID,
    ) -> ModelDelegationPayload:
        """Build a delegation request payload for a single ticket.

        Returns a ``ModelDelegationPayload`` that the orchestrator will
        publish to the delegation-request Kafka topic.
        """
        now = datetime.now(tz=UTC)
        payload: dict[str, object] = {
            "prompt": f"Run ticket-pipeline for {target.ticket_id}",
            "task_type": "research",
            "source_session_id": None,
            "source_file_path": None,
            "correlation_id": str(correlation_id),
            "max_tokens": 4096,
            "emitted_at": now.isoformat(),
        }

        return ModelDelegationPayload(
            event_type=_DELEGATION_EVENT_TYPE,
            topic=_TOPIC_DELEGATION_REQUEST,
            payload=payload,
            correlation_id=correlation_id,
        )

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
