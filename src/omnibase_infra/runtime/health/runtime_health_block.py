# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Projection of the runtime health monitor verdict onto the HTTP health payload (OMN-15217).

``ServiceRuntimeHealthMonitor`` computes the runtime's *semantic* health —
contract-discovery errors, consumer-group coverage, topic coverage — every
``RUNTIME_HEALTH_CHECK_INTERVAL`` seconds. Before OMN-15217 that verdict only
reached the container logs and the ``runtime-health-check.v1`` Kafka topic; it
was never joined to the ``/health`` HTTP payload that Docker, operators, and
promotion gates actually read.

The observed consequence (stability lane, 2026-07-27T12:58Z): ``/health``
returned ``{"status": "healthy", "details": {"healthy": true, "degraded":
false}}`` with HTTP 200 while the monitor logged
``status=DEGRADED ... errors=4`` every five minutes. Both the HTTP status code
*and* the payload body were green, so no consumer — however carefully it parsed
the response — could see the degradation.

This module is the single seam that closes that gap:

* :func:`build_runtime_health_block` renders the monitor's latest verdict into
  the ``details.runtime_health`` block of the ``/health`` payload.
* :func:`fold_runtime_verdict_into_status` degrades the payload's top-level
  ``status`` field when the monitor reports DEGRADED/CRITICAL.

Deliberate non-change: the HTTP *status code* is untouched. ``/health`` is also
the liveness probe watched by ``autoheal``; a restart-immune degradation (four
contracts that fail to import will fail to import again after a restart) must
not turn into a restart loop. Honest body, unchanged liveness code. Container
health strictness is a separate, per-lane opt-in — see
:mod:`omnibase_infra.runtime.health.container_healthcheck`.

.. versionadded:: 0.39.0
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal, cast

from omnibase_core.types import JsonType

if TYPE_CHECKING:
    from omnibase_infra.event_bus.enum_runtime_readiness_state import (
        EnumRuntimeReadinessState,
    )
    from omnibase_infra.models.health.model_runtime_health_check_event import (
        ModelRuntimeHealthCheckEvent,
    )

# Key under ``/health`` -> ``details`` carrying the monitor verdict. Consumers
# (container healthcheck, promotion gate, dashboards) key off this constant
# rather than a literal so the seam has exactly one name.
RUNTIME_HEALTH_DETAIL_KEY = "runtime_health"

_SEMANTIC_STATUS = Literal["HEALTHY", "DEGRADED", "CRITICAL"]

# Max number of dimension entries rendered into the payload. The monitor emits
# a small fixed set today; the cap keeps a future fan-out from bloating a
# response served on every Docker probe interval.
_MAX_DIMENSIONS = 32


def build_runtime_health_block(
    event: ModelRuntimeHealthCheckEvent | None,
    *,
    now: datetime | None = None,
) -> dict[str, JsonType] | None:
    """Render the monitor's latest verdict as a ``/health`` details block.

    Args:
        event: The most recent health-check event, or ``None`` when the monitor
            has not completed a cycle yet (or is not running in this profile).
        now: Injected clock for deterministic tests. Defaults to ``datetime.now(UTC)``.

    Returns:
        A JSON-serializable block, or ``None`` when no verdict exists yet.
        ``None`` is a distinct state from HEALTHY and consumers that require a
        verdict must treat it as unknown, never as healthy.
    """
    if event is None:
        return None

    current = now or datetime.now(UTC)
    observed_at = event.timestamp
    if observed_at.tzinfo is None:
        observed_at = observed_at.replace(tzinfo=UTC)
    age_seconds = max(0.0, (current - observed_at).total_seconds())

    block: dict[str, JsonType] = {
        "status": event.status,
        "observed_at": observed_at.isoformat(),
        "age_seconds": round(age_seconds, 3),
        "contract_count": event.contract_count,
        "discovery_error_count": event.discovery_error_count,
        "consumer_group_count": event.consumer_group_count,
        "empty_consumer_group_count": event.empty_consumer_group_count,
        "subscribe_topic_count": event.subscribe_topic_count,
        "uncovered_topic_count": event.uncovered_topic_count,
        "dimensions": cast(
            "JsonType",
            [
                {
                    "name": dimension.name,
                    "status": dimension.status,
                    "detail": dimension.detail,
                }
                for dimension in event.dimensions[:_MAX_DIMENSIONS]
            ],
        ),
    }
    return block


def fold_runtime_verdict_into_status(
    payload_status: Literal["healthy", "degraded", "unhealthy"],
    verdict_status: str | None,
) -> Literal["healthy", "degraded", "unhealthy"]:
    """Degrade the payload status when the monitor reports a worse verdict.

    The runtime process can be perfectly alive (handlers registered, event bus
    connected) while the runtime is semantically degraded (contracts failing to
    load). The reported status is the worse of the two.

    Args:
        payload_status: Status derived from ``RuntimeHostProcess.health_check()``.
        verdict_status: ``HEALTHY`` / ``DEGRADED`` / ``CRITICAL`` from the
            monitor, or ``None`` when no verdict exists yet (status unchanged —
            absence is not evidence of degradation, and consumers that need a
            verdict assert its presence explicitly).

    Returns:
        The worse of the two statuses.
    """
    if payload_status == "unhealthy" or verdict_status is None:
        return payload_status
    if verdict_status == "CRITICAL":
        return "unhealthy"
    if verdict_status == "DEGRADED":
        return "degraded"
    return payload_status


def fold_attach_readiness_into_status(
    payload_status: Literal["healthy", "degraded", "unhealthy"],
    readiness_state: EnumRuntimeReadinessState | None,
) -> Literal["healthy", "degraded", "unhealthy"]:
    """Degrade the payload status when a boot-wired consumer failed to attach.

    OMN-15642. Before this, ``ModelRuntimeAttachReadiness`` (OMN-15512) reached
    only ``details.components.runtime_wiring`` — a nested detail nothing
    upstream reads — while the top-level ``status`` field stayed ``"healthy"``
    with HTTP 200. That is the exact class of gap OMN-15217 already closed for
    the ``ServiceRuntimeHealthMonitor`` verdict (see the module docstring): a
    runtime can boot fully, with EVERY rollout/digest/dashboard/staleness gate
    green, while one Kafka consumer contract silently never attaches its
    subscription (``_require_contract_dispatcher_scope`` et al. raise inside
    ``subscribe_wired_contract_topics``, caught per-contract and downgraded to
    a non-fatal ``ModelContractAttachResult`` — see
    ``omnibase_infra.event_bus.model_runtime_attach_readiness``). A projection
    or reducer that stops consuming produces zero new rows for whatever it
    writes, invisible to every liveness/rollout check that never queries the
    wiring detail. OMN-15642 observed exactly this shape live on onex-dev:
    steps 30-36 of ``deploy-onex-staging`` (rollout, digest triple-match,
    dashboard health, post-deploy verification, staleness) all passed while a
    correlation-keyed projection and the unified system-event stream stayed
    silently empty, surfacing only ~60s later at the terminal business-proof
    gate. Folding the aggregate into ``status`` here means the SAME class of
    silent drop is visible at ``/health`` immediately, not only after a
    downstream consumer happens to probe by correlation_id.

    Args:
        payload_status: Status derived from ``RuntimeHostProcess.health_check()``,
            already folded with :func:`fold_runtime_verdict_into_status`.
        readiness_state: The boot attach-readiness aggregate's tri-state, or
            ``None`` before the kernel has attached it (status unchanged --
            absence is not evidence of degradation, matching
            :func:`fold_runtime_verdict_into_status`'s ``None`` handling).

    Returns:
        The worse of the two statuses. Deliberately does NOT change the HTTP
        status code -- see the module docstring's "Deliberate non-change".
    """
    if payload_status == "unhealthy" or readiness_state is None:
        return payload_status
    # Local import: EnumRuntimeReadinessState is TYPE_CHECKING-only above so
    # this module carries no runtime import-time dependency on the event_bus
    # package for callers that never pass a readiness_state.
    from omnibase_infra.event_bus.enum_runtime_readiness_state import (
        EnumRuntimeReadinessState as _EnumRuntimeReadinessState,
    )

    if readiness_state is _EnumRuntimeReadinessState.FAILED:
        return "unhealthy"
    if readiness_state is _EnumRuntimeReadinessState.DEGRADED:
        return "degraded"
    return payload_status


__all__: list[str] = [
    "RUNTIME_HEALTH_DETAIL_KEY",
    "build_runtime_health_block",
    "fold_attach_readiness_into_status",
    "fold_runtime_verdict_into_status",
]
