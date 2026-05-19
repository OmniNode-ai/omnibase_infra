# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Canonical typed model for the :8085/health aggregate response shape.

This module defines the canonical response model for the ``GET /health``
endpoint served by :class:`~omnibase_infra.services.service_health.ServiceHealth`
on port 8085.  It replaces the ``[provisional-field-source]`` annotation in
``docs/tracking/2026-04-19-runtime-state-inventory.md §E`` by naming each
response field and citing its authoritative source.

Field provenance
----------------
The ``/health`` response is assembled in two layers:

1. **``RuntimeHostProcess.health_check()``**
   (``omnibase_infra/src/omnibase_infra/runtime/service_runtime_host_process.py``,
   method ``health_check``, return dict literal at lines 5153-5179)
   — owns: ``healthy``, ``degraded``, ``startup_in_progress``, ``is_running``,
   ``is_draining``, ``pending_message_count``, ``max_concurrent_handlers``,
   ``handler_pool_size``, ``in_flight_tasks``, ``batch_response_enabled``,
   ``batch_response_pending``, ``event_bus``, ``event_bus_healthy``,
   ``failed_handlers``, ``skipped_handlers``, ``registered_handlers``,
   ``handlers``, ``handler_pools``, ``no_handlers_registered``,
   ``config_prefetch_status``, ``local_ingress``, ``components``.

2. **``ServiceHealth._handle_health()`` enrichment**
   (``omnibase_infra/src/omnibase_infra/services/service_health.py``,
   method ``_handle_health``, enrichment block at lines 1064-1076)
   — appends: ``degraded`` (overwrite), ``startup_in_progress`` (overwrite),
   ``components`` (overwrite with typed dicts).

The ``event_bus`` sub-dict is produced by
``EventBusKafka.health_check()``
(``omnibase_infra/src/omnibase_infra/event_bus/event_bus_kafka.py``,
method ``health_check``, return dict at lines 2200-2210),
which returns the fields captured in
:class:`~omnibase_infra.runtime.models.model_event_bus_aggregate_health.ModelEventBusAggregateHealth`.

OMN-9266: Cite ServiceHealth aggregate response-shape model for :8085/health.

Example:
    >>> from omnibase_infra.runtime.models.model_runtime_aggregate_health import (
    ...     ModelRuntimeAggregateHealth,
    ... )
    >>> from omnibase_infra.runtime.models.model_event_bus_aggregate_health import (
    ...     ModelEventBusAggregateHealth,
    ... )
    >>>
    >>> # Round-trip a raw health_check() dict through the model for validation
    >>> raw = {
    ...     "healthy": True,
    ...     "degraded": False,
    ...     "startup_in_progress": False,
    ...     "is_running": True,
    ...     "is_draining": False,
    ...     "pending_message_count": 0,
    ...     "max_concurrent_handlers": 10,
    ...     "handler_pool_size": 5,
    ...     "in_flight_tasks": 0,
    ...     "batch_response_enabled": False,
    ...     "batch_response_pending": 0,
    ...     "event_bus": {
    ...         "healthy": True,
    ...         "started": True,
    ...         "environment": "prod",
    ...         "bootstrap_servers": "redpanda:9092",
    ...         "circuit_state": "closed",
    ...         "subscriber_count": 374,
    ...         "topic_count": 42,
    ...         "consumer_count": 12,
    ...     },
    ...     "event_bus_healthy": True,
    ...     "failed_handlers": {},
    ...     "skipped_handlers": {},
    ...     "registered_handlers": [],
    ...     "handlers": {},
    ...     "handler_pools": {},
    ...     "no_handlers_registered": False,
    ...     "config_prefetch_status": "ok",
    ...     "local_ingress": {"enabled": False, "running": False},
    ...     "components": [],
    ... }
    >>> model = ModelRuntimeAggregateHealth.model_validate(raw, strict=False)
    >>> model.is_running
    True
    >>> model.event_bus.subscriber_count
    374
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.types import JsonType
from omnibase_infra.runtime.models.model_event_bus_aggregate_health import (
    ModelEventBusAggregateHealth,
)


class ModelRuntimeAggregateHealth(BaseModel):
    """Canonical typed model for the ``GET /health`` aggregate response shape.

    This is the source-of-truth model for the JSON body returned by
    ``ServiceHealth._handle_health()`` at ``:8085/health``.  It supersedes
    the ``[provisional-field-source]`` annotation in
    ``docs/tracking/2026-04-19-runtime-state-inventory.md §E``.

    Construction
    ------------
    The response is assembled from two authoritative sources:

    * ``RuntimeHostProcess.health_check()`` — produces the outer dict
      (``service_runtime_host_process.py``, ``health_check`` return literal,
      lines 5153-5179).
    * ``ServiceHealth._handle_health()`` enrichment — overwrites ``degraded``,
      ``startup_in_progress``, and replaces the raw ``components`` list with
      typed :class:`~omnibase_infra.runtime.models.model_component_health.ModelComponentHealth`
      dicts (``service_health.py``, lines 1064-1076).

    Validation
    ----------
    ``extra="allow"`` is intentional: future fields added to
    ``RuntimeHostProcess.health_check()`` must not break existing consumers of
    this model.  Strict validation gates live in unit tests against the known
    field set.

    Attributes:
        healthy: ``True`` only when the runtime is running, the event bus is
            healthy, no handlers failed to instantiate, all registered handlers
            are healthy, and at least one handler is registered.
        degraded: ``True`` when running with failed handlers (reduced capacity)
            or startup is in progress with a live event bus.
        startup_in_progress: ``True`` when ``_is_starting`` and not yet
            ``_is_running`` but the event bus is already healthy.
        is_running: Whether ``RuntimeHostProcess._is_running`` is ``True``
            (set after ``start()`` completes).
        is_draining: Whether the runtime is in its graceful-shutdown drain phase
            (``_is_draining``).  Load balancers should remove the instance from
            rotation when this is ``True``.
        pending_message_count: Number of in-flight messages currently being
            processed (``_pending_message_count``).
        max_concurrent_handlers: Configured concurrency ceiling for handler
            dispatch (``_max_concurrent_handlers``).
        handler_pool_size: Configured pool size per handler type
            (``_handler_pool_size``).
        in_flight_tasks: Length of ``_in_flight_tasks`` set at check time.
        batch_response_enabled: Whether a batch-response publisher is active.
        batch_response_pending: Number of pending batch-response messages
            (0 when ``batch_response_enabled`` is ``False``).
        event_bus: Nested event-bus health dict from
            ``EventBusKafka.health_check()``; typed as
            :class:`~omnibase_infra.runtime.models.model_event_bus_aggregate_health.ModelEventBusAggregateHealth`.
        event_bus_healthy: Boolean convenience flag extracted from
            ``event_bus.healthy``.
        failed_handlers: Map of handler type to error message for handlers that
            raised during ``start()``.
        skipped_handlers: Map of handler type to reason for handlers that were
            intentionally skipped (e.g. disabled via config).
        registered_handlers: List of handler types that started successfully.
        handlers: Per-handler health-check results (handler type to details dict).
        handler_pools: Per-pool health-check results from
            ``HandlerPool.health_check()`` (pool type to metrics dict).
        no_handlers_registered: ``True`` when no handlers are registered — a
            critical configuration error (runtime cannot process events).
        config_prefetch_status: Infisical config prefetch outcome.
            Values: ``"pending"``, ``"skipped"``, ``"ok"``,
            ``"degraded_no_requirements"``, ``"degraded_error"``.
        local_ingress: Typed health for the local-ingress server (Unix-domain
            socket or in-process path); serialised from
            :class:`~omnibase_infra.runtime.models.model_local_runtime_ingress_health.ModelLocalRuntimeIngressHealth`.
        components: List of per-component health dicts appended by
            ``ServiceHealth._build_component_health()`` and additional infra
            checks (``published_events_map``, ``local_ingress_component``).
    """

    model_config = ConfigDict(
        frozen=True,
        extra="allow",
        from_attributes=True,
    )

    # --- Core liveness flags ---
    healthy: bool = Field(
        ...,
        description=(
            "True only when is_running, event_bus_healthy, no failed handlers, "
            "all registered handlers healthy, and at least one handler registered"
        ),
    )
    degraded: bool = Field(
        ...,
        description=(
            "True when running with failed handlers or startup_in_progress is True"
        ),
    )
    startup_in_progress: bool = Field(
        ...,
        description=(
            "True when _is_starting=True, _is_running=False, and event bus is already healthy"
        ),
    )

    # --- Runtime state flags (OMN-9266 previously-provisional fields) ---
    is_running: bool = Field(
        ...,
        description=(
            "Source: RuntimeHostProcess._is_running — set True after start() completes, "
            "set False after stop(). Previously tagged [provisional-field-source]."
        ),
    )
    is_draining: bool = Field(
        ...,
        description=(
            "Source: RuntimeHostProcess._is_draining — True during graceful-shutdown "
            "drain window. Previously tagged [provisional-field-source]."
        ),
    )
    pending_message_count: int = Field(
        ...,
        ge=0,
        description=(
            "Source: RuntimeHostProcess._pending_message_count — in-flight messages. "
            "Previously tagged [provisional-field-source]."
        ),
    )

    # --- Concurrency metrics ---
    max_concurrent_handlers: int = Field(
        ...,
        ge=0,
        description="Configured concurrency ceiling for handler dispatch",
    )
    handler_pool_size: int = Field(
        ...,
        ge=0,
        description="Configured pool size per handler type",
    )
    in_flight_tasks: int = Field(
        ...,
        ge=0,
        description="Number of in-flight asyncio tasks at check time",
    )

    # --- Batch publisher ---
    batch_response_enabled: bool = Field(
        ...,
        description="Whether a batch-response publisher is active",
    )
    batch_response_pending: int = Field(
        ...,
        ge=0,
        description="Pending batch-response message count (0 when disabled)",
    )

    # --- Event bus sub-shape ---
    event_bus: ModelEventBusAggregateHealth = Field(
        ...,
        description=(
            "Source: EventBusKafka.health_check() return dict. "
            "Contains circuit_state, subscriber_count, topic_count, consumer_count. "
            "Previously tagged [provisional-field-source] for circuit_state."
        ),
    )
    event_bus_healthy: bool = Field(
        ...,
        description="Boolean extracted from event_bus.healthy for convenience",
    )

    # --- Handler registry ---
    failed_handlers: dict[str, JsonType] = Field(
        default_factory=dict,
        description="Map of handler_type -> error message for handlers that failed during start()",
    )
    skipped_handlers: dict[str, JsonType] = Field(
        default_factory=dict,
        description="Map of handler_type -> reason for intentionally skipped handlers",
    )
    registered_handlers: list[str] = Field(
        default_factory=list,
        description="Handler types that started successfully",
    )

    # --- Per-handler health results (OMN-9266 previously-provisional field) ---
    handlers: dict[str, JsonType] = Field(
        default_factory=dict,
        description=(
            "Source: per-handler health_check() results keyed by handler type. "
            "Previously tagged [provisional-field-source] for handler_pools."
        ),
    )
    handler_pools: dict[str, JsonType] = Field(
        default_factory=dict,
        description=(
            "Source: HandlerPool.health_check() results keyed by pool type. "
            "Previously tagged [provisional-field-source]."
        ),
    )
    no_handlers_registered: bool = Field(
        ...,
        description=(
            "True when no handlers registered — critical config error; "
            "runtime cannot process any events"
        ),
    )

    # --- Config prefetch ---
    config_prefetch_status: str = Field(
        ...,
        description=(
            "Infisical config prefetch outcome. "
            "Values: 'pending', 'skipped', 'ok', 'degraded_no_requirements', 'degraded_error'"
        ),
    )

    # --- Local ingress ---
    local_ingress: dict[str, JsonType] = Field(
        default_factory=dict,
        description=(
            "Serialised ModelLocalRuntimeIngressHealth — enabled, running, socket_path"
        ),
    )

    # --- Components list ---
    components: list[dict[str, JsonType]] = Field(
        default_factory=list,
        description=(
            "Per-component health dicts from ServiceHealth._build_component_health() "
            "plus published_events_map and local_ingress_component checks"
        ),
    )


__all__: list[str] = ["ModelRuntimeAggregateHealth"]
