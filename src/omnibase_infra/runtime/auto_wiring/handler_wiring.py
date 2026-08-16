# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# ruff: noqa: TRY400
# TRY400 disabled: logger.error is intentional to avoid leaking sensitive data in stack traces
"""Handler auto-wiring engine for OMN-7654.

Takes a :class:`ModelAutoWiringManifest` produced by contract auto-discovery
and wires handlers into the :class:`MessageDispatchEngine`:

1. Import handler modules from ``handler_routing`` paths in each contract.
2. Create dispatch callbacks that delegate to the imported handler.
3. Register routes on :class:`MessageDispatchEngine`.
4. Subscribe to Kafka topics via the event bus.
5. Detect duplicate topic ownership at package, handler, and intra-package levels.
6. Return a :class:`ModelAutoWiringReport` with per-contract outcomes.

This module performs I/O (module imports, Kafka subscriptions) — it is NOT pure.

CI gate: any PR touching this module MUST satisfy the runtime-startup gate defined in
``CLAUDE.md`` § "Runtime Startup is a First-Class CI Gate (OMN-9126)" (repo root).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import hashlib
import importlib
import inspect
import json
import logging
import math
import os
import re
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Protocol,
    cast,
    get_args,
    get_origin,
    runtime_checkable,
)
from uuid import UUID, uuid4

from pydantic import AliasChoices, AliasPath, BaseModel, ValidationError

from omnibase_core.enums.enum_core_error_code import EnumCoreErrorCode
from omnibase_core.enums.enum_database_grant_object_type import (
    EnumDatabaseGrantObjectType,
)
from omnibase_core.enums.enum_database_privilege import EnumDatabasePrivilege
from omnibase_core.enums.enum_database_schema_domain import EnumDatabaseSchemaDomain
from omnibase_core.enums.enum_handler_resolution_outcome import (
    EnumHandlerResolutionOutcome,
)
from omnibase_core.enums.enum_node_kind import EnumNodeKind
from omnibase_core.models.contracts.subcontracts.model_db_table_declaration import (
    ModelDbTableDeclaration,
)
from omnibase_core.models.core.model_deployment_topology import ModelDeploymentTopology
from omnibase_core.models.core.model_deployment_topology_database import (
    ModelDeploymentTopologyDatabase,
)
from omnibase_core.models.errors import ModelOnexError
from omnibase_core.models.resolver.model_handler_resolver_context import (
    ModelHandlerResolverContext,
)
from omnibase_core.protocols.event_bus.protocol_event_bus_subscriber import (
    ProtocolEventBusSubscriber,
)
from omnibase_core.runtime.runtime_fanout_resolver import resolve_published_topic
from omnibase_core.services.service_handler_resolver import ServiceHandlerResolver
from omnibase_core.services.service_local_handler_ownership_query import (
    ServiceLocalHandlerOwnershipQuery,
)
from omnibase_infra.errors import TopicReplicationPolicyError
from omnibase_infra.event_bus.enum_contract_attach_status import (
    EnumContractAttachStatus,
)
from omnibase_infra.event_bus.enum_topic_readiness_status import (
    EnumTopicReadinessStatus,
)
from omnibase_infra.event_bus.model_contract_attach_result import (
    ModelContractAttachResult,
)
from omnibase_infra.event_bus.model_topic_readiness_config import (
    ModelTopicReadinessConfig,
)
from omnibase_infra.event_bus.model_topic_set_readiness import (
    ModelTopicSetReadiness,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.services.service_gateway_topic_transform import (
    resolve_tenant_from_wire_topic,
)
from omnibase_infra.protocols.protocol_dispatch_result_applier import (
    ProtocolDispatchResultApplier,
)
from omnibase_infra.protocols.protocol_event_bus_like import ProtocolEventBusLike
from omnibase_infra.protocols.protocol_topic_provisioner import (
    ProtocolTopicProvisioner,
)
from omnibase_infra.runtime.auto_wiring.enum_quarantine_reason import (
    EnumQuarantineReason,
)
from omnibase_infra.runtime.auto_wiring.fanout_seam import (
    check_fanout_publish_coverage,
    is_fanout_sequence,
    normalize_fanout_sequence,
)
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelDiscoveredContract,
    ModelHandlerRef,
    ModelHandlerRoutingEntry,
)
from omnibase_infra.runtime.auto_wiring.report import (
    EnumWiringOutcome,
    ModelAutoWiringReport,
    ModelContractWiringResult,
    ModelDuplicateTopicOwnership,
    ModelQuarantinedWiring,
    ModelSkippedEntry,
    ModelWiringOutcome,
)
from omnibase_infra.runtime.contract_terminal_events import (
    envelope_terminal_payload,
    load_terminal_event_topics,
)
from omnibase_infra.runtime.dispatch_envelope_context import (
    current_dispatch_envelope,
    current_projection_tenant_authority,
)
from omnibase_infra.runtime.models.model_postgres_pool_config import (
    ModelPostgresPoolConfig,
)
from omnibase_infra.runtime.projection_tenant_authority import (
    VerifiedProjectionTenantAuthority,
    assert_projection_tenant_authority_matches_event,
    parse_canonical_tenant_uuid,
)
from omnibase_infra.runtime.protocols.protocol_contract_scoped_dispatch_engine import (
    ProtocolContractScopedDispatchEngine,
)
from omnibase_infra.runtime.providers.provider_postgres_pool import ProviderPostgresPool
from omnibase_infra.runtime.state_io.state_store_adapter import (
    CONTEXTVAR_STATE_IO_ROWS,
    StateIoUnconfiguredError,
    StateStoreAdapter,
)
from omnibase_infra.shared.tenant_stamp import stamp_verified_tenant_slug
from omnibase_infra.tools.contract_topic_extractor import read_projection_api_topics
from omnibase_infra.topology.physical_schema_mapping import (
    physical_grant_schema_for_table,
)
from omnibase_infra.utils.util_retry_optimistic import (
    OptimisticConflictError,
    retry_on_optimistic_conflict,
)
from omnibase_infra.utils.util_topic_event_type import derive_event_type_from_topic


class BoundaryDlqNotPersistedError(Exception):
    """Marks a boundary failure whose DLQ write was NOT confirmed durable.

    OMN-14498 (Lane C): ``_route_swallowed_exception`` used to return normally
    even when ``_publish_raw_to_dlq`` reported non-persistence via its
    documented ``False`` return. A callback that returns normally IS an ACK --
    ``EventBusKafka._dispatch_to_subscriber`` reads "no exception" as success
    and lets the offset advance -- so the message was acknowledged while
    existing nowhere durable. That made the OMN-15232 rewind path
    (``_rewind_after_unpersisted_dlq``) structurally unreachable for every
    auto-wired handler: the boundary swallowed its own failure before the
    consumer loop could see it.

    Raising this type in that ONE case (DLQ enabled AND the write confirmed
    non-durable) restores the invariant a NACK is supposed to carry: the
    offset is withheld and Kafka redelivers. It is deliberately NOT raised
    when the DLQ write succeeded, when the flag is off, or when no
    DLQ-capable bus is wired -- those paths keep their prior semantics.
    """

    def __init__(self, topic: str, correlation_id: object, cause: Exception) -> None:
        super().__init__(
            f"boundary DLQ write not persisted; offset must not advance "
            f"(topic={topic} correlation_id={correlation_id} "
            f"cause={type(cause).__name__})"
        )
        self.topic = topic
        self.correlation_id = correlation_id
        self.cause = cause


class BoundaryPublishError(Exception):
    """Marks a result-applier (publish) failure the outbox boundary must PROPAGATE.

    OMN-14403 §4.3: on the state_io / in-row-outbox path, a publish failure at
    the RESULT-APPLIER layer (the no-bus / external-applier shape) must NOT be
    log-and-discarded. The auto-wired boundary (`_make_event_bus_callback`)
    tags an applier failure with this type ONLY when
    `propagate_publish_failures` is set (state_io contracts), so the outer
    handler re-raises it instead of swallowing. Non-outbox contracts never set
    the flag → behavior is unchanged.

    OMN-14600 CORRECTION: this type is distinct from a conflict-retry
    exhaustion (`OptimisticConflictError`) raised INSIDE the state_io
    dispatcher itself — that exception is absorbed by
    `MessageDispatchEngine.dispatch()`'s per-dispatcher catch-all before it
    ever reaches this boundary at all, so it does NOT redeliver on this
    runtime (see the detailed note at `_make_event_bus_callback`'s
    `except (OptimisticConflictError, BoundaryPublishError)` clause). The
    state_io in_flight-lock branch self-heals inline for that reason instead
    of depending on redelivery.
    """


class HandlerDispatchFailureError(Exception):
    """A handler/coercion failure the engine reported as a FAILED dispatch RESULT.

    OMN-14716. ``MessageDispatchEngine.dispatch()`` wraps every dispatcher call in
    a catch-all (``except Exception``) that records the error and RETURNS a
    ``HANDLER_ERROR`` result instead of re-raising. A def-B handler crash (or a
    boundary coercion failure) therefore never propagates to the consume boundary
    as an exception — it arrives as a FAILED result, and
    ``DispatchResultApplier.apply()`` silently skips a non-SUCCESS result that
    carries no applicable output. The message then vanishes at HWM=0 with nothing
    terminalized (the .201 finding-aggregator incident).

    ``_raise_if_silent_dispatch_failure`` raises this from
    ``_dispatch_with_bounded_retry`` for exactly that shape so the failure flows
    into the SAME ``_route_swallowed_exception`` path a raised handler exception
    would take (loud structured metric log + best-effort DLQ under
    ``ONEX_BOUNDARY_DLQ_ENABLED``). It is classified non-retryable at that
    boundary: the FAILED result is deterministic, so retrying only burns the
    backoff budget.
    """


from omnibase_spi.protocols.runtime.protocol_handler_ownership_query import (
    ProtocolHandlerOwnershipQuery,
)

if TYPE_CHECKING:
    from omnibase_core.models.dispatch.model_dispatch_route import ModelDispatchRoute
    from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
    from omnibase_core.models.projectors.model_projection_intent import (
        ModelProjectionIntent,
    )
    from omnibase_infra.enums import EnumMessageCategory
    from omnibase_infra.models.dispatch.model_dispatch_result import (
        ModelDispatchResult,
    )
    from omnibase_infra.protocols.protocol_pattern_b_broker_transport import (
        ProtocolPatternBBrokerTransport,
    )
    from omnibase_infra.runtime.service_terminal_event_consumer import (
        TerminalEventConsumer,
    )
    from omnibase_spi.protocols.runtime import ProtocolDispatchEngine

logger = logging.getLogger(__name__)

# Matches DSNs, URLs, and connection strings that may contain credentials.
_SENSITIVE_PATTERN = re.compile(
    r"(?:postgresql|postgres|mysql|redis|amqp|kafka|mongodb|http|https)://\S*",
    re.IGNORECASE,
)
_STRICT_DISPATCHER_COVERAGE_ENV = "ONEX_STRICT_DISPATCHER_COVERAGE"
# OMN-14507: staged rollout for the auto-wired consume boundary DLQ fix.
# DEFAULT OFF (warn-first) -- see _boundary_dlq_enabled() below.
_BOUNDARY_DLQ_ENV = "ONEX_BOUNDARY_DLQ_ENABLED"
_BOUNDARY_DLQ_MAX_ATTEMPTS = 3
_BOUNDARY_DLQ_RETRY_BACKOFF_SECONDS: tuple[float, ...] = (0.1, 0.4)

# OMN-14498 / OMN-15029: strong references to the detached DLQ-routing tasks
# `_make_sync_event_publisher` schedules for a failed fire-and-forget publish
# (see `_route_sync_publisher_failure`). Without this, the Task object has no
# other referrer once `_log_publish_failure`'s local `dlq_task` variable goes
# out of scope, which is a documented footgun for tasks scheduled via
# `loop.create_task` outside of a structured-concurrency context (asyncio may
# garbage-collect a task with no external references before it completes).
# Discarded via its own done_callback once it finishes.
_DLQ_ROUTING_TASKS: set[asyncio.Task[None]] = set()

# OMN-14551: alertable counter for the boundary's one true loss window --
# retry budget exhausted AND the best-effort DLQ publish itself also failed
# (the ``message_lost=true`` branch of ``_route_swallowed_exception`` below).
# Prior to this, that path was a structured-but-unalertable ``logger.error``
# line only (forbid-verify residual ask: "a greppable log won't [page]").
# Emitted through the same process-global default Prometheus registry the
# runtime's existing metrics surface (``SinkMetricsPrometheus`` /
# ``HandlerMetricsPrometheus``'s ``/metrics`` scrape endpoint, OMN-9121
# observability profile) already exports -- mirrors the established
# module-level ``prometheus_client.Counter`` idiom in ``handler_db.py``
# (OMN-1366) rather than inventing a new observability surface. Scoped to
# this one loss-path emission only -- not a refactor of the boundary's
# other logging.
try:
    from prometheus_client import Counter as _PrometheusCounter

    _BOUNDARY_MESSAGE_LOST_COUNTER: _PrometheusCounter | None = _PrometheusCounter(
        "onex_boundary_message_lost_total",
        "Count of auto-wiring boundary messages genuinely lost: handler "
        "exception survived the bounded retry AND the best-effort DLQ "
        "publish itself also failed. MUST page -- unlike "
        "boundary_swallow_prevented (DLQ-routed, the success path), this "
        "counter incrementing means the message is gone.",
        ["topic", "error_type"],
    )
except (ImportError, ValueError):
    # ImportError: prometheus_client not installed (graceful degradation,
    # matches handler_db.py). ValueError: duplicate registration under
    # pytest-xdist/module-reimport -- idempotent fallback, not fatal.
    _BOUNDARY_MESSAGE_LOST_COUNTER = None
# OMN-14600: the state_io outbox recovery sweep (re-publish + finalize any row
# whose batch is committed but never finalized) originally ran exactly once
# per adapter lifetime (first live dispatch only) -- a row stranded later in a
# long-lived process's life was NOT re-scanned until the next boot/redeploy.
# Gating on elapsed wall-clock time instead of a boolean makes the sweep
# periodic: it re-runs opportunistically on the next dispatch once this many
# seconds have passed since the last run. select_recoverable_batches() is a
# cheap empty partial-index scan in steady state (docstring above), so a
# 30s interval costs nothing while bounding the self-heal window.
_STATE_IO_RECOVERY_SWEEP_INTERVAL_SECONDS = 30.0
# OMN-14721: terminal FSM states for the state_io emission-completeness guard.
# Mirrors the adapter's give-up sweep predicate (state_store_adapter.py
# recover_stale_rows: ``state NOT IN ('COMPLETED', 'FAILED')``) and migration
# 090's partial staleness index. A fresh seed into any NON-terminal state MUST
# carry a durable emission (a pending_emissions batch OR an in_flight marker) —
# a fresh non-terminal row committed with neither is structurally unrecoverable
# by ``select_recoverable_batches`` and silently strands the workflow.
_STATE_IO_TERMINAL_STATE_NAMES = frozenset({"COMPLETED", "FAILED"})
_TOPIC_MIGRATION_EXECUTOR_DEPS = frozenset({"provisioner", "drain_proof_gate"})
_DELEGATION_INFERENCE_INTENT_MODULE = "omnibase_core.models.delegation.wire"
_DELEGATION_INFERENCE_INTENT_NAME = "ModelInferenceIntent"
_DELEGATION_INFERENCE_INTENT_DISCRIMINATOR = "llm_inference"
_LEDGER_DB_DSN_ENV_VARS: tuple[str, ...] = ("OMNIBASE_INFRA_DB_URL", "DATABASE_URL")
# OMN-14600: pre-fix rows committed by the delegation orchestrator's OLD
# bespoke terminal carrier. Those entries recorded the CARRIER's own
# module/class_name (never the "topic" entry key -- that field did not exist
# yet) and stored its own dump verbatim: {"topic": <str>, "payload":
# {...ModelDelegationResult fields...}}. ModelDelegationEventEnvelope was the
# ONLY payload type this carrier ever wrapped in this codebase (the
# delegation orchestrator's single terminal-emit site), so the inner class is
# hardcoded here rather than re-derived generically -- there is no other
# legacy-shaped carrier to generalize for.
_LEGACY_DELEGATION_ENVELOPE_CLASS_NAME = "ModelDelegationEventEnvelope"
_LEGACY_DELEGATION_RESULT_MODULE = (
    "omnibase_core.models.delegation.wire.model_delegation_result"
)
_LEGACY_DELEGATION_RESULT_NAME = "ModelDelegationResult"


def _legacy_delegation_envelope_unwrap(
    entry: dict[str, object],
) -> tuple[str, dict[str, object]] | None:
    """Detect + unwrap a pre-fix ``ModelDelegationEventEnvelope`` outbox entry.

    Returns ``(topic, inner_payload_dict)`` when ``entry`` is the legacy
    shape (see module comment above), else ``None``. A row healed once by
    this path is re-persisted with the NEW shape by its own
    ``_finalize_outbox_row`` call, so this is a one-time migration read, not
    a permanent dual-shape burden.
    """
    if entry.get("class_name") != _LEGACY_DELEGATION_ENVELOPE_CLASS_NAME:
        return None
    if entry.get("topic"):
        # Already carries the new-shape topic key -- not the legacy shape.
        return None
    stored_payload = entry.get("payload")
    if not isinstance(stored_payload, dict):
        return None
    topic = stored_payload.get("topic")
    inner = stored_payload.get("payload")
    if isinstance(topic, str) and topic and isinstance(inner, dict):
        return topic, inner
    return None


# OMN-14403 P3a §6ii — the def-B multi-event (fan-out) publish seam. Default OFF;
# flipped in a separate PR once every wired fan-out handler is coverage-clean (repo
# rule: a gate that tightens acceptance ships behind an env flag, default OFF). A
# Sequence[BaseModel] return stops being silently dropped to output_events=[] /
# SUCCESS and becomes the published batch. This is the ONE env read for the seam
# (the OMN-11069 env-read gate approves this module); the pure logic + the mirror
# read on the RuntimeLocal path (LocalRuntimeBusAdapter) both gate on this same
# flag so the two runtimes agree. NOTE: this PR is §6ii ONLY — the §8.1
# causation/tenant carry (seam_apply_context) is a separate lane.
ENV_MULTI_EVENT_PUBLISH_SEAM = "ONEX_MULTI_EVENT_PUBLISH_SEAM"


def multi_event_seam_enabled() -> bool:
    """Return True when the def-B fan-out publish seam is enabled (default: False)."""
    return os.environ.get(ENV_MULTI_EVENT_PUBLISH_SEAM, "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _check_fanout_publish_coverage(contract: ModelDiscoveredContract) -> None:
    """Prove fan-out handlers' emittable classes are contract-declared (§2C)."""
    check_fanout_publish_coverage(
        contract,
        seam_enabled=multi_event_seam_enabled(),
        env_flag=ENV_MULTI_EVENT_PUBLISH_SEAM,
    )


def _sanitize_exc(exc: BaseException) -> str:
    """Return a sanitized one-line summary of an exception safe for logging/errors.

    Strips URLs and DSNs that may carry passwords or hostnames, then truncates.
    Only the exception type name + sanitized message is surfaced.
    """
    raw = str(exc) or type(exc).__name__
    sanitized = _SENSITIVE_PATTERN.sub("<redacted>", raw)
    return sanitized[:200]


# Deterministic signatures raised by CPython when ``asyncio.run`` /
# ``asyncio.Runner`` is invoked from inside an already-running event loop.
# OMN-9457 keys containment on the exact messages so best-effort string
# heuristics are avoided. Messages were verified empirically against CPython
# 3.11 / 3.12 source (the runtime target) and confirmed by inspecting
# ``asyncio.runners`` at runtime (asyncio/runners.py).
#
# CPython behaviour (3.11 / 3.12, verified from source):
#   * ``asyncio.run(coro)``
#     -> "asyncio.run() cannot be called from a running event loop"
#     (raised at the if-running-loop guard in asyncio/runners.py::run().)
#   * ``asyncio.Runner.run(coro)`` when another loop is active
#     -> "Runner.run() cannot be called from a running event loop"
#     (raised from asyncio/runners.py::Runner.run().)
#   * ``BaseEventLoop.run_until_complete`` nested call
#     -> "Cannot run the event loop while another loop is running"
#     (raised from asyncio/base_events.py::run_until_complete().)
#
# All three variants are matched because handlers may call any of these entry
# points, directly or transitively (e.g. a sync client that drives an async
# call with ``asyncio.run`` or ``asyncio.Runner``).
_ASYNC_INCOMPAT_MESSAGES: tuple[str, ...] = (
    "asyncio.run() cannot be called from a running event loop",
    "Runner.run() cannot be called from a running event loop",
    "Cannot run the event loop while another loop is running",
)


def _is_async_incompat_runtime_error(exc: BaseException) -> bool:
    """Return True if ``exc`` is the deterministic async-incompat signature.

    Matches ``RuntimeError`` raised by CPython's ``asyncio.run`` /
    ``asyncio.Runner.run`` when a synchronous handler constructor (or a
    dependency it resolves) calls ``asyncio.run()`` from within
    runtime-managed async boot. The detector walks the full exception
    chain — ``__cause__`` and ``__context__`` — because wrapped
    ``RuntimeError``s raised via ``raise X from original`` or propagated
    implicitly during handling may carry the original asyncio failure on
    either attribute (and per PEP 3134 both can be set simultaneously).
    Matching uses exact substring presence against the known CPython
    messages only, so unrelated ``RuntimeError``s are never misclassified.
    """
    visited: set[int] = set()
    stack: list[BaseException] = [exc]
    while stack:
        current = stack.pop()
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        if isinstance(current, RuntimeError):
            message = str(current)
            if any(needle in message for needle in _ASYNC_INCOMPAT_MESSAGES):
                return True
        # Explore BOTH branches of the exception chain. Per PEP 3134 an
        # exception may carry ``__cause__`` (explicit ``raise X from Y``)
        # and ``__context__`` (implicit propagation during handling)
        # simultaneously; skipping one branch can hide the original
        # asyncio failure when the handler constructor wraps it.
        if current.__cause__ is not None:
            stack.append(current.__cause__)
        if current.__context__ is not None:
            stack.append(current.__context__)
    return False


def _is_protocol_handler_class(handler_cls: type) -> bool:
    """Return True when a handler routing entry points at a Protocol class."""
    return bool(getattr(handler_cls, "_is_protocol", False))


def _is_delegation_inference_intent_ref(event_model: ModelHandlerRef) -> bool:
    """Return True for the canonical delegation inference-intent wire model."""
    return (
        event_model.module == _DELEGATION_INFERENCE_INTENT_MODULE
        and event_model.name == _DELEGATION_INFERENCE_INTENT_NAME
    )


def _payload_value(payload: object, key: str) -> object:
    if isinstance(payload, Mapping):
        return payload.get(key)
    return getattr(payload, key, None)


def _payload_claims_delegation_inference_intent(payload: object) -> bool:
    """Return True when a raw payload declares the inference-intent discriminator."""
    intent = _payload_value(payload, "intent")
    return intent == _DELEGATION_INFERENCE_INTENT_DISCRIMINATOR


async def _async_resolve_from_container(
    container: object,
    handler_cls: type,
) -> object | None:
    """Try to resolve handler_cls from container using get_service_async.

    Returns the resolved instance on success, None on ServiceResolutionError
    (service not registered), and re-raises any other exception.

    This avoids calling container.get_service() (sync) from inside a running
    event loop where asyncio.run() would raise RuntimeError (OMN-9410).
    """
    from omnibase_core.errors.error_service_resolution import ServiceResolutionError

    get_service_async = getattr(container, "get_service_async", None)
    if get_service_async is None:
        return None
    try:
        return await get_service_async(handler_cls)
    except ServiceResolutionError:
        return None


# Type alias matching MessageDispatchEngine.DispatcherFunc
DispatcherFunc = Callable[
    ["ModelEventEnvelope[object]"],
    Awaitable["ModelDispatchResult | None"],
]


@runtime_checkable
class ProtocolHandleable(Protocol):
    """Protocol for objects with a handle() method (auto-wired handlers)."""

    async def handle(
        self,
        envelope: ModelEventEnvelope[object],
    ) -> ModelDispatchResult | None: ...


@dataclass  # internal-dataclass-ok: holds non-serializable dispatcher callables and runtime wiring state
class PreparedWiring:
    """Data needed to register one contract entry with the dispatch engine.

    Produced by _prepare_handler_wiring (pure), consumed by
    _commit_handler_wiring (side effects only). Two phases ensure partial
    wiring never reaches the engine on later failure (OMN-8735).
    ``resolution_outcome`` / ``handler_name`` / ``skip_reason`` carry the
    resolver's per-handler outcome into the wiring report (OMN-9201).
    ``quarantine_reason`` / ``quarantine_detail`` / ``handler_module`` carry
    OMN-9457 containment state when handler construction deterministically
    failed with an async-incompatible signature.
    """

    dispatcher_id: str
    dispatcher: DispatcherFunc
    category: EnumMessageCategory
    message_types: set[str] | None
    handler_name: str = ""
    handler_module: str = ""
    resolution_outcome: EnumHandlerResolutionOutcome = (
        EnumHandlerResolutionOutcome.UNRESOLVABLE
    )
    skip_reason: str = ""
    quarantine_reason: EnumQuarantineReason | None = None
    quarantine_detail: str = ""
    route_ids: list[str] = field(default_factory=list)
    routes: list[ModelDispatchRoute] = field(default_factory=list)
    # Type-scoping predicate built from the entry's contract-declared
    # ``event_model``. When set, the dispatch engine selects this dispatcher
    # only for payloads that match the event_model, so sibling handlers on a
    # multi-handler contract are not all invoked for one message (OMN-12416).
    payload_type_matcher: Callable[[object], bool] | None = None

    @property
    def is_skip(self) -> bool:
        return (
            self.resolution_outcome
            is EnumHandlerResolutionOutcome.RESOLVED_VIA_LOCAL_OWNERSHIP_SKIP
        )

    @property
    def is_quarantined(self) -> bool:
        """True when OMN-9457 containment fired for this handler."""
        return self.quarantine_reason is not None


@dataclass  # internal-dataclass-ok: holds non-serializable dispatcher callables and runtime wiring state
class PreparedContractWiring:
    """All validated wiring data for one contract — no side effects yet.

    Produced by _prepare_contract_wiring (pure) and consumed by
    _commit_contract_wiring (side effects). Exists so wire_from_manifest can
    validate every contract before touching the dispatch engine or event bus
    (OMN-8735 — no partial state on startup abort).

    If skip_result is set, the contract was skipped and no wiring is needed.
    _commit_contract_wiring returns skip_result directly in that case.
    """

    contract: ModelDiscoveredContract
    prepared_wirings: list[PreparedWiring]
    subscription_topics: list[str]  # topics to subscribe after commit
    environment: str
    skip_result: ModelContractWiringResult | None = None


def _import_handler_class(module_path: str, class_name: str) -> type:
    """Import a handler class from its fully qualified module path.

    Args:
        module_path: Dotted module path (e.g. ``omnibase_infra.handlers.handler_foo``).
        class_name: Class name within the module.

    Returns:
        The handler class object.

    Raises:
        ImportError: If the module cannot be imported.
        TypeError: If the class is not found in the module (OMN-12408 hard-fail).
            A handler class declared in a contract but absent from its module is a
            build/contract defect — not a degradable runtime condition. TypeError is
            used (rather than AttributeError) so the caller's existing TypeError
            catch-and-reraise path (which bypasses the ONEX_WIRING_STRICT_MODE gate)
            propagates this failure as a startup crash regardless of strict mode.
    """
    mod = importlib.import_module(module_path)
    if not hasattr(mod, class_name):
        raise TypeError(
            f"CLASS_NOT_FOUND (HANDLER_LOADER_011): class '{class_name}' does not "
            f"exist in module '{module_path}'. "
            f"A contract that names a handler class that does not exist is a "
            f"build/contract defect, not a degradable condition (OMN-12408)."
        )
    return getattr(mod, class_name)  # type: ignore[no-any-return]


def _assert_is_ownership_query(obj: object) -> None:
    """Infra-boundary runtime protocol check for ProtocolHandlerOwnershipQuery.

    The core-hosted resolver types ``ownership_query`` as ``object | None``
    because ``compat → core → spi → infra`` forbids a core-to-spi import.
    Conformance MUST be verified here before the object reaches the resolver.
    See plan §Layering Invariants.
    """
    if not isinstance(obj, ProtocolHandlerOwnershipQuery):
        raise ModelOnexError(
            "handler_wiring: ownership_query does not conform to "
            f"ProtocolHandlerOwnershipQuery (got {type(obj).__name__!r})."
        )


async def _skip_dispatcher(
    envelope: ModelEventEnvelope[object],
) -> ModelDispatchResult | None:
    """Sentinel dispatcher for LOCAL_OWNERSHIP_SKIP entries; never registered."""
    return None


def _make_dispatch_callback(
    handler_instance: ProtocolHandleable,
    event_model: ModelHandlerRef | None = None,
    handler_node_kind: EnumNodeKind | None = None,
    published_event_names: frozenset[str] | None = None,
) -> DispatcherFunc:
    """Create a dispatch callback wrapping a handler instance.

    Legacy handlers receive the materialized dispatch envelope. Contract-typed
    handlers receive a validated payload model and may be sync or async. Handlers
    that declare an envelope-shaped signature keep receiving a typed envelope
    even when their contract declares ``event_model``.

    ``handler_node_kind`` carries the contract's declared archetype (from
    ``contract.node_type``). It is consulted only by ``_normalize_handler_result``
    to classify a REDUCER's bare typed / Sequence return as ``projection_intents``
    rather than events (OMN-14598); ``None`` preserves the archetype-agnostic
    classification for every other caller.

    ``published_event_names`` carries the short-name keys of the contract's
    ``published_events`` map (OMN-14794). It refines the OMN-14598 REDUCER
    classification: a REDUCER return whose model class IS a declared published
    event is emitted as an EVENT (``output_events``) instead of being captured as
    a projection — the live delegation-routing drop that stalled the FSM at
    RECEIVED. ``None`` (non-REDUCER, or a REDUCER declaring no published events)
    leaves the projection classification unchanged.

    When a handler exposes ``handle_async`` in addition to ``handle``, the async
    variant is preferred for dispatch.  This allows orchestrator handlers that
    perform side-effect publishes inside ``handle_async`` (e.g. FSM-driven swarm
    coordinators that flush command topics after each transition) to participate
    correctly in the event-bus dispatch loop.  ``handle`` stays the test/
    standalone entry point; ``handle_async`` is the runtime entry point.
    See OMN-12002.
    """
    # Prefer handle_async when the handler class explicitly declares it.
    # Performed once at wiring time so the per-message hot path has no overhead.
    # We inspect the MRO (not the instance) to exclude auto-generated attributes
    # such as MagicMock's dynamic attribute creation.
    _handle_async_method = next(
        (
            cls.__dict__["handle_async"]
            for cls in type(handler_instance).__mro__
            if "handle_async" in cls.__dict__
        ),
        None,
    )
    _candidate_handle_async = getattr(handler_instance, "handle_async", None)
    _candidate_handle = getattr(handler_instance, "handle", None)
    if (
        _handle_async_method is not None
        and callable(_handle_async_method)
        and callable(_candidate_handle_async)
    ):
        _effective_handle = cast("Callable[[object], object]", _candidate_handle_async)
    elif callable(_candidate_handle):
        _effective_handle = cast("Callable[[object], object]", _candidate_handle)
    else:

        def _missing_handle(_payload: object) -> object:
            raise ModelOnexError(
                "Auto-wired handler "
                f"{type(handler_instance).__name__} does not expose a callable "
                "handle() or handle_async() dispatch entrypoint."
            )

        _effective_handle = _missing_handle

    async def _callback(
        envelope: ModelEventEnvelope[object],
    ) -> ModelDispatchResult | None:
        handle_method = _effective_handle
        if event_model is None:
            from omnibase_infra.models.dispatch.model_dispatch_result import (
                ModelDispatchResult,
            )

            # OMN-14716: an operation_match def-B handler declares no contract
            # ``event_model``, so the typed-coercion path below never runs and the
            # engine hands the dispatcher the raw materialized wire dict. Passing
            # that dict straight to ``handle(request: ModelX)`` crashes on the
            # first attribute access (``'dict' object has no attribute ...`` — the
            # .201 finding-aggregator incident). Reach parity with runtime_local's
            # def-B coercion (``_coercion_target_model_type``, OMN-8724): when the
            # handler is a bare typed def-B handler (not an envelope handler),
            # validate the extracted domain payload into its declared input model
            # at THIS adapter boundary — never inside the handler, never by
            # wrapping the payload in a ModelEventEnvelope.
            #
            # OMN-15181 Finding 4: an operation_match handler whose ``handle()``
            # declares a CONCRETE ``ModelEventEnvelope`` annotation (e.g.
            # ``HandlerRedeployOrchestrator.handle(self, envelope:
            # ModelEventEnvelope[Any])``) was left OUT of the coercion above —
            # ``dispatch_arg`` stayed the raw materialized dict
            # (``{"payload": ..., "__bindings": {...}, "__debug_trace": {...}}``
            # from ``MessageDispatchEngine._materialize_envelope_with_bindings``),
            # never a ``ModelEventEnvelope`` instance. Only the sibling
            # ``event_model is not None`` (payload_type_match) branch below
            # called ``_materialize_typed_event_envelope`` /
            # ``_materialize_raw_event_envelope`` before invoking an
            # envelope-accepting handler. The live incident: a real,
            # grant-authorized prod redeploy dispatch crashed on
            # ``envelope.event_type`` — ``AttributeError: 'dict' object has no
            # attribute 'event_type'`` — before the orchestrator ever emitted its
            # first bus command. Materialize the same way here so both routing
            # strategies hand an envelope-annotated handler a real
            # ``ModelEventEnvelope``.
            #
            # Deliberately narrower than ``_handler_accepts_event_envelope``
            # (which also matches on the bare parameter name ``envelope``
            # regardless of annotation, to preserve untyped legacy handlers
            # such as ``handle(self, envelope: object)`` that intentionally
            # receive whatever raw object the engine handed them unchanged —
            # ``test_standard_callback_calls_async_handle``). Only a
            # CONCRETE ``ModelEventEnvelope`` annotation triggers
            # materialization here.
            dispatch_arg: object = envelope
            if _handler_declares_typed_event_envelope(handle_method):
                raw_payload = _extract_dispatch_payload(envelope)
                dispatch_arg = _materialize_raw_event_envelope(
                    envelope, raw_payload, fallback_event_type="unknown"
                )
            else:
                target_model = _resolve_def_b_input_model_type(handle_method)
                if target_model is not None:
                    # OMN-16050: pass the registered input model so the unwrap
                    # STOPS at it. ``ModelEmitRequest`` declares ``payload`` plus
                    # four transport markers, so a marker-only heuristic unwrapped
                    # through it and handed the handler the caller's inner
                    # payload — every node_event_emit_effect command DLQ'd.
                    payload = _extract_dispatch_payload(envelope, target_model)
                    if isinstance(payload, target_model):
                        dispatch_arg = payload
                    elif isinstance(payload, Mapping):
                        dispatch_arg = target_model.model_validate(payload)

            raw_result_obj = handle_method(dispatch_arg)
            raw_result = (
                await cast("Awaitable[object]", raw_result_obj)
                if asyncio.iscoroutine(raw_result_obj)
                else raw_result_obj
            )
            if raw_result is None or isinstance(raw_result, ModelDispatchResult):
                return raw_result
            if isinstance(raw_result, str):
                return cast("ModelDispatchResult | None", raw_result)
            if is_fanout_sequence(raw_result) and not any(
                isinstance(element, BaseModel)
                for element in cast("Sequence[object]", raw_result)
            ):
                # Legacy no-bus/state_io handlers use [] as a no-op fold and may
                # return list[str] intent markers. OMN-14403 only normalizes
                # actual BaseModel fan-out batches here.
                return cast("ModelDispatchResult | None", raw_result)
            # OMN-14403 §2A. A bare list/sequence used to be cast straight through
            # AS IF it were a ModelDispatchResult — strictly worse than the sibling
            # silent drop, since the applier then reads .output_events/.status off a
            # list and finds neither. Route it through the one fan-out coercion so it
            # either becomes a validated batch (seam ON) or is dropped LOUDLY (OFF).
            return _normalize_handler_result(
                raw_result, envelope, None, handler_node_kind, published_event_names
            )

        # OMN-16050: resolve the contract-declared event model BEFORE extracting so
        # the unwrap can stop at it (same fail-closed rule as the def-B branch
        # above). Resolution failure is not fatal here — the existing try/except
        # below owns that path — so the hint degrades to None and the extraction
        # keeps its pre-OMN-16050 structural behaviour.
        payload_target_model = _safe_import_event_model_class(event_model)
        payload = _extract_dispatch_payload(envelope, payload_target_model)
        handler_takes_envelope = _handler_accepts_event_envelope(
            cast("Callable[..., object]", handle_method)
        )
        try:
            model_cls = _import_event_model_class(event_model)
            typed_payload: object = (
                payload
                if isinstance(payload, model_cls)
                else model_cls.model_validate(payload)
            )
        except Exception as exc:
            # An envelope-accepting handler (a multi-step ORCHESTRATOR) declares
            # its ``event_model`` as the workflow ENTRYPOINT, yet it also consumes
            # the heterogeneous follow-up events on its other subscribe topics
            # (validated / completed / failed) whose payloads do NOT validate as
            # the entrypoint model. Such a handler coerces ``envelope.payload``
            # itself, keyed on ``event_type``. Re-hydrate the typed envelope with
            # the RAW domain payload (preserving ``event_type``) so the handler's
            # own polymorphic coercion runs — the entrypoint event still gets the
            # typed payload via the success path above (OMN-13247). Non-envelope
            # (typed-payload) handlers keep the strict fail-fast behavior.
            if handler_takes_envelope:
                fallback_envelope = _materialize_raw_event_envelope(
                    envelope, payload, event_model.name
                )
                fallback_result = handle_method(fallback_envelope)
                if asyncio.iscoroutine(fallback_result):
                    fallback_result = await cast("Awaitable[object]", fallback_result)
                return _normalize_handler_result(
                    fallback_result,
                    envelope,
                    event_model.name,
                    handler_node_kind,
                    published_event_names,
                )
            failure_result = _build_inference_intent_validation_failure_result(
                event_model=event_model,
                envelope=envelope,
                payload=payload,
                exc=exc,
            )
            if failure_result is not None:
                return failure_result
            raise
        if handler_takes_envelope:
            handler_envelope = _materialize_typed_event_envelope(
                envelope,
                cast("BaseModel", typed_payload),
                event_model.name,
            )
            envelope_result = handle_method(handler_envelope)
            if asyncio.iscoroutine(envelope_result):
                envelope_result = await cast("Awaitable[object]", envelope_result)
            return _normalize_handler_result(
                envelope_result,
                envelope,
                event_model.name,
                handler_node_kind,
                published_event_names,
            )

        typed_result = handle_method(typed_payload)
        if asyncio.iscoroutine(typed_result):
            typed_result = await cast("Awaitable[object]", typed_result)
        return _normalize_handler_result(
            typed_result,
            envelope,
            event_model.name,
            handler_node_kind,
            published_event_names,
        )

    return _callback


def _format_validation_error_detail(exc: BaseException) -> str:
    """Render a real, actionable detail string from a payload-validation failure.

    A pydantic ``ValidationError`` carries structured per-field errors
    (``loc``/``msg``/``type``); we flatten those into a compact
    ``field: message`` list so the detail is useful in a log line or a DLQ
    envelope without requiring the reader to re-run validation themselves.
    Non-pydantic exceptions (e.g. a raising custom validator) fall back to
    ``str(exc)``. Never raises.
    """
    errors = getattr(exc, "errors", None)
    if callable(errors):
        try:
            structured = errors()
        except Exception:  # noqa: BLE001 — best-effort detail formatting only
            structured = None
        if structured:
            parts = [
                f"{'.'.join(str(loc) for loc in err.get('loc', ()))}: {err.get('msg', '')}"
                for err in structured
            ]
            return "; ".join(parts) if parts else str(exc)
    return str(exc)


class PayloadTypeMatcher:
    """Callable payload-type predicate that records the real validation failure.

    Built from a contract-declared ``event_model`` (OMN-12416). Answers "does
    this payload match the handler's declared event_model?" — True when the
    payload is already an instance of the model, or when it validates against
    the model (e.g. a dict / raw envelope payload). Used by the dispatch
    engine to type-scope routing so a multi-handler contract delivers each
    message only to the handler whose event_model matches the payload.

    OMN-14492: unlike a plain predicate function, a rejecting call leaves the
    real pydantic ``ValidationError`` detail on ``last_validation_detail`` so
    the dispatch engine can distinguish "payload failed THIS handler's
    event_model validation" (publisher_malformed) from "no dispatcher was
    ever a candidate" (no_dispatcher) instead of collapsing both into the
    same unclassifiable "No dispatcher found" log line.

    The event_model class is imported lazily on first call and then cached, so
    wiring stays consistent with ``_make_dispatch_callback`` (which also defers
    the event_model import to the per-message path) and a declared-but-not-yet-
    importable model does not change failure timing. A payload that does not
    validate yields False (not an exception): "not this handler's type".
    """

    def __init__(self, event_model: ModelHandlerRef) -> None:
        self._event_model = event_model
        self._cached_model_cls: type[BaseModel] | None = None
        self.last_validation_detail: str | None = None

    def __call__(self, payload: object) -> bool:
        self.last_validation_detail = None
        if self._cached_model_cls is None:
            try:
                self._cached_model_cls = _import_event_model_class(self._event_model)
            except Exception:
                if _is_delegation_inference_intent_ref(
                    self._event_model
                ) and _payload_claims_delegation_inference_intent(payload):
                    return True
                raise
        model_cls = self._cached_model_cls
        if isinstance(payload, model_cls):
            return True
        try:
            model_cls.model_validate(payload)
        except ValidationError as exc:
            if _is_delegation_inference_intent_ref(
                self._event_model
            ) and _payload_claims_delegation_inference_intent(payload):
                return True
            self.last_validation_detail = _format_validation_error_detail(exc)
            return False
        return True


def _make_payload_type_matcher(
    event_model: ModelHandlerRef,
) -> Callable[[object], bool]:
    """Build a payload-type predicate from a contract-declared ``event_model``.

    Returns a ``PayloadTypeMatcher`` — a callable object satisfying
    ``Callable[[object], bool]`` that additionally exposes
    ``last_validation_detail`` (OMN-14492) after a rejecting call.
    """
    return PayloadTypeMatcher(event_model)


def _build_inference_intent_validation_failure_result(
    *,
    event_model: ModelHandlerRef,
    envelope: object,
    payload: object,
    exc: BaseException,
) -> ModelDispatchResult | None:
    """Build a correlated inference-response error for pre-handler validation misses.

    Delegation's inference effect publishes ``ModelInferenceResponseData`` for
    provider/runtime failures inside ``HandlerInferenceIntent.handle()``. Payload
    validation and event-model import happen before that handler is called, so
    this boundary maps only canonical ``ModelInferenceIntent`` load/validation
    failures to the same response shape. The delegation orchestrator then handles
    it through its normal inference-error path instead of leaving the caller to
    wait for a timeout.
    """
    if not _is_delegation_inference_intent_ref(event_model):
        return None
    if not _payload_claims_delegation_inference_intent(payload):
        return None

    from omnibase_core.models.delegation.wire import ModelInferenceResponseData
    from omnibase_infra.enums import EnumDispatchStatus
    from omnibase_infra.models.dispatch.model_dispatch_result import (
        ModelDispatchResult,
    )

    correlation_candidate = _extract_dispatch_correlation_id(envelope, payload)
    correlation_id = _coerce_uuid_or_none(correlation_candidate)
    if correlation_id is None:
        return None

    model_value = _payload_value(payload, "model")
    model_used = model_value if isinstance(model_value, str) else ""
    error_message = (
        "ModelInferenceIntent validation failed before "
        f"HandlerInferenceIntent.handle(): {_sanitize_exc(exc)}"
    )
    response = ModelInferenceResponseData(
        correlation_id=correlation_id,
        content="",
        model_used=model_used,
        llm_call_id="",
        latency_ms=0,
        error_message=error_message,
    )
    now = datetime.now(UTC)
    return ModelDispatchResult(
        status=EnumDispatchStatus.SUCCESS,
        topic=_extract_dispatch_topic(envelope),
        message_type=event_model.name,
        started_at=now,
        completed_at=now,
        output_count=1,
        output_events=[response],
        correlation_id=correlation_id,
    )


def _import_event_model_class(event_model: ModelHandlerRef) -> type[BaseModel]:
    module = importlib.import_module(event_model.module)
    model_cls = getattr(module, event_model.name)
    if not hasattr(model_cls, "model_validate"):
        raise TypeError(
            f"Event model {event_model.module}.{event_model.name} "
            "does not expose model_validate"
        )
    return cast("type[BaseModel]", model_cls)


def _safe_import_event_model_class(
    event_model: ModelHandlerRef | None,
) -> type[BaseModel] | None:
    """``_import_event_model_class`` that yields None instead of raising (OMN-16050).

    Used only to hint ``_extract_dispatch_payload`` with the contract-declared
    target type. An unimportable/malformed ``event_model`` must not change dispatch
    control flow from this call site — the caller's own
    ``_import_event_model_class`` inside its try/except still owns that failure.
    """
    if event_model is None:
        return None
    try:
        return _import_event_model_class(event_model)
    except Exception:  # noqa: BLE001 — hint-only resolution, never fatal here
        return None


def _handler_accepts_event_envelope(handle_method: object) -> bool:
    """Return true when a handler's first parameter is envelope-shaped."""
    try:
        signature = inspect.signature(cast("Callable[..., object]", handle_method))
    except (TypeError, ValueError):
        return False

    for parameter in signature.parameters.values():
        if parameter.name == "self":
            continue
        if parameter.kind not in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }:
            continue
        if parameter.name == "envelope":
            return True
        return _annotation_mentions_event_envelope(parameter.annotation)
    return False


def _handler_declares_typed_event_envelope(handle_method: object) -> bool:
    """Return true only when a handler's first parameter is CONCRETELY annotated
    ``ModelEventEnvelope`` — narrower than ``_handler_accepts_event_envelope``.

    ``_handler_accepts_event_envelope`` also matches on the bare parameter name
    ``envelope`` regardless of its annotation (a legacy heuristic preserved for
    untyped handlers like ``handle(self, envelope: object)`` that intentionally
    receive whatever raw object the engine handed them, unchanged). This
    stricter predicate drives ONLY the OMN-15181 Finding 4 materialization
    decision in the ``event_model is None`` (operation_match) dispatch branch:
    a handler must actually declare ``ModelEventEnvelope`` (or a
    ``ModelEventEnvelope[...]`` generic alias) to receive a materialized
    envelope instance there — a same-named-but-untyped parameter keeps
    receiving the raw dispatch input unchanged, matching its pre-existing,
    tested behavior.
    """
    try:
        signature = inspect.signature(cast("Callable[..., object]", handle_method))
    except (TypeError, ValueError):
        return False

    for parameter in signature.parameters.values():
        if parameter.name == "self":
            continue
        if parameter.kind not in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }:
            continue
        return _annotation_mentions_event_envelope(parameter.annotation)
    return False


def _annotation_mentions_event_envelope(annotation: object) -> bool:
    if annotation is inspect.Signature.empty:
        return False
    if isinstance(annotation, str):
        return "ModelEventEnvelope" in annotation
    if getattr(annotation, "__name__", "") == "ModelEventEnvelope":
        return True
    origin = get_origin(annotation)
    if getattr(origin, "__name__", "") == "ModelEventEnvelope":
        return True
    return any(_annotation_mentions_event_envelope(arg) for arg in get_args(annotation))


def _resolve_def_b_input_model_type(handle_method: object) -> type[BaseModel] | None:
    """Resolve a def-B handler's declared input model from its ``handle()`` signature.

    Mirrors runtime_local's ``_coercion_target_model_type`` (OMN-8724 /
    ``omnibase_core.runtime.runtime_local_adapter``): a canonical def-B handler
    ``handle(self, request: ModelX) -> ModelY`` reached via ``operation_match``
    (no contract-declared ``event_model``) must receive a validated ``ModelX``
    instance, not the raw materialized wire dict the engine hands the dispatcher.
    The Kafka boundary has no ``event_model`` to read for such a handler, so it
    recovers the target model the same way runtime_local does — by introspecting
    the single positional parameter's annotation.

    Returns that annotation when it is a concrete ``BaseModel`` subclass other
    than ``ModelEventEnvelope``; ``None`` otherwise (envelope handlers, ``**kwargs``
    handlers, multi-positional or unannotated params — every legacy shape keeps
    receiving the raw envelope dict unchanged). ``eval_str=True`` resolves the
    PEP 563 string annotation every node handler carries via
    ``from __future__ import annotations`` (without it the annotation is the
    literal string and the BaseModel check never matches — the OMN-8724 root
    cause), with a fall back to unevaluated annotations if resolution raises.
    """
    try:
        signature = inspect.signature(
            cast("Callable[..., object]", handle_method), eval_str=True
        )
    except (TypeError, ValueError, NameError):
        try:
            signature = inspect.signature(cast("Callable[..., object]", handle_method))
        except (TypeError, ValueError):
            return None

    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.name != "self"
        and parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    if len(positional) != 1:
        return None
    annotation = positional[0].annotation
    if (
        isinstance(annotation, type)
        and issubclass(annotation, BaseModel)
        and annotation.__name__ != "ModelEventEnvelope"
    ):
        return annotation
    return None


def _coerce_uuid_or_none(value: object) -> object | None:
    from uuid import UUID

    if isinstance(value, UUID):
        return value
    if isinstance(value, str) and value:
        try:
            return UUID(value)
        except ValueError:
            return None
    return None


def _ingress_correlation_id(message: object) -> UUID | None:
    """Recover the ingress correlation id from a message's TRANSPORT surface.

    OMN-14498: the consume boundary must be able to establish lineage without
    decoding the body, because the body is exactly what is unavailable for a
    poisoned message. Three transport shapes are supported, in order:

    * ``ModelEventMessage`` -- ``headers.correlation_id`` (a real ``UUID``);
      this is what ``EventBusKafka`` hands the callback in production.
    * a raw aiokafka ``ConsumerRecord`` -- ``headers`` as an iterable of
      ``(str, bytes)`` pairs, matching the ``correlation_id`` header both
      ``MixinKafkaDlq`` and ``DLQProducer.replay_message`` write.
    * a ``ModelEventEnvelope`` passed directly (legacy in-process call shape)
      -- its own ``correlation_id``.

    Returns ``None`` when no lineage is present on the transport, leaving the
    caller to fall back to the body and then to minting a fresh id. Never
    raises: a malformed header must not take down the consume boundary.
    """
    from uuid import UUID as _UUID

    headers = getattr(message, "headers", None)

    header_corr = getattr(headers, "correlation_id", None)
    coerced = _coerce_uuid_or_none(header_corr)
    if isinstance(coerced, _UUID):
        return coerced

    if headers is not None and not isinstance(headers, (str, bytes)):
        try:
            for entry in headers:
                key, value = entry
                if key != "correlation_id":
                    continue
                decoded = (
                    value.decode("utf-8", errors="replace")
                    if isinstance(value, bytes)
                    else value
                )
                coerced = _coerce_uuid_or_none(decoded)
                if isinstance(coerced, _UUID):
                    return coerced
        except (TypeError, ValueError):
            pass

    coerced = _coerce_uuid_or_none(getattr(message, "correlation_id", None))
    return coerced if isinstance(coerced, _UUID) else None


def _coerce_datetime_or_none(value: object) -> object | None:
    from datetime import UTC, datetime

    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
    return None


def _extract_dispatch_envelope_timestamp(envelope: object) -> object | None:
    if isinstance(envelope, Mapping):
        candidate = envelope.get("envelope_timestamp")
        if candidate is not None:
            return candidate
        candidate = envelope.get("timestamp")
        if candidate is not None:
            return candidate
        debug_trace = envelope.get("__debug_trace")
        if isinstance(debug_trace, Mapping):
            return debug_trace.get("envelope_timestamp") or debug_trace.get("timestamp")
    return getattr(envelope, "envelope_timestamp", None)


def _extract_dispatch_event_type(envelope: object) -> object | None:
    if isinstance(envelope, Mapping):
        candidate = envelope.get("event_type")
        if candidate is not None:
            return candidate
        debug_trace = envelope.get("__debug_trace")
        if isinstance(debug_trace, Mapping):
            return debug_trace.get("event_type")
    return getattr(envelope, "event_type", None)


def _materialize_raw_event_envelope(
    envelope: object,
    raw_payload: object,
    fallback_event_type: str,
) -> ModelEventEnvelope[object]:
    """Re-hydrate a typed envelope carrying the RAW domain payload.

    Used when an envelope-accepting ORCHESTRATOR consumes a follow-up event whose
    payload does not validate as its declared entrypoint ``event_model`` (the
    validated / completed / failed events of a multi-step workflow). The handler
    coerces ``envelope.payload`` itself keyed on ``event_type``, so this preserves
    the inbound ``event_type`` and hands the handler a typed
    ``ModelEventEnvelope`` (never a bare dict, which would crash
    ``envelope.event_type``) without forcing the payload into the entrypoint model
    (OMN-13247).
    """
    from datetime import UTC, datetime
    from uuid import uuid4

    from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope

    if isinstance(envelope, ModelEventEnvelope):
        updates: dict[str, object] = {"payload": raw_payload}
        if envelope.event_type is None:
            updates["event_type"] = fallback_event_type
        return envelope.model_copy(update=updates)

    correlation_id = _coerce_uuid_or_none(
        _extract_dispatch_correlation_id(envelope, raw_payload)
    )
    envelope_timestamp = _coerce_datetime_or_none(
        _extract_dispatch_envelope_timestamp(envelope)
    )
    event_type = _extract_dispatch_event_type(envelope)
    return ModelEventEnvelope[object](
        payload=raw_payload,
        correlation_id=correlation_id if correlation_id is not None else uuid4(),
        envelope_timestamp=(
            envelope_timestamp if envelope_timestamp is not None else datetime.now(UTC)
        ),
        event_type=str(event_type or fallback_event_type),
        source_tool="auto-wiring",
    )


def _materialize_typed_event_envelope(
    envelope: object,
    typed_payload: BaseModel,
    fallback_event_type: str,
) -> ModelEventEnvelope[object]:
    from datetime import UTC, datetime
    from uuid import uuid4

    from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope

    if isinstance(envelope, ModelEventEnvelope):
        updates: dict[str, object] = {"payload": typed_payload}
        if envelope.event_type is None:
            updates["event_type"] = fallback_event_type
        if envelope.payload_type is None:
            updates["payload_type"] = type(typed_payload).__name__
        return envelope.model_copy(update=updates)

    correlation_id = _coerce_uuid_or_none(
        _extract_dispatch_correlation_id(envelope, typed_payload)
    )
    envelope_timestamp = _coerce_datetime_or_none(
        _extract_dispatch_envelope_timestamp(envelope)
    )
    event_type = _extract_dispatch_event_type(envelope)

    return ModelEventEnvelope[object](
        payload=typed_payload,
        correlation_id=correlation_id if correlation_id is not None else uuid4(),
        envelope_timestamp=(
            envelope_timestamp if envelope_timestamp is not None else datetime.now(UTC)
        ),
        event_type=str(event_type or fallback_event_type),
        payload_type=type(typed_payload).__name__,
        source_tool="auto-wiring",
    )


# Transport-envelope keys the runtime adds around the domain payload. When the
# dispatch engine materializes a ModelEventEnvelope to a dict it nests the domain
# fields under ``payload`` and carries routing metadata (``partition_key`` etc.)
# alongside, so a mapping that carries a ``payload`` mapping plus any marker MAY
# be a transport envelope to unwrap. Mirrors omnimarket's
# ``_ENVELOPE_MARKER_KEYS`` predicate (OMN-12935/12936); the auto-wiring kernel
# unwraps here because it constructs the typed model itself, upstream of the
# handler's own coercion (OMN-12940).
#
# OMN-16050 — this marker set is a NECESSARY, NOT SUFFICIENT signal. The earlier
# text here asserted "domain models never declare these keys"; that invariant is
# FALSE. ``ModelEmitRequest`` (node_event_emit_effect) declares ``payload`` plus
# four of these markers (``event_type``, ``correlation_id``, ``partition_key``,
# ``event_id``), is structurally indistinguishable from a transport envelope, and
# was therefore unwrapped THROUGH — the handler got the caller's inner payload,
# ``model_validate`` raised, and every command DLQ'd. The registered-input-model
# stop condition below (``_is_registered_input_payload``) is what makes the
# heuristic safe: structure alone can never decide this.
_ENVELOPE_MARKER_KEYS: frozenset[str] = frozenset(
    {
        "partition_key",
        "event_type",
        "envelope_id",
        "event_id",
        "correlation_id",
        "__debug_trace",
    }
)


def _is_transport_envelope(value: object) -> bool:
    """True when ``value`` is envelope-SHAPED: a ``payload`` mapping plus a marker.

    Structural precondition only. A domain model may legitimately declare both a
    ``payload`` mapping and transport-plausible marker fields (OMN-16050), so this
    predicate is never sufficient on its own to justify an unwrap — see
    ``_is_registered_input_payload``, the fail-closed stop condition applied by
    ``_extract_dispatch_payload``.
    """
    return (
        isinstance(value, Mapping)
        and isinstance(value.get("payload"), Mapping)
        and bool(_ENVELOPE_MARKER_KEYS & value.keys())
    )


def _validation_alias_wire_keys(alias: object) -> set[str]:
    """Top-level wire keys a pydantic ``validation_alias`` can consume.

    ``validation_alias`` has three shapes and only the plain-string one is a
    single key. ``AliasPath("meta", "id")`` consumes the TOP-LEVEL key ``meta``
    (the remaining segments index inside that value), and ``AliasChoices`` holds
    a list of alternatives, each itself a string or an ``AliasPath``.

    Missing the non-string shapes is fail-OPEN for OMN-16050: a model aliased
    that way would fail ``_is_registered_input_payload``'s key-containment check
    even when the candidate IS the registered model, the unwrap would continue
    into the caller's payload, and the DLQ defect would return for exactly the
    contracts that use richer aliases.
    """
    if isinstance(alias, str):
        return {alias}
    if isinstance(alias, AliasPath):
        first = alias.path[0] if alias.path else None
        return {first} if isinstance(first, str) else set()
    if isinstance(alias, AliasChoices):
        keys: set[str] = set()
        for choice in alias.choices:
            keys |= _validation_alias_wire_keys(choice)
        return keys
    return set()


@lru_cache(maxsize=512)
def _model_declared_wire_keys(model: type[BaseModel]) -> frozenset[str]:
    """Every wire key ``model`` can accept: field names plus their input aliases."""
    keys: set[str] = set()
    for field_name, model_field in model.model_fields.items():
        keys.add(field_name)
        if isinstance(model_field.alias, str):
            keys.add(model_field.alias)
        keys |= _validation_alias_wire_keys(model_field.validation_alias)
    return frozenset(keys)


def _is_registered_input_payload(
    candidate: object, target_model: type[BaseModel] | None
) -> bool:
    """True when ``candidate`` IS the dispatcher's registered input model on the wire.

    The fail-closed stop condition for the recursive unwrap (OMN-16050). A
    candidate is claimed by the registered model only when BOTH hold:

    1. **Key containment** — every key present on the candidate is a declared
       field (or input alias) of ``target_model``. A real transport envelope
       always carries at least one routing/marker key the domain model does not
       declare (``source_tool``, ``envelope_id``, ``__debug_trace``,
       ``__bindings``, ``envelope_timestamp``, ...), so this alone keeps the
       OMN-12940 double-wrapped case unwrapping.
    2. **Full validation** — the candidate validates as ``target_model``, so a
       partial structural coincidence never halts the unwrap short of the domain.

    The cheap set check runs first; ``model_validate`` executes only for the rare
    candidate whose keys are entirely owned by the target model.

    Deliberately NOT a marker denylist: dropping ``event_type``/``correlation_id``
    from ``_ENVELOPE_MARKER_KEYS`` would fix ``ModelEmitRequest`` and silently
    break every genuine envelope that carries only those markers. This predicate
    keys on the CONTRACT-registered target type instead of on key spelling.
    """
    if target_model is None or not isinstance(candidate, Mapping):
        return False
    if not candidate.keys() <= _model_declared_wire_keys(target_model):
        return False
    try:
        target_model.model_validate(dict(candidate))
    except Exception:  # noqa: BLE001 — any validation failure means "not the model"
        return False
    return True


def _extract_dispatch_payload(
    envelope: object, target_model: type[BaseModel] | None = None
) -> object:
    # The runtime may deliver a DOUBLE- (or deeper-) wrapped envelope, e.g.
    # ``{"payload": {"payload": {domain}, ...markers}, "partition_key": None}``.
    # Unwrap recursively until the domain payload is reached so the kernel's
    # ``model_validate`` (and the post-handler correlation read) operate on the
    # domain, not on an intermediate envelope (OMN-12940).
    #
    # OMN-16050: stop the moment the candidate IS the dispatcher's registered
    # input model. ``target_model`` is the contract-declared type the kernel is
    # about to construct (the def-B ``handle()`` annotation, or the handler's
    # declared ``event_model``); when it is None the caller has no registered
    # type in scope and the pre-existing structural behaviour is unchanged.
    candidate: object = envelope
    if not isinstance(candidate, Mapping):
        candidate = getattr(candidate, "payload", candidate)
    while _is_transport_envelope(candidate) and not _is_registered_input_payload(
        candidate, target_model
    ):
        candidate = cast("Mapping[str, object]", candidate)["payload"]
    return candidate


def _extract_dispatch_topic(envelope: object) -> str:
    if isinstance(envelope, Mapping):
        debug_trace = envelope.get("__debug_trace")
        if isinstance(debug_trace, Mapping):
            topic = debug_trace.get("topic")
            if isinstance(topic, str) and topic:
                return topic
        topic = envelope.get("topic")
        if isinstance(topic, str) and topic:
            return topic
    topic = getattr(envelope, "topic", None)
    return topic if isinstance(topic, str) and topic else "auto-wired"


def _extract_dispatch_correlation_id(
    envelope: object, payload: object
) -> object | None:
    candidate = getattr(payload, "correlation_id", None)
    if candidate is not None:
        return candidate
    if isinstance(payload, Mapping):
        candidate = payload.get("correlation_id")
        if candidate is not None:
            return candidate
    if isinstance(envelope, Mapping):
        candidate = envelope.get("correlation_id")
        if candidate is not None:
            return candidate
        debug_trace = envelope.get("__debug_trace")
        if isinstance(debug_trace, Mapping):
            return debug_trace.get("correlation_id")
    return getattr(envelope, "correlation_id", None)


def _normalize_handler_result(
    result: object,
    envelope: object,
    message_type: str | None,
    handler_node_kind: EnumNodeKind | None = None,
    published_event_names: frozenset[str] | None = None,
) -> ModelDispatchResult | None:
    from datetime import UTC, datetime
    from uuid import UUID, uuid4

    from omnibase_core.models.dispatch.model_handler_output import ModelHandlerOutput
    from omnibase_core.models.reducer.model_intent import ModelIntent
    from omnibase_infra.enums import EnumDispatchStatus
    from omnibase_infra.models.dispatch.model_dispatch_result import (
        ModelDispatchResult,
    )

    if result is None or isinstance(result, ModelDispatchResult):
        return result

    payload = _extract_dispatch_payload(envelope)
    correlation_candidate = _extract_dispatch_correlation_id(envelope, payload)
    # Guard the coercion: a non-hex correlation candidate must fall back to a
    # fresh uuid4() rather than crash dispatch with ``ValueError: badly formed
    # hexadecimal UUID string`` (OMN-12940). ``_coerce_uuid_or_none`` already
    # guards every other correlation read site (555/702/1207/1278).
    coerced_correlation = _coerce_uuid_or_none(correlation_candidate)
    correlation_id: UUID = (
        coerced_correlation if isinstance(coerced_correlation, UUID) else uuid4()
    )

    output_events: list[BaseModel] = []
    output_intents: tuple[object, ...] = ()
    projection_intents: tuple[ModelProjectionIntent, ...] = ()

    # OMN-14598: a def-B REDUCER (contract node_type REDUCER_GENERIC) that returns
    # a bare typed projection model OR a Sequence of projection models must be
    # classified as projections — a reducer emits projections[] ONLY (core handler-
    # output contract). Computed BEFORE the isinstance/fan-out branches below so a
    # reducer's return is never misrouted to output_events or a fan-out event batch.
    reducer_projection_models: tuple[BaseModel, ...] = ()
    if (
        handler_node_kind is EnumNodeKind.REDUCER
        and not isinstance(result, ModelHandlerOutput)
        and not _is_declared_published_event_model(result, published_event_names)
    ):
        reducer_projection_models = _coerce_projection_models(result)

    if isinstance(result, ModelHandlerOutput):
        output_events = [
            event for event in result.events if isinstance(event, BaseModel)
        ]
        output_intents = tuple(result.intents)
        if isinstance(result.result, ModelIntent):
            output_intents = output_intents + (result.result,)
        elif isinstance(result.result, BaseModel):
            output_events.append(result.result)
        if result.projections:
            # OMN-14598: a ModelHandlerOutput carries projections[] only for a
            # REDUCER (validator-enforced on ModelHandlerOutput). Route them to
            # projection_intents so DispatchResultApplier's synchronous projection
            # sink fires; before this branch projections were read by NEITHER the
            # events/intents/result path and were silently dropped (e.g.
            # HandlerCodingAgentFsm's two ``for_reducer`` projections per fold).
            projection_intents = _build_projection_intents(
                result.projections, correlation_id, message_type
            )
    elif reducer_projection_models:
        projection_intents = _build_projection_intents(
            reducer_projection_models, correlation_id, message_type
        )
    elif isinstance(result, BaseModel):
        output_events = [result]
    elif is_fanout_sequence(result):
        # OMN-14403 §2A — the def-B fan-out entry. Before this branch existed a
        # Sequence return matched NEITHER branch above and fell through to
        # output_events=[] / SUCCESS: the handler's N events were dropped and the
        # dispatch reported success. That silent drop IS the defect. The applier
        # resolves each element's topic via published_events (same short-name
        # resolution the shared core resolver uses); the boot coverage gate keeps
        # that fail-closed.
        output_events = normalize_fanout_sequence(
            cast("Sequence[object]", result),
            message_type,
            seam_enabled=multi_event_seam_enabled(),
            env_flag=ENV_MULTI_EVENT_PUBLISH_SEAM,
        )

    return ModelDispatchResult(
        status=EnumDispatchStatus.SUCCESS,
        topic=_extract_dispatch_topic(envelope),
        message_type=message_type,
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        output_count=len(output_events) + len(output_intents) + len(projection_intents),
        output_events=output_events,
        # Why: Runtime wiring validates and narrows this payload shape before use.
        output_intents=output_intents,  # type: ignore[arg-type]
        projection_intents=projection_intents,
        correlation_id=correlation_id,
    )


def _is_declared_published_event_model(
    result: object,
    published_event_names: frozenset[str] | None,
) -> bool:
    """True when *result* is a single typed model the contract declares as a
    published event (OMN-14794).

    OMN-14598 classifies EVERY bare-model / ``Sequence`` return from a REDUCER
    (``node_type: REDUCER_GENERIC``) as ``projections[]``. That is correct for a
    pure FSM fold, but a REDUCER that ALSO declares a ``published_events`` entry
    for the model it returns emits that model as an EVENT — e.g.
    ``node_delegation_routing_reducer`` returns ``ModelRoutingDecision``, which its
    contract maps (``event_type: RoutingDecision``) to
    ``onex.evt.omnibase-infra.routing-decision.v1``. Without this exception the
    decision was captured into ``projection_intents`` and NEVER published as an
    event, so the delegation orchestrator's ``handle_routing_decision`` never fired
    and the workflow stalled at RECEIVED (routing-decision.v1 high-watermark flat).
    That live drop was hotpatch-validated on the stability-test runtime: excluding
    the declared-event return from the REDUCER->projection branch advanced the FSM
    RECEIVED->ROUTED->COMPLETED and moved the routing-decision.v1 HW by exactly one.

    Membership mirrors the applier's own topic resolver (``_outbox_topic_for`` /
    ``resolve_published_topic``): the class name with a leading ``Model`` stripped
    (the canonical ``event_type`` short-name), then the full class name.
    ``published_event_names`` is the key set of the contract's ``published_events``
    map (``load_published_events_map``). A ``None`` / empty set — every non-REDUCER
    caller, and any REDUCER that declares no published events — preserves the
    OMN-14598 projection classification unchanged.
    """
    if not published_event_names or not isinstance(result, BaseModel):
        return False
    class_name = type(result).__name__
    return (
        class_name.removeprefix("Model") in published_event_names
        or class_name in published_event_names
    )


def _coerce_projection_models(result: object) -> tuple[BaseModel, ...]:
    """Coerce a def-B REDUCER return into its projection models (OMN-14598).

    A reducer's canonical def-B return is either a single typed projection model
    or a Sequence of projection models — the multi-projection case, e.g.
    ``node_coding_agent_fsm_reducer`` folding to ``(advanced_state,
    trace_projection)``. A non-model / non-Sequence return yields ``()`` so the
    caller records an empty (no-op fold) dispatch result rather than
    misclassifying it as an event.
    """
    if isinstance(result, BaseModel):
        return (result,)
    if is_fanout_sequence(result):
        return tuple(
            element
            for element in cast("Sequence[object]", result)
            if isinstance(element, BaseModel)
        )
    return ()


def _derive_projector_key(model: BaseModel) -> str:
    """Derive a deterministic projector-registry key from a projection model.

    Canonical convention (OMN-14598): strip a leading ``Model`` from the class
    name and convert the remaining CamelCase to snake_case, e.g.
    ``ModelCodingAgentTraceProjection`` -> ``coding_agent_trace_projection``.
    Deterministic and self-describing so a projector can register under the same
    key without the reducer carrying routing metadata on its return value.
    """
    class_name = type(model).__name__
    stem = class_name.removeprefix("Model") or class_name
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", stem).lower()
    return snake or class_name.lower()


def _build_projection_intents(
    models: Sequence[object],
    correlation_id: UUID,
    message_type: str | None,
) -> tuple[ModelProjectionIntent, ...]:
    """Build ModelProjectionIntent entries from a reducer's projection models.

    OMN-14598: converts each projection model into a ``ModelProjectionIntent`` so
    ``ModelDispatchResult.projection_intents`` is populated (consumed by
    ``DispatchResultApplier``'s synchronous projection sink). ``projector_key`` is
    derived from the model type; ``event_type`` is the inbound message type (the
    event the reducer folded), falling back to the projection model's class name
    when the dispatch path carries no ``message_type``.
    """
    from omnibase_core.models.projectors.model_projection_intent import (
        ModelProjectionIntent,
    )

    intents: list[ModelProjectionIntent] = []
    for model in models:
        if not isinstance(model, BaseModel):
            continue
        intents.append(
            ModelProjectionIntent(
                projector_key=_derive_projector_key(model),
                event_type=message_type or type(model).__name__,
                envelope=model,
                correlation_id=correlation_id,
            )
        )
    return tuple(intents)


_TABLE_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

# OMN-13350: JSONB list columns whose name does NOT end in the _json/_jsonb
# convention and so must be explicitly JSON-adapted. A Python list bound to a
# JSONB column must be wrapped in psycopg2.extras.Json or psycopg2 sends a
# Postgres ARRAY literal, which fails against a JSONB column — and the projection
# consumer then silently commits the offset and drops the event. This set is the
# narrow allowlist for JSONB list columns that the suffix rule does not cover; it
# must NOT include genuine Postgres text[] ARRAY columns (e.g.
# swarm_runs.models_used / machines_used), which are correctly passed as raw
# lists.
# OMN-14487: recent_responses is the same defect class as corpus_errors — a JSONB
# array column (projection_delegation_inference_response_text, declared
# ``JSONB NOT NULL DEFAULT '[]'::jsonb CHECK (jsonb_typeof(...) = 'array')``)
# holding a list of objects, with no _json/_jsonb suffix. HandlerProjectionDelegation-
# InferenceResponse handed the raw list[dict] to this adapter; psycopg2 could not
# adapt the inner dicts (``can't adapt type 'dict'``), the inference-response
# projection write crashed, and — with no dlq_topics declared on that node's
# contract — the erroring event was silently dropped.
_JSONB_LIST_COLUMNS: frozenset[str] = frozenset({"corpus_errors", "recent_responses"})


def _should_jsonb_wrap_list(key: str, value: list[object]) -> bool:
    """Decide whether a ``list`` row value must be JSON-wrapped for psycopg2.

    Three independent rules; any one triggers wrapping:

    1. Column-name suffix convention (``_json`` / ``_jsonb``).
    2. The ``_JSONB_LIST_COLUMNS`` allowlist -- legacy unsuffixed JSONB list
       columns (``corpus_errors``, ``recent_responses``) that predate rule 3
       and that the structural rule below does not cover on its own (e.g. a
       JSONB column holding ``list[str]``, which is structurally
       indistinguishable from a genuine ``text[]`` ARRAY).
    3. STRUCTURAL heuristic (OMN-14494): any element of the list is itself a
       ``dict``/``list``. A genuine Postgres scalar ARRAY (``text[]``,
       ``int[]``, ...) can only ever hold scalars, so an element that is
       itself a dict/list can NEVER be a valid ARRAY member -- this case is
       unambiguously a JSONB list-of-objects/list-of-lists column. This rule
       needs no allowlist maintenance and auto-covers every future
       unsuffixed JSONB list-of-objects column, closing the recurring defect
       class behind OMN-13350 and OMN-14487 (both required a manual
       allowlist edit before this fix).

    A flat scalar list (``list[str]`` / ``list[int]`` with no dict/list
    elements) that matches none of the three rules returns ``False`` here
    and is passed raw by the caller, preserving genuine ``text[]``/``int[]``
    ARRAY semantics (e.g. ``swarm_runs.models_used`` / ``machines_used``).
    """
    if str(key).endswith(("_json", "_jsonb")):
        return True
    if str(key) in _JSONB_LIST_COLUMNS:
        return True
    return any(isinstance(item, (dict, list)) for item in value)


# Authoritative source: docs/patterns/db_url_contract.md "Per-Service Database
# URL Contract" — each OmniNode service owns its own PostgreSQL database and a
# dedicated *_DB_URL env var. This map MUST stay in parity with that table; a
# missing row makes the DB-injection auto-wiring reject a contract whose
# db_io.database names a real per-service DB (e.g. F3/OMN-13158:
# node_dispatch_outcome_bridge_effect -> database omniintelligence).
_DB_URL_ENV_MAP: dict[str, str] = {
    "omnibase_infra": "OMNIBASE_INFRA_DB_URL",
    "omniintelligence": "OMNIINTELLIGENCE_DB_URL",
    "omniclaude": "OMNICLAUDE_DB_URL",
    "omnimemory": "OMNIMEMORY_DB_URL",
    "omninode_cloud": "OMNINODE_CLOUD_DB_URL",
    "omnidash_analytics": "OMNIDASH_ANALYTICS_DB_URL",
}


@dataclass(frozen=True)
class ProjectionDatabaseBindingTarget:
    """One topology-declared workload identity and its secret-free DSN key."""

    binding_ref: str
    database_ref: str
    physical_database: str
    principal: str
    dsn_env: str


@dataclass(frozen=True)
class ProjectionCatalogBindingPolicy:
    """Composition-root choice of existing topology catalog identities."""

    read_binding: str | None = None
    write_binding: str | None = None


@dataclass(frozen=True)
class ProjectionTableTarget:
    """Topology-resolved location for one typed table declaration."""

    table: ModelDbTableDeclaration
    database_ref: str
    physical_database: str
    schema: str
    domain: EnumDatabaseSchemaDomain
    read_binding: ProjectionDatabaseBindingTarget | None
    write_binding: ProjectionDatabaseBindingTarget | None


@dataclass(frozen=True)
class ProjectionDatabaseTarget:
    """Topology-resolved, per-operation pools for one projection handler."""

    tables: tuple[ModelDbTableDeclaration, ...]
    table_targets: tuple[ProjectionTableTarget, ...]
    physical_database: str

    @property
    def database_refs(self) -> tuple[str, ...]:
        """Return the declared logical database references in stable order."""
        return tuple(sorted({target.database_ref for target in self.table_targets}))

    @property
    def schemas(self) -> tuple[str, ...]:
        """Return the declared schemas in stable order."""
        return tuple(sorted({target.schema for target in self.table_targets}))

    @property
    def domains(self) -> tuple[EnumDatabaseSchemaDomain, ...]:
        """Return the topology-derived domains in stable enum-value order."""
        return tuple(
            sorted(
                {target.domain for target in self.table_targets},
                key=lambda domain: domain.value,
            )
        )

    @property
    def bindings(self) -> tuple[ProjectionDatabaseBindingTarget, ...]:
        """Return every selected operation binding in stable order."""
        by_ref: dict[str, ProjectionDatabaseBindingTarget] = {}
        for table_target in self.table_targets:
            for binding in (table_target.read_binding, table_target.write_binding):
                if binding is not None:
                    by_ref[binding.binding_ref] = binding
        return tuple(by_ref[key] for key in sorted(by_ref))

    @property
    def dsn_envs(self) -> tuple[str, ...]:
        """Return every required DSN environment key in stable order."""
        return tuple(sorted({binding.dsn_env for binding in self.bindings}))


_TENANT_PROJECTION_BINDING = "tenant_projection"
_INTERNAL_PROJECTION_BINDING = "omninode_runtime_service"


def _resolve_projection_binding(
    database: ModelDeploymentTopologyDatabase,
    database_ref: str,
    binding_ref: str,
) -> ProjectionDatabaseBindingTarget:
    """Resolve one explicit workload binding without a physical-DB fallback."""
    binding = database.bindings.get(binding_ref)
    if binding is None:
        raise ValueError(
            f"Projection binding {binding_ref!r} is not declared for "
            f"database_ref {database_ref!r}"
        )
    if binding.database_ref != database_ref:
        raise ValueError(
            f"Projection binding {binding_ref!r} resolves to database_ref "
            f"{binding.database_ref!r}, expected {database_ref!r}"
        )
    principal = database.principals.get(binding.principal)
    if principal is None:
        raise ValueError(
            f"Projection binding {binding_ref!r} references unknown principal "
            f"{binding.principal!r}"
        )
    if not principal.login or principal.bypass_rls:
        raise ValueError(
            f"Projection principal {binding.principal!r} must be LOGIN and NOBYPASSRLS"
        )
    return ProjectionDatabaseBindingTarget(
        binding_ref=binding_ref,
        database_ref=database_ref,
        physical_database=database.physical_name,
        principal=binding.principal,
        dsn_env=binding.dsn_env,
    )


def _require_projection_binding_privileges(
    database: ModelDeploymentTopologyDatabase,
    binding: ProjectionDatabaseBindingTarget,
    table: ModelDbTableDeclaration,
    *,
    operation: str,
) -> None:
    """Prove the selected topology principal can perform the exact operation."""
    principal = database.principals[binding.principal]
    grant_schema = physical_grant_schema_for_table(table.schema, table.name)
    has_schema_usage = any(
        grant.object_type is EnumDatabaseGrantObjectType.SCHEMA
        and grant.schema == grant_schema
        and EnumDatabasePrivilege.USAGE in grant.privileges
        for grant in principal.grants
    )
    required_table_privileges = (
        {EnumDatabasePrivilege.SELECT}
        if operation == "read"
        else {
            # PostgreSQL requires SELECT as well as INSERT/UPDATE for the
            # adapter's INSERT ... ON CONFLICT DO UPDATE statement.
            EnumDatabasePrivilege.SELECT,
            EnumDatabasePrivilege.INSERT,
            EnumDatabasePrivilege.UPDATE,
        }
    )
    granted_table_privileges = {
        privilege
        for grant in principal.grants
        if grant.object_type is EnumDatabaseGrantObjectType.TABLE
        and grant.schema == grant_schema
        and table.name in grant.objects
        for privilege in grant.privileges
    }
    missing_table_privileges = sorted(
        required_table_privileges - granted_table_privileges,
        key=lambda privilege: privilege.value,
    )
    if not has_schema_usage or missing_table_privileges:
        missing = []
        if not has_schema_usage:
            missing.append(f"USAGE on schema {grant_schema!r}")
        if missing_table_privileges:
            names = ", ".join(privilege.value for privilege in missing_table_privileges)
            missing.append(f"{names} on table {table.schema}.{table.name}")
        raise ValueError(
            f"Projection binding {binding.binding_ref!r} principal "
            f"{binding.principal!r} lacks declared {operation} privileges: "
            + "; ".join(missing)
        )


def _projection_operation_bindings(
    *,
    table: ModelDbTableDeclaration,
    database: ModelDeploymentTopologyDatabase,
    domain: EnumDatabaseSchemaDomain,
    catalog_read_binding: str | None,
    catalog_write_binding: str | None,
) -> tuple[
    ProjectionDatabaseBindingTarget | None,
    ProjectionDatabaseBindingTarget | None,
]:
    """Select explicit read/write identities from domain and table access."""
    needs_read = table.access in {"read", "read_write"}
    needs_write = table.access in {"write", "read_write"}
    if domain is EnumDatabaseSchemaDomain.TENANT:
        binding_ref = _TENANT_PROJECTION_BINDING
        read_ref = binding_ref if needs_read else None
        write_ref = binding_ref if needs_write else None
    elif domain is EnumDatabaseSchemaDomain.OMNINODE_INTERNAL:
        binding_ref = _INTERNAL_PROJECTION_BINDING
        read_ref = binding_ref if needs_read else None
        write_ref = binding_ref if needs_write else None
    elif domain is EnumDatabaseSchemaDomain.PLATFORM_CATALOG:
        read_ref = catalog_read_binding if needs_read else None
        write_ref = catalog_write_binding if needs_write else None
        if needs_read and read_ref is None:
            raise ValueError(
                f"Catalog table {table.name!r} requires an explicit reader binding"
            )
        if needs_write and write_ref is None:
            raise ValueError(
                f"Catalog table {table.name!r} requires an explicit writer binding"
            )
    else:  # pragma: no cover - enum exhaustiveness guard
        raise ValueError(f"Unsupported projection database domain {domain!r}")

    read_binding = (
        _resolve_projection_binding(database, table.database_ref, read_ref)
        if read_ref is not None
        else None
    )
    write_binding = (
        _resolve_projection_binding(database, table.database_ref, write_ref)
        if write_ref is not None
        else None
    )
    if read_binding is not None:
        _require_projection_binding_privileges(
            database,
            read_binding,
            table,
            operation="read",
        )
    if write_binding is not None:
        _require_projection_binding_privileges(
            database,
            write_binding,
            table,
            operation="write",
        )
    return read_binding, write_binding


def _resolve_projection_database_target(
    db_tables: Sequence[ModelDbTableDeclaration],
    topology: ModelDeploymentTopology,
    *,
    catalog_read_binding: str | None = None,
    catalog_write_binding: str | None = None,
) -> ProjectionDatabaseTarget:
    """Resolve typed table declarations through the authoritative topology."""
    tables = tuple(db_tables)
    if not tables:
        raise ValueError("Projection database target requires at least one db_table")

    names = [table.name for table in tables]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate db_table declarations: {duplicates!r}")

    table_targets: list[ProjectionTableTarget] = []
    databases_by_physical_name: dict[str, ModelDeploymentTopologyDatabase] = {}
    for table in tables:
        database = topology.databases.get(table.database_ref)
        if database is None:
            raise ValueError(f"Unknown database_ref '{table.database_ref}'")
        domain = topology.schema_domain(table.database_ref, table.schema)
        physical_database = database.physical_name
        databases_by_physical_name[physical_database] = database
        read_binding, write_binding = _projection_operation_bindings(
            table=table,
            database=database,
            domain=domain,
            catalog_read_binding=catalog_read_binding,
            catalog_write_binding=catalog_write_binding,
        )
        table_targets.append(
            ProjectionTableTarget(
                table=table,
                database_ref=table.database_ref,
                physical_database=physical_database,
                schema=table.schema,
                domain=domain,
                read_binding=read_binding,
                write_binding=write_binding,
            )
        )

    physical_databases = tuple(sorted(databases_by_physical_name))
    if len(physical_databases) != 1:
        raise ValueError(
            "Projection handler db_tables require more than one physical database "
            f"connection, got {physical_databases!r}; split the handler or provide "
            "an explicit multi-adapter boundary"
        )
    return ProjectionDatabaseTarget(
        tables=tables,
        table_targets=tuple(table_targets),
        physical_database=physical_databases[0],
    )


_TOPIC_TO_EVENT_TYPE: dict[str, str] = {
    "node-heartbeat": "heartbeat",
    "node-introspection": "introspection",
    "node-state-change": "state_change",
}


def _derive_projection_event_type(
    topic: str,
    envelope_event_type: object,
    subscribe_topics: tuple[str, ...],
) -> str:
    """Derive projection handler event type from topic or dispatch alias."""
    topic_candidate = topic
    if not topic_candidate and len(subscribe_topics) == 1:
        topic_candidate = subscribe_topics[0]

    segment_candidates: list[str] = []
    if topic_candidate:
        segment_candidates.append(
            topic_candidate.split(".")[-2]
            if "." in topic_candidate
            else topic_candidate
        )
    if envelope_event_type:
        event_type = str(envelope_event_type).strip()
        if event_type:
            segment_candidates.append(event_type.split(".")[-1])

    for segment in segment_candidates:
        if segment in _TOPIC_TO_EVENT_TYPE:
            return _TOPIC_TO_EVENT_TYPE[segment]

    # Handlers that don't use _event_type (e.g. HandlerProjectionDelegation) receive
    # the raw segment as a passthrough. Only platform-registration projection handlers
    # require the mapped form.
    return segment_candidates[0] if segment_candidates else ""


def _materialized_dispatch_trace_value(
    envelope: object,
    key: str,
) -> object:
    """Extract trace metadata from a materialized dispatch dict."""
    if not isinstance(envelope, dict):
        return None
    trace = envelope.get("__debug_trace")
    if isinstance(trace, dict):
        return trace.get(key)
    return None


def _extract_projection_topic(envelope: object) -> str:
    """Extract projection route topic from envelope or materialized dispatch."""
    if isinstance(envelope, dict):
        value = _materialized_dispatch_trace_value(envelope, "topic")
    else:
        value = getattr(envelope, "topic", None)
        if not value:
            event_type = getattr(envelope, "event_type", None)
            if isinstance(event_type, str) and event_type.startswith("onex."):
                value = event_type
    return str(value).strip() if value else ""


def _extract_projection_event_type(envelope: object) -> object:
    """Extract event_type from envelope or materialized dispatch trace."""
    if isinstance(envelope, dict):
        return _materialized_dispatch_trace_value(envelope, "event_type")
    return getattr(envelope, "event_type", None)


def _extract_projection_payload(envelope: object) -> object:
    """Extract payload from envelope or materialized dispatch dict."""
    if isinstance(envelope, dict):
        return envelope.get("payload")
    return getattr(envelope, "payload", None)


def _extract_projection_envelope_id(envelope: object) -> object | None:
    """Return the typed, stable identity of the dispatched event envelope."""
    value = (
        envelope.get("envelope_id")
        if isinstance(envelope, dict)
        else getattr(envelope, "envelope_id", None)
    )
    return _coerce_uuid_or_none(value)


def _is_raw_event_projection_contract(contract: ModelDiscoveredContract) -> bool:
    if contract.event_bus is None:
        return False
    consumer_purpose = (contract.event_bus.consumer_purpose or "").strip().lower()
    return consumer_purpose in {"audit", "projection"}


def _raw_event_projection_enabled(
    contract: ModelDiscoveredContract,
    result_appliers_by_contract: Mapping[str, ProtocolDispatchResultApplier] | None,
) -> bool:
    """Return true when a raw projection contract has an explicit effect path.

    Raw audit/projection consumers carry Kafka `ModelEventMessage` bytes and
    usually emit intents. Wiring them without a result applier would consume
    offsets while dropping those intents, so the kernel must opt in per contract.
    """
    return _is_raw_event_projection_contract(contract) and (
        result_appliers_by_contract is not None
        and contract.name in result_appliers_by_contract
    )


def _read_dlq_topics(contract_path: Path) -> list[str]:
    """Read ``event_bus.dlq_topics`` from a contract YAML. Returns [] if absent.

    OMN-13548 (D-03): projection handlers declare the DLQ destination for
    malformed inbound events under ``event_bus.dlq_topics`` (the same field the
    omnimarket projection runners read). The typed ``ModelEventBusSubcontract``
    does not carry this field, so the wiring reads only this event-bus extension
    from the raw contract YAML. Database table locations are already typed on
    ``ModelDiscoveredContract`` and are never re-read here. The DLQ topic is
    resolved from the contract — never hardcoded in this module.

    Raises on YAML parse / file I/O failures so a broken contract is surfaced
    rather than silently degrading to a no-DLQ projection wiring.
    """
    try:
        # Why: Optional integration dependency is validated at runtime but ships incomplete typing.
        import yaml  # type: ignore[import-untyped]

        with open(contract_path) as f:
            raw = yaml.safe_load(f)
    except FileNotFoundError:
        return []
    if not isinstance(raw, dict):
        return []
    event_bus = raw.get("event_bus") or {}
    if not isinstance(event_bus, dict):
        return []
    return [str(t) for t in (event_bus.get("dlq_topics") or [])]


def _contract_declares_db_io(contract: ModelDiscoveredContract) -> bool:
    return bool(contract.db_io is not None and contract.db_io.db_tables)


def _read_state_io(contract_path: Path) -> dict[str, object]:
    """Read the top-level ``state_io`` block from a contract YAML.

    Returns ``{}`` if ``state_io`` is absent. Raises on YAML parse errors or
    unexpected file I/O failures, so a malformed contract is surfaced as a
    broken contract rather than silently treated as "no state_io". Unlike
    ``db_io``, this legacy state subcontract does not yet have a core model.

    Shape (OMN-14208 opt-in runtime dispatch seam)::

        state_io:
          database: omnibase_infra   # _DB_URL_ENV_MAP key
          table: delegation_workflow_state
          key: correlation_id        # documented; correlation_id is the
                                      # only supported read key today
          codec:
            module: <dotted module path resolved via importlib>
            name: <class name>
    """
    try:
        # Why: Optional integration dependency is validated at runtime but ships incomplete typing.
        import yaml  # type: ignore[import-untyped]

        with open(contract_path) as f:
            raw = yaml.safe_load(f)
    except FileNotFoundError:
        return {}
    if not isinstance(raw, dict):
        return {}
    state_io = raw.get("state_io") or {}
    return state_io if isinstance(state_io, dict) else {}


def _contract_declares_state_io(contract: ModelDiscoveredContract) -> bool:
    return bool(_read_state_io(contract.contract_path))


# Tenant-scoped projection tables compare ``tenant_id`` with this
# transaction-local setting in both USING and WITH CHECK policies.
_TENANT_GUC = "app.tenant_id"


def _projection_tenant_context_error(message: str) -> Exception:
    from omnibase_infra.errors.error_projection import ProjectionTenantContextError

    return ProjectionTenantContextError(message)


def _reject_canonical_tenant_field(
    values: Mapping[str, object] | None, *, domain: EnumDatabaseSchemaDomain
) -> None:
    if values is not None and "tenant_id" in values:
        raise ValueError(
            f"{domain.value} operation rejects canonical tenant_id; "
            "only tenant-domain operations may carry it"
        )


class ProjectionTableOperation:
    """Shared SQL mechanics for one topology-resolved table declaration."""

    def __init__(
        self,
        adapter: ProjectionDatabaseOperations,
        target: ProjectionTableTarget,
    ) -> None:
        self._adapter = adapter
        self._target = target

    def _assert_write_declared(self) -> None:
        if self._target.table.access not in {"write", "read_write"}:
            raise PermissionError(
                f"{self._target.schema}.{self._target.table.name} declares "
                f"access={self._target.table.access!r}; write refused"
            )

    def _assert_read_declared(self) -> None:
        if self._target.table.access not in {"read", "read_write"}:
            raise PermissionError(
                f"{self._target.schema}.{self._target.table.name} declares "
                f"access={self._target.table.access!r}; read refused"
            )

    def upsert(self, conflict_key: str, row: dict[str, object]) -> bool:
        self._assert_write_declared()
        return self._adapter._execute_upsert(
            self._target, conflict_key, row, tenant_context=None
        )

    def query(
        self, filters: dict[str, object] | None = None
    ) -> list[dict[str, object]]:
        self._assert_read_declared()
        return self._adapter._execute_query(self._target, filters, tenant_context=None)


class TenantProjectionTableOperation(ProjectionTableOperation):
    """Tenant operation whose only authority is a verified capability."""

    def _context(self) -> VerifiedProjectionTenantAuthority:
        return self._adapter._bound_tenant_context()

    def _assert_supplied_tenant(
        self,
        supplied_tenant: object,
        context: VerifiedProjectionTenantAuthority,
        *,
        operation: str,
    ) -> None:
        if isinstance(supplied_tenant, UUID):
            supplied_uuid = supplied_tenant
        elif isinstance(supplied_tenant, str):
            supplied_uuid = parse_canonical_tenant_uuid(
                supplied_tenant,
                authority=f"{self._target.table.name} {operation} compatibility field",
            )
        else:
            raise _projection_tenant_context_error(
                f"{self._target.table.name} {operation} tenant_id does not match "
                "verified projection authority"
            )
        if supplied_uuid != context.tenant_id:
            raise _projection_tenant_context_error(
                f"{self._target.table.name} {operation} tenant_id does not match "
                "verified projection authority"
            )

    def upsert(self, conflict_key: str, row: dict[str, object]) -> bool:
        self._assert_write_declared()
        context = self._context()
        attributed_row = dict(row)
        supplied_tenant = attributed_row.get("tenant_id")
        if supplied_tenant is not None:
            self._assert_supplied_tenant(supplied_tenant, context, operation="row")
        attributed_row["tenant_id"] = context.tenant_id
        return self._adapter._execute_upsert(
            self._target,
            conflict_key,
            attributed_row,
            tenant_context=context,
        )

    def query(
        self, filters: dict[str, object] | None = None
    ) -> list[dict[str, object]]:
        self._assert_read_declared()
        context = self._context()
        attributed_filters = dict(filters or {})
        supplied_tenant = attributed_filters.get("tenant_id")
        if supplied_tenant is not None:
            self._assert_supplied_tenant(supplied_tenant, context, operation="query")
        attributed_filters["tenant_id"] = context.tenant_id
        return self._adapter._execute_query(
            self._target,
            attributed_filters,
            tenant_context=context,
        )


class InternalProjectionTableOperation(ProjectionTableOperation):
    """Internal operation that never resolves or sets tenant context."""

    def upsert(self, conflict_key: str, row: dict[str, object]) -> bool:
        _reject_canonical_tenant_field(row, domain=self._target.domain)
        conflict_keys = {key.strip() for key in conflict_key.split(",")}
        if "source_tenant_id" in conflict_keys:
            raise ValueError(
                "Internal source_tenant_id is provenance only and cannot be an "
                "upsert conflict key"
            )
        return super().upsert(conflict_key, row)

    def query(
        self, filters: dict[str, object] | None = None
    ) -> list[dict[str, object]]:
        _reject_canonical_tenant_field(filters, domain=self._target.domain)
        return super().query(filters)


class CatalogProjectionTableOperation(ProjectionTableOperation):
    """Catalog operation enforcing the declaration's explicit access mode."""

    def upsert(self, conflict_key: str, row: dict[str, object]) -> bool:
        _reject_canonical_tenant_field(row, domain=self._target.domain)
        return super().upsert(conflict_key, row)

    def query(
        self, filters: dict[str, object] | None = None
    ) -> list[dict[str, object]]:
        _reject_canonical_tenant_field(filters, domain=self._target.domain)
        return super().query(filters)


class ProjectionBindingConnections:
    """Own per-binding connections, identity attestation, and transactions."""

    def __init__(
        self,
        db_urls: Mapping[str, str],
        target: ProjectionDatabaseTarget,
        psycopg2_module: object,
    ) -> None:
        required_bindings = {binding.binding_ref for binding in target.bindings}
        supplied_bindings = set(db_urls)
        if supplied_bindings != required_bindings:
            raise ValueError(
                "Projection DSN bindings must exactly match the topology target: "
                f"required={sorted(required_bindings)!r}, "
                f"supplied={sorted(supplied_bindings)!r}"
            )
        if any(not isinstance(url, str) or not url for url in db_urls.values()):
            raise ValueError("Projection DSN binding values must be non-empty strings")
        self._db_urls = dict(db_urls)
        self._connections: dict[str, object] = {}
        self._closed = False
        self._psycopg2 = psycopg2_module

    @property
    def connections(self) -> dict[str, object]:
        """Expose live connections for narrow diagnostics and cleanup proofs."""
        return self._connections

    def get(self, binding: ProjectionDatabaseBindingTarget | None) -> object:
        """Return an attested connection for one exact topology binding."""
        self.ensure_open()
        if binding is None:
            raise PermissionError(
                "Projection operation has no declared workload binding"
            )
        conn = self._connections.get(binding.binding_ref)
        if conn is None or getattr(conn, "closed", False):
            connect = self._psycopg2.connect  # type: ignore[attr-defined]
            conn = connect(self._db_urls[binding.binding_ref])
            try:
                conn.autocommit = True
                with conn.cursor() as cursor:  # type: ignore[attr-defined]
                    cursor.execute("SELECT current_user, current_database()")
                    identity = cursor.fetchone()
                expected_identity = (binding.principal, binding.physical_database)
                if (
                    not isinstance(identity, (tuple, list))
                    or tuple(identity) != expected_identity
                ):
                    raise PermissionError(
                        f"Projection binding {binding.binding_ref!r} connected as "
                        f"{identity!r}, expected {expected_identity!r}"
                    )
            except BaseException:
                conn.close()  # type: ignore[attr-defined]
                raise
            self._connections[binding.binding_ref] = conn
        return conn

    def close(self) -> None:
        """Deterministically close every per-binding connection."""
        if self._closed:
            return
        self._closed = True
        connections = tuple(self._connections.values())
        self._connections.clear()
        for conn in connections:
            if not getattr(conn, "closed", False):
                conn.close()  # type: ignore[attr-defined]

    def ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Projection database adapter is closed")

    @contextlib.contextmanager
    def tenant_transaction(
        self, conn: object, context: VerifiedProjectionTenantAuthority
    ) -> Iterator[None]:
        """Set the GUC locally from a validated context, then always end it."""
        conn.autocommit = False  # type: ignore[attr-defined]
        try:
            with conn.cursor() as cursor:  # type: ignore[attr-defined]
                cursor.execute(
                    "SELECT set_config(%s, %s, true)",
                    (_TENANT_GUC, str(context.tenant_id)),
                )
            yield
            conn.commit()  # type: ignore[attr-defined]
        except BaseException:
            conn.rollback()  # type: ignore[attr-defined]
            raise
        finally:
            conn.autocommit = True  # type: ignore[attr-defined]


class ProjectionDatabaseOperations:
    """Router over separate table operations selected from typed topology."""

    def __init__(
        self,
        db_urls: Mapping[str, str],
        target: ProjectionDatabaseTarget,
        tenant_authority: VerifiedProjectionTenantAuthority | None,
        tenant_event: object | None,
        psycopg2_module: object,
        extras_module: object,
    ) -> None:
        self._binding_connections = ProjectionBindingConnections(
            db_urls,
            target,
            psycopg2_module,
        )
        # Kept as a read-only diagnostic seam for existing runtime proofs.
        self._connections = self._binding_connections.connections
        self._extras = extras_module
        self._tenant_authority = tenant_authority
        self._tenant_event = tenant_event
        operation_types: dict[
            EnumDatabaseSchemaDomain, type[ProjectionTableOperation]
        ] = {
            EnumDatabaseSchemaDomain.TENANT: TenantProjectionTableOperation,
            EnumDatabaseSchemaDomain.OMNINODE_INTERNAL: InternalProjectionTableOperation,
            EnumDatabaseSchemaDomain.PLATFORM_CATALOG: CatalogProjectionTableOperation,
        }
        self._operations = {
            table_target.table.name: operation_types[table_target.domain](
                self, table_target
            )
            for table_target in target.table_targets
        }

    def _bound_tenant_context(self) -> VerifiedProjectionTenantAuthority:
        """Return the already-verified authority or fail before connecting."""
        self._binding_connections.ensure_open()
        if self._tenant_authority is None:
            raise _projection_tenant_context_error(
                "Tenant projection has no cryptographically verified authority"
            )
        assert_projection_tenant_authority_matches_event(
            self._tenant_authority,
            self._tenant_event,
        )
        return self._tenant_authority

    def close(self) -> None:
        """Release authority and deterministically close every connection."""
        self._tenant_authority = None
        self._tenant_event = None
        self._binding_connections.close()

    def _operation(self, table: str) -> ProjectionTableOperation:
        self._binding_connections.ensure_open()
        operation = self._operations.get(table)
        if operation is None:
            raise ValueError(
                f"Projection table {table!r} is not declared by the typed db_io contract"
            )
        return operation

    def _adapt_row(self, row: Mapping[str, object]) -> dict[str, object]:
        json_adapter = self._extras.Json  # type: ignore[attr-defined]
        return {
            key: (
                json_adapter(value)
                if isinstance(value, dict)
                or (isinstance(value, list) and _should_jsonb_wrap_list(key, value))
                else value
            )
            for key, value in row.items()
        }

    def _execute_upsert(
        self,
        target: ProjectionTableTarget,
        conflict_key: str,
        row: dict[str, object],
        *,
        tenant_context: VerifiedProjectionTenantAuthority | None,
    ) -> bool:
        conflict_keys = [key.strip() for key in conflict_key.split(",") if key.strip()]
        if not conflict_keys:
            raise ValueError("conflict_key must contain at least one column")
        if any(not _TABLE_NAME_RE.fullmatch(key) for key in conflict_keys):
            raise ValueError(f"Invalid conflict key: {conflict_key!r}")
        cols = list(row)
        bad_cols = [column for column in cols if not _TABLE_NAME_RE.fullmatch(column)]
        if bad_cols:
            raise ValueError(f"Invalid column names: {bad_cols!r}")
        missing = [key for key in conflict_keys if key not in row]
        if missing:
            raise KeyError(f"row missing conflict key(s): {missing!r}")
        quoted_cols = ", ".join(f'"{column}"' for column in cols)
        placeholders = ", ".join(f"%({column})s" for column in cols)
        conflict_columns = ", ".join(f'"{key}"' for key in conflict_keys)
        conflict_set = set(conflict_keys)
        updates = ", ".join(
            f'"{column}" = EXCLUDED."{column}"'
            for column in cols
            if column not in conflict_set
        )
        action = f"DO UPDATE SET {updates}" if updates else "DO NOTHING"
        insert_sql = " ".join(
            (
                f'INSERT INTO "{target.schema}"."{target.table.name}" ({quoted_cols})',
                f"VALUES ({placeholders})",
                f"ON CONFLICT ({conflict_columns}) {action}",
            )
        )
        conn = self._binding_connections.get(target.write_binding)
        adapted_row = self._adapt_row(row)
        if tenant_context is None:
            with conn.cursor() as cursor:  # type: ignore[attr-defined]
                cursor.execute(insert_sql, adapted_row)
        else:
            with self._binding_connections.tenant_transaction(conn, tenant_context):
                with conn.cursor() as cursor:  # type: ignore[attr-defined]
                    cursor.execute(insert_sql, adapted_row)
        return True

    def _execute_query(
        self,
        target: ProjectionTableTarget,
        filters: dict[str, object] | None,
        *,
        tenant_context: VerifiedProjectionTenantAuthority | None,
    ) -> list[dict[str, object]]:
        # Schema/table originate in validated typed declarations, never request data.
        select_sql = f'SELECT * FROM "{target.schema}"."{target.table.name}"'  # noqa: S608
        params: list[object] = []
        if filters:
            bad_keys = [
                key for key in filters if not _TABLE_NAME_RE.fullmatch(str(key))
            ]
            if bad_keys:
                raise ValueError(f"Invalid filter keys: {bad_keys!r}")
            select_sql += " WHERE " + " AND ".join(f'"{key}" = %s' for key in filters)
            params = list(filters.values())
        conn = self._binding_connections.get(target.read_binding)

        def _query() -> list[dict[str, object]]:
            cursor_factory = self._extras.RealDictCursor  # type: ignore[attr-defined]
            with conn.cursor(cursor_factory=cursor_factory) as cursor:  # type: ignore[attr-defined]
                cursor.execute(select_sql, params or None)
                return [dict(record) for record in cursor.fetchall()]

        if tenant_context is None:
            return _query()
        with self._binding_connections.tenant_transaction(conn, tenant_context):
            return _query()

    def upsert(self, table: str, conflict_key: str, row: dict[str, object]) -> bool:
        return self._operation(table).upsert(conflict_key, row)

    def query(
        self, table: str, filters: dict[str, object] | None = None
    ) -> list[dict[str, object]]:
        return self._operation(table).query(filters)


def _connect_projection_runner_db_if_needed(handler_instance: object) -> None:
    """Connect BaseProjectionRunner-style DB adapters before direct dispatch."""
    db = getattr(handler_instance, "db", None)
    if db is None or getattr(db, "_pool", None) is not None:
        return
    connect = getattr(db, "connect", None)
    if not callable(connect):
        return
    result = connect()
    if asyncio.iscoroutine(result):
        asyncio.run(result)


def _is_projection_runner_handler(handler_instance: object) -> bool:
    """Detect standalone Kafka projection runners exposed in handler_routing."""
    return (
        type(handler_instance).__name__.endswith("ProjectionRunner")
        and hasattr(handler_instance, "project_event")
        and hasattr(handler_instance, "topics")
        and hasattr(handler_instance, "db")
    )


def _extract_rows_upserted(result: object) -> int:
    """Extract the rows-written count from a projection handler's return value.

    OMN-13360: the projection terminal event must be gated on a real write.
    Projection handlers return either a ModelProjectionResult-shaped mapping
    (``{"rows_upserted": N, ...}``) or a runner-shim mapping (``{"projected":
    bool}``). This narrows both to an integer row count:

    - ``rows_upserted`` present -> coerce to int (the authoritative count).
    - only ``projected`` present (runner shim) -> 1 when truthy, else 0. The
      standalone runner returns ``{"projected": bool}`` where True already means
      a row was committed (its DB execute path raises on failure).
    - anything else -> 0 (no provable write; terminal must NOT be emitted).
    """
    if isinstance(result, dict):
        if "rows_upserted" in result:
            try:
                return int(result["rows_upserted"])
            except (TypeError, ValueError):
                return 0
        if "projected" in result:
            return 1 if bool(result["projected"]) else 0
    return 0


async def _route_projection_error_to_dlq(
    event_bus: object | None,
    dlq_topics: list[str],
    envelope: object,
    handler_name: str,
    failure_reason: str,
) -> bool:
    """Publish a malformed/erroring projection event to a DLQ/quarantine sink.

    OMN-13548 (D-03): when a projection handler raises (most commonly a
    ``ValidationError`` because the inbound event is missing a required field),
    the wiring previously logged at ERROR and dropped the message — no DLQ row,
    no durable trace. This routes the offending raw envelope to the
    contract-declared DLQ topic (``event_bus.dlq_topics[0]``) so the dropped
    event is recoverable on the bus. The DLQ envelope carries the offending
    payload, the failure reason, the handler name, and the correlation_id
    (hoisted to the top level so the failure is recoverable by correlation even
    when the payload itself is unparseable).

    OMN-14492 (OMN-14487-class silent drop): when the contract declares NO
    ``event_bus.dlq_topics``, this previously logged at ERROR and returned
    ``False`` — the event never reached any topic, only a container log line.
    That is the exact "quiet death" class OMN-14487 hit for
    ``HandlerProjectionDelegationInferenceResponse``. This now falls back to
    the platform-wide quarantine sink (``build_dlq_topic("quarantine")``) so
    every drop reaches a declared, durable topic even when the contract has no
    DLQ topic of its own.

    Generic for ALL projection handlers, not delegation-only. Best-effort:
    returns ``True`` when the DLQ/quarantine envelope was published, ``False``
    when no publishable event bus is available or the publish itself fails
    (each logged at ERROR). A DLQ publish failure never propagates, so it
    cannot wedge the consumer.
    """
    import json
    from datetime import UTC, datetime

    from omnibase_infra.enums import EnumDlqFailureClass
    from omnibase_infra.event_bus.topic_constants import build_dlq_topic

    used_quarantine_fallback = not dlq_topics
    if used_quarantine_fallback:
        dlq_topic = build_dlq_topic("quarantine")
        logger.error(
            "Projection handler %s has NO DLQ topic declared in "
            "contract.event_bus.dlq_topics — routing malformed/erroring event "
            "to the platform quarantine sink %s instead of dropping it: %s",
            handler_name,
            dlq_topic,
            failure_reason,
        )
    else:
        dlq_topic = dlq_topics[0]
    if event_bus is None or not hasattr(event_bus, "publish"):
        logger.error(
            "Projection handler %s would route malformed/erroring event to DLQ %s "
            "but no publishable event bus is bound: %s",
            handler_name,
            dlq_topic,
            failure_reason,
        )
        return False

    payload = _extract_dispatch_payload(envelope)
    correlation = _extract_dispatch_correlation_id(envelope, payload)
    correlation_id = str(correlation) if correlation is not None else str(uuid4())
    original_message: object
    model_dump = getattr(payload, "model_dump", None)
    if isinstance(payload, Mapping):
        original_message = dict(payload)
    elif callable(model_dump):
        original_message = model_dump(mode="json")
    else:
        original_message = {"raw": str(payload)}
    dlq_envelope = {
        "original_message": original_message,
        "failure_reason": failure_reason,
        "failure_class": EnumDlqFailureClass.CONSUMER_ERROR.value,
        "correlation_id": correlation_id,
        "retry_count": 0,
        "failed_at": datetime.now(UTC).isoformat(),
        "handler": handler_name,
        "quarantine_fallback": used_quarantine_fallback,
    }
    raw = json.dumps(dlq_envelope, default=str).encode("utf-8")
    publish = getattr(event_bus, "publish", None)
    if not callable(publish):
        logger.error(
            "Projection handler %s would route malformed/erroring event to DLQ %s "
            "but the bound event bus publish attribute is not callable: %s",
            handler_name,
            dlq_topic,
            failure_reason,
        )
        return False
    try:
        await publish(dlq_topic, None, raw)
    except Exception as exc:  # noqa: BLE001 — DLQ publish is best-effort; never wedge the consumer
        logger.error(
            "Projection handler %s failed to route malformed/erroring event to DLQ %s "
            "(correlation_id=%s): %s",
            handler_name,
            dlq_topic,
            correlation_id,
            _sanitize_exc(exc),
        )
        return False
    logger.warning(
        "Projection handler %s routed malformed/erroring event to DLQ %s "
        "(correlation_id=%s): %s",
        handler_name,
        dlq_topic,
        correlation_id,
        failure_reason,
    )
    return True


@dataclass(
    frozen=True
)  # internal-dataclass-ok: wiring-internal sink bundle; event_bus is a non-serializable publishable object
class ProjectionDispatchSinks:
    """Bus-side output sinks for a projection dispatch callback.

    Bundles the optional bus, terminal-event topic, and DLQ topics so the
    callback factory stays within the parameter-count budget while each sink
    remains an explicitly named, typed field. A frozen dataclass (not a Pydantic
    model) keeps this wiring-internal value object out of the model layer:
    ``event_bus`` is an arbitrary publishable object (in-memory bus, Kafka
    wiring, or a test double), so no schema validation is wanted here.
    """

    event_bus: object | None = None
    terminal_event: str | None = None
    dlq_topics: tuple[str, ...] = ()


def _make_projection_dispatch_callback(
    handler_instance: object,
    target: ProjectionDatabaseTarget,
    subscribe_topics: tuple[str, ...],
    sinks: ProjectionDispatchSinks | None = None,
) -> DispatcherFunc:
    """Create a dispatch callback for projection handlers (db_io.db_tables declared).

    Builds a synchronous psycopg2 DatabaseAdapter per call and injects it into
    input_data alongside _event_type and _topic derived from the dispatched
    envelope (OMN-13992: _topic was computed locally for logging but never
    injected, so any strict projection handler that requires
    input_data['_topic'] — e.g. HandlerProjectionLiveEvents — raised a
    ValueError on every dispatch and the event was dropped, non-fatally but
    silently, with no DLQ topic declared to catch it).

    ``sinks`` carries the bus-side outputs. When ``sinks.event_bus`` and
    ``sinks.terminal_event`` are set, a terminal event envelope is emitted to
    that topic after each successful projection so downstream Pattern-B
    consumers and golden-chain tests can observe completion.

    OMN-13548 (D-03): when ``sinks.dlq_topics`` is supplied (resolved from the
    contract's ``event_bus.dlq_topics``) and the projection handler raises, the
    offending raw envelope is routed to the DLQ topic instead of being logged +
    dropped silently. This is the robust layer for the fail-loud/observability
    guarantee: a malformed inbound event whose ``ValidationError`` escapes the
    handler is now durably captured on the bus on the REAL dispatch path, not
    only when a handler happens to catch it internally.
    """
    sinks = sinks or ProjectionDispatchSinks()
    event_bus = sinks.event_bus
    terminal_event = sinks.terminal_event
    dlq_topics = list(sinks.dlq_topics)
    handler_name = type(handler_instance).__name__
    is_projection_runner = _is_projection_runner_handler(handler_instance)
    db_urls = (
        {}
        if is_projection_runner
        else {
            binding.binding_ref: os.environ.get(binding.dsn_env, "")
            for binding in target.bindings
        }
    )
    missing_bindings = [
        binding
        for binding in target.bindings
        if not is_projection_runner and not db_urls[binding.binding_ref]
    ]
    if missing_bindings:
        raise ValueError(
            "Projection handler requires topology bindings with configured DSNs: "
            + ", ".join(
                f"{binding.binding_ref}:{binding.dsn_env}"
                for binding in missing_bindings
            )
        )

    async def _callback(
        envelope: ModelEventEnvelope[object],
    ) -> ModelDispatchResult | None:
        if is_projection_runner:
            logger.debug(
                "Projection runner skipped by DB-injection auto-wiring: handler=%s topic=%s",
                type(handler_instance).__name__,
                _extract_projection_topic(envelope) or "unknown",
            )
            return None
        projected = False
        adapter: object | None = None
        try:
            # MessageDispatchEngine hands callbacks a JSON-safe materialization.
            # The original typed envelope is retained only for stable transport
            # identity; it is never a tenant-authentication source.  Tenant
            # operations require the separate cryptographically verified
            # capability bound by trusted ingress.
            typed_envelope = current_dispatch_envelope() or envelope
            tenant_authority = current_projection_tenant_authority()
            adapter = _build_projection_db_adapter(
                db_urls,
                target,
                tenant_authority,
                typed_envelope,
            )
            topic = _extract_projection_topic(envelope)
            event_type = _derive_projection_event_type(
                topic,
                _extract_projection_event_type(envelope),
                subscribe_topics,
            )
            payload = _extract_projection_payload(envelope)
            input_data: dict[str, object] = (
                dict(payload) if isinstance(payload, dict) else {}
            )
            if hasattr(payload, "model_dump"):
                # Why: Control flow narrows this union at runtime before the attribute access.
                input_data = payload.model_dump(mode="json")  # type: ignore[union-attr]
            input_data["_db"] = adapter
            input_data["_event_type"] = event_type
            input_data["_topic"] = topic
            envelope_id = _extract_projection_envelope_id(typed_envelope)
            if envelope_id is not None:
                # Preserve the UUID at the transport boundary. Projection
                # handlers may use it as their durable idempotency key instead
                # of inventing a fresh identity for every Kafka redelivery.
                input_data["_envelope_id"] = envelope_id

            def _invoke_projection_handler() -> object:
                _connect_projection_runner_db_if_needed(handler_instance)
                # Why: Control flow narrows this union at runtime before the attribute access.
                return handler_instance.handle(input_data)  # type: ignore[union-attr, attr-defined]

            result = await asyncio.to_thread(_invoke_projection_handler)
            if asyncio.iscoroutine(result):
                result = await cast("Awaitable[object]", result)
            # OMN-13360 (deterministic-truth gate): the terminal
            # `projection-delegation-applied.v1` event asserts that a durable row
            # landed, so it must be gated on the handler's actual write outcome —
            # not merely on `handle()` not raising. The projection handler returns
            # ModelProjectionResult.model_dump() carrying rows_upserted (0 or 1+);
            # a zero-row / internally-swallowed path returns normally and would
            # otherwise emit a false-positive `projected:true`. Gate on
            # rows_upserted >= 1 and log loudly when a no-raise handler wrote zero
            # rows so the failure surfaces instead of being masked.
            rows_upserted = _extract_rows_upserted(result)
            projected = rows_upserted >= 1
            if projected:
                logger.debug(
                    "Projection handler completed: topic=%s event_type=%s "
                    "rows_upserted=%s result=%s",
                    topic,
                    event_type,
                    rows_upserted,
                    result,
                )
            else:
                logger.error(
                    "Projection handler wrote zero rows (no terminal emitted): "
                    "handler=%s topic=%s event_type=%s rows_upserted=%s result=%s",
                    type(handler_instance).__name__,
                    topic or "unknown",
                    event_type,
                    rows_upserted,
                    result,
                )
        except TypeError as exc:
            logger.error(
                "Projection handler TypeError (likely missing _db or _event_type): "
                "handler=%s topic=%s error_type=%s",
                handler_name,
                _extract_projection_topic(envelope) or "unknown",
                type(exc).__name__,
            )
            # OMN-13548 (D-03): route the offending event to the contract DLQ
            # instead of dropping it silently.
            await _route_projection_error_to_dlq(
                event_bus,
                dlq_topics,
                envelope,
                handler_name,
                f"{type(exc).__name__}: {_sanitize_exc(exc)}",
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Projection handler error: handler=%s topic=%s error_type=%s error=%s",
                handler_name,
                _extract_projection_topic(envelope) or "unknown",
                type(exc).__name__,
                exc,
            )
            # OMN-13548 (D-03): a ValidationError (e.g. a malformed delegation
            # event missing a required field) raised by the projection handler
            # on the REAL dispatch path lands here. Route the raw envelope to the
            # contract-declared DLQ topic so the dropped event is durably
            # recoverable on the bus rather than vanishing after this log line.
            await _route_projection_error_to_dlq(
                event_bus,
                dlq_topics,
                envelope,
                handler_name,
                f"{type(exc).__name__}: {_sanitize_exc(exc)}",
            )
        finally:
            if adapter is not None:
                close = getattr(adapter, "close", None)
                if callable(close):
                    close()

        if projected and event_bus is not None and terminal_event is not None:
            await _emit_projection_terminal_event(event_bus, terminal_event, envelope)

        return None

    return _callback


def _build_projection_db_adapter(
    db_urls: Mapping[str, str],
    target: ProjectionDatabaseTarget,
    tenant_authority: VerifiedProjectionTenantAuthority | None,
    tenant_event: object | None,
) -> object:
    """Build a router whose operations come only from typed topology targets."""
    # Why: Optional integration dependency ships incomplete typing.
    import psycopg2  # type: ignore[import-untyped]

    # Why: Optional integration dependency ships incomplete typing.
    import psycopg2.extras  # type: ignore[import-untyped]

    # Keep UUIDs typed through the adapter and teach psycopg2 the final wire
    # conversion, instead of stringifying correlation/tenant IDs in row data.
    psycopg2.extras.register_uuid()

    logger.debug(
        "Selecting projection adapter: database_refs=%s physical_database=%s "
        "schemas=%s domains=%s",
        target.database_refs,
        target.physical_database,
        target.schemas,
        [domain.value for domain in target.domains],
    )
    return ProjectionDatabaseOperations(
        db_urls,
        target,
        tenant_authority,
        tenant_event,
        psycopg2,
        psycopg2.extras,
    )


async def _emit_projection_terminal_event(
    event_bus: object,
    terminal_event: str,
    source_envelope: object,
) -> None:
    """Publish a terminal event after a successful DB projection.

    Propagates the source envelope's correlation_id so Pattern-B consumers
    and golden-chain tests can correlate the terminal event to the command.
    Best-effort: publish failures are logged but never propagate.
    """
    from datetime import UTC, datetime

    from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope

    try:
        source_payload = _extract_dispatch_payload(source_envelope)
        correlation_id = _coerce_uuid_or_none(
            _extract_dispatch_correlation_id(source_envelope, source_payload)
        )
        terminal_envelope = ModelEventEnvelope[object](
            payload={"projected": True},
            correlation_id=correlation_id,
            envelope_timestamp=datetime.now(UTC),
            event_type=terminal_event,
            source_tool="projection-reducer",
        )
        raw = terminal_envelope.model_dump_json().encode("utf-8")
        if hasattr(event_bus, "publish"):
            # Why: Control flow narrows this union at runtime before the attribute access.
            await event_bus.publish(terminal_event, None, raw)  # type: ignore[union-attr]
        else:
            logger.warning(
                "Projection terminal event not emitted: event_bus has no publish method "
                "(topic=%s)",
                terminal_event,
            )
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Failed to emit projection terminal event: topic=%s error_type=%s error=%s",
            terminal_event,
            type(exc).__name__,
            _sanitize_exc(exc),
        )


@dataclass(
    frozen=True
)  # internal-dataclass-ok: plain scalar extraction result, no runtime state
class StateIoMetadata:
    """Denormalized top-level columns extracted from an opaque state_io payload."""

    tenant_id: str
    state: str
    in_flight: bool


def _extract_state_io_metadata(payload_json: str) -> StateIoMetadata:
    """Extract the 3 denormalized top-level columns from an opaque state_io payload.

    omnibase_infra never decodes a state_io payload's business shape, but the
    contract-level convention (OMN-14208) is that every state_io payload
    exposes ``tenant_id`` / ``state`` / ``in_flight`` as well-known top-level
    JSON keys so this seam can populate the durable row's indexed columns
    (used by staleness sweeps) without understanding anything else about the
    payload. Missing/malformed keys degrade to safe defaults rather than
    raising — validating the payload's full business shape is the
    omnimarket-side codec's job, not a reason to fail the whole dispatch.
    """
    try:
        parsed = json.loads(payload_json)
    except (TypeError, ValueError):
        parsed = {}
    if not isinstance(parsed, dict):
        parsed = {}
    return StateIoMetadata(
        tenant_id=str(parsed.get("tenant_id") or ""),
        state=str(parsed.get("state") or ""),
        in_flight=bool(parsed.get("in_flight", False)),
    )


def _make_stateful_dispatch_callback(
    handler_instance: ProtocolHandleable,
    event_model: ModelHandlerRef | None,
    state_io: dict[str, object],
    *,
    event_bus: object | None = None,
    output_topic_map: dict[str, str] | None = None,
) -> DispatcherFunc:
    """Create a dispatch callback for contracts that declare ``state_io``.

    WRAPS (never replaces) :func:`_make_dispatch_callback` — this is a
    load-before / CAS-persist-after boundary hook around the exact same
    callback a non-stateful contract would get, preserving the OMN-13247
    envelope-coercion fallback and the ``payload_type_matcher`` scoping that
    multi-leg orchestrator contracts depend on untouched (OMN-14208).

    In-row outbox (OMN-14493 / OMN-14403 §4). When ``event_bus`` +
    ``output_topic_map`` are supplied, this wrapper is also the publish-from-row
    author for the state_io path: a leg's emitted events are persisted into the
    row's ``pending_emissions`` column WITHIN the same CAS that advances the FSM
    (commit-with-intent, ``in_flight=True``); the CAS winner then publishes those
    events FROM the committed row and CAS-finalizes (``in_flight=False``, batch
    cleared) all within the leg. This closes the CAS-retry result-selection race
    that stranded delegation rows at ``INFERENCE_COMPLETED`` — a losing/retried
    attempt can no longer lose an already-emitted intent, because the committed
    row (not the in-memory return of a possibly-losing attempt) is the publish
    source. Finalizing within the leg is load-bearing: a leg that leaves
    ``in_flight`` set would deadlock the next leg's CAS-retry forever. When no
    ``event_bus`` is supplied (the legacy/no-outbox path), the wrapper commits
    with intent and returns the result for the external applier to publish.

    Per-dispatch sequence:

    1. Load the correlation_id's current ``(payload_json, version)`` — ``None``
       if no row exists yet (e.g. this is the workflow's first leg).
    2. Set :data:`CONTEXTVAR_STATE_IO_ROWS` to ``{cid: (payload_json, version)}``
       UNCONDITIONALLY — a ``None`` payload_json for a missing row is still a
       *set* value, distinguishing "state_io active, no row yet" from
       "state_io inactive" for the ContextVar's cross-repo consumer (the
       omnimarket-side workflow-state proxy).
    3. Run the inner (unwrapped) dispatch callback — identical to what a
       non-stateful contract's dispatch would do.
    4. Call ``codec.flush(cid)`` on the contract-resolved codec instance — the
       explicit bridge to whatever the handler-side proxy decoded and mutated
       during step 3 (OMN-14208 pair-verify M1). ``None`` means the proxy
       never touched this correlation_id this dispatch (nothing to persist);
       a real payload means it did. This replaced an earlier implementation
       that expected the proxy to write its mutated payload back into
       :data:`CONTEXTVAR_STATE_IO_ROWS` itself — the ContextVar is a
       load-time input only, never a write-back channel.
    5. Persist the flushed payload: ``seed()`` if no row existed yet, else
       ``cas_update()``. A no-op dispatch (payload unchanged, or never
       populated) skips persistence entirely.
    6. The whole load -> handle -> persist unit is wrapped in
       ``retry_on_optimistic_conflict`` so a losing CAS/seed reloads the
       winning row and re-runs ``handle()`` against it — replay-safe because
       the FSM's synchronous in-flight dedup guard observes the freshly
       persisted flag on the retried attempt and folds without re-emitting.

    An exhausted retry raises ``OptimisticConflictError`` — never swallowed —
    so a resolvable-but-unresolved race is never reported as a successful
    dispatch. OMN-14600 CORRECTION: earlier revisions of this docstring
    claimed the raise "propagates — no offset commit, message redelivers".
    That is FALSE on this runtime: ``MessageDispatchEngine.dispatch()``
    catches every exception a wired dispatcher raises and converts it to a
    returned ``HANDLER_ERROR`` status rather than re-raising, and
    ``EventBusSubcontractWiring``'s consume callback has no status branch for
    ``HANDLER_ERROR`` — it commits the Kafka offset unconditionally once
    ``dispatch()`` returns. The raise here is still valuable (it surfaces the
    conflict loudly in logs/metrics and, for a caller that inspects the
    return value directly rather than going through the engine, genuinely
    propagates), but it is NOT a redelivery mechanism on the production
    consume path. The state_io in_flight-lock branch below (``_load_handle_
    persist``) therefore self-heals a stuck row INLINE rather than depending
    on this exception to trigger redelivery.
    """
    inner_callback = _make_dispatch_callback(handler_instance, event_model)

    database = str(state_io.get("database") or "omnibase_infra")
    table = str(state_io.get("table") or "")
    codec_ref = state_io.get("codec")
    if not table:
        raise ModelOnexError(
            "handler_wiring: state_io.table is required when a contract "
            "declares state_io."
        )
    if (
        not isinstance(codec_ref, dict)
        or not codec_ref.get("module")
        or not codec_ref.get("name")
    ):
        raise ModelOnexError(
            "handler_wiring: state_io.codec must declare {module, name} — "
            f"got {codec_ref!r}."
        )
    if database not in _DB_URL_ENV_MAP:
        raise ModelOnexError(
            f"handler_wiring: state_io.database {database!r} is unknown — "
            f"must be one of {sorted(_DB_URL_ENV_MAP)!r}."
        )
    db_url_env = _DB_URL_ENV_MAP[database]
    db_url = os.environ.get(db_url_env, "")
    if not db_url:
        raise StateIoUnconfiguredError(
            f"handler_wiring: contract declares state_io (table={table!r}) "
            f"but {db_url_env} is unset. state_io is a REQUIRED durability "
            "seam (OMN-14208) and fails closed at wiring time. Projection "
            "db_io topology bindings are likewise wiring-time requirements."
        )
    # Resolved once, at wiring time, not on every dispatch — mirrors the
    # fail-fast intent of every other _import_handler_class call in this
    # module. Unlike the handler class (imported but never used directly),
    # the codec IS used directly here: instantiated and called as the
    # explicit post-handle bridge (``codec.flush``) to whatever the
    # omnimarket-side ContextVar-backed workflow-state proxy decoded and
    # mutated during ``inner_callback`` (OMN-14208 pair-verify M0/M1).
    codec_cls = _import_handler_class(str(codec_ref["module"]), str(codec_ref["name"]))
    codec = codec_cls()

    pool_config = ModelPostgresPoolConfig.from_dsn(db_url, min_size=1, max_size=5)
    pool_provider = ProviderPostgresPool(pool_config)
    adapter = StateStoreAdapter(
        db_url,
        table=table,
        pool_factory=pool_provider.create,
    )
    from uuid import UUID, uuid5

    publish_topic_map: dict[str, str] = dict(output_topic_map or {})
    # 0.0 sentinel means "never run" -- always due on the first dispatch.
    _recovery_last_run_monotonic = 0.0
    _recovery_lock = asyncio.Lock()

    def _outbox_topic_for(class_name: str) -> str | None:
        """Resolve an outbox entry's Kafka topic from the published_events map.

        The contract's ``published_events`` map is authoritative (short name with
        the ``Model`` prefix stripped, then the full class name). A class with no
        mapping returns ``None`` — ``_publish_outbox_batch`` then raises rather
        than misrouting a fan-out element to a fallback topic (spec §2 Amendment C).
        """
        short = class_name.removeprefix("Model")
        return publish_topic_map.get(short) or publish_topic_map.get(class_name)

    def _build_outbox_entries(
        result: ModelDispatchResult | None,
        causation_envelope_id: str,
        cid: str,
        tenant_id: str,
    ) -> list[dict[str, object]]:
        """Serialize a leg's emitted events into row-storable outbox entries.

        Each entry carries exactly what recovery needs to rebuild the emitted
        envelope WITHOUT re-running the handler (spec §12): the event class'
        module+name, its own JSON payload, its index, and the causation /
        correlation / tenant scope for the deterministic id + tenant stamp.

        OMN-14600: the delegation terminal (the only state_io node today) is a
        BARE class emit — ``ModelDelegationCompleted`` / ``ModelDelegationFailed``,
        two distinct classes with no embedded topic field — resolved purely by
        class name via ``_outbox_topic_for`` / the contract's
        ``published_events`` map, same as every other emitted event. An
        earlier revision of this function detected an embedded-topic carrier
        (a ``ModelEventEnvelope`` or a typed payload's own ``.topic`` field)
        and fail-closed-validated it against ``allowed_output_topics`` —
        removed: dead code for delegation (its terminal carries no topic of
        its own to embed), and no other state_io node exists yet to need it.
        Add it back if a future state_io node genuinely needs an embedded,
        per-instance topic — don't carry speculative capability for a
        hypothetical node.
        """
        # A non-ModelDispatchResult return (e.g. the event_model=None cast-through
        # can hand back a raw list on clean dev — P3a's normalize coerces it) has
        # no output_events to store; the outbox only stores a real dispatch result.
        output_events = getattr(result, "output_events", None)
        if not output_events:
            return []
        entries: list[dict[str, object]] = []
        for idx, event in enumerate(output_events):
            if not isinstance(event, BaseModel):
                # OMN-14721: fail closed on a non-model element rather than the
                # prior silent ``continue``. A non-BaseModel in output_events
                # cannot be durably captured into a recoverable outbox entry;
                # dropping it silently shrinks the batch (K returned > K
                # captured) — the exact silent-emission-loss class this seam
                # exists to eliminate. Upstream ``_normalize_handler_result``
                # only ever hands BaseModel events here, so this is a fail-
                # closed invariant assertion, not a live filter.
                raise ModelOnexError(
                    message=(
                        "handler_wiring: state_io outbox capture received a "
                        f"non-BaseModel fan-out element {type(event).__name__!r} "
                        f"at index {idx} for correlation_id={cid} — cannot "
                        "durably capture the emission; failing closed rather "
                        "than silently shrinking the batch (OMN-14721)."
                    ),
                    error_code=EnumCoreErrorCode.HANDLER_EXECUTION_ERROR,
                )
            if publish_topic_map:
                # OMN-14721: resolve every captured emission's publish topic
                # through the SHARED core fan-out resolver at CAPTURE time
                # (the same resolver LocalRuntimeBusAdapter and the publish-
                # from-row path use). This fail-closes an unmapped emitted
                # class HERE, on the commit path, instead of seeding an
                # unpublishable batch the recovery sweep can never heal — it
                # mirrors ``_publish_outbox_batch``'s publish-time topic check,
                # moved one step earlier so capture and publish cannot diverge.
                resolve_published_topic(
                    publish_topic_map,
                    event,
                    message_type=type(event).__name__,
                )
            entries.append(
                {
                    "module": type(event).__module__,
                    "class_name": type(event).__name__,
                    "payload": event.model_dump(mode="json"),
                    "index": idx,
                    "causation_envelope_id": causation_envelope_id,
                    "correlation_id": cid,
                    "tenant_id": tenant_id,
                }
            )
        return entries

    def _rebuild_outbox_event(entry: dict[str, object]) -> BaseModel:
        """Rebuild the typed event model from a stored outbox entry.

        Stamps tenant_id + causation_id onto the payload FROM THE ROW (the
        applier does not, and in a fresh recovery process the input envelope is
        gone — the row is the ONLY source, spec §5 tenant-trap).

        Imports the event class via ``importlib`` (not ``_import_handler_class``)
        so the module/class recorded at commit time from an event WE emitted is
        rebuilt directly — the handler-class loader's namespace allowlist is for
        untrusted contract refs, not for a self-recorded outbox entry.

        OMN-14600: a pre-fix legacy-shaped entry (``_legacy_delegation_envelope_
        unwrap``) rebuilds the INNER ``ModelDelegationResult`` from the nested
        payload dict instead of the recorded carrier class — reconstructing the
        carrier itself would republish the double-nested bespoke shape no
        current consumer expects.
        """
        import importlib

        legacy = _legacy_delegation_envelope_unwrap(entry)
        if legacy is not None:
            _legacy_topic, inner_payload = legacy
            module = importlib.import_module(_LEGACY_DELEGATION_RESULT_MODULE)
            cls = getattr(module, _LEGACY_DELEGATION_RESULT_NAME)
            model = cls.model_validate(inner_payload)
        else:
            module = importlib.import_module(str(entry["module"]))
            cls = getattr(module, str(entry["class_name"]))
            model = cls.model_validate(entry["payload"])
        updates: dict[str, object] = {}
        fields = type(model).model_fields
        tenant_id = entry.get("tenant_id")
        if (
            tenant_id
            and "tenant_id" in fields
            and not getattr(model, "tenant_id", None)
        ):
            updates["tenant_id"] = tenant_id
        causation = entry.get("causation_envelope_id")
        if (
            causation
            and "causation_id" in fields
            and not getattr(model, "causation_id", None)
        ):
            updates["causation_id"] = UUID(str(causation))
        return model.model_copy(update=updates) if updates else model

    async def _publish_outbox_batch(entries: list[dict[str, object]]) -> int:
        """Publish a persisted outbox batch FROM THE ROW (recovery / resume).

        Rebuilds each envelope with the causation-scoped deterministic id (spec
        §8.1) + tenant, resolves its topic, and publishes via the bus.
        Idempotent: row-derived ids collapse against the original at the
        consume-path dedupe, so a duplicate re-publish is benign.

        Topic resolution: the contract's ``published_events`` class-name map
        (``_outbox_topic_for``) is authoritative — every current emit is a
        bare, un-carried class (OMN-14600). The ONE exception is
        ``_legacy_delegation_envelope_unwrap``: a pre-fix row committed by the
        old bespoke ``ModelDelegationEventEnvelope`` carrier, whose OWN nested
        dump is the only place that topic survived (the row predates the
        ``published_events`` split into completed/failed entries) — checked
        first so those specific stuck rows still resolve and self-heal.
        """
        from omnibase_core.models.events.model_event_envelope import (
            ModelEventEnvelope as _Envelope,
        )

        if event_bus is None or not hasattr(event_bus, "publish_envelope"):
            return 0
        published = 0
        for entry in entries:
            class_name = str(entry["class_name"])
            legacy = _legacy_delegation_envelope_unwrap(entry)
            topic = legacy[0] if legacy is not None else _outbox_topic_for(class_name)
            if topic is None:
                raise ModelOnexError(
                    "handler_wiring: outbox recovery cannot resolve a topic for "
                    f"fan-out class {class_name!r} — no published_events mapping."
                )
            cid_uuid = UUID(str(entry["correlation_id"]))
            causation = UUID(str(entry["causation_envelope_id"]))
            idx = int(cast("int", entry["index"]))
            envelope_id = uuid5(cid_uuid, f"{causation}:{class_name}:{idx}")
            payload = _rebuild_outbox_event(entry)
            # OMN-14743: stamp event_type from the resolved topic using the SAME
            # derivation the external applier uses (shared
            # ``derive_event_type_from_topic``). Without this the outbox emitted
            # ``event_type=None`` — this path bypasses the applier's OMN-12116
            # stamp (the state_io winner branch returns ``(1, None)``) — so the
            # routing reducer's type-scoped dispatcher (OMN-12294) dropped the
            # emission and delegation stalled at RECEIVED. A non-ONEX topic
            # derives ``None`` (the field default), so this is safe for every
            # topic shape the outbox can resolve.
            event_type = derive_event_type_from_topic(topic)
            out_envelope: _Envelope[BaseModel] = _Envelope(
                envelope_id=envelope_id,
                payload=payload,
                correlation_id=cid_uuid,
                event_type=event_type,
            )
            key: bytes | None = None
            for attr in ("entity_id", "node_id", "session_id", "correlation_id"):
                value = getattr(payload, attr, None)
                if value is not None:
                    key = str(value).encode("utf-8")
                    break
            await event_bus.publish_envelope(  # type: ignore[attr-defined]
                envelope=out_envelope, topic=topic, key=key
            )
            published += 1
        return published

    async def _finalize_outbox_row(
        cid: str,
        tenant_id: str,
        state: str,
        payload_json: str,
        expected_version: int,
    ) -> int:
        """CAS-finalize a published outbox row: in_flight=False + clear the batch.

        A CAS (not an unconditional UPDATE, spec §4.1 D1) so a concurrent
        recovery/leg that already finalized this row cannot be silently
        overwritten; a lost finalize means another path owns terminal state.
        """
        return await adapter.cas_update(
            cid,
            tenant_id=tenant_id,
            state=state,
            in_flight=False,
            payload_json=payload_json,
            expected_version=expected_version,
            pending_emissions=None,
        )

    async def _recover_outbox_batches(skip_cid: str | None = None) -> None:
        """Boot/redeploy re-publish of any in-flight row carrying a live batch.

        The adapter surfaces the recoverable rows (it has no bus); THIS wrapper
        publishes them (it does) with the same row-derived deterministic ids and
        CAS-finalizes — spec §4.1 D2 layering. Runs BEFORE recover_stale_rows so
        a recoverable row is re-emitted, not blind-FAILed (spec §4.2 R1).

        ``skip_cid`` is the correlation of the dispatch that TRIGGERED this boot
        sweep: that row is owned by the triggering leg's own in_flight-lock /
        resume path (which keys the resume on envelope_id), so the boot sweep
        must not race it — otherwise a redelivery of the crashed input would
        find its batch already recovered and needlessly re-run + fold the handler.

        OMN-14600 (Fable-gate correction): each row is recovered in its OWN
        try/except. Before this, one row that fails to recover (e.g. a stale
        pre-fix legacy-shaped entry the running code cannot resolve a topic
        for) raised out of the whole sweep — ``_ensure_stale_rows_recovered``
        never reached its ``_recovery_last_run_monotonic`` update, so the
        interval gate stayed permanently due and EVERY subsequent dispatch
        re-entered this same failing sweep and re-raised, silently dropping
        every triggering leg's own input. A row that fails here is logged and
        skipped; the sweep still completes, the timer still advances, and the
        failing row gets another attempt on the next interval (or self-heals
        via ``_legacy_delegation_envelope_unwrap`` if that was the cause).
        """
        if event_bus is None or not hasattr(adapter, "select_recoverable_batches"):
            return
        for row in await adapter.select_recoverable_batches():
            row_cid = str(row["correlation_id"])
            if skip_cid is not None and row_cid == skip_cid:
                continue
            entries = list(
                cast("list[dict[str, object]]", row.get("pending_emissions") or [])
            )
            if not entries:
                continue
            try:
                await _publish_outbox_batch(entries)
                await _finalize_outbox_row(
                    row_cid,
                    str(row["tenant_id"]),
                    str(row["state"]),
                    str(row["payload_json"]),
                    int(cast("int", row["version"])),
                )
            except Exception as exc:  # noqa: BLE001 — per-row isolation, see docstring
                logger.error(
                    "state_io recovery sweep: failed to recover row cid=%s (%s) "
                    "— skipping this row so the sweep completes and the "
                    "interval timer still advances (OMN-14600); retried on "
                    "the next sweep interval.",
                    row_cid,
                    _sanitize_exc(exc),
                )

    async def _ensure_stale_rows_recovered(skip_cid: str | None = None) -> None:
        """Run outbox re-publish + the give-up sweep, at most once per interval.

        Wiring is synchronous, so this cannot run at wiring time; the first live
        dispatch is the deterministic async point. Outbox re-publish runs FIRST
        so a recoverable in-flight batch is re-emitted before the give-up sweep
        (which fail-closes genuinely-abandoned rows) can touch it (OMN-14208 G1,
        OMN-14403 §4.1/§4.2 R1). Lock-guarded against concurrent same-tick runs.

        OMN-14600: this used to run EXACTLY ONCE per adapter lifetime (boot-time
        only, gated by a bool). A row whose winning leg crashes or otherwise
        never finalizes (e.g. a retry-storm cascade exhausting the in_flight-
        lock defer's own retries) stayed stranded until the next boot/redeploy
        -- a live, long-running process had no way to self-heal it. Gating on
        elapsed time instead makes this periodic: any dispatch that lands more
        than ``_STATE_IO_RECOVERY_SWEEP_INTERVAL_SECONDS`` after the last run
        re-triggers it, bounding how long a stranded row can survive in a busy
        process without requiring a dedicated background task.
        """
        nonlocal _recovery_last_run_monotonic
        now = time.monotonic()
        if (
            now - _recovery_last_run_monotonic
            < _STATE_IO_RECOVERY_SWEEP_INTERVAL_SECONDS
        ):
            return
        async with _recovery_lock:
            now = time.monotonic()
            if (
                now - _recovery_last_run_monotonic
                < _STATE_IO_RECOVERY_SWEEP_INTERVAL_SECONDS
            ):
                return
            await _recover_outbox_batches(skip_cid=skip_cid)
            await adapter.recover_stale_rows()
            _recovery_last_run_monotonic = time.monotonic()

    async def _find_recoverable_row(cid: str) -> dict[str, object] | None:
        """Return this cid's in-flight-with-live-batch row, if one exists.

        Backward-compatible: an adapter without ``select_recoverable_batches``
        (a legacy fake, or any non-outbox state store) has no in-row outbox, so
        the in_flight-lock is skipped and dispatch behaves exactly as pre-P3b.
        In steady state the partial index (``in_flight AND pending_emissions``)
        returns zero rows, so the per-dispatch check is a cheap empty index scan.
        """
        if not hasattr(adapter, "select_recoverable_batches"):
            return None
        for row in await adapter.select_recoverable_batches():
            if str(row["correlation_id"]) == cid:
                return row
        return None

    def _dispatch_result_from_entries(
        entries: list[dict[str, object]], cid: str
    ) -> ModelDispatchResult:
        """Rebuild a ModelDispatchResult carrying a row's batch (no-bus resume)."""
        from datetime import UTC, datetime

        from omnibase_infra.enums import EnumDispatchStatus
        from omnibase_infra.models.dispatch.model_dispatch_result import (
            ModelDispatchResult as _Result,
        )

        events = [_rebuild_outbox_event(e) for e in entries]
        return _Result(
            status=EnumDispatchStatus.SUCCESS,
            topic="",
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            output_count=len(events),
            output_events=events,
            correlation_id=UUID(cid),
        )

    async def _load_handle_persist(
        envelope: ModelEventEnvelope[object],
        cid: str,
    ) -> tuple[int, ModelDispatchResult | None]:
        # in_flight-lock (spec §4.1 D1): a row with a live, un-finalized outbox
        # batch MUST NOT run the handler — a concurrent leg would clobber the
        # winner's un-published intent. Resume-from-row if THIS input is the one
        # that created the batch (a redelivery: envelope_id == causation); else
        # defer without running the handler (branch b) so the winner's intent is
        # preserved. The resume predicate keys on envelope_id, NOT causation_id
        # (spec §4.1 E1 — a redelivered input keeps its envelope_id; its own
        # causation_id is the grandparent and would never match).
        incoming_envelope_id = getattr(envelope, "envelope_id", None)
        locked = await _find_recoverable_row(cid)
        if locked is not None:
            entries = list(
                cast("list[dict[str, object]]", locked.get("pending_emissions") or [])
            )
            causation = str(entries[0]["causation_envelope_id"]) if entries else None
            is_redelivery = (
                incoming_envelope_id is not None
                and causation is not None
                and str(incoming_envelope_id) == causation
            )
            if is_redelivery and entries:
                # Resume: re-publish the SAME batch from the row (idempotent, same
                # ids) + CAS-finalize. Handler NOT re-run (spec §4.1 E1/E2).
                if event_bus is not None:
                    await _publish_outbox_batch(entries)
                    await _finalize_outbox_row(
                        cid,
                        str(locked["tenant_id"]),
                        str(locked["state"]),
                        str(locked["payload_json"]),
                        int(cast("int", locked["version"])),
                    )
                    return 1, None
                # No bus: hand the batch back for the external applier + finalize.
                await _finalize_outbox_row(
                    cid,
                    str(locked["tenant_id"]),
                    str(locked["state"]),
                    str(locked["payload_json"]),
                    int(cast("int", locked["version"])),
                )
                return 1, _dispatch_result_from_entries(entries, cid)
            # A DIFFERENT input arrived while the batch is un-finalized. Do NOT
            # run the handler (would clobber the winner's un-published intent).
            #
            # OMN-14600 (Fable-gate correction, superseding the original
            # WIP): this branch used to return (1, None) -- a REPORTED SUCCESS
            # that commits the Kafka offset without ever running the handler
            # for THIS leg. The first fix attempt reported a conflict
            # (row_count=0) instead, relying on retry_on_optimistic_conflict's
            # in-process retries to exhaust and raise OptimisticConflictError
            # so the caller would redeliver. THAT DOES NOT WORK ON THIS
            # RUNTIME: MessageDispatchEngine.dispatch() catches every
            # exception a dispatcher raises (message_dispatch_engine.py's
            # per-dispatcher invocation loop) and converts it to a returned
            # HANDLER_ERROR status instead of re-raising; EventBusSubcontract
            # Wiring's consume callback has NO status branch for
            # HANDLER_ERROR and commits the Kafka offset unconditionally once
            # dispatch() returns. So a raised exception here NEVER triggers
            # redelivery -- it is silently absorbed and the message is gone.
            #
            # Fix: INLINE-RECOVER the winner's own batch right here instead of
            # depending on redelivery. This leg already holds the locked row +
            # its persisted entries -- publish them (idempotent: deterministic
            # uuid5 ids collapse a duplicate re-publish) and CAS-finalize using
            # the SAME helpers the is_redelivery resume branch above uses, so
            # the winner's stalled/crashed leg is completed deterministically.
            # Then report a conflict (row_count=0) so retry_on_optimistic_
            # conflict re-attempts THIS leg in-process against the now-cleared
            # lock -- it either runs the handler fresh or (if a genuinely
            # concurrent third leg raced in) defers again, never depending on
            # bus/engine redelivery semantics. No bus wired (test-only /
            # legacy-adapter path; production always passes one, see the
            # wiring call site) means we cannot publish from here -- the
            # winner's finalize is left to the external applier or the
            # periodic recovery sweep, unchanged from before.
            if event_bus is not None:
                await _publish_outbox_batch(entries)
                await _finalize_outbox_row(
                    cid,
                    str(locked["tenant_id"]),
                    str(locked["state"]),
                    str(locked["payload_json"]),
                    int(cast("int", locked["version"])),
                )
            logger.warning(
                "state_io in_flight-lock: deferring leg (cid=%s) as a conflict "
                "— a prior leg's outbox batch was committed but not yet "
                "finalized; inline-recovered (published + finalized) the "
                "winner's batch instead of relying on redelivery (OMN-14600 — "
                "redelivery is a dead path on this runtime, see comment "
                "above). Reported as row_count=0 so the caller retries this "
                "leg against the now-cleared lock.",
                cid,
            )
            return 0, None

        loaded = await adapter.load(cid)
        payload_json, version = loaded if loaded is not None else (None, 0)
        token = CONTEXTVAR_STATE_IO_ROWS.set({cid: (payload_json, version)})
        try:
            result = await inner_callback(envelope)
            flushed_payload_json = codec.flush(cid)
            new_payload_json = (
                flushed_payload_json
                if flushed_payload_json is not None
                else payload_json
            )
        finally:
            CONTEXTVAR_STATE_IO_ROWS.reset(token)

        if new_payload_json is None or (
            loaded is not None and new_payload_json == payload_json
        ):
            # Nothing changed (a no-op leg, e.g. the in-flight dedup guard
            # rejecting a duplicate without touching state) — skip
            # persistence entirely rather than bump version on a byte-
            # identical write. Reported as a successful (non-conflict) attempt.
            return 1, result

        metadata = _extract_state_io_metadata(new_payload_json)
        causation_envelope_id = (
            str(incoming_envelope_id) if incoming_envelope_id is not None else cid
        )
        # The in-row outbox is active only when the adapter supports it. A legacy
        # adapter (or a non-outbox state store) whose seed/cas_update have no
        # pending_emissions kwarg keeps the pre-P3b commit-then-return behavior
        # exactly (no batch persisted, no publish-from-row) — fully backward-compat.
        outbox_supported = hasattr(adapter, "select_recoverable_batches")
        entries = (
            _build_outbox_entries(
                result, causation_envelope_id, cid, metadata.tenant_id
            )
            if outbox_supported
            else []
        )
        # Commit-with-intent (spec §4.1): persist the advanced FSM state AND the
        # intended emissions in ONE CAS. in_flight=True marks "batch committed,
        # awaiting publish+finalize" so a crash between here and publish is
        # recoverable from the row alone.
        commit_in_flight = bool(entries) or metadata.in_flight
        # Pass pending_emissions only to an outbox-capable adapter; a legacy
        # adapter's seed/cas_update have no such kwarg (backward-compat).
        pending = entries or None

        # OMN-14721: fail-closed emission-completeness guard. A leg that NEWLY
        # seeds a fresh workflow (``loaded is None``) into a NON-terminal FSM
        # state MUST carry a durable emission — either a committed outbox batch
        # (``pending_emissions``) or an ``in_flight`` marker awaiting a follow-up
        # leg. A fresh, non-terminal row committed with in_flight=False AND an
        # empty batch (``not commit_in_flight``) is STRUCTURALLY unrecoverable:
        # ``select_recoverable_batches`` only re-publishes rows that are
        # ``in_flight AND jsonb_array_length(pending_emissions) > 0``, and
        # ``recover_stale_rows`` only give-up-FAILs it after the stale TTL — so
        # the emission the handler was supposed to produce to progress the
        # workflow is silently dropped forever and the row stalls (the OMN-14721
        # delegation routing-intent regression: RECEIVED / in_flight=false /
        # pending=∅). Convert that silent permanent drop into a LOUD dispatch
        # failure so it is surfaced + DLQ'd (via the OMN-14716 silent-dispatch-
        # failure guard) and can never recur unobserved.
        if (
            loaded is None
            and outbox_supported
            and not commit_in_flight
            and metadata.state not in _STATE_IO_TERMINAL_STATE_NAMES
        ):
            raise ModelOnexError(
                message=(
                    "handler_wiring: state_io refused to seed an unrecoverable "
                    f"dead row for correlation_id={cid} — a fresh workflow "
                    f"committing NON-terminal state={metadata.state!r} with "
                    "in_flight=False and an EMPTY outbox batch can never be "
                    "re-published by the recovery sweep "
                    "(select_recoverable_batches requires in_flight AND a "
                    "non-empty pending_emissions batch). The handler captured no "
                    "durable emission for a state that must emit to progress; "
                    "failing closed so the drop is surfaced and DLQ'd rather "
                    "than silently stranding the workflow (OMN-14721)."
                ),
                error_code=EnumCoreErrorCode.HANDLER_EXECUTION_ERROR,
            )

        if loaded is None:
            if outbox_supported:
                won = await adapter.seed(
                    cid,
                    tenant_id=metadata.tenant_id,
                    state=metadata.state,
                    in_flight=commit_in_flight,
                    payload_json=new_payload_json,
                    pending_emissions=pending,
                )
            else:
                won = await adapter.seed(
                    cid,
                    tenant_id=metadata.tenant_id,
                    state=metadata.state,
                    in_flight=commit_in_flight,
                    payload_json=new_payload_json,
                )
            row_count = 1 if won else 0
            committed_version = 0
        elif outbox_supported:
            row_count = await adapter.cas_update(
                cid,
                tenant_id=metadata.tenant_id,
                state=metadata.state,
                in_flight=commit_in_flight,
                payload_json=new_payload_json,
                expected_version=version,
                pending_emissions=pending,
            )
            committed_version = version + 1
        else:
            row_count = await adapter.cas_update(
                cid,
                tenant_id=metadata.tenant_id,
                state=metadata.state,
                in_flight=commit_in_flight,
                payload_json=new_payload_json,
                expected_version=version,
            )
            committed_version = version + 1

        if row_count == 0:
            # Lost the CAS — a concurrent winner advanced the row. Retry reloads
            # against the winner (which committed + publishes ITS own intent), so
            # this attempt's events are correctly dropped, never lost.
            return 0, result

        # Winner. If the wrapper owns a bus AND emitted a batch, publish-from-row
        # + CAS-finalize WITHIN this leg — leaving the row in_flight would deadlock
        # the next leg's CAS-retry (the stuck-at-INFERENCE_COMPLETED symptom).
        # Return None so the external applier does NOT re-publish the same batch
        # (no double-publish — OMN-14403 §7 decision 1).
        if entries and event_bus is not None:
            try:
                await _publish_outbox_batch(entries)
            except Exception as exc:
                raise BoundaryPublishError(
                    "handler_wiring: state_io outbox publish-from-row failed"
                ) from exc
            await _finalize_outbox_row(
                cid,
                metadata.tenant_id,
                metadata.state,
                new_payload_json,
                committed_version,
            )
            return 1, None

        # No-bus path (test commit / non-emitting leg): hand the result back for
        # the external applier to publish; the row keeps its committed intent.
        return 1, result

    async def _callback(
        envelope: ModelEventEnvelope[object],
    ) -> ModelDispatchResult | None:
        payload = _extract_dispatch_payload(envelope)
        correlation_id = _coerce_uuid_or_none(
            _extract_dispatch_correlation_id(envelope, payload)
        )
        if correlation_id is None:
            raise ModelOnexError(
                "handler_wiring: state_io dispatch requires a correlation_id "
                "on every leg — got none. Legs 2-5 of a multi-leg "
                "orchestrator carry no tenant_id on the wire but MUST carry "
                "correlation_id (the state_io read key)."
            )
        cid = str(correlation_id)

        await _ensure_stale_rows_recovered(skip_cid=cid)

        _row_count, result = await retry_on_optimistic_conflict(
            lambda: _load_handle_persist(envelope, cid),
            check_conflict=lambda outcome: outcome[0] == 0,
            correlation_id=cast("UUID", correlation_id),
        )
        return result

    return _callback


def _raise_if_silent_dispatch_failure(
    result: object,
    topic: str,
) -> None:
    """Raise ``HandlerDispatchFailureError`` for a FAILED dispatch result that
    would otherwise vanish silently (OMN-14716).

    ``MessageDispatchEngine.dispatch()`` converts a dispatcher crash (a def-B
    handler AttributeError, a boundary coercion ``ValidationError``, ...) into a
    ``HANDLER_ERROR``/``INTERNAL_ERROR`` result rather than re-raising, and
    ``DispatchResultApplier.apply()`` silently skips a non-SUCCESS result that
    carries no applicable output. That combination is the exact
    "handler crashed, both terminal topics stayed HWM=0, nothing surfaced"
    incident shape. Detect only that shape — an error status with NO
    output_events / output_intents / projection_intents — and raise so the
    boundary routes it through ``_route_swallowed_exception`` (loud metric log +
    best-effort DLQ).

    A partial-success ``HANDLER_ERROR`` (one sibling handler failed but another
    produced output the applier publishes) is intentionally NOT surfaced here —
    it is not a silent drop. Non-error statuses (SUCCESS/NO_DISPATCHER/... — the
    latter has its own engine-side DLQ routing) are left untouched. Guarded on a
    real ``ModelDispatchResult`` instance so a ``MagicMock`` result in a test
    never trips it.
    """
    from omnibase_infra.enums import EnumDispatchStatus
    from omnibase_infra.models.dispatch.model_dispatch_result import (
        ModelDispatchResult,
    )

    if not isinstance(result, ModelDispatchResult):
        return
    if result.status not in (
        EnumDispatchStatus.HANDLER_ERROR,
        EnumDispatchStatus.INTERNAL_ERROR,
    ):
        return
    has_applicable_output = bool(
        result.output_events or result.output_intents or result.projection_intents
    )
    if has_applicable_output:
        return
    raise HandlerDispatchFailureError(
        f"dispatch to topic={topic} returned status={result.status.value} with no "
        f"terminal output (dispatcher_id={result.dispatcher_id}): "
        f"{result.error_message or 'handler/coercion failure'}"
    )


def _normalize_contract_dispatcher_scope(
    dispatcher_ids: Collection[str] | None,
    *,
    contract_name: str,
    allow_empty: bool,
) -> frozenset[str]:
    """Return a canonical unique dispatcher scope without trusting transport state.

    A contract-owned Kafka callback must never fall back to process-global
    dispatch. Two contracts can intentionally consume the same topic under
    distinct groups; global fan-out from each callback executes both contracts'
    handlers once per group (OMN-15474). The live engine owner registry, not a
    serialized wiring report, is authoritative for completeness.
    """
    if dispatcher_ids is None:
        raise ModelOnexError(
            message=(
                "handler_wiring: contract-scoped subscription is missing its "
                f"dispatcher scope for contract {contract_name!r}; refusing "
                "process-global fan-out."
            ),
            error_code=EnumCoreErrorCode.INVALID_CONFIGURATION,
        )
    raw_dispatcher_ids = tuple(dispatcher_ids)
    seen: set[str] = set()
    duplicate_ids: set[str] = set()
    for dispatcher_id in raw_dispatcher_ids:
        if dispatcher_id in seen:
            duplicate_ids.add(dispatcher_id)
        seen.add(dispatcher_id)
    if (
        (not raw_dispatcher_ids and not allow_empty)
        or any(
            not dispatcher_id or dispatcher_id != dispatcher_id.strip()
            for dispatcher_id in raw_dispatcher_ids
        )
        or duplicate_ids
    ):
        raise ModelOnexError(
            message=(
                "handler_wiring: contract-scoped subscription has an empty or "
                f"invalid dispatcher scope for contract {contract_name!r} "
                f"(duplicates={sorted(duplicate_ids)}); "
                "refusing process-global fan-out."
            ),
            error_code=EnumCoreErrorCode.INVALID_CONFIGURATION,
        )
    return frozenset(raw_dispatcher_ids)


def _require_contract_dispatcher_scope(
    dispatcher_ids: Collection[str] | None,
    *,
    contract_name: str,
) -> frozenset[str]:
    """Return a canonical non-empty dispatcher scope before subscribing."""
    return _normalize_contract_dispatcher_scope(
        dispatcher_ids,
        contract_name=contract_name,
        allow_empty=False,
    )


def _require_unique_canonical_contract_names(
    contract_names: Sequence[str],
    *,
    identity_source: str,
) -> frozenset[str]:
    """Return an exact identity set or reject aliases and duplicate rows."""
    raw_names = tuple(contract_names)
    noncanonical_names = tuple(
        sorted(
            {
                contract_name
                for contract_name in raw_names
                if not contract_name or contract_name != contract_name.strip()
            }
        )
    )
    if noncanonical_names:
        raise ModelOnexError(
            message=(
                f"handler_wiring: noncanonical {identity_source} contract names "
                f"are not valid subscription identities: {noncanonical_names}"
            ),
            error_code=EnumCoreErrorCode.INVALID_CONFIGURATION,
        )

    seen: set[str] = set()
    duplicate_names: set[str] = set()
    for contract_name in raw_names:
        if contract_name in seen:
            duplicate_names.add(contract_name)
        seen.add(contract_name)
    if duplicate_names:
        raise ModelOnexError(
            message=(
                f"handler_wiring: duplicate {identity_source} contract names "
                "would collapse or schedule repeated consumer attachment: "
                f"{sorted(duplicate_names)}"
            ),
            error_code=EnumCoreErrorCode.INVALID_CONFIGURATION,
        )
    return frozenset(raw_names)


def _validate_initial_subscription_contract_identities(
    manifest: ModelAutoWiringManifest,
    report: ModelAutoWiringReport,
) -> None:
    """Require a canonical, exact bijection between report and manifest names.

    OMN-15474 ruling 4 (re-affirmed by OMN-15621 after PR #2609 narrowed this
    to a report-subset-of-manifest check, contrary to the ruling). Both
    directions are load-bearing for single-owner dispatch:

    1. **Uniqueness** on each side. A repeated contract name would schedule a
       repeated consumer attachment for one identity — the same
       execute-the-command-twice class this ticket exists to close.
    2. **report ⊆ manifest.** A report row naming a contract the manifest never
       declared is an identity error: it would attach a consumer for a contract
       this boot does not own.
    3. **manifest ⊆ report.** A manifest contract with no report row at all is
       indistinguishable from one that silently vanished from the wiring
       pass — exactly the class of bug that produced process-global dispatch
       (OMN-15474). This direction is safe to assert unconditionally because
       the report is contractually TOTAL over the manifest it was built from:
       :func:`wire_from_manifest` backfills one explicit SKIPPED row per
       uncovered contract via :func:`build_unwired_contract_results` before it
       returns (see the "OMN-15474 totality post-condition" comment there), so
       a contract that failed to wire, was resolver-skipped, or was
       quarantined still produces a row — it is simply not ``WIRED``. A
       missing row is therefore never legitimate; it means some caller handed
       this function a report that was never produced by the real producer
       (e.g. a hand-truncated report in a test), or a manifest that differs
       from the one the report was built against. Both are boot-time bugs.

    A prior revision of this docstring claimed the reverse direction "refused
    the boot outright against the full shipped manifest (missing_from_report
    = 118 contracts)". That is not reproducible against the current producer:
    a live run of the real main-profile manifest (118 contracts) through
    :func:`wire_from_manifest` yields a report that is already exactly total
    (0 missing, 0 unexpected) before this check ever runs. The 118 in that
    docstring was the full manifest size, not a genuine gap — see OMN-15621.
    """
    manifest_names = _require_unique_canonical_contract_names(
        tuple(contract.name for contract in manifest.contracts),
        identity_source="manifest",
    )
    report_names = _require_unique_canonical_contract_names(
        tuple(result.contract_name for result in report.results),
        identity_source="report",
    )
    if report_names != manifest_names:
        raise ModelOnexError(
            message=(
                "handler_wiring: report and manifest contract-name mismatch; "
                "initial subscription requires an exact bijection "
                f"(missing_from_report={sorted(manifest_names - report_names)}, "
                f"unexpected_in_report={sorted(report_names - manifest_names)})"
            ),
            error_code=EnumCoreErrorCode.INVALID_CONFIGURATION,
        )


def _validate_not_ready_contract_identities(
    manifest: ModelAutoWiringManifest,
    not_ready_results: Sequence[ModelContractAttachResult],
) -> tuple[ModelContractAttachResult, ...]:
    """Return uniquely named NOT_READY rows forming a valid manifest subset."""
    manifest_names = _require_unique_canonical_contract_names(
        tuple(contract.name for contract in manifest.contracts),
        identity_source="manifest",
    )
    pending_results = tuple(
        result
        for result in not_ready_results
        if result.status is EnumContractAttachStatus.NOT_READY
    )
    not_ready_names = _require_unique_canonical_contract_names(
        tuple(result.contract_name for result in pending_results),
        identity_source="NOT_READY",
    )
    unexpected_names = not_ready_names.difference(manifest_names)
    if unexpected_names:
        raise ModelOnexError(
            message=(
                "handler_wiring: NOT_READY and manifest contract-name mismatch; "
                "reattach identities must be a manifest subset "
                f"(unexpected={sorted(unexpected_names)})"
            ),
            error_code=EnumCoreErrorCode.INVALID_CONFIGURATION,
        )
    return pending_results


def _require_contract_scoped_dispatch_engine(
    dispatch_engine: object,
    *,
    contract_name: str,
) -> ProtocolContractScopedDispatchEngine:
    """Resolve the explicit scoped-dispatch capability before consumer attach."""
    scoped_dispatch = getattr(dispatch_engine, "dispatch_scoped", None)
    if not callable(scoped_dispatch):
        raise ModelOnexError(
            message=(
                "handler_wiring: contract-scoped subscription requires an "
                f"explicit scoped dispatch capability for contract {contract_name!r}; "
                "refusing to attach a consumer that could fail after delivery."
            ),
            error_code=EnumCoreErrorCode.INVALID_CONFIGURATION,
        )
    return cast("ProtocolContractScopedDispatchEngine", dispatch_engine)


def _validate_registered_contract_dispatcher_scope(
    dispatch_engine: object,
    dispatcher_scope: frozenset[str],
    *,
    contract_name: str,
) -> frozenset[str]:
    """Compare one normalized scope with the complete live engine owner set."""
    scoped_engine = _require_contract_scoped_dispatch_engine(
        dispatch_engine,
        contract_name=contract_name,
    )
    ownership_validator = getattr(
        dispatch_engine,
        "validate_contract_dispatcher_scope",
        None,
    )
    if not callable(ownership_validator):
        raise ModelOnexError(
            message=(
                "handler_wiring: contract-scoped subscription requires a "
                "dispatcher ownership validation capability for contract "
                f"{contract_name!r}; refusing to attach without proving "
                "current engine membership."
            ),
            error_code=EnumCoreErrorCode.INVALID_CONFIGURATION,
        )
    return scoped_engine.validate_contract_dispatcher_scope(
        contract_name,
        dispatcher_scope,
    )


def _require_registered_contract_dispatcher_scope(
    dispatch_engine: object,
    dispatcher_ids: Collection[str] | None,
    *,
    contract_name: str,
) -> frozenset[str]:
    """Require one non-empty exact scope to exist on the current engine."""
    dispatcher_scope = _require_contract_dispatcher_scope(
        dispatcher_ids,
        contract_name=contract_name,
    )
    return _validate_registered_contract_dispatcher_scope(
        dispatch_engine,
        dispatcher_scope,
        contract_name=contract_name,
    )


def _validate_contract_dispatcher_ownership(
    dispatch_engine: object,
    dispatcher_scopes: Sequence[tuple[str, Collection[str]]],
    *,
    allow_empty_scopes: bool = False,
) -> None:
    """Validate current engine membership and one-contract ownership.

    Reports and persisted NOT_READY results are typed transport artifacts, not
    engine authority. Validate every referenced dispatcher before provisioning
    or attaching any consumer. One contract may own multiple unique dispatcher
    IDs; one dispatcher ID may never be claimed by multiple contracts.
    """
    normalized_scopes: list[tuple[str, frozenset[str]]] = []
    owners_by_dispatcher: dict[str, set[str]] = defaultdict(set)
    for contract_name, dispatcher_ids in dispatcher_scopes:
        dispatcher_scope = _normalize_contract_dispatcher_scope(
            dispatcher_ids,
            contract_name=contract_name,
            allow_empty=allow_empty_scopes,
        )
        normalized_scopes.append((contract_name, dispatcher_scope))
        for dispatcher_id in dispatcher_scope:
            owners_by_dispatcher[dispatcher_id].add(contract_name)

    multiply_owned = {
        dispatcher_id: tuple(sorted(owners))
        for dispatcher_id, owners in owners_by_dispatcher.items()
        if len(owners) > 1
    }
    if multiply_owned:
        raise ModelOnexError(
            message=(
                "handler_wiring: contract-scoped subscription assigns "
                "dispatcher IDs to multiple contracts; refusing consumer "
                f"attach: {multiply_owned}"
            ),
            error_code=EnumCoreErrorCode.INVALID_CONFIGURATION,
        )

    for contract_name, dispatcher_scope in normalized_scopes:
        _validate_registered_contract_dispatcher_scope(
            dispatch_engine,
            dispatcher_scope,
            contract_name=contract_name,
        )


async def _dispatch_to_contract_scope(
    dispatch_engine: ProtocolContractScopedDispatchEngine,
    topic: str,
    envelope: ModelEventEnvelope[object],
    allowed_dispatcher_ids: frozenset[str],
) -> ModelDispatchResult:
    """Dispatch through the engine while preserving callback ownership."""
    return await dispatch_engine.dispatch_scoped(
        topic,
        envelope,
        allowed_dispatcher_ids=allowed_dispatcher_ids,
    )


def _make_event_bus_callback(
    topic: str,
    dispatch_engine: ProtocolDispatchEngine,
    result_applier: ProtocolDispatchResultApplier | None = None,
    *,
    tenant_scoped: bool = False,
    event_bus: object | None = None,
    propagate_publish_failures: bool = False,
    allowed_dispatcher_ids: Collection[str] | None = None,
) -> Callable[..., Awaitable[None]]:
    """Create a Kafka on_message callback that deserializes and dispatches to engine.

    Mirrors EventBusSubcontractWiring._create_dispatch_callback but stripped of
    DLQ/idempotency concerns. When a result applier is supplied, dispatcher
    outputs are applied on the same auto-wired path that owns the subscription.

    ``tenant_scoped`` (OMN-14349, OMN-14208 Path A): when True, derives a
    verified tenant_id from this topic's ``tenant-<slug>.`` wire prefix and
    stamps it into the payload before ``dispatch_engine.dispatch()`` is ever
    called -- overwriting any client-supplied value. This is the layer where
    ``topic`` is genuinely in scope with the envelope still mutable (proven:
    it already derives ``event_type`` from ``topic`` below); the per-handler
    dispatch callback further downstream (``_make_dispatch_callback``) never
    sees ``topic`` at all, so the stamp cannot happen there. A topic with no
    ``tenant-<slug>.`` prefix is left completely unstamped -- never given a
    defaulted or guessed tenant (Stage-1 warn semantics, OMN-14208 §5.1).

    ``event_bus`` (OMN-14507): optional handle to the bus this topic was
    subscribed on. Duck-typed for a ``_publish_raw_to_dlq`` method exactly
    like ``EventBusSubcontractWiring._publish_to_dlq`` -- when present AND
    ``_boundary_dlq_enabled()`` is True, a handler exception that survives
    the bounded retry is routed there instead of vanishing. ``None`` (the
    default) preserves the historical no-DLQ callback shape for any
    caller/test that does not pass one.
    """
    import json

    from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope

    dispatcher_scope = _require_contract_dispatcher_scope(
        allowed_dispatcher_ids,
        contract_name=topic,
    )
    scoped_dispatch_engine = _require_contract_scoped_dispatch_engine(
        dispatch_engine,
        contract_name=topic,
    )

    def _derive_event_type_from_topic(topic: str) -> str | None:
        parts = topic.split(".")
        if len(parts) >= 5 and parts[0] == "onex":
            return f"{parts[2]}.{parts[3]}"
        return None

    async def _dispatch_with_bounded_retry(
        envelope: ModelEventEnvelope[object],
    ) -> None:
        """Dispatch + apply, retrying a bounded number of times on failure.

        A single attempt (no retry, no sleep) when the boundary-DLQ flag is
        off -- matches the pre-OMN-14507 call shape exactly (dispatch once,
        apply once). Only the dispatch/apply step is retried: a deserialize
        failure above this point is a content error (malformed JSON/schema)
        that retrying can never fix.

        Non-retryable classification (OMN-14507 review, gap G2): a Pydantic
        ``ValidationError`` or ``ProtocolConfigurationError`` raised BY the
        dispatch/apply step is itself a content/config error -- e.g. a
        handler-level wire-model rejecting an unknown field under
        ``extra="forbid"`` (the exact §7 death signal this boundary exists to
        carry), or ``ProtocolConfigurationError`` from a missing dispatcher.
        Both are deterministic: retrying burns the full backoff budget for a
        guaranteed-identical failure. These break out of the loop on the
        FIRST occurrence and go straight to the caller's DLQ/log handling,
        matching the sibling classifier's (``EventBusSubcontractWiring``)
        non-retryable treatment of content errors. Everything else (network
        blips, transient infra errors) is retried up to
        ``_BOUNDARY_DLQ_MAX_ATTEMPTS`` times.

        Idempotency note (gap G4): a handler that performs a side effect
        before raising will have that side effect repeated on each retry
        attempt within a single delivery (in addition to Kafka's own
        at-least-once redelivery). Handlers on this boundary are expected to
        be idempotent already for that reason; this does not introduce a new
        assumption, only a higher chance of exercising it within one message.
        """
        from pydantic import ValidationError as PydanticValidationError

        from omnibase_infra.errors import ProtocolConfigurationError

        attempts = _BOUNDARY_DLQ_MAX_ATTEMPTS if _boundary_dlq_enabled() else 1
        last_exc: Exception | None = None
        for attempt in range(attempts):
            try:
                if not await _wait_for_dispatch_engine_freeze(topic, dispatch_engine):
                    return
                result = await _dispatch_to_contract_scope(
                    scoped_dispatch_engine,
                    topic,
                    envelope,
                    dispatcher_scope,
                )
                if result_applier is not None and result is not None:
                    try:
                        await result_applier.apply(result, envelope.correlation_id)
                    except Exception as apply_exc:
                        # OMN-14403 §4.3: on the outbox path a publish failure
                        # must PROPAGATE (redeliver), never be retried-then-
                        # swallowed. Tag it so the loop breaks and the outer
                        # handler re-raises. Off the outbox path this is a no-op:
                        # the original exception rides the generic retry arm.
                        if propagate_publish_failures:
                            raise BoundaryPublishError(
                                "outbox publish failed"
                            ) from apply_exc
                        raise
                # OMN-14716: the engine catch-all converts a dispatcher crash (a
                # def-B handler AttributeError, a boundary coercion failure) into a
                # FAILED result instead of re-raising, and the applier silently
                # skips a non-SUCCESS result with no output. Surface that shape so
                # it is logged + best-effort-DLQ'd here instead of vanishing at
                # HWM=0 -- routed through the same _route_swallowed_exception path
                # a raised handler exception takes.
                _raise_if_silent_dispatch_failure(result, topic)
                return
            except (
                PydanticValidationError,
                ProtocolConfigurationError,
                BoundaryPublishError,
                HandlerDispatchFailureError,
            ) as exc:
                # Non-retryable: deterministic content/config error, an outbox
                # publish failure that must propagate (not retry), or a FAILED
                # dispatch result the engine already produced deterministically
                # (OMN-14716). No backoff, no further attempts -- see docstring
                # gap G2 + OMN-14403 §4.3.
                last_exc = exc
                break
            except Exception as exc:  # noqa: BLE001 — bounded-retry loop; re-raised below on exhaustion
                last_exc = exc
                if attempt < attempts - 1:
                    backoff = _BOUNDARY_DLQ_RETRY_BACKOFF_SECONDS[
                        min(attempt, len(_BOUNDARY_DLQ_RETRY_BACKOFF_SECONDS) - 1)
                    ]
                    logger.warning(
                        "Auto-wiring callback retry: topic=%s attempt=%d/%d "
                        "error_type=%s error=%s backoff_s=%.2f",
                        topic,
                        attempt + 1,
                        attempts,
                        type(exc).__name__,
                        _sanitize_exc(exc),
                        backoff,
                    )
                    await asyncio.sleep(backoff)
        assert last_exc is not None
        raise last_exc

    async def _route_swallowed_exception(
        exc: Exception,
        message: object,
        correlation_id: UUID,
    ) -> None:
        """Handle a handler exception that survived dispatch (OMN-14507).

        flag OFF (default): identical swallow-and-ACK semantics to the
        pre-fix behavior -- one ``logger.error`` -- plus a structured
        metric-shaped log line so the swallow is at least observable (the
        DEFAULT-OFF, warn-first stage of the rollout).

        flag ON: additionally attempts to durably preserve the message in
        the topic's DLQ via the same duck-typed ``_publish_raw_to_dlq``
        contract ``EventBusSubcontractWiring._publish_to_dlq`` already
        depends on. If no ``event_bus`` was supplied, the bus does not
        expose that method, or the DLQ publish itself raises, this degrades
        to the same loud-log-only path -- it never raises and never blocks
        the boundary from returning.

        Honesty note (OMN-14507 review, gap G1): this is BEST-EFFORT DLQ
        delivery, not true at-least-once. The offset always advances (no
        nack/redelivery here) -- if the retry budget is exhausted AND the
        DLQ publish itself fails, the message IS lost, just loudly instead of
        silently. Strictly better than the pre-fix 100% swallow, but callers
        must not read "flag ON" as a guarantee that no message can ever be
        lost; that would require true nack/redelivery, deliberately deferred
        to a follow-up (see G1 in the PR review).

        Metric naming (gap G3): only the case that ACTUALLY prevented loss
        (``dlq_routed=true``) is logged as ``boundary_swallow_prevented``. The
        flag-off path and the DLQ-unavailable/DLQ-publish-failed paths are
        logged as ``boundary_swallow_observed`` -- nothing was prevented
        there, only observed; conflating the two would mislead an operator
        alerting on the "prevented" counter into believing the message survived.
        """
        from omnibase_infra.enums import EnumInfraTransportType
        from omnibase_infra.errors import ModelInfraErrorContext
        from omnibase_infra.utils.util_error_sanitization import (
            sanitize_error_message,
        )

        sanitized = sanitize_error_message(exc)
        logger.error(
            "Auto-wiring callback error: topic=%s error_type=%s error=%s "
            "correlation_id=%s",
            topic,
            type(exc).__name__,
            sanitized,
            correlation_id,
        )
        dlq_enabled = _boundary_dlq_enabled()
        publish_dlq_fn = (
            getattr(event_bus, "_publish_raw_to_dlq", None)
            if dlq_enabled and event_bus is not None
            else None
        )
        if publish_dlq_fn is None or not callable(publish_dlq_fn):
            # metric surface: a structured, greppable log line stands in for a
            # counter emission in this DRAFT (see PR body for the follow-up).
            # Nothing was prevented here -- see gap G3 above.
            logger.error(
                "metric_name=boundary_swallow_observed dlq_routed=false "
                "dlq_enabled=%s topic=%s error_type=%s correlation_id=%s",
                dlq_enabled,
                topic,
                type(exc).__name__,
                correlation_id,
            )
            return

        def _increment_message_lost_counter() -> None:
            # OMN-14551: this IS the alertable signal -- the log line at each
            # call site is greppable but not pageable. Never let metric
            # emission itself become a new swallow site.
            if _BOUNDARY_MESSAGE_LOST_COUNTER is None:
                return
            try:
                _BOUNDARY_MESSAGE_LOST_COUNTER.labels(
                    topic=topic, error_type=type(exc).__name__
                ).inc()
            except Exception as metric_exc:  # noqa: BLE001 — metric emission must never crash the consumer
                context = ModelInfraErrorContext.with_correlation(
                    correlation_id=correlation_id,
                    transport_type=EnumInfraTransportType.KAFKA,
                    operation="increment_message_lost_counter",
                    target_name=topic,
                    original_error_type=type(metric_exc).__name__,
                )
                logger.warning(
                    "Failed to increment onex_boundary_message_lost_total "
                    "metric for topic=%s (message loss above is still "
                    "authoritative): context=%s",
                    topic,
                    context.model_dump(mode="json", exclude_none=True),
                )

        try:
            from omnibase_infra.event_bus.topic_constants import (
                get_dlq_topic_for_original,
            )

            dlq_persisted = await publish_dlq_fn(
                original_topic=topic,
                raw_msg=message,
                error=exc,
                correlation_id=correlation_id,
                failure_type="handler_exception",
                consumer_group="auto-wiring",
                dlq_topic=get_dlq_topic_for_original(topic),
            )
            if dlq_persisted:
                logger.error(
                    "metric_name=boundary_swallow_prevented dlq_routed=true "
                    "dlq_enabled=%s topic=%s error_type=%s correlation_id=%s",
                    dlq_enabled,
                    topic,
                    type(exc).__name__,
                    correlation_id,
                )
            else:
                # OMN-14936: a False return means the publish did NOT
                # durably persist (rejected input, producer unavailable, or
                # the send itself failed/timed out) WITHOUT raising -- the
                # message is lost exactly like the except-branch below, just
                # signaled through the return value instead of an exception.
                # Reusing "dlq_publish_failed=true" here (rather than a new
                # token) keeps this the same alertable shape as the
                # exception path for any existing log-based consumer.
                logger.error(
                    "metric_name=boundary_swallow_observed dlq_routed=false "
                    "dlq_enabled=%s dlq_publish_failed=true message_lost=true "
                    "topic=%s error_type=%s correlation_id=%s",
                    dlq_enabled,
                    topic,
                    type(exc).__name__,
                    correlation_id,
                )
                _increment_message_lost_counter()
                # OMN-14498: a NACK must never ACK the offset. Returning
                # normally here IS an ACK -- _dispatch_to_subscriber reads
                # "no exception" as success and lets the offset advance --
                # so the record would be acknowledged while existing nowhere
                # durable, and the OMN-15232 rewind path would never see it.
                raise BoundaryDlqNotPersistedError(topic, correlation_id, exc)
        except BoundaryDlqNotPersistedError:
            raise
        except Exception as dlq_exc:
            # Best-effort DLQ failed too -- the message IS lost here (gap G1).
            # Loud, not silent, but not prevented -- see gap G3 above.
            logger.error(
                "metric_name=boundary_swallow_observed dlq_routed=false "
                "dlq_enabled=%s dlq_publish_failed=true message_lost=true "
                "topic=%s error_type=%s dlq_error=%s correlation_id=%s",
                dlq_enabled,
                topic,
                type(exc).__name__,
                sanitize_error_message(dlq_exc),
                correlation_id,
            )
            _increment_message_lost_counter()
            # Same invariant as the False-return branch above: the DLQ write
            # is not durable, so the offset must be withheld rather than
            # advanced over a message that exists nowhere.
            raise BoundaryDlqNotPersistedError(topic, correlation_id, exc) from dlq_exc

    async def callback(message: object) -> None:
        from uuid import uuid4

        # OMN-14498: seed lineage from the INGRESS transport headers before
        # anything can fail. The body is not a reliable lineage source -- a
        # poisoned message (truncated/undecodable JSON) raises inside
        # json.loads below, before either body-derived recovery
        # (envelope.correlation_id / data["correlation_id"]) can run, and the
        # boundary then fell through to the DLQ still holding a freshly
        # minted uuid4. That produced a VALID id with the WRONG lineage: the
        # DLQ record, and every faithful replay of it, carried a fabricated
        # ancestry, so the resulting terminal joined to nothing upstream.
        # Precedence is ingress header -> body -> mint, so a decodable
        # envelope still wins (it is the authoritative in-band value) and a
        # message with no lineage anywhere still gets a usable id.
        correlation_id: UUID = _ingress_correlation_id(message) or uuid4()
        try:
            raw = getattr(message, "value", None)
            if raw is not None:
                data = json.loads(
                    raw.decode("utf-8") if isinstance(raw, bytes) else raw
                )
                from pydantic import ValidationError as PydanticValidationError

                try:
                    envelope: ModelEventEnvelope[object] = ModelEventEnvelope[
                        object
                    ].model_validate(data)
                except PydanticValidationError:
                    # Raw command payload (no envelope wrapper) — synthesize one.
                    from datetime import UTC, datetime

                    raw_corr = (
                        data.get("correlation_id") if isinstance(data, dict) else None
                    )
                    corr = _coerce_uuid_or_none(raw_corr) or uuid4()
                    derived = _derive_event_type_from_topic(topic)
                    envelope = ModelEventEnvelope[object](
                        payload=data,
                        correlation_id=corr,
                        envelope_timestamp=datetime.now(UTC),
                        event_type=derived or topic,
                        source_tool="auto-wiring",
                    )
                explicit_event_type = (
                    data.get("event_type") if isinstance(data, dict) else None
                )
                if explicit_event_type:
                    envelope = envelope.model_copy(
                        update={"event_type": explicit_event_type}
                    )
                else:
                    derived_event_type = _derive_event_type_from_topic(topic)
                    if derived_event_type is not None:
                        envelope = envelope.model_copy(
                            update={"event_type": derived_event_type}
                        )
                if tenant_scoped:
                    envelope = _stamp_tenant_id_from_topic_prefix(topic, envelope)
            else:
                if not isinstance(message, ModelEventEnvelope):
                    logger.warning(
                        "Auto-wiring callback: message has no 'value' and is not a ModelEventEnvelope"
                        " — dropping. topic=%s message_type=%s",
                        topic,
                        type(message).__name__,
                    )
                    return
                envelope = message
            if envelope.correlation_id is not None:
                correlation_id = envelope.correlation_id
            await _dispatch_with_bounded_retry(envelope)
        except (OptimisticConflictError, BoundaryPublishError) as exc:
            # OMN-14403 §4.3, OMN-14600 CORRECTION: this except-tuple's
            # OptimisticConflictError arm is effectively DEAD for a dispatcher
            # registered on MessageDispatchEngine (the state_io stateful
            # callback IS such a dispatcher) — dispatch_engine.dispatch()
            # catches every exception its per-dispatcher invocation loop sees
            # and returns a HANDLER_ERROR status instead of re-raising, so an
            # OptimisticConflictError raised inside _load_handle_persist's
            # retry_on_optimistic_conflict call never survives to reach here.
            # BoundaryPublishError IS live: it is raised by
            # _dispatch_with_bounded_retry AFTER dispatch_engine.dispatch()
            # already returned (from result_applier.apply() failing), which is
            # outside that catch-all, so it genuinely propagates from this
            # callback to its caller (no offset commit here). Whether that
            # caller's non-commit actually produces a Kafka redelivery is a
            # property of the OUTER consumer wiring, not verified at this
            # layer — do not assume it without checking that caller.
            if propagate_publish_failures:
                if isinstance(exc, BoundaryPublishError) and exc.__cause__:
                    raise exc.__cause__
                raise
            await _route_swallowed_exception(exc, message, correlation_id)
        except Exception as exc:  # noqa: BLE001 — boundary: never unsubscribe; route to _route_swallowed_exception
            await _route_swallowed_exception(exc, message, correlation_id)

    return callback


def _stamp_tenant_id_from_topic_prefix(
    topic: str,
    envelope: ModelEventEnvelope[object],
) -> ModelEventEnvelope[object]:
    """Overwrite payload["tenant_id"] with the slug from a tenant-<slug>. wire prefix.

    OMN-14349 (OMN-14208 Path A). The config-bound identity always wins: this
    overwrites any client-supplied ``tenant_id``, it never merges-if-absent
    and never falls back to one. A topic with no matching prefix leaves the
    payload completely untouched -- never a defaulted or guessed tenant
    (Stage-1 warn semantics; a missing/self-reported value is handled by the
    existing OMN-14058 flow downstream, not masked here).

    OMN-15792: this is the subscribe/dispatch-side call site of the single
    runtime topic resolver. ``resolve_tenant_from_wire_topic`` is the same
    resolver the gateway forwarder's publish-side ``HandlerForwardOutbound``
    resolves through (via ``prefix_topic``) -- previously this function
    hand-rolled its own regex extraction with no slug validation, which is
    exactly the two-independent-resolvers-disagreeing class OMN-15757/
    OMN-15778 hit. A reserved or malformed slug embedded in a prefix-shaped
    topic now raises (routed to the existing swallowed-exception boundary
    handling below) instead of being silently stamped.
    """
    slug, _canonical_topic = resolve_tenant_from_wire_topic(topic)
    if slug is None:
        return envelope
    if not isinstance(envelope.payload, dict):
        return envelope
    # OMN-14367: route through the single canonical stamp so this producer and
    # the gateway forwarder's consume_inbound cannot diverge on the shape again.
    stamped_payload = stamp_verified_tenant_slug(envelope.payload, slug)
    return envelope.model_copy(update={"payload": stamped_payload})


def _make_raw_event_projection_callback(
    topic: str,
    dispatch_engine: ProtocolDispatchEngine,
    result_applier: ProtocolDispatchResultApplier,
    *,
    allowed_dispatcher_ids: Collection[str] | None = None,
) -> Callable[..., Awaitable[None]]:
    """Create a callback for raw Kafka `ModelEventMessage` projection contracts."""
    from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
    from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage

    dispatcher_scope = _require_contract_dispatcher_scope(
        allowed_dispatcher_ids,
        contract_name=topic,
    )
    scoped_dispatch_engine = _require_contract_scoped_dispatch_engine(
        dispatch_engine,
        contract_name=topic,
    )

    async def callback(message: object) -> None:
        try:
            raw_message = (
                message
                if isinstance(message, ModelEventMessage)
                else ModelEventMessage.model_validate(message)
            )
            if not await _wait_for_dispatch_engine_freeze(topic, dispatch_engine):
                return
            envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
                payload=cast("object", raw_message.model_dump(mode="json")),
                correlation_id=raw_message.headers.correlation_id,
                envelope_timestamp=raw_message.headers.timestamp,
                event_type=(
                    _derive_event_type_alias_from_topic(topic)
                    or raw_message.headers.event_type
                ),
                source_tool=raw_message.headers.source,
            )
            result = await _dispatch_to_contract_scope(
                scoped_dispatch_engine,
                topic,
                envelope,
                dispatcher_scope,
            )
            if result is not None:
                await result_applier.apply(result, envelope.correlation_id)
        except Exception as exc:  # noqa: BLE001 — consumer boundary; log and continue
            logger.error(
                "Raw projection callback error: topic=%s error_type=%s error=%s",
                topic,
                type(exc).__name__,
                exc,
            )

    return callback


async def _wait_for_dispatch_engine_freeze(
    topic: str,
    dispatch_engine: object,
) -> bool:
    """Wait until the dispatch engine is frozen before consuming startup messages."""
    if bool(getattr(dispatch_engine, "is_frozen", True)):
        return True

    timeout_seconds = _dispatch_freeze_wait_timeout_seconds()
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    logger.info(
        "Auto-wiring callback waiting for MessageDispatchEngine freeze: topic=%s",
        topic,
    )

    while not bool(getattr(dispatch_engine, "is_frozen", True)):
        if asyncio.get_running_loop().time() >= deadline:
            logger.error(
                "Auto-wiring callback timed out waiting for MessageDispatchEngine "
                "freeze; dropping message: topic=%s timeout_seconds=%.1f",
                topic,
                timeout_seconds,
            )
            return False
        await asyncio.sleep(0.1)

    logger.info(
        "Auto-wiring callback resumed after MessageDispatchEngine freeze: topic=%s",
        topic,
    )
    return True


def _dispatch_freeze_wait_timeout_seconds() -> float:
    raw = os.environ.get("ONEX_DISPATCH_FREEZE_WAIT_TIMEOUT_SECONDS", "900")
    try:
        timeout_seconds = float(raw)
    except ValueError:
        logger.warning(
            "Invalid ONEX_DISPATCH_FREEZE_WAIT_TIMEOUT_SECONDS=%r; using 900s",
            raw,
        )
        return 900.0
    if not math.isfinite(timeout_seconds):
        logger.warning(
            "Invalid ONEX_DISPATCH_FREEZE_WAIT_TIMEOUT_SECONDS=%r; using 900s",
            raw,
        )
        return 900.0
    return max(timeout_seconds, 0.1)


def _derive_route_id(
    contract_name: str,
    handler_key: str,
    topic: str,
) -> str:
    """Derive a route ID from contract name, handler entry key, and full topic path.

    Uses the full topic path (sanitized) to guarantee uniqueness across topics
    that share a common segment (OMN-8735).

    When two routing entries reference the same handler class for different
    operations (e.g. ``HandlerLlmCliSubprocess`` for both ``inference.gemini_cli``
    and ``inference.codex_cli``) and subscribe to the same topic, the
    ``handler + topic`` pair alone produces a collision.  The handler entry key
    includes the sanitized operation suffix when present, guaranteeing each
    entry gets a distinct route ID (OMN-9461 / OMN-10447).
    """
    safe_topic = re.sub(r"[.\-]", "_", topic)
    return f"route.auto.{contract_name}.{handler_key}.{safe_topic}"


def _derive_dispatcher_id(contract_name: str, handler_key: str) -> str:
    """Derive a dispatcher ID from contract name and handler entry key.

    When two routing entries in the same contract reference the same handler
    class (e.g. ``HandlerLlmCliSubprocess`` wired for both ``inference.gemini_cli``
    and ``inference.codex_cli``), the plain handler name alone produces a
    collision.  The entry key includes the sanitized operation suffix and keeps
    dispatcher IDs distinct (OMN-9461 / OMN-10447).
    """
    return f"dispatcher.auto.{contract_name}.{handler_key}"


def _derive_handler_entry_key(entry: ModelHandlerRoutingEntry) -> str:
    """Return the stable per-entry handler key used for pre-resolution and IDs.

    The key preserves the legacy plain handler name when neither ``operation``
    nor ``topic`` is present. When ``operation`` is present, it appends a
    sanitized operation label plus a short digest, preventing collisions when
    a contract uses the same handler class for multiple operations.

    OMN-14580: a ``topic_match`` contract can legitimately route the SAME
    operation to the SAME handler from several distinct topics (e.g. one
    reducer operation invoked from N event sources, each with its own
    ``event_model`` — see ``node_swarm_subtask_state_reducer``) — operation
    alone no longer disambiguates that shape and produced a dispatcher-ID
    collision (ONEX_CORE_064_DUPLICATE_REGISTRATION) on cold boot. When the
    entry declares its own ``topic``, it is folded into the digest (not the
    human-readable label, to avoid re-duplicating the topic that
    ``_derive_route_id`` already appends) so each topic still gets its own
    key.
    """
    handler_name = entry.handler.name
    operation = entry.operation or ""
    topic = entry.topic.strip() if entry.topic else ""
    if not operation and not topic:
        return handler_name

    normalized_op = ""
    if operation:
        normalized_op = re.sub(r"[^A-Za-z0-9_]+", "_", operation.strip()).strip("_")

    digest_source = f"{operation}|{topic}" if topic else operation
    digest = hashlib.sha1(digest_source.encode()).hexdigest()[:8]
    safe_op = f"{normalized_op}_{digest}" if normalized_op else digest
    return f"{handler_name}.{safe_op}"


def _required_handler_init_params(handler_cls: type) -> frozenset[str]:
    """Return required constructor parameter names for a handler class."""
    sig = inspect.signature(handler_cls)
    return frozenset(
        name
        for name, param in sig.parameters.items()
        if name != "self"
        and param.kind
        in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
        and param.default is inspect.Parameter.empty
    )


def _should_skip_sync_container_resolution(handler_cls: type) -> bool:
    """Return True when sync container resolution is unnecessary for handler_cls.

    Zero-arg handlers can be constructed directly by the resolver, and handlers
    that require only runtime-known ports can be constructed from materialized
    dependencies. In both cases, calling a sync container from runtime-managed
    async boot is unnecessary and can trip ``asyncio.run()`` crashes.
    """
    required_params = _required_handler_init_params(handler_cls)
    return not required_params or required_params <= frozenset(
        {
            "event_bus",
            "event_publisher",
            "event_consumer",
            "dispatch_port",
            "provisioner",
            "drain_proof_gate",
        }
    )


async def _await_event_bus_publish(awaitable: Awaitable[object]) -> None:
    await awaitable


async def _route_sync_publisher_failure(
    exc: Exception,
    *,
    event_bus: object,
    handler_name: str,
    topic: str,
    payload: bytes,
) -> None:
    """Best-effort DLQ routing for a sync-handler's fire-and-forget publish
    failure (OMN-14498 / OMN-15029).

    ``_make_sync_event_publisher``'s ``_publish`` schedules the actual
    downstream publish as a detached asyncio Task/Future and only logged its
    failure via ``_log_publish_failure``'s ``add_done_callback`` — no DLQ, no
    metric, no durable trace that the event ever existed. Confirmed still
    live on ``origin/dev`` by the OMN-15029 false-Done reopen. There is
    nothing to re-raise into here: the sync handler that issued the publish
    has already returned by the time this callback runs, so propagating the
    exception synchronously is not possible.

    Mirrors the ``_make_event_bus_callback._route_swallowed_exception`` idiom
    (OMN-14507) for the consume boundary: when the event bus exposes the
    duck-typed ``_publish_raw_to_dlq`` contract, the payload that failed to
    publish is durably preserved on that topic's DLQ instead of vanishing
    with nothing but a log line. Unlike the consume-boundary version this is
    UNCONDITIONAL — not gated behind ``ONEX_BOUNDARY_DLQ_ENABLED``: this is a
    pure best-effort recovery channel layered on top of the EXISTING
    fire-and-forget publish (no new retry, no change to offset/delivery
    semantics, no control-flow change on success), so there is no
    staged-rollout risk to hold behind a flag — leaving it flag-gated would
    reproduce the exact live-off, still-swallowing state OMN-15029 exists to
    close.

    Never raises: a failure in the DLQ path itself is logged loudly
    (``message_lost=true``) rather than crashing the kernel loop's task
    processing.
    """
    from uuid import uuid4

    from omnibase_infra.event_bus.topic_constants import get_dlq_topic_for_original

    correlation_id = uuid4()
    publish_dlq_fn = getattr(event_bus, "_publish_raw_to_dlq", None)
    if publish_dlq_fn is None or not callable(publish_dlq_fn):
        logger.error(
            "metric_name=boundary_swallow_observed dlq_routed=false "
            "message_lost=true handler=%s topic=%s error_type=%s "
            "correlation_id=%s",
            handler_name,
            topic,
            type(exc).__name__,
            correlation_id,
        )
        return

    from types import SimpleNamespace

    raw_msg = SimpleNamespace(value=payload, key=None, offset=None, partition=None)
    try:
        dlq_persisted = await publish_dlq_fn(
            original_topic=topic,
            raw_msg=raw_msg,
            error=exc,
            correlation_id=correlation_id,
            failure_type="sync_publisher_publish_failed",
            consumer_group="auto-wiring-sync-publisher",
            dlq_topic=get_dlq_topic_for_original(topic),
        )
        if dlq_persisted:
            logger.error(
                "metric_name=boundary_swallow_prevented dlq_routed=true "
                "handler=%s topic=%s error_type=%s correlation_id=%s",
                handler_name,
                topic,
                type(exc).__name__,
                correlation_id,
            )
            return
        # A False return means the DLQ publish did NOT durably persist
        # (rejected input, producer unavailable, or the send itself
        # failed/timed out) WITHOUT raising -- the message is lost exactly
        # like the except-branch below, just signaled via the return value.
        logger.error(
            "metric_name=boundary_swallow_observed dlq_routed=false "
            "dlq_publish_failed=true message_lost=true handler=%s topic=%s "
            "error_type=%s correlation_id=%s",
            handler_name,
            topic,
            type(exc).__name__,
            correlation_id,
        )
    except Exception as dlq_exc:  # noqa: BLE001 — DLQ publish is itself a boundary; never let it crash the kernel loop
        logger.error(
            "metric_name=boundary_swallow_observed dlq_routed=false "
            "dlq_publish_failed=true message_lost=true handler=%s topic=%s "
            "error_type=%s dlq_error=%s correlation_id=%s",
            handler_name,
            topic,
            type(exc).__name__,
            _sanitize_exc(dlq_exc),
            correlation_id,
        )


def _make_sync_event_publisher(
    *,
    event_bus: object,
    handler_name: str,
    terminal_topics: frozenset[str] = frozenset(),
) -> Callable[[str, bytes], None]:
    """Adapt async runtime event-bus publish to legacy sync handler publishers.

    The publisher is constructed during ``wire_from_manifest`` while the runtime
    kernel's event loop is running, so that loop is captured here as the owning
    loop. Legacy sync handlers (e.g. ``HandlerContextRoiRunner``) execute on a
    ``ThreadPoolExecutor`` worker thread — the dispatch engine offloads blocking
    sync handlers via ``run_in_executor`` — so a publish issued from inside such
    a handler runs on a thread that does not own the kernel loop.

    The publish awaitable returned by the event bus binds its internal Futures to
    the kernel loop. Running that awaitable on a *foreign* loop (the previous
    behavior: ``asyncio.run`` spun a throwaway loop in the worker thread once
    ``get_running_loop`` raised ``RuntimeError``) produced the ``got Future
    attached to a different loop`` warning and 2-3 minute terminal-emission retry
    delays (OMN-13658). Scheduling the coroutine back onto the owning kernel loop
    via ``asyncio.run_coroutine_threadsafe`` keeps every Future on its loop, so
    the publish completes immediately from any thread.

    ``terminal_topics`` (OMN-15468) carries the publishing contract's declared
    terminal topics — every site: ``terminal_event``, ``terminal_events`` and
    ``runtime_dispatch.terminal_events``, read through the same function route
    discovery uses. A publish to one of those topics is a TERMINAL emission and
    is wrapped in a ``ModelEventEnvelope`` here, at the one factory that hands
    every def-B handler its publisher.

    Why here and not in the handlers: the other half of this same wiring
    (``DispatchResultApplier``) already publishes the def-B return value as a
    full envelope, so before this change a single contract emitted its SUCCESS
    terminal enveloped and its handler-emitted FAILURE terminal raw — and the
    Pattern B broker's terminal path decodes envelopes. Live on the ``.201`` dev
    lane at merged ``5dc68190`` (2026-07-30T17:13Z), with #2560 already
    subscribing the broker to the failure topic, a forced node-generation
    failure still returned ``ok=true`` / ``status=completed`` / ``error=null``,
    byte-identical to the success control, because the record waiting on the
    failure topic was raw. Fixing that per node would mean editing every handler
    that self-publishes a terminal; fixing it here covers the whole declared-
    terminal set at once. Any topic that is not a declared terminal is forwarded
    byte-for-byte unchanged.
    """
    publish = getattr(event_bus, "publish", None)
    if not callable(publish):
        raise ModelOnexError(
            "handler_wiring: handler "
            f"{handler_name!r} declares event_publisher, but event_bus "
            f"{type(event_bus).__name__!r} has no callable publish()."
        )

    try:
        kernel_loop = asyncio.get_running_loop()
    except RuntimeError as exc:
        raise ModelOnexError(
            "handler_wiring: the sync event_publisher for handler "
            f"{handler_name!r} must be constructed on the runtime kernel event "
            "loop (wire_from_manifest runs on it), but no running loop was found."
        ) from exc

    def _publish(topic: str, payload: bytes) -> None:
        payload = envelope_terminal_payload(
            topic=topic,
            payload=payload,
            terminal_topics=terminal_topics,
        )
        result = publish(topic, None, payload)
        if not inspect.isawaitable(result):
            return

        publish_awaitable = cast("Awaitable[object]", result)

        def _log_publish_failure(
            done: asyncio.Future[None] | concurrent.futures.Future[None],
        ) -> None:
            try:
                done.result()
            except Exception as exc:  # noqa: BLE001 — publish boundary logging
                logger.error(
                    "Auto-wired event_publisher publish failed: "
                    "handler=%s topic=%s error_type=%s error=%s",
                    handler_name,
                    topic,
                    type(exc).__name__,
                    exc,
                )
                # OMN-14498 / OMN-15029: previously the failure vanished here
                # -- logged, never routed anywhere durable. This callback is
                # sync (it cannot itself await), so the best-effort DLQ
                # routing is scheduled as a task on the kernel loop instead.
                # Safe to call unconditionally: `_log_publish_failure` always
                # executes on `kernel_loop`'s own thread, whether it was
                # registered on an asyncio Task (the same-loop branch below)
                # or chained from `run_coroutine_threadsafe` (the
                # cross-thread branch) -- `_chain_future` resolves the
                # `concurrent.futures.Future` from inside the loop's own
                # callback dispatch, which is also where that Future invokes
                # its done-callbacks.
                dlq_task = kernel_loop.create_task(
                    _route_sync_publisher_failure(
                        exc,
                        event_bus=event_bus,
                        handler_name=handler_name,
                        topic=topic,
                        payload=payload,
                    )
                )
                _DLQ_ROUTING_TASKS.add(dlq_task)
                dlq_task.add_done_callback(_DLQ_ROUTING_TASKS.discard)

        try:
            running_loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop is kernel_loop:
            # Publishing from the kernel loop's own thread (async handler path):
            # schedule the coroutine directly on it.
            task = kernel_loop.create_task(_await_event_bus_publish(publish_awaitable))
            task.add_done_callback(_log_publish_failure)
            return

        # Publishing from a ThreadPoolExecutor worker thread (or any thread that
        # does not own the kernel loop): hand the coroutine to the kernel loop in
        # a thread-safe way so the publish awaitable's Futures stay on their
        # owning loop. This avoids the "got Future attached to a different loop"
        # warning and the 2-3 minute retry delay (OMN-13658).
        future = asyncio.run_coroutine_threadsafe(
            _await_event_bus_publish(publish_awaitable), kernel_loop
        )
        future.add_done_callback(_log_publish_failure)

    return _publish


def _make_sync_event_consumer(
    *,
    event_bus: object,
    handler_name: str,
) -> TerminalEventConsumer:
    """Materialize the blocking terminal-event consumer for request/response handlers.

    Mirror of ``_make_sync_event_publisher`` for the consume leg. Some EFFECT
    handlers (e.g. ``HandlerContextRoiRunner``, OMN-13005) publish a command and
    then block on the correlated terminal event, reading result fields back from
    its payload. They declare an injectable ``event_consumer``.

    The returned object is directly callable with the legacy single-call shape
    ``(terminal_topic, correlation_id, timeout_seconds) -> dict | None`` and also
    exposes ``.open(topic) -> TerminalConsumerSession`` for the two-phase
    (subscribe-before-publish) protocol introduced in OMN-13012: the handler
    positions the consumer (assign + seek_to_end) BEFORE publishing its command,
    then waits AFTER, so a terminal emitted in the publish→wait gap is not seeked
    past.

    Without this injection the handler falls back to its own no-op default that
    returns ``None`` immediately — never honoring the timeout, so every result
    row is a degenerate generation-failure even though the terminal event arrives
    moments later. The concrete consumer runs the correlate-and-wait loop on an
    isolated event loop in a worker thread so blocking does not deadlock the
    runtime dispatch loop that delivers the awaited terminal.
    """
    from omnibase_infra.runtime.service_terminal_event_consumer import (
        make_terminal_event_consumer,
    )

    return make_terminal_event_consumer(
        event_bus=event_bus,
        handler_name=handler_name,
    )


def _contracts_root_for_runtime_dependencies() -> Path:
    raw = os.environ.get("ONEX_CONTRACTS_DIR")
    if raw:
        return Path(raw)
    return Path(__file__).resolve().parents[2] / "nodes"


def _build_topic_migration_executor_dependencies() -> dict[str, object]:
    """Build concrete collaborators for HandlerTopicMigrationExecutor.

    The handler declares these as required constructor services. Materializing
    them here keeps generic resolver Step 2 deterministic while preserving the
    handler's strict constructor contract.
    """
    from aiokafka import AIOKafkaConsumer
    from aiokafka.admin import AIOKafkaAdminClient

    from omnibase_infra.event_bus.kafka_auth import build_aiokafka_auth_kwargs_from_env
    from omnibase_infra.event_bus.service_topic_manager import TopicProvisioner
    from omnibase_infra.migration.adapter_kafka_admin_lag import AdapterKafkaAdminLag
    from omnibase_infra.migration.service_consumer_lag_observer import (
        ServiceConsumerLagObserver,
    )
    from omnibase_infra.migration.service_drain_proof_gate import ServiceDrainProofGate

    bootstrap_servers = os.environ["KAFKA_BOOTSTRAP_SERVERS"]  # ONEX_EXCLUDE: env
    auth_kwargs = build_aiokafka_auth_kwargs_from_env()
    admin = AIOKafkaAdminClient(bootstrap_servers=bootstrap_servers, **auth_kwargs)
    consumer = AIOKafkaConsumer(bootstrap_servers=bootstrap_servers, **auth_kwargs)
    lag_admin = AdapterKafkaAdminLag(admin, consumer)
    observer = ServiceConsumerLagObserver(lag_admin)
    return {
        "provisioner": TopicProvisioner(
            bootstrap_servers=bootstrap_servers,
            contracts_root=_contracts_root_for_runtime_dependencies(),
        ),
        "drain_proof_gate": ServiceDrainProofGate(observer),
    }


def _handler_requires_delegation_dispatch_port(handler_cls: type) -> bool:
    parameter = inspect.signature(handler_cls).parameters.get("dispatch_port")
    if parameter is None:
        return False
    annotation = parameter.annotation
    if isinstance(annotation, str):
        return "ProtocolDelegationDispatchPort" in annotation
    if getattr(annotation, "__name__", "") == "ProtocolDelegationDispatchPort":
        return True
    return any(
        getattr(arg, "__name__", "") == "ProtocolDelegationDispatchPort"
        for arg in get_args(annotation)
    )


def _materialize_known_handler_dependencies(
    *,
    handler_name: str,
    handler_cls: type,
    materialized_explicit_dependencies: dict[str, dict[str, object]] | None,
    event_bus: object | None,
    container: object | None,
    ownership_query: object | None,
    terminal_topics: frozenset[str] = frozenset(),
) -> dict[str, dict[str, object]] | None:
    """Materialize infra-known constructor deps for core resolver Step 2.

    Runtime images may carry a core resolver that only direct-injects
    ``event_bus``. Threading ``container`` / ``ownership_query`` through the
    existing explicit-dependency map keeps infra wiring deterministic without
    requiring a synchronized core release.
    """
    signature = inspect.signature(handler_cls)
    constructor_params = frozenset(
        name
        for name, param in signature.parameters.items()
        if name != "self"
        and param.kind
        in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
    )
    requires_delegation_port = _handler_requires_delegation_dispatch_port(handler_cls)
    required_params = _required_handler_init_params(handler_cls)
    requires_event_publisher = "event_publisher" in constructor_params
    requires_event_consumer = "event_consumer" in constructor_params
    if (
        not required_params
        and not requires_delegation_port
        and not requires_event_publisher
        and not requires_event_consumer
    ):
        return materialized_explicit_dependencies
    available = {
        name: value
        for name, value in (
            ("event_bus", event_bus),
            ("container", container),
            ("ownership_query", ownership_query),
        )
        if value is not None
    }
    if requires_event_publisher and event_bus is not None:
        available["event_publisher"] = _make_sync_event_publisher(
            event_bus=event_bus,
            handler_name=handler_name,
            terminal_topics=terminal_topics,
        )
    if requires_event_consumer and event_bus is not None:
        available["event_consumer"] = _make_sync_event_consumer(
            event_bus=event_bus,
            handler_name=handler_name,
        )
    if required_params.issubset(_TOPIC_MIGRATION_EXECUTOR_DEPS):
        available.update(_build_topic_migration_executor_dependencies())
    if not (
        requires_event_publisher
        or requires_event_consumer
        or required_params.intersection(
            {"container", "ownership_query", "dispatch_port"}
        )
        or ("dispatch_port" in constructor_params and requires_delegation_port)
        or required_params.issubset(_TOPIC_MIGRATION_EXECUTOR_DEPS)
    ):
        return materialized_explicit_dependencies
    if "dispatch_port" in constructor_params and requires_delegation_port:
        # Pure Kafka delegation chain (OMN-12294): the delegate-skill handler
        # dispatches via the Kafka-backed RuntimeDelegationDispatchPort. The
        # delegation orchestrator consumes the command on its own bus
        # subscription and emits the terminal event the broker awaits — there is
        # no in-process bridge.
        if event_bus is not None:
            from omnibase_infra.runtime.service_delegation_dispatch_port import (
                RuntimeDelegationDispatchPort,
            )

            available["dispatch_port"] = RuntimeDelegationDispatchPort(
                cast("ProtocolPatternBBrokerTransport", event_bus)
            )
    if not required_params <= set(available):
        return materialized_explicit_dependencies
    if requires_event_publisher and "event_publisher" not in available:
        return materialized_explicit_dependencies

    merged = dict(materialized_explicit_dependencies or {})
    handler_dependencies = dict(merged.get(handler_name, {}))
    for name in required_params:
        handler_dependencies.setdefault(name, available[name])
    if "dispatch_port" in constructor_params and "dispatch_port" in available:
        handler_dependencies.setdefault("dispatch_port", available["dispatch_port"])
    if "db_dsn" in constructor_params:
        from omnibase_infra.runtime.overlay.contract_env_ref import (
            expand_contract_env_refs,
        )

        db_dsn = next(
            (
                resolved.strip()
                for name in _LEDGER_DB_DSN_ENV_VARS
                if (resolved := expand_contract_env_refs(f"${{env.{name}:}}").strip())
            ),
            "",
        )
        if db_dsn:
            handler_dependencies.setdefault("db_dsn", db_dsn)
    if requires_event_publisher and "event_publisher" in available:
        handler_dependencies.setdefault("event_publisher", available["event_publisher"])
    if requires_event_consumer and "event_consumer" in available:
        handler_dependencies.setdefault("event_consumer", available["event_consumer"])
    merged[handler_name] = handler_dependencies
    return merged


def _derive_topic_pattern_from_topic(topic: str) -> str:
    """Derive a topic pattern from a fully qualified topic string.

    Replaces the first segment (realm prefix) with a wildcard.
    Example: ``onex.evt.platform.node-introspection.v1`` -> ``*.evt.platform.node-introspection.*``

    For ONEX 5-segment topics, wildcards are placed at positions 1 and 5.
    """
    parts = topic.split(".")
    if len(parts) >= 5:
        # Standard ONEX 5-segment: onex.<kind>.<producer>.<event-name>.v<n>
        parts[0] = "*"
        parts[-1] = "*"
        return ".".join(parts)
    # Fallback: exact match
    return topic


def _derive_message_category(topic: str) -> str:
    """Derive message category string from ONEX topic naming convention.

    Convention: ``onex.<kind>.<producer>.<event-name>.v<n>``
    where ``<kind>`` is one of: evt, cmd, intent.

    Returns lowercase values matching EnumMessageCategory enum values.
    """
    parts = topic.split(".")
    if len(parts) >= 2:
        kind = parts[1]
        if kind == "evt":
            return "event"
        if kind == "cmd":
            return "command"
        if kind == "intent":
            return "intent"
    return "event"


def _node_kind_from_node_type(node_type: str | None) -> EnumNodeKind | None:
    """Map a contract ``node_type`` (e.g. ``REDUCER_GENERIC``) to EnumNodeKind.

    Only the archetype prefix matters. Returns ``None`` for an unrecognized or
    empty node_type so the dispatch adapter keeps its archetype-agnostic
    classification (events / intents / fan-out). Used to tell
    ``_normalize_handler_result`` that a bare/Sequence return from a REDUCER is a
    projection, not an event (OMN-14598).
    """
    prefix = (node_type or "").strip().upper()
    if prefix.startswith("REDUCER"):
        return EnumNodeKind.REDUCER
    if prefix.startswith("EFFECT"):
        return EnumNodeKind.EFFECT
    if prefix.startswith("COMPUTE"):
        return EnumNodeKind.COMPUTE
    if prefix.startswith("ORCHESTRATOR"):
        return EnumNodeKind.ORCHESTRATOR
    return None


def _derive_event_type_alias_from_topic(topic: str) -> str | None:
    """Derive the dispatch-engine event_type alias from an ONEX topic."""
    parts = topic.split(".")
    if len(parts) >= 5 and parts[0] == "onex":
        return f"{parts[2]}.{parts[3]}"
    return None


def _topics_for_handler_entry(
    contract: ModelDiscoveredContract,
    entry: ModelHandlerRoutingEntry,
) -> tuple[str, ...]:
    """Return subscribe topics that can be deterministically assigned to entry."""
    if contract.event_bus is None:
        return ()

    topics = contract.event_bus.subscribe_topics

    # OMN-13825: honor a contract-declared per-handler topic (topic_match
    # strategy). When a handler entry names its own subscribe topic, that
    # entry deterministically owns exactly that topic — the reducer's
    # topic_match contract (e.g. node_projection_swarm, two handlers each
    # declaring one of two subscribe topics) previously fell through to the
    # multi-handler ambiguity guard (return ()) and registered ZERO dispatch
    # routes, orphaning the dispatcher ("No dispatcher found"). An entry_topic
    # that is not an actual subscribe topic returns () so a real contract
    # error surfaces rather than silently mis-routing.
    entry_topic = entry.topic.strip() if entry.topic else ""
    if entry_topic:
        return (entry_topic,) if entry_topic in topics else ()

    event_type_alias = entry.event_type.strip() if entry.event_type else ""
    if event_type_alias:
        matched = tuple(
            topic
            for topic in topics
            if topic == event_type_alias
            or _derive_event_type_alias_from_topic(topic) == event_type_alias
        )
        return matched

    if entry.event_model is None:
        return topics

    if len(topics) == 1:
        return topics

    # OMN-12848: a sole handler entry unambiguously owns every subscribe topic.
    # The ambiguity guard below (return ()) only applies when MULTIPLE handler
    # entries compete for the same topics without per-handler event_type
    # disambiguation. A single-handler contract with an event_model and more
    # than one subscribe topic (e.g. node_generation_consumer subscribing to
    # both node-generation-requested and node-deploy) previously fell through to
    # return () and registered ZERO dispatch routes — the command was consumed
    # then DLQ'd ("No dispatcher found"). Assign all topics to the sole handler.
    if (
        contract.handler_routing is not None
        and len(contract.handler_routing.handlers) == 1
    ):
        return topics

    return ()


def _literal_event_type_aliases_from_topics(
    subscribe_topics: tuple[str, ...],
) -> set[str]:
    """Return literal wire-topic aliases accepted as envelope event_type keys."""
    return {topic.strip() for topic in subscribe_topics if topic.strip()}


def _strict_dispatcher_coverage_enabled() -> bool:
    """Return True when strict orchestrator dispatcher coverage is enabled."""
    return os.environ.get(_STRICT_DISPATCHER_COVERAGE_ENV, "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _wiring_strict_mode_enabled() -> bool:
    """Return True when ONEX_WIRING_STRICT_MODE is active (OMN-9126).

    In strict mode no failure is demoted: per-handler resolution failures
    re-raise (preserving the pre-OMN-13203 boot-crash invariant) instead of
    being quarantined. Mirrors the env check at ``wire_from_manifest`` so a
    single source defines strict semantics.
    """
    return os.environ.get("ONEX_WIRING_STRICT_MODE", "").lower() in ("1", "true")


def _boundary_dlq_enabled() -> bool:
    """Return True when the auto-wired consume boundary must not silently
    discard a handler exception (OMN-14507).

    ``_make_event_bus_callback`` catches every exception raised while
    dispatching a consumed message and, historically, only logged it -- the
    message itself (and any evidence it ever arrived) then vanished: no DLQ,
    no redelivery, no metric. That is the root mechanism behind this
    session's silent-death theme (reference_autowired_boundary_swallows_no_redelivery).

    DEFAULT OFF (staged, warn-first rollout -- CLAUDE.md's rule that a
    strict-tightening gate ships behind a default-off flag until downstream
    compliance is proven): when unset/false, the boundary keeps its exact
    historical swallow-and-ACK behavior, with one addition -- a structured
    metric-shaped log line so the swallow is at least observable. When
    enabled, an exception that survives the bounded retry is routed to the
    topic's DLQ on a BEST-EFFORT basis (reusing the same
    ``EventBusKafka._publish_raw_to_dlq`` / ``get_dlq_topic_for_original``
    machinery ``EventBusSubcontractWiring._publish_to_dlq`` already uses)
    instead of only being logged. This is NOT true at-least-once: the
    consumer offset always advances regardless (no nack/redelivery), so if
    the DLQ publish itself fails the message is still lost -- loudly
    (``message_lost=true`` in the log) rather than silently. See
    ``_route_swallowed_exception``'s docstring for the full gap (OMN-14507
    review, G1).
    """
    return os.environ.get(_BOUNDARY_DLQ_ENV, "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _is_orchestrator_contract(contract: ModelDiscoveredContract) -> bool:
    """Return True when a discovered contract is an orchestrator variant."""
    return "orchestrator" in contract.node_type.lower()


def _start_event_type_aliases(contract: ModelDiscoveredContract) -> tuple[str, ...]:
    """Return topic-derived event_type aliases for orchestrator start commands."""
    if contract.event_bus is None:
        return ()

    aliases: list[str] = []
    for topic in contract.event_bus.subscribe_topics:
        parts = topic.split(".")
        if (
            len(parts) < 5
            or parts[0] != "onex"
            or parts[1] != "cmd"
            or not parts[3].endswith("-start")
            or parts[4] != "v1"
        ):
            continue
        alias = _derive_event_type_alias_from_topic(topic)
        if alias is not None:
            aliases.append(alias)
    return tuple(dict.fromkeys(aliases))


def _live_message_types(pcw: PreparedContractWiring) -> set[str]:
    """Return message types that will be committed for non-skipped handlers."""
    message_types: set[str] = set()
    for prepared in pcw.prepared_wirings:
        if prepared.is_skip or prepared.is_quarantined:
            continue
        if prepared.message_types is not None:
            message_types.update(prepared.message_types)
    return message_types


def _preflight_prepared_registration_ids(
    prepared_contracts: Sequence[PreparedContractWiring],
    dispatch_engine: object,
    *,
    dynamic_materialization_authorized: bool = False,
) -> None:
    """Validate every derived dispatcher/route ID before the first commit.

    Preparation is deliberately side-effect-free. Preserve that transaction
    boundary by detecting cross-contract, same-contract, and normalization
    collisions across the complete manifest before any engine registration or
    Kafka subscription becomes visible.
    """
    dispatcher_origins: dict[str, list[str]] = defaultdict(list)
    dispatcher_owners: dict[str, str] = {}
    route_origins: dict[str, list[str]] = defaultdict(list)
    for prepared_contract in prepared_contracts:
        if prepared_contract.skip_result is not None:
            continue
        contract_name = prepared_contract.contract.name
        for prepared in prepared_contract.prepared_wirings:
            if prepared.is_skip or prepared.is_quarantined:
                continue
            origin = f"{contract_name}:{prepared.handler_name}"
            dispatcher_origins[prepared.dispatcher_id].append(origin)
            dispatcher_owners[prepared.dispatcher_id] = contract_name
            for route_id in prepared.route_ids:
                route_origins[route_id].append(origin)

    duplicate_dispatcher_ids = {
        dispatcher_id: tuple(origins)
        for dispatcher_id, origins in dispatcher_origins.items()
        if len(origins) > 1
    }
    if duplicate_dispatcher_ids:
        raise ModelOnexError(
            message=(
                "handler_wiring: duplicate prepared dispatcher IDs across the "
                f"manifest: {duplicate_dispatcher_ids}"
            ),
            error_code=EnumCoreErrorCode.DUPLICATE_REGISTRATION,
        )

    duplicate_route_ids = {
        route_id: tuple(origins)
        for route_id, origins in route_origins.items()
        if len(origins) > 1
    }
    if duplicate_route_ids:
        raise ModelOnexError(
            message=(
                "handler_wiring: duplicate prepared route IDs across the manifest "
                f"(including normalized topic IDs): {duplicate_route_ids}"
            ),
            error_code=EnumCoreErrorCode.DUPLICATE_REGISTRATION,
        )

    from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

    if isinstance(dispatch_engine, MessageDispatchEngine):
        dispatch_engine.validate_registration_batch(
            tuple(dispatcher_origins),
            tuple(route_origins),
            dispatcher_owners=dispatcher_owners,
            allow_frozen=dynamic_materialization_authorized,
        )


def _collect_orchestrator_dispatcher_coverage_gaps(
    prepared_contracts: list[PreparedContractWiring],
    failed_gaps: list[str] | None = None,
) -> tuple[str, ...]:
    """Find orchestrator start topics that lack a matching live dispatcher."""
    gaps: list[str] = list(failed_gaps or [])
    for pcw in prepared_contracts:
        contract = pcw.contract
        if not _is_orchestrator_contract(contract):
            continue
        start_aliases = _start_event_type_aliases(contract)
        if not start_aliases:
            continue
        message_types = _live_message_types(pcw)
        for alias in start_aliases:
            if alias in message_types:
                continue
            gaps.append(
                f"{contract.name}: missing dispatcher for {alias} "
                f"(node_type={contract.node_type}, contract={contract.contract_path})"
            )
    return tuple(gaps)


def _assert_orchestrator_dispatcher_coverage(
    prepared_contracts: list[PreparedContractWiring],
    failed_gaps: list[str] | None = None,
) -> None:
    """Raise when strict mode finds an orchestrator start topic without a dispatcher."""
    gaps = _collect_orchestrator_dispatcher_coverage_gaps(
        prepared_contracts,
        failed_gaps,
    )
    if gaps:
        raise ModelOnexError(
            "Strict dispatcher coverage failed for orchestrator start topics: "
            + "; ".join(gaps)
        )


ENV_SINGLE_OWNER_COMMAND_TOPICS = "ONEX_SINGLE_OWNER_COMMAND_TOPICS"


def _single_owner_command_topics_strict() -> bool:
    """True when the OMN-15474 single-owner command-topic gate must fail closed."""
    return os.environ.get(ENV_SINGLE_OWNER_COMMAND_TOPICS, "").strip().lower() in (
        "1",
        "true",
    )


def _assert_single_owner_command_topics(
    manifest: ModelAutoWiringManifest,
) -> None:
    """Fail closed when a COMMAND topic has more than one in-process consumer.

    OMN-15474. A command is an instruction to execute exactly once. Every wired
    contract joins its own consumer group (``compute_consumer_group_id`` keys on
    node identity), so two contracts subscribed to one ``onex.cmd.*`` topic in
    one process means the broker delivers the accepted command to BOTH: the
    whole reducer chain runs twice, both executions carry the SAME ingress
    correlation id, and every terminal event, projection row, LLM judge call and
    cost line is doubled. That is the live defect measured on ``onex-dev``
    (73 duplicated ``(correlation_id, topic)`` pairs in 48h; two quality-gate
    evaluations returning DIFFERENT scores for one command).

    ``_detect_duplicate_topics`` already SAW this — it logged
    ``Duplicate topic ownership detected`` on every affected boot — but it ran
    AFTER Phase 2 had already committed the subscriptions, and only at WARNING.
    Detection that arrives after the side effect and cannot fail the boot is not
    enforcement ([[feedback_a_rule_is_not_a_mechanism]]). This is the mechanism:
    a preflight, before any subscription is attached.

    EVENT topics are deliberately untouched. Fan-out is their contract — many
    independent consumers legitimately observe one event on their own groups.
    Only the command category carries the execute-exactly-once obligation.

    STRICT MODE IS OFF BY DEFAULT, and that is deliberate. This repo's standing
    rule is that a strict invariant "lands AFTER all downstream consumers are
    compliant... if a strict gate must ship first, it ships behind an env flag
    (default OFF) and is flipped in a separate PR once compliance is merged"
    (CLAUDE.md, Testing and CI). Compliance is NOT met today. Measured
    2026-08-01 by running THIS gate's own detection over
    ``discover_contracts()`` (109 contracts, omnibase_infra only —
    ``omnimarket`` is not installed in that venv, so its contracts are not
    discoverable and are NOT counted here), 3 command topics have more than
    one in-process owner:

    - ``onex.cmd.omnibase-infra.build-loop-append.v1``
      -> node_build_loop_write_effect, node_ledger_projection_compute
    - ``onex.cmd.omnibase-infra.chain-learn.v1``
      -> node_chain_orchestrator, node_chain_retrieval_effect
    - ``onex.cmd.platform.request-introspection.v1``
      -> node_ledger_projection_compute, node_registration_orchestrator

    An earlier revision of this docstring claimed "8 (1 in omnibase_infra;
    7 in omnimarket)". That is not reproducible: infra alone is 3, not 1. The
    omnimarket figure cannot be measured from this repo's venv at all. A raw,
    unfiltered ``contract.yaml`` scan across the infra worktree plus the
    canonical omnimarket clone yields 16 topics with >1 declared subscriber,
    but that is a strict superset (it applies none of the discovery, package-
    activation, or plugin_managed filtering the gate applies). Re-measure with
    the gate's own code path in the target deployment before flipping the flag;
    do not trust any count in this docstring as the deployed number.

    Raising unconditionally here would refuse the runtime boot on
    the very next deploy. So: OFF ⇒ log an ERROR naming every violation
    (louder than the pre-existing post-commit WARNING, and now emitted BEFORE
    the subscriptions attach); ON ⇒ raise before any side effect. Flip
    ``ONEX_SINGLE_OWNER_COMMAND_TOPICS=1`` in a follow-up once those 8 are
    resolved.
    """
    from omnibase_infra.enums import EnumMessageCategory

    command_topic_owners: dict[str, list[str]] = defaultdict(list)
    for contract in manifest.contracts:
        if contract.event_bus is None:
            continue
        for topic in contract.event_bus.subscribe_topics:
            if _derive_message_category(topic) == EnumMessageCategory.COMMAND.value:
                command_topic_owners[topic].append(contract.name)

    violations = [
        f"{topic} owned by {sorted(owners)}"
        for topic, owners in sorted(command_topic_owners.items())
        if len(owners) > 1
    ]
    if not violations:
        return

    detail = (
        "Command topics are single-owner: a command must execute exactly once, "
        "but these command topics have more than one in-process consumer, so "
        "every accepted command is dispatched once per owner, under one "
        f"correlation id (OMN-15474): {'; '.join(violations)}. Give each "
        "command topic exactly one owning contract, or move the additional "
        "consumers onto an event topic."
    )
    if _single_owner_command_topics_strict():
        raise ModelOnexError(detail, error_code=EnumCoreErrorCode.INVALID_STATE)
    logger.error(
        "%s (non-strict — set %s=1 to refuse the boot instead of doubling "
        "every accepted command on these topics)",
        detail,
        ENV_SINGLE_OWNER_COMMAND_TOPICS,
    )


def _detect_duplicate_topics(
    manifest: ModelAutoWiringManifest,
) -> list[ModelDuplicateTopicOwnership]:
    """Detect duplicate topic ownership across contracts.

    Checks three levels:
    - **package-level**: Two contracts from different packages subscribe to same topic.
    - **handler-level**: Two contracts (any package) subscribe to same topic.
    - **intra-package**: Two contracts from the same package subscribe to same topic.
    """
    # Map topic -> list of (contract_name, package_name)
    topic_owners: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for contract in manifest.contracts:
        if contract.event_bus:
            for topic in contract.event_bus.subscribe_topics:
                topic_owners[topic].append((contract.name, contract.package_name))

    duplicates: list[ModelDuplicateTopicOwnership] = []
    for topic, owners in topic_owners.items():
        if len(owners) <= 1:
            continue

        owner_names = tuple(name for name, _ in owners)
        packages = {pkg for _, pkg in owners}

        if len(packages) > 1:
            level = "package"
        elif len(packages) == 1:
            level = "intra-package"
        else:
            level = "handler"

        duplicates.append(
            ModelDuplicateTopicOwnership(
                topic=topic,
                owners=owner_names,
                level=level,
            )
        )

    return duplicates


UNWIRED_BACKFILL_REASON = (
    "no wiring result produced for this manifest contract "
    "(wiring-report totality backfill, OMN-15474)"
)


def build_unwired_contract_results(
    manifest: ModelAutoWiringManifest,
    *,
    reason: str,
    already_reported: Collection[str] = (),
) -> tuple[ModelContractWiringResult, ...]:
    """Return one explicit "did not wire" row per uncovered manifest contract.

    The wiring report consumed by :func:`subscribe_wired_contract_topics` is
    TOTAL over the manifest: every discovered contract either wired, failed, or
    carries an explicit :attr:`EnumWiringOutcome.SKIPPED` row naming why it did
    not. Totality is what makes the initial-subscription identity check
    (:func:`_validate_initial_subscription_contract_identities`) a decision
    instead of a guess — a report that merely *omits* a contract is
    indistinguishable from one where the contract silently vanished, and
    silently-vanished contracts are exactly the class of bug that produced
    process-global dispatch (OMN-15474).

    This is the canonical constructor for those rows. Every producer of a
    :class:`ModelAutoWiringReport` — including test doubles standing in for the
    wiring engine — MUST use it rather than emitting a partial report, so the
    "did not wire" set is derived from the manifest the runtime actually holds
    rather than hand-mirrored against it.

    Args:
        manifest: The manifest the report must be total over.
        reason: Human-readable reason recorded on each synthesized row.
        already_reported: Contract names that already have a result row.

    Returns:
        One SKIPPED result per manifest contract absent from
        ``already_reported``, in manifest order. Empty when the report is
        already total.
    """
    covered = frozenset(already_reported)
    return tuple(
        ModelContractWiringResult(
            contract_name=contract.name,
            package_name=contract.package_name,
            outcome=EnumWiringOutcome.SKIPPED,
            reason=reason,
        )
        for contract in manifest.contracts
        if contract.name not in covered
    )


async def wire_from_manifest(
    manifest: ModelAutoWiringManifest,
    dispatch_engine: ProtocolDispatchEngine,
    event_bus: object | None = None,
    environment: str = "dev",
    container: object | None = None,
    *,
    subscribe_immediately: bool = True,
    result_appliers_by_contract: Mapping[str, ProtocolDispatchResultApplier]
    | None = None,
    materialized_explicit_dependencies: dict[str, dict[str, object]] | None = None,
    topology: ModelDeploymentTopology | None = None,
    catalog_binding_policy: ProjectionCatalogBindingPolicy | None = None,
) -> ModelAutoWiringReport:
    """Wire all discovered contracts into the dispatch engine and event bus.

    For each contract in the manifest that has both ``handler_routing`` and
    ``event_bus`` declarations:

    1. Import handler modules from ``handler_routing.handlers[].handler``.
    2. Instantiate handler classes via DI container (if provided) or zero-arg ctor.
    3. Create dispatch callbacks wrapping each handler.
    4. Register dispatchers and routes on the dispatch engine.
    5. Subscribe to Kafka topics via the event bus (if provided).

    Contracts without ``handler_routing`` or ``event_bus`` are skipped.
    Per-contract failures are collected across the full scan; after all contracts
    are processed, if any failures exist a ``ModelOnexError`` is raised listing
    all of them (OMN-8735 strict invariant).

    Args:
        manifest: The auto-wiring manifest from discovery.
        dispatch_engine: The MessageDispatchEngine to register routes on.
        event_bus: Optional event bus for Kafka subscriptions. When None,
            topic subscriptions are skipped (dispatchers + routes still registered).
        environment: Environment name for consumer group derivation.
        container: Optional DI container used to resolve handler constructor
            deps. Threaded into ``ModelHandlerResolverContext`` and consumed
            by ``ServiceHandlerResolver`` at precedence Step 3 (OMN-9199).
        subscribe_immediately: When True (default), commit Kafka subscriptions
            during this call. When False, only dispatchers/routes are registered;
            callers must invoke ``subscribe_wired_contract_topics()`` after the
            dispatch engine is frozen.
        result_appliers_by_contract: Optional per-contract dispatch result
            appliers. Only contracts present in this mapping apply dispatcher
            outputs from auto-wired callbacks.
        materialized_explicit_dependencies: Optional pre-built constructor
            dependencies keyed by handler name for resolver Step 2.
        topology: Checked-in deployment topology loaded by the composition
            boundary. Required for every contract that declares ``db_io``.
        catalog_binding_policy: Explicit topology binding names for catalog read
            and write operations. Missing choices fail catalog wiring closed.

    Returns:
        A :class:`ModelAutoWiringReport` with per-contract outcomes.
    """
    _require_unique_canonical_contract_names(
        tuple(contract.name for contract in manifest.contracts),
        identity_source="manifest",
    )

    # Construct the resolver + ownership query ONCE per wiring pass from the
    # manifest itself (OMN-9201). The ownership query is set-membership
    # against the locally discovered node_name set — no I/O, no SQL. See
    # omnibase_core/services/service_local_handler_ownership_query.py.
    resolver = ServiceHandlerResolver()
    ownership_query: object = ServiceLocalHandlerOwnershipQuery(
        local_node_names=frozenset(c.name for c in manifest.contracts)
    )
    # Infra-boundary protocol conformance check. This is the ONLY place where
    # core+spi types meet via isinstance; see plan §Layering Invariants.
    _assert_is_ownership_query(ownership_query)

    # Phase 0: Async pre-resolution — resolve handler instances from container via
    # get_service_async before entering the sync _prepare_contract_wiring loop.
    # This avoids calling container.get_service() (sync) from inside a running event
    # loop where the underlying asyncio.run() raises RuntimeError (OMN-9410).
    # Pre-resolved instances are threaded as pre_resolved_handlers so the sync resolver
    # can skip its container Step 3 entirely for these handlers.
    pre_resolved_handlers: dict[str, object] = {}
    if container is not None:
        for contract in manifest.contracts:
            if _is_raw_event_projection_contract(
                contract
            ) and not _raw_event_projection_enabled(
                contract, result_appliers_by_contract
            ):
                continue
            if contract.handler_routing is None:
                continue
            for entry in contract.handler_routing.handlers:
                handler_name = entry.handler.name
                handler_key = _derive_handler_entry_key(entry)
                if handler_key in pre_resolved_handlers:
                    continue
                try:
                    handler_cls = _import_handler_class(
                        entry.handler.module, handler_name
                    )
                    instance = await _async_resolve_from_container(
                        container, handler_cls
                    )
                    if instance is not None:
                        pre_resolved_handlers[handler_key] = instance
                        logger.debug(
                            "Auto-wiring: pre-resolved %s.%s key=%s via container (async)",
                            entry.handler.module,
                            handler_name,
                            handler_key,
                        )
                except Exception:  # noqa: BLE001 — import errors are caught per-contract in Phase 1
                    pass

    # Phase 1: Validate and prepare ALL contracts — no engine/bus side effects yet.
    # Failures are collected; if any exist, we raise before touching anything (OMN-8735).
    prepared_contracts: list[PreparedContractWiring] = []
    failed_results: list[ModelContractWiringResult] = []
    dispatcher_coverage_failed_gaps: list[str] = []
    for contract in manifest.contracts:
        try:
            prepared = _prepare_contract_wiring(
                contract=contract,
                dispatch_engine=dispatch_engine,
                resolver=resolver,
                ownership_query=ownership_query,
                event_bus=event_bus,
                environment=environment,
                container=container,
                pre_resolved_handlers=pre_resolved_handlers
                if container is not None
                else None,
                result_appliers_by_contract=result_appliers_by_contract,
                materialized_explicit_dependencies=materialized_explicit_dependencies,
                topology=topology,
                catalog_binding_policy=catalog_binding_policy,
            )
            prepared_contracts.append(prepared)
        except TypeError:
            # OMN-8735 invariant: resolver-exhaustion TypeError must NOT be
            # demoted to a collectable failure. Propagate unchanged so the
            # kernel crashes loudly at boot.
            raise
        except StateIoUnconfiguredError:
            # OMN-14484 invariant: a REQUIRED state_io seam without its DSN is a
            # startup-FATAL config error (OMN-14208) — never a per-contract
            # failure to collect under non-strict mode. Propagate so boot crashes
            # loudly instead of booting "healthy" with the orchestrator silently
            # dead and every one of its messages routed to the DLQ.
            raise
        except Exception as exc:  # noqa: BLE001 — collect per-contract, raise after scan
            exc_summary = _sanitize_exc(exc)
            logger.error(
                "Auto-wiring contract '%s' from package '%s' raised: %s",
                contract.name,
                contract.package_name,
                type(exc).__name__,
            )
            failed_results.append(
                ModelContractWiringResult(
                    contract_name=contract.name,
                    package_name=contract.package_name,
                    outcome=EnumWiringOutcome.FAILED,
                    reason=f"{type(exc).__name__}: {exc_summary}",
                )
            )
            if _is_orchestrator_contract(contract):
                for alias in _start_event_type_aliases(contract):
                    dispatcher_coverage_failed_gaps.append(
                        f"{contract.name}: missing dispatcher for {alias} "
                        f"because handler preparation failed "
                        f"({type(exc).__name__}: {exc_summary})"
                    )

    # Check for failures before committing any side effects.
    # ONEX_WIRING_STRICT_MODE=1 raises on any failure (default OFF per OMN-9126:
    # strict gate ships after all downstream consumers are compliant).
    failures = failed_results
    if failures:
        failed_reasons = [f"{r.contract_name}: {r.reason}" for r in failures]
        if os.environ.get("ONEX_WIRING_STRICT_MODE", "").lower() in ("1", "true"):
            raise ModelOnexError(
                f"Auto-wiring failed for {len(failures)} contract(s): "
                + "; ".join(failed_reasons)
            )
        logger.warning(
            "Auto-wiring failed for %d contract(s) (non-strict — set ONEX_WIRING_STRICT_MODE=1 to enforce): %s",
            len(failures),
            "; ".join(failed_reasons),
        )

    if _strict_dispatcher_coverage_enabled():
        _assert_orchestrator_dispatcher_coverage(
            prepared_contracts,
            dispatcher_coverage_failed_gaps,
        )

    # OMN-15474: single-owner command topics, asserted BEFORE Phase 2 commits any
    # subscription. Must stay above the commit loop — the post-commit
    # _detect_duplicate_topics warning below is diagnosis, not a gate.
    _assert_single_owner_command_topics(manifest)

    _preflight_prepared_registration_ids(prepared_contracts, dispatch_engine)

    # Phase 2: All contracts validated — commit registrations and subscriptions.
    # Failed contracts are included in results so total_failed is accurate.
    # service_kernel respects the flag before asserting total_failed == 0.
    results: list[ModelContractWiringResult] = list(failed_results)
    for pcw in prepared_contracts:
        result = await _commit_contract_wiring(
            pcw,
            dispatch_engine,
            event_bus,
            subscribe_immediately=subscribe_immediately,
            result_applier=(result_appliers_by_contract or {}).get(pcw.contract.name),
        )
        results.append(result)

    # OMN-15474 totality post-condition. Phase 1 + Phase 2 above are written so
    # that every manifest contract yields exactly one row (a prepared contract
    # commits a row, a preparation failure collects one). That is a property of
    # two loops, not of this function's signature, so a future refactor can
    # break it silently — and the only downstream symptom would be
    # subscribe_wired_contract_topics aborting the kernel at boot on a
    # report/manifest bijection failure. Backfill instead: any manifest
    # contract with no row gets an explicit SKIPPED row naming why, so the
    # report this function returns is TOTAL by contract rather than by
    # accident. This does NOT relax the downstream identity check — that check
    # is unchanged and still rejects report rows with no manifest contract,
    # which is the direction no backfill can repair.
    unwired_backfill = build_unwired_contract_results(
        manifest,
        reason=UNWIRED_BACKFILL_REASON,
        already_reported=tuple(r.contract_name for r in results),
    )
    if unwired_backfill:
        logger.error(
            "Auto-wiring produced no result row for %d manifest contract(s); "
            "backfilling explicit unwired rows to keep the report total "
            "(OMN-15474). This is a wiring-engine bug, not a contract bug: %s",
            len(unwired_backfill),
            sorted(r.contract_name for r in unwired_backfill),
        )
        results.extend(unwired_backfill)

    duplicates = _detect_duplicate_topics(manifest)

    for dup in duplicates:
        logger.warning(
            "Duplicate topic ownership detected: topic=%s owners=%s level=%s",
            dup.topic,
            dup.owners,
            dup.level,
        )

    # OMN-9457: flatten per-contract quarantines into a report-level list so
    # callers can enumerate every contained handler without walking every
    # result. Order mirrors the per-contract scan so the flat list is
    # deterministic across runs.
    all_quarantined: list[ModelQuarantinedWiring] = []
    for result in results:
        all_quarantined.extend(result.quarantined_handlers)

    report = ModelAutoWiringReport(
        results=tuple(results),
        duplicates=tuple(duplicates),
        quarantined_handlers=tuple(all_quarantined),
    )

    if all_quarantined:
        # High-visibility summary: operators tailing runtime-effects logs
        # on first boot need to see the quarantined set without digging
        # through per-contract DEBUG lines.
        summary = ", ".join(
            f"{q.contract_name}:{q.handler_name}={q.reason.value}"
            for q in all_quarantined
        )
        logger.warning(
            "Auto-wiring quarantined %d handler(s) — runtime will continue "
            "without them. Follow-up migration required: %s",
            len(all_quarantined),
            summary,
        )

    logger.info(
        "Auto-wiring complete: wired=%d skipped=%d failed=%d "
        "quarantined=%d duplicates=%d",
        report.total_wired,
        report.total_skipped,
        report.total_failed,
        report.total_quarantined,
        len(report.duplicates),
    )

    return report


def _contract_provision_topics(contract: ModelDiscoveredContract) -> tuple[str, ...]:
    """Return the topic set this contract owns at boot (OMN-13237 §3.6, OMN-15330,
    OMN-15832).

    Subscribe topics (the consumers that attach) UNION the contract's owned
    publish topics UNION its declared ``event_bus.dlq_topics`` UNION its served
    ``projection_api`` topics. Names come from the contract's own declarations
    only — never a Python literal.

    OMN-15330 — DLQ topics used to be excluded here and left to the best-effort
    universe warm. That delegation broke the moment the warm was switched off:
    ``ONEX_BOOT_UNIVERSE_PROVISION=0`` is the standing onex-dev setting (added
    after the 2026-07-27 >1000-topic broker near-meltdown), and with the warm
    off NOTHING created the declared DLQ topics. The first malformed event then
    hit ``[ONEX_CORE_041_INVALID_CONFIGURATION] Topic '<dlq>' not found on
    broker`` inside ``_route_projection_error_to_dlq`` and the record was
    dropped — observed live on onex-dev 2026-07-28T16:29Z for
    ``onex.dlq.omnimarket.projection-delegation-inference-response-malformed.v1``
    and four siblings.

    The DLQ names are read with ``_read_dlq_topics`` — the SAME reader the
    projection auto-wiring uses to build ``ModelProjectionSinks.dlq_topics`` —
    so the provisioned string is byte-identical to the routing target by
    construction, rather than by a second parser that can drift. DLQ topics
    enter the readiness confirm alongside the rest: attaching a consumer whose
    dead-letter sink is not ready guarantees silent loss on the first malformed
    event, so this fails closed (a NOT_READY contract is retried by the
    OMN-15215 reconciliation loop).

    OMN-15832 — the same universe-warm-off gap applies to ``projection_api``
    (``onex.snapshot.*``) topics: nothing else creates them at boot, and the
    contract's ``event_bus`` union above never scanned that section at all.
    ``read_projection_api_topics`` (``omnibase_infra.tools.contract_topic_extractor``)
    is the SAME parser ``ContractTopicExtractor.extract``'s global scan uses —
    one source of parsing truth, scoped to ``expose: true`` AND
    ``bus_backed: true`` exposures, so the boot provision set can never diverge
    from what ``omnimarket.projection.discovery.build_projection_topic_map``
    will actually serve. Explicitly NOT a fix to re-enable
    ``ONEX_BOOT_UNIVERSE_PROVISION`` or to have the consumer
    (``SnapshotCache``) self-provision its own topics — both remain out of
    scope by standing decision; this stays a governed, boot-side, per-contract
    addition to the same confirm path DLQ topics already use.
    """
    if contract.event_bus is None:
        return ()
    ordered = list(contract.event_bus.subscribe_topics)
    ordered.extend(contract.event_bus.publish_topics)
    typed_dlq_topics = getattr(contract.event_bus, "dlq_topics", ())
    if typed_dlq_topics:
        ordered.extend(typed_dlq_topics)
    else:
        try:
            ordered.extend(_read_dlq_topics(contract.contract_path))
        except Exception:  # noqa: BLE001 — per-contract boot boundary
            # ``_interleave_contract`` runs under ``asyncio.gather(...)`` with no
            # ``return_exceptions=True``, so a raise here would abort the ENTIRE
            # boot subscribe pass for every contract. Degrading this one contract
            # to its pre-OMN-15330 behaviour (no DLQ provisioning) is strictly less
            # bad, and the warning names the contract that needs fixing.
            logger.warning(
                "Could not read event_bus.dlq_topics for contract '%s' from %s — "
                "its DLQ topics will NOT be provisioned at boot (OMN-15330)",
                contract.name,
                contract.contract_path,
                exc_info=True,
            )
    try:
        ordered.extend(read_projection_api_topics(contract.contract_path))
    except Exception:  # noqa: BLE001 — per-contract boot boundary, see DLQ comment above
        logger.warning(
            "Could not read projection_api topics for contract '%s' from %s — "
            "its snapshot topics will NOT be provisioned at boot (OMN-15832)",
            contract.name,
            contract.contract_path,
            exc_info=True,
        )
    return tuple(dict.fromkeys(t for t in ordered if t and t.strip()))


async def subscribe_wired_contract_topics(
    manifest: ModelAutoWiringManifest,
    report: ModelAutoWiringReport,
    dispatch_engine: ProtocolDispatchEngine,
    event_bus: object | None,
    environment: str = "dev",
    result_appliers_by_contract: Mapping[str, ProtocolDispatchResultApplier]
    | None = None,
    *,
    provisioner: ProtocolTopicProvisioner | None = None,
    readiness_config: ModelTopicReadinessConfig | None = None,
    attach_results_out: list[ModelContractAttachResult] | None = None,
    core_runtime_topics: frozenset[str] = frozenset(),
    core_runtime_owners: Mapping[str, str] | None = None,
) -> dict[str, tuple[str, ...]]:
    """Subscribe Kafka topics for contracts that already wired successfully.

    This is the post-freeze companion to ``wire_from_manifest(...,
    subscribe_immediately=False)``. It preserves the kernel invariant that
    consumers only start after the dispatch engine becomes read-only.

    OMN-13237 — when *provisioner* is supplied, the boot interleaves
    provision -> confirm-ready -> attach PER WIRED CONTRACT (replacing the
    global create-all-then-subscribe-all big-bang that caused the cold-broker
    crash-loop). Each contract keeps the provision->ready->attach ORDER
    invariant (§3.9); bounded parallelism across contracts is allowed via
    ``readiness_config.max_concurrent_contract_attach``. A contract whose
    provision/readiness fails is recorded NOT-READY and SKIPPED for attach — it
    never aborts the kernel or recycles the process (§3.5, §3.8). Per-contract
    attach outcomes are appended to *attach_results_out* when provided so the
    caller can build the runtime readiness tri-state.

    Returns the map of attached contract -> attached topics (the contracts that
    actually subscribed). Backward-compatible: with no *provisioner* the
    behavior is the original concurrent subscribe (no readiness gate).
    """
    _validate_initial_subscription_contract_identities(manifest, report)
    if event_bus is None:
        return {}

    report_dispatcher_scopes = tuple(
        (result.contract_name, result.dispatchers_registered)
        for result in report.results
    )
    _validate_contract_dispatcher_ownership(
        dispatch_engine,
        report_dispatcher_scopes,
        allow_empty_scopes=True,
    )

    contract_by_name = {contract.name: contract for contract in manifest.contracts}

    # Collect eligible contracts in priority order (projection appliers first).
    eligible: list[tuple[ModelContractWiringResult, ModelDiscoveredContract]] = []
    for result in _prioritize_subscription_results(
        report,
        result_appliers_by_contract,
    ):
        if result.outcome is not EnumWiringOutcome.WIRED:
            continue
        contract = contract_by_name.get(result.contract_name)
        if contract is None:
            continue
        if not result.dispatchers_registered:
            # Resolver-owned skips and quarantines intentionally register no
            # local dispatcher. They therefore own no consume callback. The
            # old path still subscribed them and a process-global dispatch
            # could execute some other contract's matching handler; keeping
            # them unsubscribed is the only truthful zero-owner state.
            logger.info(
                "Auto-wiring (deferred): skipping Kafka subscription for "
                "contract '%s' because it owns zero dispatchers (OMN-15474)",
                contract.name,
            )
            continue
        if _is_raw_event_projection_contract(contract) and (
            result_appliers_by_contract is None
            or contract.name not in result_appliers_by_contract
        ):
            continue
        # plugin_managed: domain plugin owns Kafka subscription (OMN-10864).
        if contract.event_bus is not None and contract.event_bus.plugin_managed:
            logger.info(
                "Auto-wiring (deferred): skipping Kafka subscription for "
                "plugin-managed contract '%s' (OMN-10864)",
                contract.name,
            )
            continue
        eligible.append((result, contract))

    knobs = readiness_config or ModelTopicReadinessConfig()
    # Bounded parallelism across contracts; each contract keeps its own
    # provision->ready->attach order (§3.9).
    semaphore = asyncio.Semaphore(knobs.max_concurrent_contract_attach)

    async def _provision_ready_attach(
        result: ModelContractWiringResult,
        contract: ModelDiscoveredContract,
    ) -> ModelContractAttachResult:
        async with semaphore:
            return await _interleave_contract(
                name=result.contract_name,
                contract=contract,
                dispatch_engine=dispatch_engine,
                event_bus=event_bus,
                environment=environment,
                result_applier=(result_appliers_by_contract or {}).get(
                    result.contract_name
                ),
                allowed_dispatcher_ids=result.dispatchers_registered,
                provisioner=provisioner,
                readiness_config=knobs,
                core_runtime_topics=core_runtime_topics,
                core_runtime_owners=core_runtime_owners,
            )

    attach_results = await asyncio.gather(
        *(_provision_ready_attach(result, contract) for result, contract in eligible)
    )

    if attach_results_out is not None:
        attach_results_out.extend(attach_results)

    subscribed: dict[str, tuple[str, ...]] = {}
    for ar in attach_results:
        if ar.status is EnumContractAttachStatus.ATTACHED:
            subscribed[ar.contract_name] = ar.topics_subscribed
    return subscribed


async def _interleave_contract(
    *,
    name: str,
    contract: ModelDiscoveredContract,
    dispatch_engine: ProtocolDispatchEngine,
    event_bus: object,
    environment: str,
    result_applier: ProtocolDispatchResultApplier | None,
    allowed_dispatcher_ids: Collection[str] | None,
    provisioner: ProtocolTopicProvisioner | None,
    readiness_config: ModelTopicReadinessConfig,
    core_runtime_topics: frozenset[str] = frozenset(),
    core_runtime_owners: Mapping[str, str] | None = None,
) -> ModelContractAttachResult:
    """Provision -> confirm-ready -> attach for ONE contract (§3.2, OMN-13237).

    The order invariant is enforced here: every ``ensure_topic_exists`` for the
    contract precedes its readiness confirm, which precedes consumer attach.
    """
    dispatcher_scope = _require_registered_contract_dispatcher_scope(
        dispatch_engine,
        allowed_dispatcher_ids,
        contract_name=name,
    )
    provision_topics = _contract_provision_topics(contract)

    readiness: ModelTopicSetReadiness | None = None
    if provisioner is not None and provision_topics:
        # (1) Provision the contract's topics (idempotent), in declared order.
        for topic in provision_topics:
            try:
                await provisioner.ensure_topic_exists(topic_name=topic)
            except TopicReplicationPolicyError:
                # OMN-15395: a durability-policy violation is fail-closed and
                # must escape this best-effort boundary. Attaching a consumer to
                # a contract whose topics were silently skipped because one of
                # them declares RF1 on MSK is the exact outcome the policy
                # exists to prevent.
                raise
            except Exception:  # noqa: BLE001 — boundary: per-contract, never fatal
                logger.warning(
                    "Topic provisioning failed for contract '%s' topic '%s' "
                    "(non-fatal, contract will be NOT-READY)",
                    name,
                    topic,
                    exc_info=True,
                )
        # (2) Confirm broker metadata converged before attaching the consumer.
        try:
            readiness = await provisioner.confirm_topics_ready(
                provision_topics,
                config=readiness_config,
            )
        except Exception:  # noqa: BLE001 — boundary: per-contract, never fatal
            logger.warning(
                "Topic readiness confirm raised for contract '%s' "
                "(non-fatal, contract will be NOT-READY)",
                name,
                exc_info=True,
            )
            readiness = ModelTopicSetReadiness(
                topics=provision_topics,
                status=EnumTopicReadinessStatus.UNAVAILABLE,
            )
        if not readiness.is_ready:
            logger.warning(
                "Contract '%s' NOT-READY: topic metadata did not converge "
                "(status=%s failures=%s) — skipping consumer attach, runtime "
                "stays live (OMN-13237)",
                name,
                readiness.status.value,
                [f.topic for f in readiness.failures],
            )
            return ModelContractAttachResult(
                contract_name=name,
                status=EnumContractAttachStatus.NOT_READY,
                dispatcher_ids=tuple(sorted(dispatcher_scope)),
                readiness=readiness,
                detail=f"readiness {readiness.status.value}",
            )

    # (3) Attach the consumer (readiness passed or no provisioner supplied).
    try:
        topics_subscribed = await _subscribe_contract_topics(
            contract=contract,
            dispatch_engine=dispatch_engine,
            event_bus=event_bus,
            environment=environment,
            result_applier=result_applier,
            allowed_dispatcher_ids=dispatcher_scope,
            core_runtime_topics=core_runtime_topics,
            core_runtime_owners=core_runtime_owners,
        )
    except Exception as exc:  # noqa: BLE001 — boundary: per-contract, never fatal
        logger.warning(
            "Contract '%s' consumer attach FAILED after readiness (non-fatal): %s",
            name,
            type(exc).__name__,
            exc_info=True,
        )
        return ModelContractAttachResult(
            contract_name=name,
            status=EnumContractAttachStatus.FAILED,
            dispatcher_ids=tuple(sorted(dispatcher_scope)),
            readiness=readiness,
            detail=type(exc).__name__,
        )

    return ModelContractAttachResult(
        contract_name=name,
        status=EnumContractAttachStatus.ATTACHED,
        dispatcher_ids=tuple(sorted(dispatcher_scope)),
        topics_subscribed=tuple(topics_subscribed),
        readiness=readiness,
    )


# Bounded background NOT_READY reconciliation (OMN-15215, OMN-13237 follow-up).
DEFAULT_NOT_READY_RETRY_INITIAL_DELAY_SECONDS: float = 30.0
DEFAULT_NOT_READY_RETRY_BACKOFF_SECONDS: float = 30.0
DEFAULT_NOT_READY_RETRY_MAX_ATTEMPTS: int = 5


async def reattach_not_ready_contracts(
    manifest: ModelAutoWiringManifest,
    not_ready_results: Sequence[ModelContractAttachResult],
    dispatch_engine: ProtocolDispatchEngine,
    event_bus: object | None,
    environment: str = "dev",
    result_appliers_by_contract: Mapping[str, ProtocolDispatchResultApplier]
    | None = None,
    *,
    provisioner: ProtocolTopicProvisioner | None = None,
    readiness_config: ModelTopicReadinessConfig | None = None,
    core_runtime_topics: frozenset[str] = frozenset(),
    core_runtime_owners: Mapping[str, str] | None = None,
) -> tuple[dict[str, tuple[str, ...]], tuple[ModelContractAttachResult, ...]]:
    """Re-attempt provision -> confirm-ready -> attach for contracts still NOT_READY.

    OMN-15215 (CONFIRMED root cause): ``subscribe_wired_contract_topics`` makes
    exactly ONE provision->confirm->attach attempt per contract via
    ``_interleave_contract``. A contract whose topic metadata has not converged
    within the bounded readiness poll (``ModelTopicReadinessConfig``, 30s/60
    attempts by default) is recorded NOT_READY and its consumer attach is
    skipped — PERMANENTLY, for the rest of the process lifetime, because
    nothing ever calls ``_interleave_contract`` for it again. For a
    wide-topic-count contract (e.g. ``node_ledger_projection_compute``'s 26
    topics, OMN-15006/OMN-15168) a transient cold-broker topic-creation race on
    a handful of just-provisioned topics starves the ENTIRE contract's
    consumer: since ``_subscribe_contract_topics`` subscribes a contract's
    topics as one all-or-nothing unit, zero of its 26 topics ever get a Kafka
    consumer group — not even the ones unrelated to the race. Live evidence
    (fresh stability-test boot, 2026-07-27): ``NOT-READY: topic metadata did
    not converge (status=not_ready failures=[4 OCC governance topics])``
    logged exactly once at boot, followed by a ZERO count of "Auto-wired
    subscription ... node=node_ledger_projection_compute" log lines across the
    container's entire observed lifetime (3 separate contract-discovery
    passes, ~11 minutes) — the OMN-13237 "runtime stays live" framing implies
    eventual recoverability that was never actually implemented.

    This is NOT the ``handler_routing_loader`` "Unknown routing_strategy
    'topic_match'" fallback warning (that code path is a separate, informational
    ``RuntimeContractConfigLoader`` boot-summary pass — its output is never
    consumed by ``auto_wiring``'s wire/attach decision, confirmed by a real,
    unmocked repro: the current ``discovery.py`` + ``handler_wiring.py`` path
    already wires and attaches 26/26 topic_match entries for
    ``node_ledger_projection_compute`` via the topic-folded dispatcher-ID
    derivation from OMN-14580/OMN-13825). Fixing the loader's
    ``VALID_ROUTING_STRATEGIES`` alone would NOT have unblocked OMN-15169 —
    only closing this NOT_READY-has-no-retry gap does.

    Re-runs the SAME provision->confirm->attach interleave
    (``_interleave_contract``) for each contract still in NOT_READY status,
    returning newly-attached topics and updated per-contract results. Callers
    invoke this repeatedly (bounded, with backoff — see
    ``run_not_ready_reconciliation_loop``) until every contract attaches or a
    bounded retry budget is exhausted.
    """
    pending_results = _validate_not_ready_contract_identities(
        manifest,
        not_ready_results,
    )
    if event_bus is None:
        return {}, ()

    contract_by_name = {contract.name: contract for contract in manifest.contracts}
    still_not_ready_names = tuple(result.contract_name for result in pending_results)
    if not still_not_ready_names:
        return {}, ()

    _validate_contract_dispatcher_ownership(
        dispatch_engine,
        tuple(
            (result.contract_name, result.dispatcher_ids) for result in pending_results
        ),
    )

    knobs = readiness_config or ModelTopicReadinessConfig()
    semaphore = asyncio.Semaphore(knobs.max_concurrent_contract_attach)

    not_ready_by_name = {result.contract_name: result for result in pending_results}

    async def _retry_one(name: str) -> ModelContractAttachResult | None:
        contract = contract_by_name.get(name)
        previous_result = not_ready_by_name.get(name)
        if contract is None or previous_result is None:
            return None
        async with semaphore:
            return await _interleave_contract(
                name=name,
                contract=contract,
                dispatch_engine=dispatch_engine,
                event_bus=event_bus,
                environment=environment,
                result_applier=(result_appliers_by_contract or {}).get(name),
                allowed_dispatcher_ids=previous_result.dispatcher_ids,
                provisioner=provisioner,
                readiness_config=knobs,
                core_runtime_topics=core_runtime_topics,
                core_runtime_owners=core_runtime_owners,
            )

    retried = await asyncio.gather(
        *(_retry_one(name) for name in still_not_ready_names)
    )
    results = tuple(r for r in retried if r is not None)

    newly_subscribed: dict[str, tuple[str, ...]] = {
        r.contract_name: r.topics_subscribed
        for r in results
        if r.status is EnumContractAttachStatus.ATTACHED
    }
    return newly_subscribed, results


async def run_not_ready_reconciliation_loop(
    manifest: ModelAutoWiringManifest,
    initial_not_ready: Sequence[ModelContractAttachResult],
    dispatch_engine: ProtocolDispatchEngine,
    event_bus: object | None,
    environment: str = "dev",
    result_appliers_by_contract: Mapping[str, ProtocolDispatchResultApplier]
    | None = None,
    *,
    provisioner: ProtocolTopicProvisioner | None = None,
    readiness_config: ModelTopicReadinessConfig | None = None,
    core_runtime_topics: frozenset[str] = frozenset(),
    core_runtime_owners: Mapping[str, str] | None = None,
    initial_delay_seconds: float = DEFAULT_NOT_READY_RETRY_INITIAL_DELAY_SECONDS,
    backoff_seconds: float = DEFAULT_NOT_READY_RETRY_BACKOFF_SECONDS,
    max_attempts: int = DEFAULT_NOT_READY_RETRY_MAX_ATTEMPTS,
    on_attempt: Callable[
        [dict[str, tuple[str, ...]], tuple[ModelContractAttachResult, ...]], None
    ]
    | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> tuple[ModelContractAttachResult, ...]:
    """Bounded background retry of NOT_READY contracts (OMN-15215).

    Sleeps ``initial_delay_seconds``, then re-attempts every still-NOT_READY
    contract via ``reattach_not_ready_contracts``, up to ``max_attempts`` times
    with ``backoff_seconds`` between attempts. Stops early once every contract
    has attached. Never raises on a still-NOT_READY outcome — this preserves
    the OMN-13237 fail-open boot contract (a contract that never converges
    stays degraded, not crash-looping); this loop makes "runtime stays live"
    actually recoverable instead of a permanent skip. ``on_attempt`` is an
    optional caller hook (e.g. to fold newly-subscribed topics into shared
    boot-time bookkeeping such as topic-collision detection) invoked after
    each attempt with ``(newly_subscribed, results)``. ``sleep`` is injectable
    so tests can drive the loop without real wall-clock delay.
    """
    validated_not_ready = _validate_not_ready_contract_identities(
        manifest,
        initial_not_ready,
    )
    _validate_contract_dispatcher_ownership(
        dispatch_engine,
        tuple(
            (result.contract_name, result.dispatcher_ids)
            for result in validated_not_ready
        ),
    )
    pending: dict[str, ModelContractAttachResult] = {
        r.contract_name: r for r in validated_not_ready
    }
    if not pending:
        return ()

    await sleep(initial_delay_seconds)
    latest: dict[str, ModelContractAttachResult] = {}
    for attempt in range(1, max_attempts + 1):
        if not pending:
            break
        newly_subscribed, results = await reattach_not_ready_contracts(
            manifest,
            tuple(pending.values()),
            dispatch_engine,
            event_bus,
            environment,
            result_appliers_by_contract,
            provisioner=provisioner,
            readiness_config=readiness_config,
            core_runtime_topics=core_runtime_topics,
            core_runtime_owners=core_runtime_owners,
        )
        for result in results:
            latest[result.contract_name] = result
            if result.status is EnumContractAttachStatus.ATTACHED:
                pending.pop(result.contract_name, None)
            else:
                pending[result.contract_name] = result
        if on_attempt is not None:
            on_attempt(newly_subscribed, results)
        logger.info(
            "NOT_READY reconciliation attempt %d/%d: resolved=%d remaining=%d "
            "(OMN-15215)",
            attempt,
            max_attempts,
            len(newly_subscribed),
            len(pending),
        )
        if pending and attempt < max_attempts:
            await sleep(backoff_seconds)

    if pending:
        logger.warning(
            "NOT_READY reconciliation exhausted after %d attempts, still "
            "not-ready: %s (OMN-15215/OMN-13237, runtime stays live degraded)",
            max_attempts,
            sorted(pending),
        )
    return tuple(latest.values())


def _prioritize_subscription_results(
    report: ModelAutoWiringReport,
    result_appliers_by_contract: Mapping[str, ProtocolDispatchResultApplier]
    | None = None,
) -> tuple[ModelContractWiringResult, ...]:
    """Return wired results with explicitly-applied contracts first.

    Contract-specific result appliers represent durable side effects such as
    projection writes. Subscribing them before the generic backlog avoids
    missing readiness-critical events during long Kafka group-join startup.
    """
    if not result_appliers_by_contract:
        return tuple(report.results)

    priority_contracts = frozenset(result_appliers_by_contract)
    indexed_results = tuple(enumerate(report.results))
    return tuple(
        result
        for _, result in sorted(
            indexed_results,
            key=lambda item: (
                0 if item[1].contract_name in priority_contracts else 1,
                item[0],
            ),
        )
    )


def _prepare_contract_wiring(
    *,
    contract: ModelDiscoveredContract,
    dispatch_engine: object,
    resolver: ServiceHandlerResolver,
    ownership_query: object,
    event_bus: object | None,
    environment: str,
    container: object | None = None,
    materialized_explicit_dependencies: (dict[str, dict[str, object]] | None) = None,
    pre_resolved_handlers: dict[str, object] | None = None,
    result_appliers_by_contract: Mapping[str, ProtocolDispatchResultApplier]
    | None = None,
    topology: ModelDeploymentTopology | None = None,
    catalog_binding_policy: ProjectionCatalogBindingPolicy | None = None,
) -> PreparedContractWiring:
    """Prepare one contract for wiring — NO side effects.

    Skipped contracts encode the skip on ``skip_result``. Handler-preparation
    failures raise ``ModelOnexError`` (caller collects across contracts).
    Resolver-Step-6 ``TypeError`` propagates unchanged to preserve the
    OMN-8735 fail-fast invariant.
    """
    if contract.handler_routing is None:
        return PreparedContractWiring(
            contract=contract,
            prepared_wirings=[],
            subscription_topics=[],
            environment=environment,
            skip_result=ModelContractWiringResult(
                contract_name=contract.name,
                package_name=contract.package_name,
                outcome=EnumWiringOutcome.SKIPPED,
                reason="No handler_routing declared in contract",
            ),
        )

    if contract.event_bus is None or not contract.event_bus.subscribe_topics:
        return PreparedContractWiring(
            contract=contract,
            prepared_wirings=[],
            subscription_topics=[],
            environment=environment,
            skip_result=ModelContractWiringResult(
                contract_name=contract.name,
                package_name=contract.package_name,
                outcome=EnumWiringOutcome.SKIPPED,
                reason="No event_bus.subscribe_topics declared in contract",
            ),
        )

    if _is_raw_event_projection_contract(
        contract
    ) and not _raw_event_projection_enabled(contract, result_appliers_by_contract):
        consumer_purpose = (contract.event_bus.consumer_purpose or "").strip().lower()
        # OMN-14516/OMN-14530: a contract that DECLARES consumer_purpose=audit|
        # projection AND subscribe_topics is asserting it must consume and persist.
        # Reaching this branch means no result applier is wired for it — a kernel
        # MISCONFIGURATION, not a valid opt-out. It is FAILED, never SKIPPED.
        #
        # SKIPPED is not counted by total_failed, so the runtime booted GREEN with
        # the contract silently unwired: zero handlers, zero subscriptions, class
        # never constructed. node_ledger_projection_compute died exactly this way —
        # event_ledger held ZERO rows while 1,028,463 events flowed past, on
        # stability-test AND prod, because nobody hand-added its name to the
        # kernel's result-applier allowlist. That allowlist is now DELETED: an
        # audit/projection consumer wires itself by declaring
        # intent_consumption.intent_routing_table (the kernel DERIVES the applier).
        # If it reaches here it declared neither an applier nor a resolvable routing
        # table, so FAILED makes the gap loud and lets ONEX_WIRING_STRICT_MODE catch
        # it (the strict assert reads total_failed, which SKIPPED sailed past).
        #
        # Same fail-closed reasoning as the StateIoUnconfiguredError seam
        # (OMN-14484): a declared-but-unconfigured durability seam is a startup
        # error, not a per-contract shrug. Do NOT soften this back to SKIPPED to
        # make a boot go green — declare the routing table (kernel derivation wires
        # it) or remove consumer_purpose from the contract.
        return PreparedContractWiring(
            contract=contract,
            prepared_wirings=[],
            subscription_topics=[],
            environment=environment,
            skip_result=ModelContractWiringResult(
                contract_name=contract.name,
                package_name=contract.package_name,
                outcome=EnumWiringOutcome.FAILED,
                reason=(
                    f"consumer_purpose={consumer_purpose!r} declares a raw event "
                    f"projection but no result applier is wired for contract "
                    f"{contract.name!r} — it would consume offsets and drop every "
                    f"intent. Declare intent_consumption.intent_routing_table so the "
                    f"kernel derives a DispatchResultApplier, or remove "
                    f"consumer_purpose."
                ),
            ),
        )

    # OMN-14403 §2C: prove every fan-out handler's emittable classes are declared
    # in this contract's published_events BEFORE registering any dispatch route —
    # an unmapped fan-out element would fall back to the single output_topic and
    # silently misroute. Warn-only while the seam is OFF; fail-closed once ON.
    _check_fanout_publish_coverage(contract)

    prepared_wirings: list[PreparedWiring] = []
    for entry in contract.handler_routing.handlers:
        try:
            prepared = _prepare_handler_wiring(
                contract=contract,
                entry=entry,
                dispatch_engine=dispatch_engine,
                resolver=resolver,
                ownership_query=ownership_query,
                event_bus=event_bus,
                container=container,
                materialized_explicit_dependencies=materialized_explicit_dependencies,
                pre_resolved_handlers=pre_resolved_handlers,
                topology=topology,
                catalog_binding_policy=catalog_binding_policy,
            )
            prepared_wirings.append(prepared)
        except TypeError:
            # OMN-8735 invariant: resolver Step 6 exhaustion must NOT be
            # wrapped. Propagate unchanged so the kernel crashes loudly.
            raise
        except StateIoUnconfiguredError:
            # OMN-14484 invariant: an unconfigured REQUIRED state_io durability
            # seam is a startup-FATAL configuration error (OMN-14208), not a
            # per-handler wiring bug to wrap-and-collect. Propagate UNWRAPPED so
            # wire_from_manifest re-raises it and boot fails loudly. Wrapping it
            # into a generic ModelOnexError + collecting it under non-strict mode
            # turned this fail-CLOSED seam into fail-SILENT: it dropped every
            # dispatcher of the contract while the runtime booted "healthy", so
            # node_delegation_orchestrator (the only state_io contract) DLQ'd
            # 100% of its command/event traffic on any lane missing the DSN.
            raise
        except Exception as exc:
            exc_summary = _sanitize_exc(exc)
            logger.error(
                "Failed to prepare handler '%s' for contract '%s' (package '%s'): %s",
                entry.handler.name,
                contract.name,
                contract.package_name,
                type(exc).__name__,
            )
            raise ModelOnexError(
                f"Auto-wiring contract '{contract.name}' failed: "
                f"handler={entry.handler.name}: {type(exc).__name__}: {exc_summary}"
            ) from exc

    # plugin_managed: domain plugin owns Kafka subscription for this contract's
    # topics (OMN-10864). Dispatch routes are still registered so the engine
    # can route messages consumed via the plugin's EventBusSubcontractWiring.
    subscription_topics: list[str] = (
        []
        if contract.event_bus.plugin_managed
        else list(contract.event_bus.subscribe_topics)
    )
    if contract.event_bus.plugin_managed:
        logger.info(
            "Auto-wiring: skipping Kafka subscription for plugin-managed contract "
            "'%s' — domain plugin owns topic subscription (OMN-10864)",
            contract.name,
        )

    return PreparedContractWiring(
        contract=contract,
        prepared_wirings=prepared_wirings,
        subscription_topics=subscription_topics,
        environment=environment,
    )


async def _commit_contract_wiring(
    pcw: PreparedContractWiring,
    dispatch_engine: object,
    event_bus: object | None,
    *,
    subscribe_immediately: bool = True,
    result_applier: ProtocolDispatchResultApplier | None = None,
    dynamic_materialization_authorized: bool = False,
) -> ModelContractWiringResult:
    """Commit a validated PreparedContractWiring to the engine and event bus.

    All side effects (dispatcher/route registration, Kafka subscriptions)
    happen here. OMN-8735 requires every contract in the manifest has been
    prepared successfully before this is called. Per-handler resolver
    outcomes are projected into ``ModelContractWiringResult.wirings``;
    LOCAL_OWNERSHIP_SKIP entries land in ``skipped_handlers`` (OMN-9201).
    """
    if pcw.skip_result is not None:
        # Why: Runtime validation guarantees the returned value matches the contract.
        return pcw.skip_result  # type: ignore[return-value]

    # Why: Runtime compatibility requires assigning through a broader static type.
    contract: ModelDiscoveredContract = pcw.contract  # type: ignore[assignment]
    dispatchers_registered: list[str] = []
    routes_registered: list[str] = []
    topics_subscribed: list[str] = []
    wirings: list[ModelWiringOutcome] = []
    skipped_handlers: list[ModelSkippedEntry] = []
    quarantined: list[ModelQuarantinedWiring] = []

    for prepared in pcw.prepared_wirings:
        dispatcher_id, route_ids = _commit_handler_wiring(
            prepared,
            dispatch_engine,
            owner_contract_name=contract.name,
            dynamic_materialization_authorized=dynamic_materialization_authorized,
        )
        if prepared.is_quarantined:
            assert prepared.quarantine_reason is not None  # narrow for mypy
            quarantined.append(
                ModelQuarantinedWiring(
                    contract_name=contract.name,
                    package_name=contract.package_name,
                    handler_module=prepared.handler_module,
                    handler_name=prepared.handler_name,
                    reason=prepared.quarantine_reason,
                    detail=prepared.quarantine_detail,
                )
            )
        elif prepared.is_skip:
            skipped_handlers.append(
                ModelSkippedEntry(
                    handler_name=prepared.handler_name,
                    reason=prepared.skip_reason,
                )
            )
        else:
            dispatchers_registered.append(dispatcher_id)
            routes_registered.extend(route_ids)
        wirings.append(
            ModelWiringOutcome(
                handler_name=prepared.handler_name,
                resolution_outcome=prepared.resolution_outcome,
                skipped_reason=prepared.skip_reason,
            )
        )

    # Fail-closed phantom-wiring guard (OMN-14141). A contract that declares
    # subscribe topics but registered ZERO dispatchers — and quarantined /
    # resolver-skipped NOTHING — is a silent phantom-wire: the topic would be
    # consumed and its Kafka offsets committed with no handler ever running.
    # This is the wiring-side backstop for the flat-schema silent-zero-parse
    # defect (the parse guard in discovery._parse_handler_routing is the first
    # line). When all four hold, pcw.prepared_wirings was empty — handler_routing
    # produced no parseable handlers. Legacy top-level ``handler:`` fallbacks and
    # resolver-ownership skips always leave a dispatcher, a skipped_handler, or a
    # quarantine, so this never fires on them. Returned as FAILED (not WIRED) so
    # total_failed is accurate and ONEX_WIRING_STRICT_MODE crashes boot loudly;
    # the topic is NOT subscribed either way.
    if (
        pcw.subscription_topics
        and not dispatchers_registered
        and not skipped_handlers
        and not quarantined
    ):
        return ModelContractWiringResult(
            contract_name=contract.name,
            package_name=contract.package_name,
            outcome=EnumWiringOutcome.FAILED,
            reason=(
                "phantom wiring: contract declares "
                f"{len(pcw.subscription_topics)} subscribe topic(s) but "
                "registered zero dispatchers (handler_routing produced no "
                "parseable handlers). Convert flat handler_class/handler_module "
                "entries to nested handler:{name,module} (OMN-14141)."
            ),
            wirings=tuple(wirings),
            skipped_handlers=tuple(skipped_handlers),
            quarantined_handlers=tuple(quarantined),
        )

    if (
        subscribe_immediately
        and event_bus is not None
        and pcw.subscription_topics
        and dispatchers_registered
    ):
        topics_subscribed.extend(
            await _subscribe_contract_topics(
                contract=contract,
                dispatch_engine=dispatch_engine,
                event_bus=event_bus,
                environment=pcw.environment,
                result_applier=result_applier,
                allowed_dispatcher_ids=dispatchers_registered,
            )
        )
    elif (
        subscribe_immediately
        and event_bus is not None
        and pcw.subscription_topics
        and not dispatchers_registered
    ):
        logger.info(
            "Auto-wiring: skipping Kafka subscription for contract '%s' "
            "because it owns zero dispatchers (OMN-15474)",
            contract.name,
        )

    # OMN-9457: when every prepared handler was quarantined, report SKIPPED
    # with reason "all handlers quarantined" — there is nothing wired on
    # the dispatch engine and the quarantine is the reason. A mixed
    # contract where some handlers were resolver-skipped (not quarantined)
    # and the rest quarantined does NOT take this path: "all handlers
    # quarantined" must mean *every* handler quarantined, not "no live
    # handlers and at least one quarantined". Mixed skip+quarantine
    # contracts fall through to the normal WIRED return below so the
    # existing resolver-skip reasoning remains authoritative for the
    # skipped handlers.
    all_handlers_quarantined = bool(pcw.prepared_wirings) and all(
        p.is_quarantined for p in pcw.prepared_wirings
    )
    if all_handlers_quarantined:
        return ModelContractWiringResult(
            contract_name=contract.name,
            package_name=contract.package_name,
            outcome=EnumWiringOutcome.SKIPPED,
            reason="all handlers quarantined",
            wirings=tuple(wirings),
            skipped_handlers=tuple(skipped_handlers),
            quarantined_handlers=tuple(quarantined),
        )

    return ModelContractWiringResult(
        contract_name=contract.name,
        package_name=contract.package_name,
        outcome=EnumWiringOutcome.WIRED,
        dispatchers_registered=tuple(dispatchers_registered),
        routes_registered=tuple(routes_registered),
        topics_subscribed=tuple(topics_subscribed),
        wirings=tuple(wirings),
        skipped_handlers=tuple(skipped_handlers),
        quarantined_handlers=tuple(quarantined),
    )


async def _subscribe_contract_topics(
    *,
    contract: ModelDiscoveredContract,
    dispatch_engine: object,
    event_bus: object,
    environment: str,
    result_applier: ProtocolDispatchResultApplier | None = None,
    allowed_dispatcher_ids: Collection[str] | None = None,
    core_runtime_topics: frozenset[str] = frozenset(),
    core_runtime_owners: Mapping[str, str] | None = None,
) -> list[str]:
    """Subscribe all declared event-bus topics for a wired contract.

    OMN-14758 (S6): topics in ``core_runtime_topics`` are owned by the ONE core
    ``RuntimeDispatch`` loop, not the legacy push path. The legacy callback is NOT
    built/subscribed for those topics — this split is the mechanism that makes the
    ``RuntimeDispatch ⟂ legacy`` single-owner assertion hold. Default EMPTY ⇒ every
    topic is subscribed by the legacy path exactly as before (zero behavior change).

    OMN-14771 (S8 §D1=4b): ``core_runtime_owners`` maps an allowlisted topic to the ONE
    contract whose consumption MOVED to the core runtime. For a genuine fan-out topic the
    legacy callback is skipped ONLY for the OWNER; the topic's OTHER subscribers stay
    legacy fan-out consumers on their own distinct consumer groups. When a topic has no
    owner entry (a single-owner allowlist topic, or owners unavailable), the S6 behavior
    is preserved: the sole subscriber's legacy callback is skipped.
    """
    owner_by_topic = dict(core_runtime_owners or {})
    if contract.event_bus is None or not contract.event_bus.subscribe_topics:
        return []

    dispatcher_scope = _require_registered_contract_dispatcher_scope(
        dispatch_engine,
        allowed_dispatcher_ids,
        contract_name=contract.name,
    )

    from omnibase_infra.enums import EnumConsumerGroupPurpose
    from omnibase_infra.models import ModelNodeIdentity
    from omnibase_infra.runtime.event_bus_subcontract_wiring import (
        load_published_events_map,
    )
    from omnibase_infra.runtime.service_dispatch_result_applier import (
        DispatchResultApplier,
    )
    from omnibase_infra.utils import compute_consumer_group_id

    typed_bus: ProtocolEventBusSubscriber = cast(
        "ProtocolEventBusSubscriber", event_bus
    )
    effective_result_applier = result_applier
    output_topic = _select_dispatch_result_output_topic(contract)
    if (
        effective_result_applier is None
        and contract.event_bus is not None
        and output_topic is not None
        and not _contract_declares_db_io(contract)
    ):
        # ProtocolEventBusLike is @runtime_checkable; isinstance both narrows
        # the type for mypy and provides a runtime use of the import (avoiding
        # CodeQL py/unused-import false positive when cast() is the only ref).
        assert isinstance(event_bus, ProtocolEventBusLike), (
            f"event_bus must implement ProtocolEventBusLike, got {type(event_bus).__name__}"
        )
        # Route each returned model to ITS declared topic via the contract's
        # published_events map (event_type short-name -> topic). Without this,
        # a multi-publish-topic contract (e.g. the LLM call effect publishing
        # both delegation-call-completed and inference-response) would route
        # every returned model to the single ``output_topic`` fallback (the
        # first publish topic), mis-routing the inference response (OMN-12416).
        # Resolved from the contract's own discovered path, so the map is read
        # from the installed contract regardless of cwd.
        output_topic_map = load_published_events_map(contract.contract_path)
        effective_result_applier = DispatchResultApplier(
            event_bus=event_bus,
            output_topic=output_topic,
            output_topic_map=output_topic_map,
            allowed_output_topics=contract.event_bus.publish_topics,
            # OMN-15468 AC2: hand the applier the contract's DECLARED failure
            # terminal so a returned model that states a failure verdict cannot
            # be republished onto the success terminal by map-miss fallback.
            # Read through the same single reader the Pattern B broker's
            # subscription set is built from, so the two cannot disagree about
            # which topics are terminal for this contract.
            failure_terminal_topics=_declared_failure_terminal_topics(
                contract, success_topic=output_topic
            ),
        )
    node_identity = ModelNodeIdentity(
        env=environment,
        service=contract.package_name,
        node_name=contract.name,
        version=str(contract.contract_version),
    )
    consumer_group = compute_consumer_group_id(
        node_identity, EnumConsumerGroupPurpose.CONSUME
    )

    # Build callbacks for all topics first (synchronous, no I/O).
    topic_callbacks: list[tuple[str, Callable[..., Awaitable[None]]]] = []
    for topic in contract.event_bus.subscribe_topics:
        # OMN-14758 (S6) / OMN-14771 (S8 §D1=4b): the ONE core RuntimeDispatch owns this
        # topic's OWNER route. Skip the legacy push callback for the OWNER only, so
        # ownership is disjoint (single-owner invariant §c.3). A designated-but-different
        # owner means this contract is a legacy fan-out consumer that KEEPS its
        # subscription on its own distinct consumer group. No owner entry ⇒ S6 behavior
        # (skip the sole subscriber).
        if topic in core_runtime_topics:
            owner = owner_by_topic.get(topic)
            if owner is None or owner == contract.name:
                logger.info(
                    "Auto-wiring: skipping legacy subscription for topic=%s node=%s "
                    "(ownership=core-runtime, OMN-14758/OMN-14771)",
                    topic,
                    contract.name,
                )
                continue
            logger.info(
                "Auto-wiring: keeping legacy fan-out subscription for topic=%s node=%s "
                "(core-runtime owner=%s, non-owner stays legacy on its own group, "
                "OMN-14771 §4b)",
                topic,
                contract.name,
                owner,
            )
        if _is_raw_event_projection_contract(contract):
            if effective_result_applier is None:
                raise ModelOnexError(
                    f"Raw event projection contract {contract.name!r} requires "
                    "an explicit result applier to avoid dropping intents."
                )
            callback = _make_raw_event_projection_callback(
                topic,
                # Why: Runtime wiring validates and narrows this payload shape before use.
                dispatch_engine,  # type: ignore[arg-type]
                effective_result_applier,
                allowed_dispatcher_ids=dispatcher_scope,
            )
        else:
            callback = _make_event_bus_callback(
                topic,
                # Why: Runtime wiring validates and narrows this payload shape before use.
                dispatch_engine,  # type: ignore[arg-type]
                result_applier=effective_result_applier,
                tenant_scoped=contract.event_bus.tenant_scoped_ingress,
                # OMN-14507: gives the boundary a DLQ target (duck-typed
                # _publish_raw_to_dlq) when ONEX_BOUNDARY_DLQ_ENABLED is set.
                event_bus=typed_bus,
                # OMN-14493 §4.3: on the state_io / in-row-outbox path ONLY, a
                # publish-from-row failure + conflict-retry exhaustion PROPAGATE
                # (redeliver) instead of being log-and-discarded. Non-state_io
                # contracts keep the historical swallow behavior unchanged.
                propagate_publish_failures=_contract_declares_state_io(contract),
                allowed_dispatcher_ids=dispatcher_scope,
            )
        topic_callbacks.append((topic, callback))

    # Subscribe all topics concurrently.  Each subscribe() triggers a Kafka
    # consumer group-join (5-10 s per topic); running them in parallel reduces
    # cold-start time from O(n*t) to O(t) for n topics.
    async def _subscribe_one(
        topic: str,
        cb: Callable[..., Awaitable[None]],
    ) -> str:
        await typed_bus.subscribe(
            topic=topic,
            node_identity=node_identity,
            on_message=cb,
        )
        logger.info(
            "Auto-wired subscription: topic=%s consumer_group=%s node=%s",
            topic,
            consumer_group,
            contract.name,
        )
        return topic

    topics_subscribed: list[str] = list(
        await asyncio.gather(*(_subscribe_one(t, cb) for t, cb in topic_callbacks))
    )

    return topics_subscribed


def _declared_failure_terminal_topics(
    contract: ModelDiscoveredContract,
    *,
    success_topic: str,
) -> tuple[str, ...]:
    """Return the contract's declared FAILURE terminal topics (OMN-15468 AC2).

    A failure terminal is any contract-declared terminal topic that is (a) not
    the success terminal the applier falls back to and (b) actually publishable
    by this contract. Both conditions matter: publishing to an undeclared topic
    would violate the contract's own publish allowlist, and re-routing to the
    success terminal would be a no-op.

    Read through :func:`load_terminal_event_topics` — the SAME reader the
    Pattern B broker's subscription set is built from — so the applier's idea of
    which topics are terminal cannot drift from the broker's. A second
    hand-rolled reader here is exactly the seam mismatch that produced this
    ticket.
    """
    if contract.event_bus is None or not contract.event_bus.publish_topics:
        return ()
    publishable = set(contract.event_bus.publish_topics)
    declared = load_terminal_event_topics(contract.contract_path)
    return tuple(
        sorted(
            topic
            for topic in declared
            if topic != success_topic and topic in publishable
        )
    )


def _select_dispatch_result_output_topic(
    contract: ModelDiscoveredContract,
) -> str | None:
    """Choose the fallback output topic for generic dispatch results.

    Prefer the contract's declared terminal_event when it is also publishable.
    Otherwise fall back to the first publish topic to preserve older contracts.
    """
    if contract.event_bus is None or not contract.event_bus.publish_topics:
        return None
    if (
        contract.terminal_event
        and contract.terminal_event in contract.event_bus.publish_topics
    ):
        return contract.terminal_event
    return contract.event_bus.publish_topics[0]


async def _wire_single_contract(
    *,
    contract: ModelDiscoveredContract,
    dispatch_engine: ProtocolDispatchEngine,
    event_bus: object | None,
    environment: str,
    container: object | None = None,
    topology: ModelDeploymentTopology | None = None,
    catalog_binding_policy: ProjectionCatalogBindingPolicy | None = None,
    dynamic_materialization_authorized: bool = False,
) -> ModelContractWiringResult:
    """Wire a single discovered contract into the dispatch engine.

    Thin wrapper around _prepare_contract_wiring + _commit_contract_wiring.
    Kept for backwards compatibility. New code should use wire_from_manifest
    which validates all contracts before committing any side effects.

    Constructs a single-contract ownership query locally so the resolver's
    ownership-skip step resolves affirmatively for the caller's contract.
    """
    resolver = ServiceHandlerResolver()
    ownership_query: object = ServiceLocalHandlerOwnershipQuery(
        local_node_names=frozenset({contract.name})
    )
    _assert_is_ownership_query(ownership_query)

    prepared = _prepare_contract_wiring(
        contract=contract,
        dispatch_engine=dispatch_engine,
        resolver=resolver,
        ownership_query=ownership_query,
        event_bus=event_bus,
        environment=environment,
        container=container,
        topology=topology,
        catalog_binding_policy=catalog_binding_policy,
    )
    _preflight_prepared_registration_ids(
        (prepared,),
        dispatch_engine,
        dynamic_materialization_authorized=dynamic_materialization_authorized,
    )
    return await _commit_contract_wiring(
        prepared,
        dispatch_engine,
        event_bus,
        dynamic_materialization_authorized=dynamic_materialization_authorized,
    )


def _prepare_handler_wiring(
    *,
    contract: ModelDiscoveredContract,
    entry: ModelHandlerRoutingEntry,
    dispatch_engine: object,
    resolver: ServiceHandlerResolver,
    ownership_query: object,
    event_bus: object | None = None,
    container: object | None = None,
    materialized_explicit_dependencies: (dict[str, dict[str, object]] | None) = None,
    pre_resolved_handlers: dict[str, object] | None = None,
    topology: ModelDeploymentTopology | None = None,
    catalog_binding_policy: ProjectionCatalogBindingPolicy | None = None,
) -> PreparedWiring:
    """Prepare one handler entry — delegates construction to the resolver.

    The full precedence chain (ownership skip → node registry → container →
    event_bus → zero-arg → TypeError) lives in
    ``omnibase_core.services.service_handler_resolver.ServiceHandlerResolver``
    (OMN-9199). No engine mutation here; side effects happen in
    :func:`_commit_handler_wiring` (OMN-8735 two-phase invariant).

    OMN-8735 fail-fast is preserved: the resolver's Step 6 ``TypeError`` is
    NOT caught here; it propagates unchanged to the caller. ``is_skip``
    entries returned from this function MUST NOT be committed.

    pre_resolved_handlers: Instances already resolved via get_service_async in
    Phase 0 of wire_from_manifest (OMN-9410). When present for a handler, the
    resolver's container Step 3 is bypassed — the pre-resolved instance is used
    directly. This avoids asyncio.run() inside a running event loop.
    """
    from omnibase_core.enums.enum_handler_resolution_outcome import (
        EnumHandlerResolutionOutcome,
    )
    from omnibase_core.models.dispatch.model_dispatch_route import ModelDispatchRoute
    from omnibase_core.models.resolver.model_handler_resolution import (
        ModelHandlerResolution,
    )
    from omnibase_infra.enums import EnumMessageCategory

    handler_ref = entry.handler
    handler_key = _derive_handler_entry_key(entry)

    # Determine category/message types before importing or constructing the
    # handler. Some domain-owned contracts use auto-wiring for subscriptions
    # only; those entries must report accurate metadata while avoiding generic
    # direct-handler dispatcher registration.
    _category_str_early = "event"
    if entry.message_category:
        _category_str_early = entry.message_category.strip().lower()
    elif contract.event_bus and contract.event_bus.subscribe_topics:
        _category_str_early = _derive_message_category(
            contract.event_bus.subscribe_topics[0]
        )
    _early_category = EnumMessageCategory(_category_str_early)

    message_types: set[str] | None = None
    if entry.event_model is not None:
        message_types = {entry.event_model.name}
    # OMN-9215: index the dispatcher under the contract-declared event_type alias
    # in addition to the Pydantic class name. Publishers set
    # ModelEventEnvelope.event_type to the dot-path string; without this alias,
    # the dispatcher lookup falls back to type(payload).__name__ which resolves
    # to "dict" on object-erased envelopes and never matches the class-name key.
    # Strip surrounding whitespace so registration matches the dispatch-engine
    # normalization (message_dispatch_engine.py normalizes via .strip()).
    event_type_alias = entry.event_type.strip() if entry.event_type else ""
    subscribe_topics = contract.event_bus.subscribe_topics if contract.event_bus else ()
    literal_topic_aliases = _literal_event_type_aliases_from_topics(subscribe_topics)
    if literal_topic_aliases:
        message_types = (message_types or set()).union(literal_topic_aliases)
    if event_type_alias:
        message_types = (message_types or set()) | {event_type_alias}
    elif contract.event_bus:
        topic_aliases = {
            alias
            for topic in subscribe_topics
            if (alias := _derive_event_type_alias_from_topic(topic)) is not None
        }
        if topic_aliases:
            message_types = (message_types or set()).union(topic_aliases)

    handler_cls = _import_handler_class(handler_ref.module, handler_ref.name)
    handler_constructor_params = inspect.signature(handler_cls).parameters
    if (
        "event_publisher" in handler_constructor_params
        and event_bus is None
        and contract.event_bus is not None
        and contract.event_bus.publish_topics
    ):
        raise ModelOnexError(
            "handler_wiring: handler "
            f"{handler_ref.name!r} declares event_publisher for publishing "
            f"{tuple(contract.event_bus.publish_topics)!r}, but no runtime "
            "event_bus was provided."
        )

    # Fast path: if Phase 0 pre-resolved this handler via get_service_async,
    # skip the sync resolver's container Step 3 entirely (OMN-9410).
    pre_resolved_instance = (
        pre_resolved_handlers.get(handler_key) if pre_resolved_handlers else None
    )
    if pre_resolved_handlers is not None and _should_skip_sync_container_resolution(
        handler_cls
    ):
        _effective_container = None
    elif container is not None:
        _effective_container = container
    elif dispatch_engine is not None:
        _effective_container = getattr(dispatch_engine, "_container", None)
    else:
        _effective_container = None
    _effective_materialized_dependencies = _materialize_known_handler_dependencies(
        handler_name=handler_ref.name,
        handler_cls=handler_cls,
        materialized_explicit_dependencies=materialized_explicit_dependencies,
        event_bus=event_bus,
        container=_effective_container,
        ownership_query=ownership_query,
        # OMN-15468: the publishing contract's own declared terminal topics,
        # read through the same function route discovery uses, so the wiring's
        # notion of "this publish is a terminal" cannot drift from the set the
        # Pattern B broker subscribes to.
        terminal_topics=load_terminal_event_topics(contract.contract_path),
    )

    def _quarantine_prepared(
        *,
        reason: EnumQuarantineReason,
        detail: str,
    ) -> PreparedWiring:
        """Return a containment-only PreparedWiring for a known bad handler.

        Known containment-worthy declaration/construction failures are
        surfaced in the wiring report instead of partially registering a broken
        dispatcher. Resolver Step-6 constructor exhaustion TypeError remains
        boot-fatal outside these explicit reasons.
        """
        logger.warning(
            "Auto-wiring: quarantining handler %s.%s "
            "(contract=%s, package=%s, reason=%s): %s. Runtime-effects boot "
            "will continue; follow-up migration required.",
            handler_ref.module,
            handler_ref.name,
            contract.name,
            contract.package_name,
            reason.value,
            detail,
        )
        return PreparedWiring(
            dispatcher_id="",
            dispatcher=_skip_dispatcher,
            category=_early_category,
            message_types=None,
            handler_name=handler_ref.name,
            handler_module=handler_ref.module,
            resolution_outcome=EnumHandlerResolutionOutcome.UNRESOLVABLE,
            skip_reason=f"quarantined:{reason.value}",
            quarantine_reason=reason,
            quarantine_detail=detail,
        )

    if pre_resolved_instance is not None:
        resolution = ModelHandlerResolution(
            outcome=EnumHandlerResolutionOutcome.RESOLVED_VIA_CONTAINER,
            handler_instance=pre_resolved_instance,
        )
        logger.debug(
            "Auto-wiring: using pre-resolved instance for %s.%s key=%s",
            handler_ref.module,
            handler_ref.name,
            handler_key,
        )
    else:
        # node_name=contract.name: established ONEX naming convention — see
        # ModelNodeIdentity construction at _commit_contract_wiring below.
        ctx = ModelHandlerResolverContext(
            handler_cls=handler_cls,
            handler_module=handler_ref.module,
            handler_name=handler_ref.name,
            contract_name=contract.name,
            node_name=contract.name,
            explicit_dependency_shape=None,
            materialized_explicit_dependencies=_effective_materialized_dependencies,
            event_bus=event_bus,
            container=_effective_container,
            ownership_query=ownership_query,
        )
        try:
            resolution = resolver.resolve(ctx)
        except TypeError as exc:
            # OMN-12501: Protocol interfaces are non-instantiable by design.
            # They are invalid as handler_routing targets, but should be
            # reported as contract migration work rather than crashing
            # runtime-effects boot under the generic resolver TypeError path.
            if _is_protocol_handler_class(handler_cls):
                return _quarantine_prepared(
                    reason=EnumQuarantineReason.PROTOCOL_HANDLER_DECLARATION,
                    detail=_sanitize_exc(exc),
                )
            # OMN-13203: a bare resolver TypeError that is NOT a Protocol target
            # is exactly the unsatisfiable-ctor (ServiceHandlerResolver Step 6)
            # or ctor-arg-mismatch (Step 2) per-handler wiring bug. These are
            # the ONLY `raise TypeError` sites in the resolver (Steps 1a/2/6),
            # are deterministic, never recoverable runtime state, and never an
            # infra outage — broker/DB/secret failures surface as ModelOnexError
            # / InfraConnectionError / ConnectionError / OSError, never a bare
            # resolver TypeError. Before this change the bare re-raise here
            # propagated through the OMN-8735 TypeError guards and crashed the
            # whole runtime-effects boot (every healthy handler with it). Quarantine
            # the single bad handler and continue so the runtime binds its health
            # server and reports failed=N. Strict mode re-raises (preserves the
            # boot-crash invariant) so the gate can still fail closed.
            if _wiring_strict_mode_enabled():
                raise
            return _quarantine_prepared(
                reason=EnumQuarantineReason.UNRESOLVABLE_HANDLER,
                detail=_sanitize_exc(exc),
            )
        except ValueError as exc:
            # OMN-13203: a per-handler ValueError from resolver/context construction
            # (not-handle-shaped handler, blank-required field) is the same class
            # of deterministic per-handler wiring bug as the unsatisfiable-ctor
            # TypeError above — contain it identically. ValueError is NOT raised by
            # broker/DB/secret transports (those raise ModelOnexError /
            # InfraConnectionError / ConnectionError / OSError, caught by the
            # `except Exception` arms upstream which still propagate), so this does
            # not over-broaden the catch into infra outages. Strict mode re-raises.
            if _wiring_strict_mode_enabled():
                raise
            return _quarantine_prepared(
                reason=EnumQuarantineReason.UNRESOLVABLE_HANDLER,
                detail=_sanitize_exc(exc),
            )
        except RuntimeError as exc:
            # OMN-9457: deterministic containment for handlers whose
            # construction path calls asyncio.run() inside runtime-managed
            # async boot. Any other RuntimeError propagates unchanged.
            if _is_async_incompat_runtime_error(exc):
                return _quarantine_prepared(
                    reason=EnumQuarantineReason.ASYNC_INCOMPATIBLE,
                    detail=_sanitize_exc(exc),
                )
            raise

    # _early_category was computed up-front so the quarantine sentinel could
    # carry consistent reporting metadata; reuse it here for the live path.
    category = _early_category

    if (
        resolution.outcome
        is EnumHandlerResolutionOutcome.RESOLVED_VIA_LOCAL_OWNERSHIP_SKIP
    ):
        # Deliberate skip — caller records it in skipped_handlers; nothing
        # is registered on the dispatch engine (OMN-9201).
        return PreparedWiring(
            dispatcher_id="",
            dispatcher=_skip_dispatcher,
            category=category,
            message_types=message_types,
            handler_name=handler_ref.name,
            handler_module=handler_ref.module,
            resolution_outcome=resolution.outcome,
            skip_reason=resolution.skipped_reason,
        )

    # Narrow at the infra boundary: core types handler_instance as
    # object | None per §Layering Invariants; non-skip outcomes guarantee
    # a constructed handler.
    handler_instance = cast("ProtocolHandleable", resolution.handler_instance)

    # Use projection callback when contract declares db_io.db_tables, or the
    # opt-in stateful callback when it declares state_io (OMN-14208). These
    # two wiring arms are disjoint by construction: db_io projection handlers
    # own their own persistence and terminal-event emission, while state_io
    # wraps the standard dispatch callback with a load-before/CAS-persist-
    # after boundary hook and returns a normal ModelDispatchResult through
    # the standard result-applier path. A contract declaring both is a
    # wiring-time contract defect, not a case to silently prioritize one arm.
    db_tables = tuple(contract.db_io.db_tables) if contract.db_io is not None else ()
    state_io = _read_state_io(contract.contract_path)
    if db_tables and state_io:
        raise ModelOnexError(
            f"handler_wiring: contract {contract.name!r} declares BOTH "
            "db_io and state_io — these are disjoint wiring arms "
            "(OMN-14208); a contract must declare exactly one."
        )
    if db_tables:
        if topology is None:
            raise ModelOnexError(
                f"handler_wiring: contract {contract.name!r} declares db_io but "
                "wire_from_manifest received no checked-in ModelDeploymentTopology"
            )
        catalog_policy = catalog_binding_policy or ProjectionCatalogBindingPolicy()
        target = _resolve_projection_database_target(
            db_tables,
            topology,
            catalog_read_binding=catalog_policy.read_binding,
            catalog_write_binding=catalog_policy.write_binding,
        )
        subscribe_topics = (
            contract.event_bus.subscribe_topics if contract.event_bus else ()
        )
        projection_terminal_event = (
            contract.terminal_event
            if contract.terminal_event
            and contract.event_bus is not None
            and contract.terminal_event in contract.event_bus.publish_topics
            else None
        )
        # OMN-13548 (D-03): resolve the malformed-event DLQ destination from the
        # contract's typed event-bus declaration. Filesystem-discovered legacy
        # contracts retain the raw-YAML fallback during the migration window.
        typed_dlq_topics = (
            getattr(contract.event_bus, "dlq_topics", ())
            if contract.event_bus is not None
            else ()
        )
        projection_dlq_topics = (
            list(typed_dlq_topics)
            if typed_dlq_topics
            else _read_dlq_topics(contract.contract_path)
        )
        callback = _make_projection_dispatch_callback(
            handler_instance,
            target,
            subscribe_topics,
            sinks=ProjectionDispatchSinks(
                event_bus=event_bus,
                terminal_event=projection_terminal_event,
                dlq_topics=tuple(projection_dlq_topics),
            ),
        )
        logger.info(
            "Auto-wired projection handler with DB injection: handler=%s db_tables=%s "
            "terminal_event=%s dlq_topics=%s",
            handler_ref.name,
            [table.name for table in target.tables],
            projection_terminal_event,
            projection_dlq_topics,
        )
        # Projection handlers route by topic/db_io, not event_model; leave
        # them untyped so the projection dispatch path is unchanged.
        payload_type_matcher: Callable[[object], bool] | None = None
    elif state_io:
        # OMN-14493 in-row outbox: give the stateful wrapper the bus + the
        # contract's published_events (class -> topic) map so it can publish-from-
        # row + CAS-finalize WITHIN the leg (production has-bus path) and re-publish
        # a crash-recovered batch on boot. Without these the wrapper falls back to
        # commit-then-return (external applier publishes) — which is the exact
        # CAS-retry loss OMN-14493 fixes, so production MUST pass them.
        from omnibase_infra.runtime.event_bus_subcontract_wiring import (
            load_published_events_map,
        )

        _outbox_topic_map = (
            load_published_events_map(contract.contract_path)
            if contract.event_bus is not None
            else None
        )
        callback = _make_stateful_dispatch_callback(
            handler_instance,
            entry.event_model,
            state_io,
            event_bus=event_bus,
            output_topic_map=_outbox_topic_map,
        )
        logger.info(
            "Auto-wired stateful handler with state_io in-row outbox "
            "(publish-from-row + CAS-finalize): handler=%s table=%s "
            "outbox_topics=%d",
            handler_ref.name,
            state_io.get("table"),
            len(_outbox_topic_map or {}),
        )
        # state_io wraps the standard dispatch callback, so the same
        # event_model type-scoping applies as the non-stateful path
        # (OMN-12416) — a multi-leg orchestrator's sibling handler entries
        # still route by their own declared event_model.
        payload_type_matcher = (
            _make_payload_type_matcher(entry.event_model)
            if entry.event_model is not None
            else None
        )
    else:
        _handler_node_kind = _node_kind_from_node_type(contract.node_type)
        # OMN-14794: a REDUCER that DECLARES a published event and returns that
        # event model must emit it as an EVENT, not a projection. Thread the
        # contract's published_events short-names so _normalize_handler_result
        # routes a declared-event return to output_events instead of the
        # OMN-14598 REDUCER->projection capture (live delegation-routing drop that
        # stalled the FSM at RECEIVED). Only loaded for REDUCERs — the sole
        # archetype whose projection classification the set can refine.
        _published_event_names: frozenset[str] | None = None
        if (
            _handler_node_kind is EnumNodeKind.REDUCER
            and contract.event_bus is not None
        ):
            from omnibase_infra.runtime.event_bus_subcontract_wiring import (
                load_published_events_map,
            )

            _published_event_names = frozenset(
                load_published_events_map(contract.contract_path)
            )
        callback = _make_dispatch_callback(
            handler_instance,
            entry.event_model,
            handler_node_kind=_handler_node_kind,
            published_event_names=_published_event_names,
        )
        # Type-scope the dispatcher on its declared event_model so a
        # multi-handler contract routes each message to the single handler
        # whose event_model matches the payload (OMN-12416). Untyped
        # (operation-only) handlers stay un-scoped — legacy matching applies.
        payload_type_matcher = (
            _make_payload_type_matcher(entry.event_model)
            if entry.event_model is not None
            else None
        )
    dispatcher_id = _derive_dispatcher_id(contract.name, handler_key)

    # Pre-compute routes (no engine calls yet)
    route_ids: list[str] = []
    routes: list[ModelDispatchRoute] = []
    if contract.event_bus:
        for topic in _topics_for_handler_entry(contract, entry):
            route_id = _derive_route_id(contract.name, handler_key, topic)
            topic_pattern = _derive_topic_pattern_from_topic(topic)

            route = ModelDispatchRoute(
                route_id=route_id,
                topic_pattern=topic_pattern,
                message_category=category,
                handler_id=dispatcher_id,
            )
            route_ids.append(route_id)
            routes.append(route)

    return PreparedWiring(
        dispatcher_id=dispatcher_id,
        dispatcher=callback,
        category=category,
        message_types=message_types,
        handler_name=handler_ref.name,
        handler_module=handler_ref.module,
        resolution_outcome=resolution.outcome,
        route_ids=route_ids,
        routes=routes,
        payload_type_matcher=payload_type_matcher,
    )


def _commit_handler_wiring(
    prepared: PreparedWiring,
    dispatch_engine: object,
    *,
    owner_contract_name: str | None = None,
    dynamic_materialization_authorized: bool = False,
) -> tuple[str, list[str]]:
    """Register a prepared handler wiring with the dispatch engine (side effects only).

    Must only be called after :func:`_prepare_handler_wiring` has succeeded for
    ALL handlers in a contract, ensuring the engine is never mutated for a
    partially-valid contract (OMN-8735).

    Skip entries (``prepared.is_skip``) are no-ops — the resolver emitted
    ``RESOLVED_VIA_LOCAL_OWNERSHIP_SKIP`` for this handler, so nothing is
    registered on the dispatch engine (OMN-9201). Quarantined entries
    (``prepared.is_quarantined``) are also no-ops — OMN-9457 containment
    keeps async-incompatible handlers off the dispatch engine so they
    cannot poison runtime-effects boot.

    When ``dynamic_materialization_authorized=True`` and the engine is frozen,
    the private dynamic registration methods are used instead of the standard
    ones. This flag MUST only be set by ``materialize_cached_contract()`` after
    full contract validation — never by general application code (OMN-11246).
    ``owner_contract_name`` records the contract provenance used by the
    pre-subscribe ownership validator. Auto-wiring callers always supply it;
    direct/manual registrations remain deliberately unowned.

    Returns:
        Tuple of (dispatcher_id, list of route_ids registered). Returns
        ``("", [])`` for skip / quarantined entries.
    """
    if prepared.is_skip or prepared.is_quarantined:
        return "", []

    from omnibase_core.enums.enum_core_error_code import EnumCoreErrorCode
    from omnibase_core.models.errors import ModelOnexError
    from omnibase_infra.runtime.message_dispatch_engine import (
        MessageDispatchEngine,
    )

    engine = dispatch_engine
    if isinstance(engine, MessageDispatchEngine):
        if engine.is_frozen:
            if not dynamic_materialization_authorized:
                raise ModelOnexError(
                    message="Post-freeze registration requires explicit dynamic "
                    "materialization authorization.",
                    error_code=EnumCoreErrorCode.INVALID_STATE,
                )
            engine._register_dispatcher_dynamic(
                dispatcher_id=prepared.dispatcher_id,
                dispatcher=prepared.dispatcher,
                category=prepared.category,
                message_types=prepared.message_types,
                payload_type_matcher=prepared.payload_type_matcher,
                owner_contract_name=owner_contract_name,
            )
            for route in prepared.routes:
                engine._register_route_dynamic(route)
        else:
            engine.register_dispatcher(
                dispatcher_id=prepared.dispatcher_id,
                dispatcher=prepared.dispatcher,
                category=prepared.category,
                message_types=prepared.message_types,
                payload_type_matcher=prepared.payload_type_matcher,
                owner_contract_name=owner_contract_name,
            )
            for route in prepared.routes:
                engine.register_route(route)

    return prepared.dispatcher_id, prepared.route_ids


def _wire_handler_entry(
    *,
    contract: ModelDiscoveredContract,
    entry: ModelHandlerRoutingEntry,
    dispatch_engine: object,
    event_bus: object | None = None,
    container: object | None = None,
    topology: ModelDeploymentTopology | None = None,
    catalog_binding_policy: ProjectionCatalogBindingPolicy | None = None,
) -> tuple[str, list[str]]:
    """Prepare and immediately commit one handler entry (single-contract shortcut).

    Kept for backwards compatibility with call sites that don't need the
    two-phase split.  New code should call _prepare_handler_wiring +
    _commit_handler_wiring directly.
    """
    resolver = ServiceHandlerResolver()
    ownership_query: object = ServiceLocalHandlerOwnershipQuery(
        local_node_names=frozenset({contract.name})
    )
    _assert_is_ownership_query(ownership_query)

    prepared = _prepare_handler_wiring(
        contract=contract,
        entry=entry,
        dispatch_engine=dispatch_engine,
        resolver=resolver,
        ownership_query=ownership_query,
        event_bus=event_bus,
        container=container,
        topology=topology,
        catalog_binding_policy=catalog_binding_policy,
    )
    return _commit_handler_wiring(
        prepared,
        dispatch_engine,
        owner_contract_name=contract.name,
    )
