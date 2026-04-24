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

import importlib
import logging
import os
import re
from collections import defaultdict
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

from omnibase_core.enums.enum_handler_resolution_outcome import (
    EnumHandlerResolutionOutcome,
)
from omnibase_core.models.errors import ModelOnexError
from omnibase_core.models.resolver.model_handler_resolver_context import (
    ModelHandlerResolverContext,
)
from omnibase_core.protocols.event_bus.protocol_event_bus_subscriber import (
    ProtocolEventBusSubscriber,
)
from omnibase_core.services.service_handler_resolver import ServiceHandlerResolver
from omnibase_core.services.service_local_handler_ownership_query import (
    ServiceLocalHandlerOwnershipQuery,
)
from omnibase_infra.protocols.protocol_dispatch_result_applier import (
    ProtocolDispatchResultApplier,
)
from omnibase_infra.runtime.auto_wiring.enum_quarantine_reason import (
    EnumQuarantineReason,
)
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelDiscoveredContract,
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
from omnibase_spi.protocols.runtime.protocol_handler_ownership_query import (
    ProtocolHandlerOwnershipQuery,
)

if TYPE_CHECKING:
    from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
    from omnibase_infra.enums import EnumMessageCategory
    from omnibase_infra.models.dispatch.model_dispatch_result import (
        ModelDispatchResult,
    )
    from omnibase_infra.models.dispatch.model_dispatch_route import ModelDispatchRoute
    from omnibase_infra.protocols.protocol_dispatch_engine import (
        ProtocolDispatchEngine,
    )

logger = logging.getLogger(__name__)

# Matches DSNs, URLs, and connection strings that may contain credentials.
_SENSITIVE_PATTERN = re.compile(
    r"(?:postgresql|postgres|mysql|redis|amqp|kafka|mongodb|http|https)://\S*",
    re.IGNORECASE,
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


@dataclass
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


@dataclass
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
        AttributeError: If the class is not found in the module.
    """
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls


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
) -> DispatcherFunc:
    """Create a dispatch callback wrapping a handler instance.

    The callback calls ``handler_instance.handle(envelope)`` and returns the
    result. This matches the ``DispatcherFunc`` signature expected by
    ``MessageDispatchEngine.register_dispatcher``.
    """

    async def _callback(
        envelope: ModelEventEnvelope[object],
    ) -> ModelDispatchResult | None:
        handle_method = handler_instance.handle
        return await handle_method(envelope)

    return _callback


_TABLE_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

_DB_URL_ENV_MAP: dict[str, str] = {
    "omnidash_analytics": "OMNIDASH_ANALYTICS_DB_URL",
    "omnibase_infra": "OMNIBASE_INFRA_DB_URL",
}

_TOPIC_TO_EVENT_TYPE: dict[str, str] = {
    "node-heartbeat": "heartbeat",
    "node-introspection": "introspection",
}


def _read_db_io_tables(contract_path: Path) -> list[dict[str, str]]:
    """Read db_io.db_tables from a contract YAML. Returns [] if db_io is absent.

    Raises on YAML parse errors or unexpected file I/O failures so the caller
    can mark the contract as broken rather than silently falling back to the
    non-projection wiring path.
    """
    try:
        import yaml  # type: ignore[import-untyped]

        with open(contract_path) as f:
            raw = yaml.safe_load(f)
    except FileNotFoundError:
        return []
    if not isinstance(raw, dict):
        return []
    db_io = raw.get("db_io") or {}
    return list(db_io.get("db_tables") or [])


def _build_sync_db_adapter(db_url: str) -> object:
    """Build a synchronous psycopg2-backed DatabaseAdapter from a DSN.

    Separated from _make_projection_dispatch_callback to allow test patching.
    """
    import psycopg2  # type: ignore[import-untyped]
    import psycopg2.extras  # type: ignore[import-untyped]

    class SyncPsycopg2Adapter:
        _conn: object

        def __init__(self, dsn: str) -> None:
            self._dsn = dsn
            self._conn = None

        def _get_conn(self) -> object:
            if self._conn is None or getattr(self._conn, "closed", False):
                conn = psycopg2.connect(self._dsn)
                conn.autocommit = True
                self._conn = conn
            return self._conn

        def upsert(self, table: str, conflict_key: str, row: dict[str, object]) -> bool:
            if not _TABLE_NAME_RE.match(table):
                raise ValueError(f"Invalid table name: {table!r}")
            if not _TABLE_NAME_RE.match(conflict_key):
                raise ValueError(f"Invalid conflict key: {conflict_key!r}")
            conn = self._get_conn()
            cols = list(row.keys())
            bad_cols = [c for c in cols if not _TABLE_NAME_RE.match(str(c))]
            if bad_cols:
                raise ValueError(f"Invalid column names: {bad_cols!r}")
            quoted_cols = ", ".join(f'"{c}"' for c in cols)
            placeholders = ", ".join(f"%({c})s" for c in cols)
            updates = ", ".join(
                f'"{c}" = EXCLUDED."{c}"' for c in cols if c != conflict_key
            )
            # table/conflict_key/cols validated by _TABLE_NAME_RE — not raw user input
            parts = [
                f'INSERT INTO "{table}" ({quoted_cols})',
                f"VALUES ({placeholders})",
                f'ON CONFLICT ("{conflict_key}") DO UPDATE SET {updates}',
            ]
            insert_sql = " ".join(parts)
            with conn.cursor() as cur:  # type: ignore[union-attr, attr-defined]
                cur.execute(insert_sql, row)
            return True

        def query(
            self, table: str, filters: dict[str, object] | None = None
        ) -> list[dict[str, object]]:
            if not _TABLE_NAME_RE.match(table):
                raise ValueError(f"Invalid table name: {table!r}")
            conn = self._get_conn()
            # table validated by _TABLE_NAME_RE — not user input
            select_sql = f'SELECT * FROM "{table}"'  # noqa: S608
            params: list[object] = []
            if filters:
                bad_keys = [k for k in filters if not _TABLE_NAME_RE.match(str(k))]
                if bad_keys:
                    raise ValueError(f"Invalid filter keys: {bad_keys!r}")
                clauses = [f'"{k}" = %s' for k in filters]
                select_sql += " WHERE " + " AND ".join(clauses)
                params = list(filters.values())
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:  # type: ignore[union-attr, attr-defined]
                cur.execute(select_sql, params or None)
                return [dict(r) for r in cur.fetchall()]

    return SyncPsycopg2Adapter(db_url)


def _make_projection_dispatch_callback(
    handler_instance: object,
    db_tables: list[dict[str, str]],
    subscribe_topics: tuple[str, ...],
) -> DispatcherFunc:
    """Create a dispatch callback for projection handlers (db_io.db_tables declared).

    Builds a synchronous psycopg2 DatabaseAdapter per call and injects it into
    input_data alongside _event_type derived from the topic name.
    """
    database = (
        db_tables[0].get("database", "omnidash_analytics")
        if db_tables
        else "omnidash_analytics"
    )
    if database not in _DB_URL_ENV_MAP:
        raise ValueError(
            f"Unknown database {database!r} in contract db_io — "
            f"must be one of {sorted(_DB_URL_ENV_MAP)!r}"
        )
    db_url_env = _DB_URL_ENV_MAP[database]

    async def _callback(
        envelope: ModelEventEnvelope[object],
    ) -> ModelDispatchResult | None:
        db_url = os.environ.get(db_url_env, "")
        if not db_url:
            logger.error(
                "Projection handler skipped: %s not set (database=%s)",
                db_url_env,
                database,
            )
            return None
        try:
            adapter = _build_sync_db_adapter(db_url)
            topic = getattr(envelope, "topic", "") or ""
            topic_segment = topic.split(".")[-2] if "." in topic else topic
            if topic_segment not in _TOPIC_TO_EVENT_TYPE:
                raise ValueError(
                    f"Unknown topic segment {topic_segment!r} from topic {topic!r} — "
                    f"must be one of {sorted(_TOPIC_TO_EVENT_TYPE)!r}"
                )
            event_type = _TOPIC_TO_EVENT_TYPE[topic_segment]
            payload = getattr(envelope, "payload", None)
            input_data: dict[str, object] = (
                dict(payload) if isinstance(payload, dict) else {}
            )
            if hasattr(payload, "model_dump"):
                input_data = payload.model_dump(mode="python")  # type: ignore[union-attr]  # noqa: model-dump-bare
            input_data["_db"] = adapter
            input_data["_event_type"] = event_type
            result = handler_instance.handle(input_data)  # type: ignore[union-attr, attr-defined]
            logger.debug(
                "Projection handler completed: topic=%s event_type=%s result=%s",
                topic,
                event_type,
                result,
            )
        except TypeError as exc:
            logger.error(
                "Projection handler TypeError (likely missing _db or _event_type): "
                "handler=%s topic=%s error_type=%s",
                type(handler_instance).__name__,
                getattr(envelope, "topic", "unknown"),
                type(exc).__name__,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Projection handler error: handler=%s topic=%s error_type=%s",
                type(handler_instance).__name__,
                getattr(envelope, "topic", "unknown"),
                type(exc).__name__,
            )
        return None

    return _callback


def _make_event_bus_callback(
    topic: str,
    dispatch_engine: ProtocolDispatchEngine,
    result_applier: ProtocolDispatchResultApplier | None = None,
) -> Callable[..., Awaitable[None]]:
    """Create a Kafka on_message callback that deserializes and dispatches to engine.

    Mirrors EventBusSubcontractWiring._create_dispatch_callback but stripped of
    DLQ/idempotency concerns. When a result applier is supplied, dispatcher
    outputs are applied on the same auto-wired path that owns the subscription.
    """
    import json

    from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope

    async def callback(message: object) -> None:
        try:
            raw = getattr(message, "value", None)
            if raw is not None:
                data = json.loads(
                    raw.decode("utf-8") if isinstance(raw, bytes) else raw
                )
                envelope: ModelEventEnvelope[object] = ModelEventEnvelope[
                    object
                ].model_validate(data)
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
            result = await dispatch_engine.dispatch(topic, envelope)
            if result_applier is not None and result is not None:
                await result_applier.apply(result, envelope.correlation_id)
        except Exception as exc:  # noqa: BLE001 — boundary: log and discard; unsubscribe unavailable here
            logger.error(
                "Auto-wiring callback error: topic=%s error_type=%s error=%s",
                topic,
                type(exc).__name__,
                exc,
            )

    return callback


def _derive_route_id(contract_name: str, handler_name: str, topic: str) -> str:
    """Derive a route ID from contract name, handler name, and full topic path.

    Uses the full topic path (sanitized) to guarantee uniqueness across topics
    that share a common segment (OMN-8735).
    """
    safe_topic = re.sub(r"[.\-]", "_", topic)
    return f"route.auto.{contract_name}.{handler_name}.{safe_topic}"


def _derive_dispatcher_id(contract_name: str, handler_name: str) -> str:
    """Derive a dispatcher ID from contract and handler names."""
    return f"dispatcher.auto.{contract_name}.{handler_name}"


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

    Returns:
        A :class:`ModelAutoWiringReport` with per-contract outcomes.
    """
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
            if contract.handler_routing is None:
                continue
            for entry in contract.handler_routing.handlers:
                handler_name = entry.handler.name
                if handler_name in pre_resolved_handlers:
                    continue
                try:
                    handler_cls = _import_handler_class(
                        entry.handler.module, handler_name
                    )
                    instance = await _async_resolve_from_container(
                        container, handler_cls
                    )
                    if instance is not None:
                        pre_resolved_handlers[handler_name] = instance
                        logger.debug(
                            "Auto-wiring: pre-resolved %s.%s via container (async)",
                            entry.handler.module,
                            handler_name,
                        )
                except Exception:  # noqa: BLE001 — import errors are caught per-contract in Phase 1
                    pass

    # Phase 1: Validate and prepare ALL contracts — no engine/bus side effects yet.
    # Failures are collected; if any exist, we raise before touching anything (OMN-8735).
    prepared_contracts: list[PreparedContractWiring] = []
    failed_results: list[ModelContractWiringResult] = []
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
                if pre_resolved_handlers
                else None,
            )
            prepared_contracts.append(prepared)
        except TypeError:
            # OMN-8735 invariant: resolver-exhaustion TypeError must NOT be
            # demoted to a collectable failure. Propagate unchanged so the
            # kernel crashes loudly at boot.
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


async def subscribe_wired_contract_topics(
    manifest: ModelAutoWiringManifest,
    report: ModelAutoWiringReport,
    dispatch_engine: ProtocolDispatchEngine,
    event_bus: object | None,
    environment: str = "dev",
    result_appliers_by_contract: Mapping[str, ProtocolDispatchResultApplier]
    | None = None,
) -> dict[str, tuple[str, ...]]:
    """Subscribe Kafka topics for contracts that already wired successfully.

    This is the post-freeze companion to ``wire_from_manifest(...,
    subscribe_immediately=False)``. It preserves the kernel invariant that
    consumers only start after the dispatch engine becomes read-only.
    """
    if event_bus is None:
        return {}

    contract_by_name = {contract.name: contract for contract in manifest.contracts}
    subscriptions_by_contract: dict[str, tuple[str, ...]] = {}

    for result in report.results:
        if result.outcome is not EnumWiringOutcome.WIRED:
            continue
        contract = contract_by_name.get(result.contract_name)
        if contract is None:
            continue
        topics_subscribed = await _subscribe_contract_topics(
            contract=contract,
            dispatch_engine=dispatch_engine,
            event_bus=event_bus,
            environment=environment,
            result_applier=(result_appliers_by_contract or {}).get(contract.name),
        )
        subscriptions_by_contract[contract.name] = tuple(topics_subscribed)

    return subscriptions_by_contract


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
            )
            prepared_wirings.append(prepared)
        except TypeError:
            # OMN-8735 invariant: resolver Step 6 exhaustion must NOT be
            # wrapped. Propagate unchanged so the kernel crashes loudly.
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

    return PreparedContractWiring(
        contract=contract,
        prepared_wirings=prepared_wirings,
        subscription_topics=list(contract.event_bus.subscribe_topics),
        environment=environment,
    )


async def _commit_contract_wiring(
    pcw: PreparedContractWiring,
    dispatch_engine: object,
    event_bus: object | None,
    *,
    subscribe_immediately: bool = True,
    result_applier: ProtocolDispatchResultApplier | None = None,
) -> ModelContractWiringResult:
    """Commit a validated PreparedContractWiring to the engine and event bus.

    All side effects (dispatcher/route registration, Kafka subscriptions)
    happen here. OMN-8735 requires every contract in the manifest has been
    prepared successfully before this is called. Per-handler resolver
    outcomes are projected into ``ModelContractWiringResult.wirings``;
    LOCAL_OWNERSHIP_SKIP entries land in ``skipped_handlers`` (OMN-9201).
    """
    if pcw.skip_result is not None:
        return pcw.skip_result  # type: ignore[return-value]

    contract: ModelDiscoveredContract = pcw.contract  # type: ignore[assignment]
    dispatchers_registered: list[str] = []
    routes_registered: list[str] = []
    topics_subscribed: list[str] = []
    wirings: list[ModelWiringOutcome] = []
    skipped_handlers: list[ModelSkippedEntry] = []
    quarantined: list[ModelQuarantinedWiring] = []

    for prepared in pcw.prepared_wirings:
        dispatcher_id, route_ids = _commit_handler_wiring(prepared, dispatch_engine)
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

    if subscribe_immediately and event_bus is not None and pcw.subscription_topics:
        topics_subscribed.extend(
            await _subscribe_contract_topics(
                contract=contract,
                dispatch_engine=dispatch_engine,
                event_bus=event_bus,
                environment=pcw.environment,
                result_applier=result_applier,
            )
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
) -> list[str]:
    """Subscribe all declared event-bus topics for a wired contract."""
    if contract.event_bus is None or not contract.event_bus.subscribe_topics:
        return []

    from omnibase_infra.enums import EnumConsumerGroupPurpose
    from omnibase_infra.models import ModelNodeIdentity
    from omnibase_infra.utils import compute_consumer_group_id

    typed_bus: ProtocolEventBusSubscriber = cast(
        "ProtocolEventBusSubscriber", event_bus
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

    topics_subscribed: list[str] = []
    for topic in contract.event_bus.subscribe_topics:
        callback = _make_event_bus_callback(
            topic,
            dispatch_engine,  # type: ignore[arg-type]
            result_applier=result_applier,
        )
        await typed_bus.subscribe(
            topic=topic,
            node_identity=node_identity,
            on_message=callback,
        )
        topics_subscribed.append(topic)
        logger.info(
            "Auto-wired subscription: topic=%s consumer_group=%s node=%s",
            topic,
            consumer_group,
            contract.name,
        )

    return topics_subscribed


async def _wire_single_contract(
    *,
    contract: ModelDiscoveredContract,
    dispatch_engine: ProtocolDispatchEngine,
    event_bus: object | None,
    environment: str,
    container: object | None = None,
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
    )
    return await _commit_contract_wiring(prepared, dispatch_engine, event_bus)


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
    from omnibase_core.models.resolver.model_handler_resolution import (
        ModelHandlerResolution,
    )
    from omnibase_infra.enums import EnumMessageCategory
    from omnibase_infra.models.dispatch.model_dispatch_route import ModelDispatchRoute

    handler_ref = entry.handler
    handler_cls = _import_handler_class(handler_ref.module, handler_ref.name)

    _effective_container = container or (
        getattr(dispatch_engine, "_container", None)
        if dispatch_engine is not None
        else None
    )

    # Fast path: if Phase 0 pre-resolved this handler via get_service_async,
    # skip the sync resolver's container Step 3 entirely (OMN-9410).
    pre_resolved_instance = (
        pre_resolved_handlers.get(handler_ref.name) if pre_resolved_handlers else None
    )

    # Determine category up-front so the quarantine sentinel below (which
    # bypasses the regular resolve/construct path) can still carry consistent
    # reporting metadata.
    _category_str_early = "EVENT"
    if contract.event_bus and contract.event_bus.subscribe_topics:
        _category_str_early = _derive_message_category(
            contract.event_bus.subscribe_topics[0]
        )
    _early_category = EnumMessageCategory(_category_str_early)

    def _quarantine_prepared(exc: BaseException) -> PreparedWiring:
        """Return a containment-only PreparedWiring for an async-incompat handler.

        OMN-9457: the handler's constructor raised ``RuntimeError: asyncio.run()
        cannot be called from a running event loop``. We deterministically
        contain it — no dispatcher / route registration — and surface it on the
        wiring report so follow-up migration is visible rather than
        partially-broken runtime state.
        """
        detail = _sanitize_exc(exc)
        logger.warning(
            "Auto-wiring: quarantining async-incompatible handler %s.%s "
            "(contract=%s, package=%s): %s. Runtime-effects boot will "
            "continue; convert the handler to async-safe construction to "
            "re-enable it.",
            handler_ref.module,
            handler_ref.name,
            contract.name,
            contract.package_name,
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
            skip_reason=f"quarantined:{EnumQuarantineReason.ASYNC_INCOMPATIBLE.value}",
            quarantine_reason=EnumQuarantineReason.ASYNC_INCOMPATIBLE,
            quarantine_detail=detail,
        )

    if pre_resolved_instance is not None:
        resolution = ModelHandlerResolution(
            outcome=EnumHandlerResolutionOutcome.RESOLVED_VIA_CONTAINER,
            handler_instance=pre_resolved_instance,
        )
        logger.debug(
            "Auto-wiring: using pre-resolved instance for %s.%s",
            handler_ref.module,
            handler_ref.name,
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
            materialized_explicit_dependencies=materialized_explicit_dependencies,
            event_bus=event_bus,
            container=_effective_container,
            ownership_query=ownership_query,
        )
        try:
            resolution = resolver.resolve(ctx)
        except RuntimeError as exc:
            # OMN-9457: deterministic containment for handlers whose
            # construction path calls asyncio.run() inside runtime-managed
            # async boot. Any other RuntimeError propagates unchanged.
            if _is_async_incompat_runtime_error(exc):
                return _quarantine_prepared(exc)
            raise

    # _early_category was computed up-front so the quarantine sentinel could
    # carry consistent reporting metadata; reuse it here for the live path.
    category = _early_category

    message_types: set[str] | None = None
    if entry.event_model is not None:
        message_types = {entry.event_model.name}
    # OMN-9215: index the dispatcher under the contract-declared event_type alias
    # in addition to the Pydantic class name. Publishers set
    # ModelEventEnvelope.event_type to the dot-path string; without this alias,
    # the dispatcher lookup falls back to type(payload).__name__ which resolves
    # to "dict" on object-erased envelopes and never matches the class-name key.
    # Strip surrounding whitespace so registration matches the dispatch-engine
    # normalization (service_message_dispatch_engine.py normalizes via .strip()).
    event_type_alias = entry.event_type.strip() if entry.event_type else ""
    if event_type_alias:
        message_types = (message_types or set()) | {event_type_alias}

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

    # Use projection callback when contract declares db_io.db_tables.
    # Projection handlers implement handle(input_data: dict) with _db injected
    # rather than the standard async handle(envelope) protocol.
    db_tables = _read_db_io_tables(contract.contract_path)
    if db_tables:
        subscribe_topics = (
            contract.event_bus.subscribe_topics if contract.event_bus else ()
        )
        callback = _make_projection_dispatch_callback(
            handler_instance,
            db_tables,
            subscribe_topics,
        )
        logger.info(
            "Auto-wired projection handler with DB injection: handler=%s db_tables=%s",
            handler_ref.name,
            [t.get("name") for t in db_tables],
        )
    else:
        callback = _make_dispatch_callback(handler_instance)
    dispatcher_id = _derive_dispatcher_id(contract.name, handler_ref.name)

    # Pre-compute routes (no engine calls yet)
    route_ids: list[str] = []
    routes: list[ModelDispatchRoute] = []
    if contract.event_bus:
        for topic in contract.event_bus.subscribe_topics:
            route_id = _derive_route_id(contract.name, handler_ref.name, topic)
            topic_pattern = _derive_topic_pattern_from_topic(topic)

            route = ModelDispatchRoute(
                route_id=route_id,
                topic_pattern=topic_pattern,
                message_category=category,
                dispatcher_id=dispatcher_id,
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
    )


def _commit_handler_wiring(
    prepared: PreparedWiring,
    dispatch_engine: object,
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

    Returns:
        Tuple of (dispatcher_id, list of route_ids registered). Returns
        ``("", [])`` for skip / quarantined entries.
    """
    if prepared.is_skip or prepared.is_quarantined:
        return "", []

    from omnibase_infra.runtime.service_message_dispatch_engine import (
        MessageDispatchEngine,
    )

    engine = dispatch_engine
    if isinstance(engine, MessageDispatchEngine):
        engine.register_dispatcher(
            dispatcher_id=prepared.dispatcher_id,
            dispatcher=prepared.dispatcher,
            category=prepared.category,
            message_types=prepared.message_types,
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
    )
    return _commit_handler_wiring(prepared, dispatch_engine)
