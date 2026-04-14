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
"""

from __future__ import annotations

import importlib
import inspect
import logging
import os
import re
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

from omnibase_core.protocols.event_bus.protocol_event_bus_subscriber import (
    ProtocolEventBusSubscriber,
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

    Produced by _prepare_handler_wiring (pure, no side effects) and consumed by
    _commit_handler_wiring (side effects only). Separating the two phases ensures
    that partial wiring never reaches the engine when a later entry fails (OMN-8735).
    """

    dispatcher_id: str
    dispatcher: DispatcherFunc
    category: EnumMessageCategory
    message_types: set[str] | None
    route_ids: list[str] = field(default_factory=list)
    routes: list[ModelDispatchRoute] = field(default_factory=list)


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

    contract: object  # ModelDiscoveredContract
    prepared_wirings: list[PreparedWiring]
    subscription_topics: list[str]  # topics to subscribe after commit
    environment: str
    skip_result: object = None  # ModelContractWiringResult | None


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
) -> Callable[..., Awaitable[None]]:
    """Create a Kafka on_message callback that deserializes and dispatches to engine.

    Mirrors EventBusSubcontractWiring._create_dispatch_callback but stripped of
    DLQ/idempotency concerns — auto-wired nodes rely on the simplified path.
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
            await dispatch_engine.dispatch(topic, envelope)
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
        container: Optional DI container used to resolve handler constructor deps.
            When provided, handlers with required params are resolved via
            ``container.get_service(handler_cls)`` before falling back to
            event_bus injection or raising (OMN-8735).

    Returns:
        A :class:`ModelAutoWiringReport` with per-contract outcomes.
    """
    # Phase 1: Validate and prepare ALL contracts — no engine/bus side effects yet.
    # Failures are collected; if any exist, we raise before touching anything (OMN-8735).
    prepared_contracts: list[PreparedContractWiring] = []
    failed_results: list[ModelContractWiringResult] = []
    for contract in manifest.contracts:
        try:
            prepared = _prepare_contract_wiring(
                contract=contract,
                dispatch_engine=dispatch_engine,
                event_bus=event_bus,
                environment=environment,
                container=container,
            )
            prepared_contracts.append(prepared)
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
    failures = failed_results
    if failures:
        from omnibase_core.models.errors import ModelOnexError

        failed_reasons = [f"{r.contract_name}: {r.reason}" for r in failures]
        raise ModelOnexError(
            f"Auto-wiring failed for {len(failures)} contract(s): "
            + "; ".join(failed_reasons)
        )

    # Phase 2: All contracts validated — commit registrations and subscriptions.
    results: list[ModelContractWiringResult] = []
    for pcw in prepared_contracts:
        result = await _commit_contract_wiring(pcw, dispatch_engine, event_bus)
        results.append(result)

    duplicates = _detect_duplicate_topics(manifest)

    for dup in duplicates:
        logger.warning(
            "Duplicate topic ownership detected: topic=%s owners=%s level=%s",
            dup.topic,
            dup.owners,
            dup.level,
        )

    report = ModelAutoWiringReport(
        results=tuple(results),
        duplicates=tuple(duplicates),
    )

    logger.info(
        "Auto-wiring complete: wired=%d skipped=%d failed=%d duplicates=%d",
        report.total_wired,
        report.total_skipped,
        report.total_failed,
        len(report.duplicates),
    )

    return report


def _prepare_contract_wiring(
    *,
    contract: ModelDiscoveredContract,
    dispatch_engine: object,
    event_bus: object | None,
    environment: str,
    container: object | None = None,
) -> PreparedContractWiring:
    """Validate and prepare one contract for wiring — NO side effects.

    Skipped contracts are encoded as PreparedContractWiring with skip_result set.
    Wirable contracts have prepared_wirings populated and skip_result=None.

    Raises on handler preparation failure so :func:`wire_from_manifest` can
    collect all failures before touching the engine or event bus (OMN-8735).
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
                event_bus=event_bus,
                container=container,
            )
            prepared_wirings.append(prepared)
        except Exception as exc:
            exc_summary = _sanitize_exc(exc)
            logger.error(
                "Failed to prepare handler '%s' for contract '%s' (package '%s'): %s",
                entry.handler.name,
                contract.name,
                contract.package_name,
                type(exc).__name__,
            )
            from omnibase_core.models.errors import ModelOnexError

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
) -> ModelContractWiringResult:
    """Commit a validated :class:`PreparedContractWiring` to the engine and event bus.

    All side effects (dispatcher/route registration, Kafka subscriptions) live here.
    Must only be called after every contract in the manifest has been successfully
    prepared (OMN-8735).
    """
    if pcw.skip_result is not None:
        return pcw.skip_result  # type: ignore[return-value]

    contract: ModelDiscoveredContract = pcw.contract  # type: ignore[assignment]
    dispatchers_registered: list[str] = []
    routes_registered: list[str] = []
    topics_subscribed: list[str] = []

    for prepared in pcw.prepared_wirings:
        dispatcher_id, route_ids = _commit_handler_wiring(prepared, dispatch_engine)
        dispatchers_registered.append(dispatcher_id)
        routes_registered.extend(route_ids)

    if event_bus is not None and pcw.subscription_topics:
        from omnibase_infra.enums import EnumConsumerGroupPurpose
        from omnibase_infra.models import ModelNodeIdentity
        from omnibase_infra.utils import compute_consumer_group_id

        for topic in pcw.subscription_topics:
            node_identity = ModelNodeIdentity(
                env=pcw.environment,
                service=contract.package_name,
                node_name=contract.name,
                version=str(contract.contract_version),
            )
            consumer_group = compute_consumer_group_id(
                node_identity, EnumConsumerGroupPurpose.CONSUME
            )
            callback = _make_event_bus_callback(topic, dispatch_engine)  # type: ignore[arg-type]
            typed_bus: ProtocolEventBusSubscriber = cast(
                "ProtocolEventBusSubscriber", event_bus
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

    return ModelContractWiringResult(
        contract_name=contract.name,
        package_name=contract.package_name,
        outcome=EnumWiringOutcome.WIRED,
        dispatchers_registered=tuple(dispatchers_registered),
        routes_registered=tuple(routes_registered),
        topics_subscribed=tuple(topics_subscribed),
    )


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
    """
    prepared = _prepare_contract_wiring(
        contract=contract,
        dispatch_engine=dispatch_engine,
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
    event_bus: object | None = None,
    container: object | None = None,
) -> PreparedWiring:
    """Validate and prepare one handler entry for wiring — NO side effects.

    Instantiates the handler class, creates its dispatch callback, and
    pre-computes all route data.  Nothing is registered with the engine here;
    that is deferred to :func:`_commit_handler_wiring` so that the engine is
    only mutated after every handler in the contract has been validated
    successfully (OMN-8735).

    Returns:
        A :class:`PreparedWiring` ready for :func:`_commit_handler_wiring`.
    """
    # Deferred imports to avoid circular dependencies
    from omnibase_infra.enums import EnumMessageCategory
    from omnibase_infra.models.dispatch.model_dispatch_route import ModelDispatchRoute

    handler_ref = entry.handler
    handler_cls = _import_handler_class(handler_ref.module, handler_ref.name)

    # Resolve handler via DI container if available and the class has constructor deps,
    # otherwise fall back to event_bus injection or zero-arg construction (OMN-8735).
    handler_instance: ProtocolHandleable
    sig = inspect.signature(handler_cls)
    params = sig.parameters
    # Only concrete named params (POSITIONAL_OR_KEYWORD / KEYWORD_ONLY) without defaults
    # are considered DI deps. VAR_POSITIONAL (*args) and VAR_KEYWORD (**kwargs) excluded.
    _concrete_kinds = (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )
    non_self_params = {
        k: v
        for k, v in params.items()
        if k != "self"
        and v.kind in _concrete_kinds
        and v.default is inspect.Parameter.empty
    }

    # Resolve the effective container: explicit param takes precedence over
    # dispatch_engine._container (legacy path kept for backwards compat).
    _effective_container = container or (
        getattr(dispatch_engine, "_container", None)
        if dispatch_engine is not None
        else None
    )

    if _effective_container is not None:
        try:
            handler_instance = _effective_container.get_service(handler_cls)  # type: ignore[union-attr]
            logger.debug(
                "Resolved %s.%s via DI container",
                handler_ref.module,
                handler_ref.name,
            )
        except Exception:  # noqa: BLE001 — DI resolution failed; try other paths
            if event_bus is not None and set(non_self_params) == {"event_bus"}:
                handler_instance = handler_cls(event_bus=event_bus)
                logger.debug(
                    "Auto-wired event_bus into %s.%s (container resolution failed)",
                    handler_ref.module,
                    handler_ref.name,
                )
            elif non_self_params:
                # Container resolution failed and deps are unsatisfiable — raise loudly
                # so startup fails with a clear message rather than a cryptic TypeError
                # from handler_cls() with missing args (OMN-8735).
                dep_names = list(non_self_params)
                raise TypeError(
                    f"Handler {handler_ref.module}.{handler_ref.name} requires "
                    f"constructor parameters {dep_names!r} but DI container "
                    f"resolution also failed."
                )
            else:
                handler_instance = handler_cls()
    elif event_bus is not None and set(non_self_params) == {"event_bus"}:
        handler_instance = handler_cls(event_bus=event_bus)
        logger.debug(
            "Auto-wired event_bus into %s.%s",
            handler_ref.module,
            handler_ref.name,
        )
    elif non_self_params:
        # Handler has constructor deps but no container and no event_bus — unsatisfiable.
        # Raise immediately so startup fails loudly (OMN-8735).
        dep_names = list(non_self_params)
        raise TypeError(
            f"Handler {handler_ref.module}.{handler_ref.name} requires constructor "
            f"parameters {dep_names!r} but no DI container is available to satisfy them."
        )
    else:
        handler_instance = handler_cls()

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

    # Determine message category from subscribe topics
    category_str = "EVENT"
    if contract.event_bus and contract.event_bus.subscribe_topics:
        category_str = _derive_message_category(contract.event_bus.subscribe_topics[0])

    category = EnumMessageCategory(category_str)

    # Determine message types from entry
    message_types: set[str] | None = None
    if entry.event_model is not None:
        message_types = {entry.event_model.name}

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

    Returns:
        Tuple of (dispatcher_id, list of route_ids registered).
    """
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
    prepared = _prepare_handler_wiring(
        contract=contract,
        entry=entry,
        dispatch_engine=dispatch_engine,
        event_bus=event_bus,
        container=container,
    )
    return _commit_handler_wiring(prepared, dispatch_engine)
