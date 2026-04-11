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
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

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
    from omnibase_core.protocols.event_bus.protocol_event_bus_subscriber import (
        ProtocolEventBusSubscriber,
    )
    from omnibase_infra.models.dispatch.model_dispatch_result import (
        ModelDispatchResult,
    )
    from omnibase_infra.protocols.protocol_dispatch_engine import (
        ProtocolDispatchEngine,
    )

logger = logging.getLogger(__name__)

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
                envelope = message  # type: ignore[assignment]
            await dispatch_engine.dispatch(topic, envelope)
        except Exception as exc:  # noqa: BLE001 — boundary: log and discard; unsubscribe unavailable here
            logger.error(
                "Auto-wiring callback error: topic=%s error_type=%s error=%s",
                topic,
                type(exc).__name__,
                exc,
            )

    return callback


def _derive_route_id(contract_name: str, handler_name: str) -> str:
    """Derive a route ID from contract and handler names."""
    return f"route.auto.{contract_name}.{handler_name}"


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
) -> ModelAutoWiringReport:
    """Wire all discovered contracts into the dispatch engine and event bus.

    For each contract in the manifest that has both ``handler_routing`` and
    ``event_bus`` declarations:

    1. Import handler modules from ``handler_routing.handlers[].handler``.
    2. Instantiate handler classes (zero-arg constructor).
    3. Create dispatch callbacks wrapping each handler.
    4. Register dispatchers and routes on the dispatch engine.
    5. Subscribe to Kafka topics via the event bus (if provided).

    Contracts without ``handler_routing`` or ``event_bus`` are skipped.
    Errors on individual contracts are captured — they never abort the scan.

    Args:
        manifest: The auto-wiring manifest from discovery.
        dispatch_engine: The MessageDispatchEngine to register routes on.
        event_bus: Optional event bus for Kafka subscriptions. When None,
            topic subscriptions are skipped (dispatchers + routes still registered).
        environment: Environment name for consumer group derivation.

    Returns:
        A :class:`ModelAutoWiringReport` with per-contract outcomes.
    """
    results: list[ModelContractWiringResult] = []

    for contract in manifest.contracts:
        result = await _wire_single_contract(
            contract=contract,
            dispatch_engine=dispatch_engine,
            event_bus=event_bus,
            environment=environment,
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


async def _wire_single_contract(
    *,
    contract: ModelDiscoveredContract,
    dispatch_engine: ProtocolDispatchEngine,
    event_bus: object | None,
    environment: str,
) -> ModelContractWiringResult:
    """Wire a single discovered contract into the dispatch engine.

    Returns a result capturing success, skip, or failure.
    """
    # Skip contracts without handler routing
    if contract.handler_routing is None:
        return ModelContractWiringResult(
            contract_name=contract.name,
            package_name=contract.package_name,
            outcome=EnumWiringOutcome.SKIPPED,
            reason="No handler_routing declared in contract",
        )

    # Skip contracts without event bus subscriptions
    if contract.event_bus is None or not contract.event_bus.subscribe_topics:
        return ModelContractWiringResult(
            contract_name=contract.name,
            package_name=contract.package_name,
            outcome=EnumWiringOutcome.SKIPPED,
            reason="No event_bus.subscribe_topics declared in contract",
        )

    dispatchers_registered: list[str] = []
    routes_registered: list[str] = []
    topics_subscribed: list[str] = []

    try:
        # Import and wire each handler from handler_routing
        for entry in contract.handler_routing.handlers:
            dispatcher_id, route_ids = _wire_handler_entry(
                contract=contract,
                entry=entry,
                dispatch_engine=dispatch_engine,
                event_bus=event_bus,
            )
            dispatchers_registered.append(dispatcher_id)
            routes_registered.extend(route_ids)

        # Subscribe to Kafka topics via event bus
        if event_bus is not None and contract.event_bus:
            from omnibase_infra.enums import EnumConsumerGroupPurpose
            from omnibase_infra.models import ModelNodeIdentity
            from omnibase_infra.utils import compute_consumer_group_id

            for topic in contract.event_bus.subscribe_topics:
                node_identity = ModelNodeIdentity(
                    env=environment,
                    service=contract.package_name,
                    node_name=contract.name,
                    version="v1",
                )
                consumer_group = compute_consumer_group_id(
                    node_identity, EnumConsumerGroupPurpose.CONSUME
                )
                callback = _make_event_bus_callback(topic, dispatch_engine)
                typed_bus = cast("ProtocolEventBusSubscriber", event_bus)
                unsubscribe = await typed_bus.subscribe(
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

    except Exception as exc:  # noqa: BLE001 — boundary: structured diagnostics for auto-wiring
        logger.error(
            "Failed to auto-wire contract '%s' from package '%s': %s",
            contract.name,
            contract.package_name,
            type(exc).__name__,
        )
        return ModelContractWiringResult(
            contract_name=contract.name,
            package_name=contract.package_name,
            outcome=EnumWiringOutcome.FAILED,
            reason=f"{type(exc).__name__}: {exc}",
        )

    return ModelContractWiringResult(
        contract_name=contract.name,
        package_name=contract.package_name,
        outcome=EnumWiringOutcome.WIRED,
        dispatchers_registered=tuple(dispatchers_registered),
        routes_registered=tuple(routes_registered),
        topics_subscribed=tuple(topics_subscribed),
    )


def _wire_handler_entry(
    *,
    contract: ModelDiscoveredContract,
    entry: ModelHandlerRoutingEntry,
    dispatch_engine: object,
    event_bus: object | None = None,
) -> tuple[str, list[str]]:
    """Import a handler, create callback, register dispatcher + routes.

    Returns:
        Tuple of (dispatcher_id, list of route_ids registered).
    """
    # Deferred imports to avoid circular dependencies
    from omnibase_infra.enums import EnumMessageCategory
    from omnibase_infra.models.dispatch.model_dispatch_route import ModelDispatchRoute
    from omnibase_infra.runtime.service_message_dispatch_engine import (
        MessageDispatchEngine,
    )

    handler_ref = entry.handler
    handler_cls = _import_handler_class(handler_ref.module, handler_ref.name)

    # Inject event_bus if the handler's __init__ declares it as a keyword parameter.
    # Handlers that accept event_bus receive the runtime event bus so they can publish
    # phase-transition events. All other handlers are constructed with zero args.
    handler_instance: ProtocolHandleable
    if (
        event_bus is not None
        and "event_bus" in inspect.signature(handler_cls).parameters
    ):
        handler_instance = handler_cls(event_bus=event_bus)
        logger.debug(
            "Auto-wired event_bus into %s.%s",
            handler_ref.module,
            handler_ref.name,
        )
    else:
        handler_instance = handler_cls()

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

    # Register dispatcher
    engine = dispatch_engine
    if isinstance(engine, MessageDispatchEngine):
        engine.register_dispatcher(
            dispatcher_id=dispatcher_id,
            dispatcher=callback,
            category=category,
            message_types=message_types,
        )

    # Register routes for each subscribe topic
    route_ids: list[str] = []
    if contract.event_bus:
        for topic in contract.event_bus.subscribe_topics:
            route_id = _derive_route_id(contract.name, topic.split(".")[-2])
            topic_pattern = _derive_topic_pattern_from_topic(topic)

            route = ModelDispatchRoute(
                route_id=route_id,
                topic_pattern=topic_pattern,
                message_category=category,
                dispatcher_id=dispatcher_id,
            )

            if isinstance(engine, MessageDispatchEngine):
                engine.register_route(route)

            route_ids.append(route_id)

    return dispatcher_id, route_ids
