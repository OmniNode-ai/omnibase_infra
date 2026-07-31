# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Cross-boundary regression for contract-scoped Kafka dispatch (OMN-15474).

Two contracts may intentionally consume the same topic under distinct consumer
groups (for example, an orchestrator and a projection).  Each Kafka callback
must dispatch only to the handlers registered by the contract that owns that
callback.  Calling the process-global dispatch fan-out from both callbacks
executes every matching handler twice.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from omnibase_infra.models import ModelNodeIdentity
from omnibase_infra.runtime.auto_wiring import (
    ModelAutoWiringManifest,
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
    ModelHandlerRef,
    ModelHandlerRouting,
    ModelHandlerRoutingEntry,
    subscribe_wired_contract_topics,
    wire_from_manifest,
)
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

if TYPE_CHECKING:
    from uuid import UUID

    from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
    from omnibase_infra.models.dispatch.model_dispatch_result import (
        ModelDispatchResult,
    )

pytestmark = pytest.mark.integration

_SHARED_TOPIC = "onex.cmd.omn15474.shared-command.v1"


class HandlerContractA:
    """First owning handler on the shared topic."""

    calls = 0

    async def handle(self, envelope: object) -> None:
        del envelope
        type(self).calls += 1


class HandlerContractB:
    """Second owning handler on the shared topic."""

    calls = 0

    async def handle(self, envelope: object) -> None:
        del envelope
        type(self).calls += 1


class _RecordingApplier:
    def __init__(self) -> None:
        self.results: list[ModelDispatchResult] = []

    async def apply(
        self,
        result: ModelDispatchResult | None,
        correlation_id: UUID | None = None,
    ) -> None:
        del correlation_id
        if result is not None:
            self.results.append(result)


class _RecordingBus:
    """Kafka-boundary double preserving every distinct consumer group."""

    def __init__(self) -> None:
        self.subscriptions: list[
            tuple[str, ModelNodeIdentity, Callable[..., Awaitable[None]]]
        ] = []

    async def subscribe(
        self,
        *,
        topic: str,
        node_identity: ModelNodeIdentity,
        on_message: Callable[..., Awaitable[None]],
    ) -> object:
        self.subscriptions.append((topic, node_identity, on_message))

        async def _unsubscribe() -> None:
            return None

        return _unsubscribe

    async def deliver_to_every_group(
        self, envelope: ModelEventEnvelope[object]
    ) -> None:
        message = type(
            "KafkaMessage", (), {"value": envelope.model_dump_json().encode()}
        )()
        for _topic, _identity, callback in self.subscriptions:
            await callback(message)


class _ProtocolOnlyDispatchEngine:
    """Published SPI shape without the opt-in scoped-dispatch capability."""

    is_frozen = True

    async def dispatch(
        self,
        topic: str,
        envelope: ModelEventEnvelope[object],
    ) -> ModelDispatchResult | None:
        del topic, envelope
        return None

    async def dispatch_with_transaction(
        self,
        *,
        topic: str,
        envelope: ModelEventEnvelope[object],
        tx: object,
    ) -> ModelDispatchResult | None:
        del topic, envelope, tx
        return None


def _contract(name: str, handler_name: str) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name=name,
        node_type="ORCHESTRATOR_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path(f"/tmp/{name}/contract.yaml"),  # noqa: S108
        entry_point_name=name,
        package_name="omn15474-fixture",
        event_bus=ModelEventBusWiring(subscribe_topics=(_SHARED_TOPIC,)),
        handler_routing=ModelHandlerRouting(
            routing_strategy="operation_match",
            handlers=(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(name=handler_name, module=__name__),
                    event_model=None,
                ),
            ),
        ),
    )


@pytest.mark.asyncio
async def test_shared_topic_dispatch_is_scoped_to_each_contract_owner() -> None:
    """One broker record executes each owning handler/applier exactly once."""
    from datetime import UTC, datetime
    from uuid import uuid4

    from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope

    HandlerContractA.calls = 0
    HandlerContractB.calls = 0
    contract_a = _contract("node_contract_a", "HandlerContractA")
    contract_b = _contract("node_contract_b", "HandlerContractB")
    manifest = ModelAutoWiringManifest(contracts=(contract_a, contract_b), errors=())
    engine = MessageDispatchEngine()
    bus = _RecordingBus()
    applier_a = _RecordingApplier()
    applier_b = _RecordingApplier()
    appliers = {
        contract_a.name: applier_a,
        contract_b.name: applier_b,
    }

    report = await wire_from_manifest(
        manifest,
        engine,
        event_bus=bus,
        environment="test",
        subscribe_immediately=False,
        result_appliers_by_contract=appliers,
    )
    engine.freeze()
    subscriptions = await subscribe_wired_contract_topics(
        manifest,
        report,
        engine,
        bus,
        "test",
        appliers,
    )

    assert subscriptions == {
        contract_a.name: (_SHARED_TOPIC,),
        contract_b.name: (_SHARED_TOPIC,),
    }
    assert len(bus.subscriptions) == 2
    assert {
        identity.node_name for _topic, identity, _callback in bus.subscriptions
    } == {contract_a.name, contract_b.name}

    envelope = ModelEventEnvelope[object](
        payload={"prompt": "one broker record"},
        correlation_id=uuid4(),
        envelope_timestamp=datetime.now(UTC),
        event_type="omn15474.shared-command",
        source_tool="contract-scoped-dispatch-test",
    )
    await bus.deliver_to_every_group(envelope)

    assert HandlerContractA.calls == 1
    assert HandlerContractB.calls == 1
    assert len(applier_a.results) == 1
    assert len(applier_b.results) == 1

    result_by_contract = {result.contract_name: result for result in report.results}
    dispatcher_a = result_by_contract[contract_a.name].dispatchers_registered[0]
    dispatcher_b = result_by_contract[contract_b.name].dispatchers_registered[0]
    assert applier_a.results[0].dispatcher_id == dispatcher_a
    assert applier_b.results[0].dispatcher_id == dispatcher_b


@pytest.mark.asyncio
async def test_immediate_subscriptions_preserve_contract_scope() -> None:
    """The pre-freeze immediate attach path captures the same exact scope."""
    from datetime import UTC, datetime
    from uuid import uuid4

    from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope

    HandlerContractA.calls = 0
    HandlerContractB.calls = 0
    contract_a = _contract("node_immediate_a", "HandlerContractA")
    contract_b = _contract("node_immediate_b", "HandlerContractB")
    manifest = ModelAutoWiringManifest(contracts=(contract_a, contract_b), errors=())
    engine = MessageDispatchEngine()
    bus = _RecordingBus()
    applier_a = _RecordingApplier()
    applier_b = _RecordingApplier()

    report = await wire_from_manifest(
        manifest,
        engine,
        event_bus=bus,
        environment="test",
        result_appliers_by_contract={
            contract_a.name: applier_a,
            contract_b.name: applier_b,
        },
    )
    engine.freeze()

    assert len(bus.subscriptions) == 2
    assert all(
        result.topics_subscribed == (_SHARED_TOPIC,) for result in report.results
    )

    envelope = ModelEventEnvelope[object](
        payload={"prompt": "immediate path"},
        correlation_id=uuid4(),
        envelope_timestamp=datetime.now(UTC),
        event_type="omn15474.shared-command",
        source_tool="contract-scoped-dispatch-test",
    )
    await bus.deliver_to_every_group(envelope)

    assert HandlerContractA.calls == 1
    assert HandlerContractB.calls == 1
    assert len(applier_a.results) == 1
    assert len(applier_b.results) == 1


@pytest.mark.asyncio
async def test_wired_subscription_without_dispatcher_scope_fails_closed() -> None:
    """A wired callback can never silently regain process-global fan-out."""
    from omnibase_core.models.errors import ModelOnexError
    from omnibase_infra.runtime.auto_wiring.handler_wiring import (
        _subscribe_contract_topics,
    )

    with pytest.raises(ModelOnexError, match="missing its dispatcher scope"):
        await _subscribe_contract_topics(
            contract=_contract("node_missing_scope", "HandlerContractA"),
            dispatch_engine=MessageDispatchEngine(),
            event_bus=_RecordingBus(),
            environment="test",
        )


@pytest.mark.asyncio
async def test_engine_without_scoped_dispatch_fails_before_subscribe() -> None:
    """Protocol-only engines cannot discover incompatibility after consuming."""
    from omnibase_core.models.errors import ModelOnexError
    from omnibase_infra.runtime.auto_wiring.handler_wiring import (
        _subscribe_contract_topics,
    )

    bus = _RecordingBus()

    with pytest.raises(ModelOnexError, match="scoped dispatch capability"):
        await _subscribe_contract_topics(
            contract=_contract("node_protocol_only", "HandlerContractA"),
            dispatch_engine=_ProtocolOnlyDispatchEngine(),
            event_bus=bus,
            environment="test",
            allowed_dispatcher_ids={"dispatcher.protocol-only"},
        )

    assert bus.subscriptions == []


@pytest.mark.asyncio
async def test_dispatch_engine_rejects_empty_contract_scope() -> None:
    """An explicitly scoped dispatch cannot use an empty allowlist."""
    from datetime import UTC, datetime
    from uuid import uuid4

    from omnibase_core.models.errors import ModelOnexError
    from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope

    engine = MessageDispatchEngine()
    engine.freeze()
    envelope = ModelEventEnvelope[object](
        payload={},
        correlation_id=uuid4(),
        envelope_timestamp=datetime.now(UTC),
        event_type="omn15474.shared-command",
        source_tool="contract-scoped-dispatch-test",
    )

    with pytest.raises(ModelOnexError, match="at least one allowed dispatcher"):
        await engine.dispatch(
            _SHARED_TOPIC,
            envelope,
            allowed_dispatcher_ids=frozenset(),
        )


@pytest.mark.asyncio
async def test_dispatch_engine_rejects_unknown_contract_scope() -> None:
    """A stale or forged dispatcher ID cannot silently route process-wide."""
    from datetime import UTC, datetime
    from uuid import uuid4

    from omnibase_core.models.errors import ModelOnexError
    from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope

    engine = MessageDispatchEngine()
    engine.freeze()
    envelope = ModelEventEnvelope[object](
        payload={},
        correlation_id=uuid4(),
        envelope_timestamp=datetime.now(UTC),
        event_type="omn15474.shared-command",
        source_tool="contract-scoped-dispatch-test",
    )

    with pytest.raises(ModelOnexError, match="not registered on this engine"):
        await engine.dispatch_scoped(
            _SHARED_TOPIC,
            envelope,
            allowed_dispatcher_ids={"dispatcher.unknown"},
        )
