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

import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from omnibase_infra.event_bus.enum_contract_attach_status import (
    EnumContractAttachStatus,
)
from omnibase_infra.event_bus.enum_topic_readiness_status import (
    EnumTopicReadinessStatus,
)
from omnibase_infra.event_bus.model_contract_attach_result import (
    ModelContractAttachResult,
)
from omnibase_infra.event_bus.model_topic_set_readiness import (
    ModelTopicSetReadiness,
)
from omnibase_infra.models import ModelNodeIdentity
from omnibase_infra.runtime.auto_wiring import (
    ModelAutoWiringManifest,
    ModelAutoWiringReport,
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
    from omnibase_infra.topics.model_topic_spec import ModelTopicSpec

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


class _RecordingReadyProvisioner:
    """Provisioner double that exposes any pre-validation side effect."""

    def __init__(self) -> None:
        self.ensure_calls: list[str] = []
        self.confirm_calls: list[tuple[str, ...]] = []

    async def ensure_topic_exists(
        self,
        topic_name: str,
        spec: ModelTopicSpec | None = None,
        correlation_id: UUID | None = None,
    ) -> bool:
        del spec, correlation_id
        self.ensure_calls.append(topic_name)
        return True

    async def confirm_topics_ready(
        self,
        topics: Sequence[str],
        *,
        expected_specs: Mapping[str, ModelTopicSpec] | None = None,
        config: object | None = None,
        correlation_id: UUID | None = None,
    ) -> ModelTopicSetReadiness:
        del expected_specs, config, correlation_id
        normalized_topics = tuple(topics)
        self.confirm_calls.append(normalized_topics)
        return ModelTopicSetReadiness(
            topics=normalized_topics,
            status=EnumTopicReadinessStatus.READY,
            ready_topics=normalized_topics,
            attempts=1,
        )


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
    return _contract_with_handlers(name, (handler_name,))


def _contract_with_handlers(
    name: str,
    handler_names: tuple[str, ...],
    *,
    topics: tuple[str, ...] = (_SHARED_TOPIC,),
) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name=name,
        node_type="ORCHESTRATOR_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path(f"/tmp/{name}/contract.yaml"),  # noqa: S108
        entry_point_name=name,
        package_name="omn15474-fixture",
        event_bus=ModelEventBusWiring(subscribe_topics=topics),
        handler_routing=ModelHandlerRouting(
            routing_strategy="operation_match",
            handlers=tuple(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(name=handler_name, module=__name__),
                    event_model=None,
                )
                for handler_name in handler_names
            ),
        ),
    )


def _revalidate_report_with_scopes(
    report: ModelAutoWiringReport,
    scopes_by_contract: Mapping[str, tuple[str, ...]],
) -> ModelAutoWiringReport:
    """Forge only through the public serialization/validation boundary."""
    payload = json.loads(report.model_dump_json())
    for result in payload["results"]:
        contract_name = result["contract_name"]
        if contract_name in scopes_by_contract:
            result["dispatchers_registered"] = list(scopes_by_contract[contract_name])
    return ModelAutoWiringReport.model_validate_json(json.dumps(payload))


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


@pytest.mark.parametrize("subscribe_immediately", [True, False])
@pytest.mark.asyncio
async def test_cross_contract_derived_id_collision_fails_before_manifest_commit(
    monkeypatch: pytest.MonkeyPatch,
    *,
    subscribe_immediately: bool,
) -> None:
    """A batch-wide derived-ID collision cannot expose partial live state."""
    from omnibase_core.models.errors import ModelOnexError
    from omnibase_infra.runtime.auto_wiring import handler_wiring
    from omnibase_infra.runtime.auto_wiring.handler_wiring import (
        _derive_dispatcher_id,
        _derive_handler_entry_key,
        _derive_route_id,
    )

    contract_a = _contract("node.alpha", "HandlerContractA")
    contract_b = _contract("node", "alpha.HandlerContractA")
    manifest = ModelAutoWiringManifest(contracts=(contract_a, contract_b), errors=())
    handler_key_a = _derive_handler_entry_key(
        contract_a.handler_routing.handlers[0]  # type: ignore[union-attr]
    )
    handler_key_b = _derive_handler_entry_key(
        contract_b.handler_routing.handlers[0]  # type: ignore[union-attr]
    )
    assert _derive_dispatcher_id(contract_a.name, handler_key_a) == (
        _derive_dispatcher_id(contract_b.name, handler_key_b)
    )
    assert _derive_route_id(contract_a.name, handler_key_a, _SHARED_TOPIC) == (
        _derive_route_id(contract_b.name, handler_key_b, _SHARED_TOPIC)
    )

    monkeypatch.setattr(
        handler_wiring,
        "_import_handler_class",
        lambda _module, _name: HandlerContractA,
    )
    engine = MessageDispatchEngine()
    bus = _RecordingBus()

    with pytest.raises(ModelOnexError, match="duplicate prepared dispatcher IDs"):
        await wire_from_manifest(
            manifest,
            engine,
            event_bus=bus,
            environment="test",
            subscribe_immediately=subscribe_immediately,
        )

    assert engine.dispatcher_count == 0
    assert engine.route_count == 0
    assert bus.subscriptions == []


@pytest.mark.parametrize("subscribe_immediately", [True, False])
@pytest.mark.asyncio
async def test_normalized_route_collision_fails_before_manifest_commit(
    *,
    subscribe_immediately: bool,
) -> None:
    """Topic normalization collisions are rejected before engine mutation."""
    from omnibase_core.models.errors import ModelOnexError
    from omnibase_infra.runtime.auto_wiring.handler_wiring import (
        _derive_handler_entry_key,
        _derive_route_id,
    )

    topics = (
        "onex.cmd.foo-bar.name.v1",
        "onex.cmd.foo.bar-name.v1",
    )
    contract = _contract_with_handlers(
        "node_normalized_route_collision",
        ("HandlerContractA",),
        topics=topics,
    )
    handler_key = _derive_handler_entry_key(
        contract.handler_routing.handlers[0]  # type: ignore[union-attr]
    )
    assert _derive_route_id(contract.name, handler_key, topics[0]) == _derive_route_id(
        contract.name,
        handler_key,
        topics[1],
    )
    engine = MessageDispatchEngine()
    bus = _RecordingBus()

    with pytest.raises(ModelOnexError, match="duplicate prepared route IDs"):
        await wire_from_manifest(
            ModelAutoWiringManifest(contracts=(contract,), errors=()),
            engine,
            event_bus=bus,
            environment="test",
            subscribe_immediately=subscribe_immediately,
        )

    assert engine.dispatcher_count == 0
    assert engine.route_count == 0
    assert bus.subscriptions == []


@pytest.mark.asyncio
async def test_duplicate_manifest_names_fail_at_wire_entry() -> None:
    """Immediate wiring rejects typed name collisions before preparation."""
    from omnibase_core.models.errors import ModelOnexError

    HandlerContractA.calls = 0
    HandlerContractB.calls = 0
    contract_a = _contract("node_wire_duplicate_a", "HandlerContractA")
    contract_b = _contract("node_wire_duplicate_b", "HandlerContractB")
    original_manifest = ModelAutoWiringManifest(
        contracts=(contract_a, contract_b),
        errors=(),
    )
    payload = json.loads(original_manifest.model_dump_json())
    payload["contracts"][1]["name"] = contract_a.name
    duplicate_manifest = ModelAutoWiringManifest.model_validate_json(
        json.dumps(payload)
    )
    engine = MessageDispatchEngine()
    bus = _RecordingBus()
    applier = _RecordingApplier()
    caught: ModelOnexError | None = None

    try:
        await wire_from_manifest(
            duplicate_manifest,
            engine,
            event_bus=bus,
            environment="test",
            result_appliers_by_contract={contract_a.name: applier},
        )
    except ModelOnexError as exc:
        caught = exc

    assert caught is not None, (
        "wire_from_manifest accepted a serialized/Pydantic HandlerA/HandlerB "
        "manifest name collision "
        f"(dispatchers={engine.dispatcher_count}, subscriptions={len(bus.subscriptions)})"
    )
    assert "duplicate manifest contract names" in str(caught)
    assert engine.dispatcher_count == 0
    assert bus.subscriptions == []
    assert HandlerContractA.calls == 0
    assert HandlerContractB.calls == 0
    assert applier.results == []


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
async def test_revalidated_cross_contract_scope_fails_before_provisioning() -> None:
    """A typed report cannot assign one dispatcher to two contract owners."""
    from omnibase_core.models.errors import ModelOnexError

    contract_a = _contract("node_forged_owner_a", "HandlerContractA")
    contract_b = _contract("node_forged_owner_b", "HandlerContractB")
    manifest = ModelAutoWiringManifest(contracts=(contract_a, contract_b), errors=())
    engine = MessageDispatchEngine()
    bus = _RecordingBus()
    report = await wire_from_manifest(
        manifest,
        engine,
        event_bus=bus,
        environment="test",
        subscribe_immediately=False,
    )
    engine.freeze()

    by_name = {result.contract_name: result for result in report.results}
    dispatcher_a = by_name[contract_a.name].dispatchers_registered[0]
    forged_report = _revalidate_report_with_scopes(
        report,
        {
            contract_a.name: (dispatcher_a,),
            contract_b.name: (dispatcher_a,),
        },
    )
    provisioner = _RecordingReadyProvisioner()

    with pytest.raises(ModelOnexError, match="multiple contracts"):
        await subscribe_wired_contract_topics(
            manifest,
            forged_report,
            engine,
            bus,
            "test",
            provisioner=provisioner,
        )

    assert provisioner.ensure_calls == []
    assert provisioner.confirm_calls == []
    assert bus.subscriptions == []


@pytest.mark.parametrize("scope_variant", ["proper_subset", "empty"])
@pytest.mark.asyncio
async def test_revalidated_initial_scope_must_equal_live_owner_set(
    scope_variant: str,
) -> None:
    """A typed initial report cannot suppress an owner's registered handlers."""
    from omnibase_core.models.errors import ModelOnexError

    contract = _contract_with_handlers(
        "node_initial_complete_scope",
        ("HandlerContractA", "HandlerContractB"),
    )
    manifest = ModelAutoWiringManifest(contracts=(contract,), errors=())
    engine = MessageDispatchEngine()
    report = await wire_from_manifest(
        manifest,
        engine,
        event_bus=_RecordingBus(),
        environment="test",
        subscribe_immediately=False,
    )
    engine.freeze()
    owned_scope = report.results[0].dispatchers_registered
    assert len(owned_scope) == 2
    forged_scope = owned_scope[:1] if scope_variant == "proper_subset" else ()
    forged_report = _revalidate_report_with_scopes(
        report,
        {contract.name: forged_scope},
    )
    bus = _RecordingBus()
    provisioner = _RecordingReadyProvisioner()
    attach_results: list[ModelContractAttachResult] = []

    with pytest.raises(ModelOnexError, match="complete live owner set"):
        await subscribe_wired_contract_topics(
            manifest,
            forged_report,
            engine,
            bus,
            "test",
            provisioner=provisioner,
            attach_results_out=attach_results,
        )

    assert provisioner.ensure_calls == []
    assert provisioner.confirm_calls == []
    assert bus.subscriptions == []
    assert attach_results == []


@pytest.mark.asyncio
async def test_revalidated_duplicate_report_names_fail_before_side_effects() -> None:
    """Typed duplicate result identities cannot schedule a contract twice."""
    from omnibase_core.models.errors import ModelOnexError

    contract = _contract("node_duplicate_report", "HandlerContractA")
    manifest = ModelAutoWiringManifest(contracts=(contract,), errors=())
    engine = MessageDispatchEngine()
    report = await wire_from_manifest(
        manifest,
        engine,
        event_bus=_RecordingBus(),
        environment="test",
        subscribe_immediately=False,
    )
    engine.freeze()
    payload = json.loads(report.model_dump_json())
    payload["results"].append(dict(payload["results"][0]))
    duplicate_report = ModelAutoWiringReport.model_validate_json(json.dumps(payload))
    bus = _RecordingBus()
    provisioner = _RecordingReadyProvisioner()
    attach_results: list[ModelContractAttachResult] = []

    with pytest.raises(ModelOnexError, match="duplicate report contract names"):
        await subscribe_wired_contract_topics(
            manifest,
            duplicate_report,
            engine,
            bus,
            "test",
            provisioner=provisioner,
            attach_results_out=attach_results,
        )

    assert provisioner.ensure_calls == []
    assert provisioner.confirm_calls == []
    assert bus.subscriptions == []
    assert attach_results == []


@pytest.mark.asyncio
async def test_revalidated_duplicate_manifest_names_fail_before_side_effects() -> None:
    """Two different handlers cannot collapse under one manifest identity."""
    from omnibase_core.models.errors import ModelOnexError

    contract_a = _contract("node_manifest_a", "HandlerContractA")
    contract_b = _contract("node_manifest_b", "HandlerContractB")
    original_manifest = ModelAutoWiringManifest(
        contracts=(contract_a, contract_b),
        errors=(),
    )
    engine = MessageDispatchEngine()
    report = await wire_from_manifest(
        original_manifest,
        engine,
        event_bus=_RecordingBus(),
        environment="test",
        subscribe_immediately=False,
    )
    engine.freeze()
    payload = json.loads(original_manifest.model_dump_json())
    payload["contracts"][1]["name"] = contract_a.name
    duplicate_manifest = ModelAutoWiringManifest.model_validate_json(
        json.dumps(payload)
    )
    bus = _RecordingBus()
    provisioner = _RecordingReadyProvisioner()
    attach_results: list[ModelContractAttachResult] = []

    with pytest.raises(ModelOnexError, match="duplicate manifest contract names"):
        await subscribe_wired_contract_topics(
            duplicate_manifest,
            report,
            engine,
            bus,
            "test",
            provisioner=provisioner,
            attach_results_out=attach_results,
        )

    assert provisioner.ensure_calls == []
    assert provisioner.confirm_calls == []
    assert bus.subscriptions == []
    assert attach_results == []


@pytest.mark.asyncio
async def test_initial_contract_names_must_be_canonical_and_bijective() -> None:
    """Whitespace aliases and missing report identities both fail synchronously."""
    from omnibase_core.models.errors import ModelOnexError

    contract_a = _contract("node_identity_a", "HandlerContractA")
    contract_b = _contract("node_identity_b", "HandlerContractB")
    manifest = ModelAutoWiringManifest(contracts=(contract_a, contract_b), errors=())
    engine = MessageDispatchEngine()
    report = await wire_from_manifest(
        manifest,
        engine,
        event_bus=_RecordingBus(),
        environment="test",
        subscribe_immediately=False,
    )
    engine.freeze()

    noncanonical_payload = json.loads(report.model_dump_json())
    noncanonical_payload["results"][0]["contract_name"] = f" {contract_a.name} "
    noncanonical_report = ModelAutoWiringReport.model_validate_json(
        json.dumps(noncanonical_payload)
    )
    missing_payload = json.loads(report.model_dump_json())
    missing_payload["results"] = missing_payload["results"][:1]
    missing_report = ModelAutoWiringReport.model_validate_json(
        json.dumps(missing_payload)
    )
    noncanonical_manifest_payload = json.loads(manifest.model_dump_json())
    noncanonical_manifest_payload["contracts"][0]["name"] = f" {contract_a.name} "
    noncanonical_manifest = ModelAutoWiringManifest.model_validate_json(
        json.dumps(noncanonical_manifest_payload)
    )

    for forged_manifest, forged_report, error_match in (
        (
            noncanonical_manifest,
            report,
            "noncanonical manifest contract names",
        ),
        (manifest, noncanonical_report, "noncanonical report contract names"),
        (manifest, missing_report, "report.*manifest contract-name mismatch"),
    ):
        bus = _RecordingBus()
        provisioner = _RecordingReadyProvisioner()
        attach_results: list[ModelContractAttachResult] = []
        with pytest.raises(ModelOnexError, match=error_match):
            await subscribe_wired_contract_topics(
                forged_manifest,
                forged_report,
                engine,
                bus,
                "test",
                provisioner=provisioner,
                attach_results_out=attach_results,
            )
        assert provisioner.ensure_calls == []
        assert provisioner.confirm_calls == []
        assert bus.subscriptions == []
        assert attach_results == []


@pytest.mark.asyncio
async def test_revalidated_wrong_registered_owner_fails_before_provisioning() -> None:
    """A unique registered ID still cannot be reassigned to another contract."""
    from omnibase_core.models.errors import ModelOnexError

    contract_a = _contract("node_wrong_owner_a", "HandlerContractA")
    contract_b = _contract("node_wrong_owner_b", "HandlerContractB")
    manifest = ModelAutoWiringManifest(contracts=(contract_a, contract_b), errors=())
    engine = MessageDispatchEngine()
    bus = _RecordingBus()
    report = await wire_from_manifest(
        manifest,
        engine,
        event_bus=bus,
        environment="test",
        subscribe_immediately=False,
    )
    engine.freeze()
    by_name = {result.contract_name: result for result in report.results}
    dispatcher_b = by_name[contract_b.name].dispatchers_registered[0]
    forged_report = _revalidate_report_with_scopes(
        report,
        {
            contract_a.name: (dispatcher_b,),
            contract_b.name: (),
        },
    )
    provisioner = _RecordingReadyProvisioner()

    with pytest.raises(ModelOnexError, match="not owned by contract"):
        await subscribe_wired_contract_topics(
            manifest,
            forged_report,
            engine,
            bus,
            "test",
            provisioner=provisioner,
        )

    assert provisioner.ensure_calls == []
    assert provisioner.confirm_calls == []
    assert bus.subscriptions == []


@pytest.mark.asyncio
async def test_revalidated_unknown_report_scope_fails_before_provisioning() -> None:
    """Every dispatcher cited by a typed wiring report must exist now."""
    from omnibase_core.models.errors import ModelOnexError

    contract = _contract("node_forged_unknown", "HandlerContractA")
    manifest = ModelAutoWiringManifest(contracts=(contract,), errors=())
    engine = MessageDispatchEngine()
    bus = _RecordingBus()
    report = await wire_from_manifest(
        manifest,
        engine,
        event_bus=bus,
        environment="test",
        subscribe_immediately=False,
    )
    engine.freeze()
    forged_report = _revalidate_report_with_scopes(
        report,
        {contract.name: ("dispatcher.forged-unknown",)},
    )
    provisioner = _RecordingReadyProvisioner()

    with pytest.raises(ModelOnexError, match="not registered on this engine"):
        await subscribe_wired_contract_topics(
            manifest,
            forged_report,
            engine,
            bus,
            "test",
            provisioner=provisioner,
        )

    assert provisioner.ensure_calls == []
    assert provisioner.confirm_calls == []
    assert bus.subscriptions == []


@pytest.mark.asyncio
async def test_direct_unknown_scope_fails_before_subscribe() -> None:
    """The lowest direct subscription boundary also checks membership."""
    from omnibase_core.models.errors import ModelOnexError
    from omnibase_infra.runtime.auto_wiring.handler_wiring import (
        _subscribe_contract_topics,
    )

    engine = MessageDispatchEngine()
    engine.freeze()
    bus = _RecordingBus()

    with pytest.raises(ModelOnexError, match="not registered on this engine"):
        await _subscribe_contract_topics(
            contract=_contract("node_direct_unknown", "HandlerContractA"),
            dispatch_engine=engine,
            event_bus=bus,
            environment="test",
            allowed_dispatcher_ids={"dispatcher.direct-unknown"},
        )

    assert bus.subscriptions == []


@pytest.mark.parametrize("scope_variant", ["proper_subset", "empty"])
@pytest.mark.asyncio
async def test_direct_scope_must_equal_live_owner_set(scope_variant: str) -> None:
    """The direct subscription seam requires the complete immutable owner set."""
    from omnibase_core.models.errors import ModelOnexError
    from omnibase_infra.runtime.auto_wiring.handler_wiring import (
        _subscribe_contract_topics,
    )

    contract = _contract_with_handlers(
        "node_direct_complete_scope",
        ("HandlerContractA", "HandlerContractB"),
    )
    manifest = ModelAutoWiringManifest(contracts=(contract,), errors=())
    engine = MessageDispatchEngine()
    report = await wire_from_manifest(
        manifest,
        engine,
        event_bus=_RecordingBus(),
        environment="test",
        subscribe_immediately=False,
    )
    engine.freeze()
    owned_scope = report.results[0].dispatchers_registered
    forged_scope = owned_scope[:1] if scope_variant == "proper_subset" else ()
    bus = _RecordingBus()

    with pytest.raises(
        ModelOnexError,
        match=r"complete live owner set|empty or invalid dispatcher scope",
    ):
        await _subscribe_contract_topics(
            contract=contract,
            dispatch_engine=engine,
            event_bus=bus,
            environment="test",
            allowed_dispatcher_ids=forged_scope,
        )

    assert bus.subscriptions == []


@pytest.mark.asyncio
async def test_revalidated_not_ready_unknown_scope_fails_before_provisioning() -> None:
    """A persisted NOT_READY scope is revalidated against the live engine."""
    from omnibase_core.models.errors import ModelOnexError
    from omnibase_infra.runtime.auto_wiring.handler_wiring import (
        reattach_not_ready_contracts,
    )

    contract = _contract("node_not_ready_unknown", "HandlerContractA")
    manifest = ModelAutoWiringManifest(contracts=(contract,), errors=())
    engine = MessageDispatchEngine()
    bus = _RecordingBus()
    await wire_from_manifest(
        manifest,
        engine,
        event_bus=bus,
        environment="test",
        subscribe_immediately=False,
    )
    engine.freeze()
    forged_result = ModelContractAttachResult.model_validate_json(
        json.dumps(
            {
                "contract_name": contract.name,
                "status": EnumContractAttachStatus.NOT_READY.value,
                "dispatcher_ids": ["dispatcher.not-ready-unknown"],
            }
        )
    )
    provisioner = _RecordingReadyProvisioner()

    with pytest.raises(ModelOnexError, match="not registered on this engine"):
        await reattach_not_ready_contracts(
            manifest,
            (forged_result,),
            engine,
            bus,
            "test",
            provisioner=provisioner,
        )

    assert provisioner.ensure_calls == []
    assert provisioner.confirm_calls == []
    assert bus.subscriptions == []


@pytest.mark.parametrize("scope_variant", ["proper_subset", "empty"])
@pytest.mark.asyncio
async def test_revalidated_not_ready_scope_must_equal_live_owner_set(
    scope_variant: str,
) -> None:
    """A persisted retry scope cannot omit any dispatcher owned by its contract."""
    from omnibase_core.models.errors import ModelOnexError
    from omnibase_infra.runtime.auto_wiring.handler_wiring import (
        reattach_not_ready_contracts,
    )

    contract = _contract_with_handlers(
        "node_not_ready_complete_scope",
        ("HandlerContractA", "HandlerContractB"),
    )
    manifest = ModelAutoWiringManifest(contracts=(contract,), errors=())
    engine = MessageDispatchEngine()
    report = await wire_from_manifest(
        manifest,
        engine,
        event_bus=_RecordingBus(),
        environment="test",
        subscribe_immediately=False,
    )
    engine.freeze()
    owned_scope = report.results[0].dispatchers_registered
    forged_scope = owned_scope[:1] if scope_variant == "proper_subset" else ()
    forged_result = ModelContractAttachResult.model_validate_json(
        json.dumps(
            {
                "contract_name": contract.name,
                "status": EnumContractAttachStatus.NOT_READY.value,
                "dispatcher_ids": list(forged_scope),
            }
        )
    )
    bus = _RecordingBus()
    provisioner = _RecordingReadyProvisioner()

    with pytest.raises(
        ModelOnexError,
        match=r"complete live owner set|empty or invalid dispatcher scope",
    ):
        await reattach_not_ready_contracts(
            manifest,
            (forged_result,),
            engine,
            bus,
            "test",
            provisioner=provisioner,
        )

    assert provisioner.ensure_calls == []
    assert provisioner.confirm_calls == []
    assert bus.subscriptions == []


@pytest.mark.asyncio
async def test_not_ready_cross_contract_scope_fails_before_provisioning() -> None:
    """Reattach rejects one dispatcher persisted under two contracts."""
    from omnibase_core.models.errors import ModelOnexError
    from omnibase_infra.runtime.auto_wiring.handler_wiring import (
        reattach_not_ready_contracts,
    )

    contract_a = _contract("node_not_ready_owner_a", "HandlerContractA")
    contract_b = _contract("node_not_ready_owner_b", "HandlerContractB")
    manifest = ModelAutoWiringManifest(contracts=(contract_a, contract_b), errors=())
    engine = MessageDispatchEngine()
    bus = _RecordingBus()
    report = await wire_from_manifest(
        manifest,
        engine,
        event_bus=bus,
        environment="test",
        subscribe_immediately=False,
    )
    engine.freeze()
    by_name = {result.contract_name: result for result in report.results}
    shared_dispatcher = by_name[contract_a.name].dispatchers_registered[0]
    results = tuple(
        ModelContractAttachResult.model_validate_json(
            json.dumps(
                {
                    "contract_name": contract_name,
                    "status": EnumContractAttachStatus.NOT_READY.value,
                    "dispatcher_ids": [shared_dispatcher],
                }
            )
        )
        for contract_name in (contract_a.name, contract_b.name)
    )
    provisioner = _RecordingReadyProvisioner()

    with pytest.raises(ModelOnexError, match="multiple contracts"):
        await reattach_not_ready_contracts(
            manifest,
            results,
            engine,
            bus,
            "test",
            provisioner=provisioner,
        )

    assert provisioner.ensure_calls == []
    assert provisioner.confirm_calls == []
    assert bus.subscriptions == []


@pytest.mark.asyncio
async def test_duplicate_typed_not_ready_names_fail_before_side_effects() -> None:
    """Persisted duplicate identities cannot schedule two reattach attempts."""
    from omnibase_core.models.errors import ModelOnexError
    from omnibase_infra.runtime.auto_wiring.handler_wiring import (
        reattach_not_ready_contracts,
    )

    contract = _contract("node_duplicate_not_ready", "HandlerContractA")
    manifest = ModelAutoWiringManifest(contracts=(contract,), errors=())
    engine = MessageDispatchEngine()
    report = await wire_from_manifest(
        manifest,
        engine,
        event_bus=_RecordingBus(),
        environment="test",
        subscribe_immediately=False,
    )
    engine.freeze()
    dispatcher_id = report.results[0].dispatchers_registered[0]
    serialized_result = json.dumps(
        {
            "contract_name": contract.name,
            "status": EnumContractAttachStatus.NOT_READY.value,
            "dispatcher_ids": [dispatcher_id],
        }
    )
    duplicate_results = tuple(
        ModelContractAttachResult.model_validate_json(serialized_result)
        for _ in range(2)
    )
    bus = _RecordingBus()
    provisioner = _RecordingReadyProvisioner()

    with pytest.raises(ModelOnexError, match="duplicate NOT_READY contract names"):
        await reattach_not_ready_contracts(
            manifest,
            duplicate_results,
            engine,
            bus,
            "test",
            provisioner=provisioner,
        )

    assert provisioner.ensure_calls == []
    assert provisioner.confirm_calls == []
    assert bus.subscriptions == []


@pytest.mark.asyncio
async def test_duplicate_initial_not_ready_fails_before_loop_side_effects() -> None:
    """The public reconciliation loop validates before dict collapse or sleep."""
    from omnibase_core.models.errors import ModelOnexError
    from omnibase_infra.runtime.auto_wiring.handler_wiring import (
        run_not_ready_reconciliation_loop,
    )

    contract = _contract("node_duplicate_initial_not_ready", "HandlerContractA")
    manifest = ModelAutoWiringManifest(contracts=(contract,), errors=())
    engine = MessageDispatchEngine()
    report = await wire_from_manifest(
        manifest,
        engine,
        event_bus=_RecordingBus(),
        environment="test",
        subscribe_immediately=False,
    )
    engine.freeze()
    baseline_dispatcher_count = engine.dispatcher_count
    dispatcher_id = report.results[0].dispatchers_registered[0]
    serialized_result = json.dumps(
        {
            "contract_name": contract.name,
            "status": EnumContractAttachStatus.NOT_READY.value,
            "dispatcher_ids": [dispatcher_id],
        }
    )
    duplicate_initial_not_ready = tuple(
        ModelContractAttachResult.model_validate_json(serialized_result)
        for _ in range(2)
    )
    bus = _RecordingBus()
    provisioner = _RecordingReadyProvisioner()
    applier = _RecordingApplier()
    sleep_calls: list[float] = []
    attempt_results: list[tuple[ModelContractAttachResult, ...]] = []
    returned_results: tuple[ModelContractAttachResult, ...] = ()
    caught: ModelOnexError | None = None

    async def _record_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    def _record_attempt(
        _subscribed: dict[str, tuple[str, ...]],
        results: tuple[ModelContractAttachResult, ...],
    ) -> None:
        attempt_results.append(results)

    try:
        returned_results = await run_not_ready_reconciliation_loop(
            manifest,
            duplicate_initial_not_ready,
            engine,
            bus,
            "test",
            {contract.name: applier},
            provisioner=provisioner,
            max_attempts=1,
            on_attempt=_record_attempt,
            sleep=_record_sleep,
        )
    except ModelOnexError as exc:
        caught = exc

    assert caught is not None, (
        "reconciliation accepted duplicate typed initial NOT_READY identities "
        f"(sleep={sleep_calls}, ensure={provisioner.ensure_calls}, "
        f"confirm={provisioner.confirm_calls}, subscriptions={len(bus.subscriptions)}, "
        f"results={returned_results})"
    )
    assert "duplicate NOT_READY contract names" in str(caught)
    assert engine.dispatcher_count == baseline_dispatcher_count
    assert sleep_calls == []
    assert provisioner.ensure_calls == []
    assert provisioner.confirm_calls == []
    assert bus.subscriptions == []
    assert returned_results == ()
    assert attempt_results == []
    assert applier.results == []


@pytest.mark.asyncio
async def test_forged_initial_not_ready_scope_fails_before_loop_sleep() -> None:
    """The public retry wrapper validates live scope before its first sleep."""
    from omnibase_core.models.errors import ModelOnexError
    from omnibase_infra.runtime.auto_wiring.handler_wiring import (
        run_not_ready_reconciliation_loop,
    )

    contract = _contract("node_forged_initial_scope", "HandlerContractA")
    manifest = ModelAutoWiringManifest(contracts=(contract,), errors=())
    engine = MessageDispatchEngine()
    await wire_from_manifest(
        manifest,
        engine,
        event_bus=_RecordingBus(),
        environment="test",
        subscribe_immediately=False,
    )
    engine.freeze()
    baseline_dispatcher_count = engine.dispatcher_count
    forged_result = ModelContractAttachResult.model_validate_json(
        json.dumps(
            {
                "contract_name": contract.name,
                "status": EnumContractAttachStatus.NOT_READY.value,
                "dispatcher_ids": ["dispatcher.forged-before-sleep"],
            }
        )
    )
    bus = _RecordingBus()
    provisioner = _RecordingReadyProvisioner()
    applier = _RecordingApplier()
    sleep_calls: list[float] = []
    attempt_results: list[tuple[ModelContractAttachResult, ...]] = []

    async def _record_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    def _record_attempt(
        _subscribed: dict[str, tuple[str, ...]],
        results: tuple[ModelContractAttachResult, ...],
    ) -> None:
        attempt_results.append(results)

    with pytest.raises(ModelOnexError, match="not registered on this engine"):
        await run_not_ready_reconciliation_loop(
            manifest,
            (forged_result,),
            engine,
            bus,
            "test",
            {contract.name: applier},
            provisioner=provisioner,
            max_attempts=1,
            on_attempt=_record_attempt,
            sleep=_record_sleep,
        )

    assert engine.dispatcher_count == baseline_dispatcher_count
    assert sleep_calls == []
    assert provisioner.ensure_calls == []
    assert provisioner.confirm_calls == []
    assert bus.subscriptions == []
    assert attempt_results == []
    assert applier.results == []


@pytest.mark.asyncio
async def test_not_ready_names_must_be_canonical_manifest_subset() -> None:
    """Reattach rejects whitespace aliases and names absent from the manifest."""
    from omnibase_core.models.errors import ModelOnexError
    from omnibase_infra.runtime.auto_wiring.handler_wiring import (
        reattach_not_ready_contracts,
    )

    contract = _contract("node_not_ready_identity", "HandlerContractA")
    manifest = ModelAutoWiringManifest(contracts=(contract,), errors=())
    engine = MessageDispatchEngine()
    report = await wire_from_manifest(
        manifest,
        engine,
        event_bus=_RecordingBus(),
        environment="test",
        subscribe_immediately=False,
    )
    engine.freeze()
    dispatcher_id = report.results[0].dispatchers_registered[0]

    for contract_name, error_match in (
        (f" {contract.name} ", "noncanonical NOT_READY contract names"),
        ("node_not_in_manifest", "NOT_READY.*manifest contract-name mismatch"),
    ):
        forged_result = ModelContractAttachResult.model_validate_json(
            json.dumps(
                {
                    "contract_name": contract_name,
                    "status": EnumContractAttachStatus.NOT_READY.value,
                    "dispatcher_ids": [dispatcher_id],
                }
            )
        )
        bus = _RecordingBus()
        provisioner = _RecordingReadyProvisioner()
        with pytest.raises(ModelOnexError, match=error_match):
            await reattach_not_ready_contracts(
                manifest,
                (forged_result,),
                engine,
                bus,
                "test",
                provisioner=provisioner,
            )
        assert provisioner.ensure_calls == []
        assert provisioner.confirm_calls == []
        assert bus.subscriptions == []


@pytest.mark.asyncio
async def test_not_ready_wrong_registered_owner_fails_before_provisioning() -> None:
    """A reattach scope must retain its original live contract provenance."""
    from omnibase_core.models.errors import ModelOnexError
    from omnibase_infra.runtime.auto_wiring.handler_wiring import (
        reattach_not_ready_contracts,
    )

    contract_a = _contract("node_not_ready_wrong_a", "HandlerContractA")
    contract_b = _contract("node_not_ready_wrong_b", "HandlerContractB")
    manifest = ModelAutoWiringManifest(contracts=(contract_a, contract_b), errors=())
    engine = MessageDispatchEngine()
    bus = _RecordingBus()
    report = await wire_from_manifest(
        manifest,
        engine,
        event_bus=bus,
        environment="test",
        subscribe_immediately=False,
    )
    engine.freeze()
    by_name = {result.contract_name: result for result in report.results}
    dispatcher_b = by_name[contract_b.name].dispatchers_registered[0]
    forged_result = ModelContractAttachResult.model_validate_json(
        json.dumps(
            {
                "contract_name": contract_a.name,
                "status": EnumContractAttachStatus.NOT_READY.value,
                "dispatcher_ids": [dispatcher_b],
            }
        )
    )
    provisioner = _RecordingReadyProvisioner()

    with pytest.raises(ModelOnexError, match="not owned by contract"):
        await reattach_not_ready_contracts(
            manifest,
            (forged_result,),
            engine,
            bus,
            "test",
            provisioner=provisioner,
        )

    assert provisioner.ensure_calls == []
    assert provisioner.confirm_calls == []
    assert bus.subscriptions == []


@pytest.mark.asyncio
async def test_one_contract_may_own_two_unique_dispatchers() -> None:
    """Ownership uniqueness is per dispatcher, not one dispatcher per contract."""
    from datetime import UTC, datetime
    from uuid import uuid4

    from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope

    HandlerContractA.calls = 0
    HandlerContractB.calls = 0
    contract = _contract_with_handlers(
        "node_two_dispatchers",
        ("HandlerContractA", "HandlerContractB"),
    )
    manifest = ModelAutoWiringManifest(contracts=(contract,), errors=())
    engine = MessageDispatchEngine()
    bus = _RecordingBus()
    report = await wire_from_manifest(
        manifest,
        engine,
        event_bus=bus,
        environment="test",
        subscribe_immediately=False,
    )
    engine.freeze()

    owned_dispatchers = report.results[0].dispatchers_registered
    assert len(owned_dispatchers) == 2
    assert len(set(owned_dispatchers)) == 2
    subscriptions = await subscribe_wired_contract_topics(
        manifest,
        report,
        engine,
        bus,
        "test",
    )
    assert subscriptions == {contract.name: (_SHARED_TOPIC,)}
    assert len(bus.subscriptions) == 1

    envelope = ModelEventEnvelope[object](
        payload={"prompt": "one contract, two dispatchers"},
        correlation_id=uuid4(),
        envelope_timestamp=datetime.now(UTC),
        event_type="omn15474.shared-command",
        source_tool="contract-scoped-dispatch-test",
    )
    await bus.deliver_to_every_group(envelope)

    assert HandlerContractA.calls == 1
    assert HandlerContractB.calls == 1


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
