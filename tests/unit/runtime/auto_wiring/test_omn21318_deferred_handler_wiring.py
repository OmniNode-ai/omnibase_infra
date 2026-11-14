# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21318: a demand-deferred contract is not wired at boot at all.

The boot subscription split (OMN-18087) already kept an idle contract's
consumer groups from joining at boot, but its handlers were still imported
and its dispatchers still registered. Deferral now happens one step earlier:
``wire_from_manifest`` leaves a deferred contract entirely unwired — one
``DEFERRED`` row keeps the report total over the manifest (OMN-15474) — and
``subscribe_wired_contract_topics`` watches its topics instead of attaching
them. When the first record reaches one of those topics, the wirer built by
``make_deferred_contract_wirer`` imports and registers the contract on the
frozen engine, then joins its groups from ``earliest`` through the normal
provision -> ready -> attach interleave, so the record that started the
contract is the first one its groups read.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Collection
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import pytest

from omnibase_core.models.errors import ModelOnexError
from omnibase_infra.enums import EnumConsumerGroupPurpose
from omnibase_infra.event_bus.enum_contract_attach_exclusion_reason import (
    EnumContractAttachExclusionReason,
)
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.model_contract_attach_exclusion import (
    ModelContractAttachExclusion,
)
from omnibase_infra.event_bus.model_contract_attach_result import (
    ModelContractAttachResult,
)
from omnibase_infra.event_bus.model_topic_readiness_config import (
    ModelTopicReadinessConfig,
)
from omnibase_infra.event_bus.models import ModelEventMessage
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.handlers.boot_readiness_checks import verify_roster_partition
from omnibase_infra.handlers.boot_wiring_deferral import (
    DEFERRED_WIRING_REASON,
    BootWiringDeferral,
    DeferredContractWirer,
    make_deferred_contract_wirer,
    plan_boot_wiring_deferral,
)
from omnibase_infra.models import ModelNodeIdentity
from omnibase_infra.runtime.auto_wiring import handler_wiring
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    subscribe_wired_contract_topics,
    wire_from_manifest,
)
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelDiscoveredContract,
    ModelEventBusWiring,
)
from omnibase_infra.runtime.auto_wiring.report import (
    EnumWiringOutcome,
    ModelAutoWiringReport,
    ModelContractWiringResult,
)
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine
from tests.unit.runtime.auto_wiring.test_per_contract_boot_interleave import (
    RecordingProvisioner,
    _contract,
    _fake_handler_cls,
)

if TYPE_CHECKING:
    from omnibase_infra.handlers.handler_kafka_topic_demand_source import (
        HandlerKafkaTopicDemandSource,
    )

pytestmark = pytest.mark.unit

ACTIVE = "onex.cmd.test.delegate-in-use.v1"
IDLE = "onex.cmd.test.never-requested.v1"
# Every contract the shared _contract helper builds declares this handler, so
# each wiring pass asks for it exactly once per contract it actually wires.
FAKE_MODULE = "fake.module"
FAKE_HANDLER = "FakeHandler"


class RecordingKafkaBus(EventBusKafka):
    """A Kafka bus whose subscribe records the join instead of joining."""

    def __init__(self) -> None:
        super().__init__(
            config=ModelKafkaEventBusConfig(bootstrap_servers="localhost:9092")
        )
        self.joins: list[tuple[str, str | None]] = []
        self.joined = asyncio.Event()

    async def subscribe(
        self,
        topic: str,
        node_identity: ModelNodeIdentity | None = None,
        on_message: Callable[[ModelEventMessage], Awaitable[None]] | None = None,
        *,
        group_id: str | None = None,
        purpose: EnumConsumerGroupPurpose = EnumConsumerGroupPurpose.CONSUME,
        required_for_readiness: bool = False,
        auto_offset_reset: str | None = None,
    ) -> Callable[[], Awaitable[None]]:
        self.joins.append((topic, auto_offset_reset))
        self.joined.set()

        async def _unsubscribe() -> None:
            return None

        return _unsubscribe


class FakeDemandSource:
    """Reports which topics hold records and hands back the first-record hook."""

    def __init__(self, in_use: frozenset[str]) -> None:
        self._in_use = in_use
        self.watched: frozenset[str] = frozenset()
        self.on_first_record: Callable[[str], Awaitable[None]] | None = None
        self.stopped = False

    async def watch_idle_topics(
        self,
        topics: Collection[str],
        on_first_record: Callable[[str], Awaitable[None]],
    ) -> tuple[frozenset[str], Callable[[], Awaitable[None]]]:
        self.watched = frozenset(topics) - self._in_use
        self.on_first_record = on_first_record

        async def stop() -> None:
            self.stopped = True

        return frozenset(topics) & self._in_use, stop


DeferredWirerFactory = Callable[
    [ModelAutoWiringManifest, MessageDispatchEngine, RecordingKafkaBus],
    DeferredContractWirer | None,
]
"""Builds the boot's first-request wiring step from the boot's own pieces."""


def _recording_import(imports: list[tuple[str, str]]) -> Callable[[str, str], type]:
    """A ``_import_handler_class`` stand-in recording each (module, name) asked."""

    def _import(module_path: str, class_name: str) -> type:
        imports.append((module_path, class_name))
        return _fake_handler_cls()

    return _import


def _real_wirer(
    manifest: ModelAutoWiringManifest,
    engine: MessageDispatchEngine,
    bus: RecordingKafkaBus,
) -> DeferredContractWirer:
    """The production first-request wiring step, bound to the boot's pieces."""
    return make_deferred_contract_wirer(
        manifest=manifest,
        dispatch_engine=engine,
        event_bus=bus,
        environment="local",
        container=None,
        result_appliers_by_contract=None,
        materialized_explicit_dependencies=None,
        topology=None,
        wiring=handler_wiring,
    )


def _failed_wirer(
    _manifest: ModelAutoWiringManifest,
    _engine: MessageDispatchEngine,
    _bus: RecordingKafkaBus,
) -> DeferredContractWirer:
    """A first-request wiring step whose result is a FAILED row, built directly."""

    async def wire(contract: ModelDiscoveredContract) -> ModelContractWiringResult:
        return ModelContractWiringResult(
            contract_name=contract.name,
            package_name=contract.package_name,
            outcome=EnumWiringOutcome.FAILED,
            reason="first-request handler import refused (test)",
        )

    return wire


async def _plan(
    manifest: ModelAutoWiringManifest,
    demand: FakeDemandSource,
    wire: DeferredContractWirer | None,
    priority: tuple[str, ...] = (),
) -> BootWiringDeferral:
    return await plan_boot_wiring_deferral(
        manifest,
        demand_source=cast("HandlerKafkaTopicDemandSource", demand),
        readiness_config=ModelTopicReadinessConfig(priority_contracts=priority),
        result_appliers_by_contract=None,
        wire=wire,
        wiring=handler_wiring,
    )


async def _boot(
    demand: FakeDemandSource,
    make_wire_deferred: DeferredWirerFactory,
    *,
    first_record_before_freeze: bool = False,
) -> tuple[
    RecordingKafkaBus,
    MessageDispatchEngine,
    ModelAutoWiringReport,
    list[ModelContractAttachExclusion],
    list[ModelContractAttachResult],
]:
    """Boot the in-use + idle manifest the way the kernel does, idle one deferred.

    The caller must hold the ``_import_handler_class`` patch: the boot imports
    the in-use contract's handler, and a first request later imports the idle
    contract's through *make_wire_deferred*'s step.
    """
    manifest = ModelAutoWiringManifest(
        contracts=(
            _contract("node_in_use", (ACTIVE,)),
            _contract("node_idle", (IDLE,)),
        )
    )
    engine = MessageDispatchEngine()
    bus = RecordingKafkaBus()
    exclusions: list[ModelContractAttachExclusion] = []
    results: list[ModelContractAttachResult] = []
    deferral = await _plan(manifest, demand, make_wire_deferred(manifest, engine, bus))
    wired_report = await wire_from_manifest(
        deferral.wired_manifest,
        engine,
        event_bus=bus,
        environment="local",
        subscribe_immediately=False,
    )
    report = deferral.complete_report(wired_report)
    engine.freeze()
    if first_record_before_freeze:
        assert demand.on_first_record is not None
        await demand.on_first_record(IDLE)
    await subscribe_wired_contract_topics(
        manifest=deferral.wired_manifest,
        report=wired_report,
        dispatch_engine=engine,
        event_bus=bus,
        environment="local",
        provisioner=RecordingProvisioner(),
        readiness_config=ModelTopicReadinessConfig(),
        attach_results_out=results,
        exclusions_out=exclusions,
    )
    await deferral.watch(
        dispatch_engine=engine,
        event_bus=bus,
        environment="local",
        provisioner=RecordingProvisioner(),
        exclusions_out=exclusions,
        stops_out=[],
    )
    return bus, engine, report, exclusions, results


@pytest.mark.asyncio
async def test_boot_wires_zero_handlers_of_deferred_contracts() -> None:
    """Twenty idle contracts defer: only the in-use contract's handler imports."""
    idle_names = tuple(f"node_idle_{index:02d}" for index in range(20))
    idle_topics = tuple(f"onex.cmd.test.idle-{index:02d}.v1" for index in range(20))
    manifest = ModelAutoWiringManifest(
        contracts=(
            _contract("node_in_use", (ACTIVE,)),
            *(
                _contract(name, (topic,))
                for name, topic in zip(idle_names, idle_topics, strict=True)
            ),
        )
    )
    engine = MessageDispatchEngine()
    bus = RecordingKafkaBus()
    imports: list[tuple[str, str]] = []
    deferral = await _plan(manifest, FakeDemandSource(frozenset({ACTIVE})), None)
    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        side_effect=_recording_import(imports),
    ):
        wired_report = await wire_from_manifest(
            deferral.wired_manifest,
            engine,
            event_bus=bus,
            environment="local",
            subscribe_immediately=False,
        )

    # Exactly one import: the in-use contract's handler. The deferred
    # contracts' handler module was never asked for.
    assert imports == [(FAKE_MODULE, FAKE_HANDLER)]
    assert deferral.deferred_names == frozenset(idle_names)
    assert wired_report.total_wired == 1
    # Nothing was registered for a deferred contract: the engine holds only the
    # in-use contract's dispatcher, and the bus joined nothing.
    assert engine.dispatcher_count == 1
    assert bus.joins == []
    # The completed report stays total over the manifest: one SKIPPED row, with
    # the deferral reason, per deferred contract.
    report = deferral.complete_report(wired_report)
    assert {r.contract_name for r in report.results} == {
        c.name for c in manifest.contracts
    }
    deferred_rows = [r for r in report.results if r.reason == DEFERRED_WIRING_REASON]
    assert {r.contract_name for r in deferred_rows} == set(idle_names)
    assert all(r.outcome is EnumWiringOutcome.SKIPPED for r in deferred_rows)


@pytest.mark.asyncio
async def test_without_a_demand_source_nothing_is_deferred() -> None:
    """The in-memory bus has no broker to ask: every contract wires at boot."""
    manifest = ModelAutoWiringManifest(
        contracts=(_contract("node_a", (ACTIVE,)), _contract("node_b", (IDLE,)))
    )

    deferral = await plan_boot_wiring_deferral(
        manifest,
        demand_source=None,
        readiness_config=ModelTopicReadinessConfig(),
        result_appliers_by_contract=None,
        wire=None,
        wiring=handler_wiring,
    )

    assert deferral.deferred_names == frozenset()
    assert [c.name for c in deferral.wired_manifest.contracts] == ["node_a", "node_b"]


@pytest.mark.asyncio
async def test_golden_first_request_wires_then_attaches_once() -> None:
    """The first record wires the deferred contract once, then attaches it."""
    demand = FakeDemandSource(frozenset({ACTIVE}))
    imports: list[tuple[str, str]] = []
    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        side_effect=_recording_import(imports),
    ):
        bus, engine, _report, exclusions, _results = await _boot(demand, _real_wirer)
        assert [e.reason for e in exclusions if e.contract_name == "node_idle"] == [
            EnumContractAttachExclusionReason.DEMAND_DEFERRED
        ]

        # At boot only the in-use contract joined. The recorder holds the
        # in-use contract's one import, and only its dispatcher is registered.
        assert [topic for topic, _ in bus.joins] == [ACTIVE]
        assert demand.watched == {IDLE}
        assert imports == [(FAKE_MODULE, FAKE_HANDLER)]
        assert engine.dispatcher_count == 1

        bus.joined.clear()
        assert demand.on_first_record is not None
        await demand.on_first_record(IDLE)
        await asyncio.wait_for(bus.joined.wait(), timeout=2)

        # The first request imported the deferred contract's handler exactly
        # once, registered its dispatcher on the frozen engine, then joined its
        # topic from earliest: the record that started it is the first it reads.
        assert bus.joins == [(ACTIVE, None), (IDLE, "earliest")]
        assert imports == [(FAKE_MODULE, FAKE_HANDLER)] * 2
        assert engine.dispatcher_count == 2

        # A second record on the same topic starts nothing new.
        await demand.on_first_record(IDLE)
        for _ in range(5):
            await asyncio.sleep(0)

        assert [join for join in bus.joins if join[0] == IDLE] == [(IDLE, "earliest")]
        assert len(imports) == 2
        assert engine.dispatcher_count == 2


@pytest.mark.asyncio
async def test_a_record_that_lands_before_the_engine_froze_is_replayed() -> None:
    """A first record during wiring is held, then starts the contract once armed."""
    demand = FakeDemandSource(frozenset({ACTIVE}))
    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        side_effect=_recording_import([]),
    ):
        bus, engine, _report, _exclusions, _results = await _boot(
            demand, _real_wirer, first_record_before_freeze=True
        )
        await asyncio.wait_for(bus.joined.wait(), timeout=2)
        for _ in range(5):
            await asyncio.sleep(0)

        assert [join for join in bus.joins if join[0] == IDLE] == [(IDLE, "earliest")]
        assert engine.dispatcher_count == 2


@pytest.mark.asyncio
async def test_error_chain_a_first_request_wiring_that_fails_attaches_nothing() -> None:
    """A FAILED first-request wiring attaches nothing and raises nothing."""
    demand = FakeDemandSource(frozenset({ACTIVE}))
    imports: list[tuple[str, str]] = []
    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        side_effect=_recording_import(imports),
    ):
        bus, engine, _report, _exclusions, _results = await _boot(demand, _failed_wirer)
        assert demand.on_first_record is not None

        await demand.on_first_record(IDLE)
        for _ in range(10):
            await asyncio.sleep(0)

    assert bus.joins == [(ACTIVE, None)]
    assert imports == [(FAKE_MODULE, FAKE_HANDLER)]
    assert engine.dispatcher_count == 1


@pytest.mark.asyncio
async def test_error_chain_a_handler_import_that_raises_is_a_failed_row() -> None:
    """The real wirer turns a preparation error into a FAILED row, not a crash."""
    manifest = ModelAutoWiringManifest(contracts=(_contract("node_idle", (IDLE,)),))
    engine = MessageDispatchEngine()
    engine.freeze()
    wire = _real_wirer(manifest, engine, RecordingKafkaBus())
    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        side_effect=ImportError("handler module is gone (test)"),
    ):
        row = await wire(manifest.contracts[0])

    assert row.outcome is EnumWiringOutcome.FAILED
    assert "ImportError" in row.reason
    assert engine.dispatcher_count == 0


@pytest.mark.asyncio
async def test_error_chain_a_duplicate_name_among_deferred_contracts_is_refused_before_the_watch() -> (
    None
):
    """A duplicate contract name fails the plan before any watch starts."""
    manifest = ModelAutoWiringManifest(
        contracts=(
            _contract("node_in_use", (ACTIVE,)),
            _contract("node_idle", (IDLE,)),
            _contract("node_idle", ("onex.cmd.test.also-never-requested.v1",)),
        )
    )
    demand = FakeDemandSource(frozenset({ACTIVE}))

    with pytest.raises(ModelOnexError):
        await _plan(manifest, demand, None)

    # The manifest-wide name check ran before the broker was measured, so the
    # watch never started: nothing holds its first-record hook.
    assert demand.on_first_record is None


@pytest.mark.asyncio
async def test_error_chain_a_command_topic_with_two_owners_is_refused_when_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The strict single-owner gate fails the plan before any watch starts."""
    # Without the full-manifest check the two deferred contracts would pass
    # silently, because the wired manifest holds neither.
    monkeypatch.setenv("ONEX_SINGLE_OWNER_COMMAND_TOPICS", "1")
    manifest = ModelAutoWiringManifest(
        contracts=(
            _contract("node_owner_a", (IDLE,)),
            _contract("node_owner_b", (IDLE,)),
        )
    )
    demand = FakeDemandSource(frozenset())

    with pytest.raises(ModelOnexError):
        await _plan(manifest, demand, None)

    assert demand.on_first_record is None


@pytest.mark.asyncio
async def test_selection_never_defers_priority_in_use_or_plugin_managed() -> None:
    """Only a non-priority contract whose every topic is idle is deferred."""
    priority_topic = "onex.cmd.test.priority-listed.v1"
    plugin_topic = "onex.cmd.test.plugin-owned.v1"
    manifest = ModelAutoWiringManifest(
        contracts=(
            _contract("node_priority", (priority_topic,)),
            _contract("node_in_use", (ACTIVE,)),
            _contract("node_idle", (IDLE,)),
            # _contract has no plugin_managed option, so overwrite the bus.
            _contract("node_plugin", (plugin_topic,)).model_copy(
                update={
                    "event_bus": ModelEventBusWiring(
                        subscribe_topics=(plugin_topic,),
                        publish_topics=(),
                        plugin_managed=True,
                    )
                }
            ),
        )
    )

    deferral = await _plan(
        manifest, FakeDemandSource(frozenset({ACTIVE})), None, ("node_priority",)
    )

    assert deferral.deferred_names == {"node_idle"}


@pytest.mark.asyncio
async def test_an_unmeasurable_broker_defers_nothing() -> None:
    """A broker that reports every topic asked as in use defers no contract."""
    manifest = ModelAutoWiringManifest(
        contracts=(_contract("node_a", (ACTIVE,)), _contract("node_idle", (IDLE,)))
    )

    deferral = await _plan(manifest, FakeDemandSource(frozenset({ACTIVE, IDLE})), None)

    assert deferral.deferred_names == frozenset()
    assert [c.name for c in deferral.wired_manifest.contracts] == [
        "node_a",
        "node_idle",
    ]


@pytest.mark.asyncio
async def test_roster_partition_accounts_for_deferred_contracts() -> None:
    """The deferred contract is excluded, the wired one attempted: no unaccounted."""
    demand = FakeDemandSource(frozenset({ACTIVE}))
    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        side_effect=_recording_import([]),
    ):
        _bus, _engine, report, exclusions, results = await _boot(demand, _real_wirer)

    partition = verify_roster_partition(
        [row.contract_name for row in report.results], results, exclusions
    )

    assert partition.is_partition, partition.describe_violations()
    assert partition.attempted == ("node_in_use",)
    assert partition.excluded == ("node_idle",)


def _raising_wirer(
    _manifest: ModelAutoWiringManifest,
    _engine: MessageDispatchEngine,
    _bus: RecordingKafkaBus,
) -> DeferredContractWirer:
    """A first-request wiring step that raises instead of returning a row."""

    async def wire(contract: ModelDiscoveredContract) -> ModelContractWiringResult:
        raise RuntimeError(f"wiring {contract.name} blew up (test)")

    return wire


@pytest.mark.asyncio
async def test_error_chain_a_start_that_raises_is_logged_and_attaches_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An exception out of a first-request start is logged, never lost or fatal."""
    demand = FakeDemandSource(frozenset({ACTIVE}))
    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        side_effect=_recording_import([]),
    ):
        bus, _engine, _report, _exclusions, _results = await _boot(
            demand, _raising_wirer
        )
        assert demand.on_first_record is not None
        with caplog.at_level("ERROR"):
            await demand.on_first_record(IDLE)
            for _ in range(10):
                await asyncio.sleep(0)

    assert bus.joins == [(ACTIVE, None)]
    assert any("ended with an exception" in r.getMessage() for r in caplog.records)
