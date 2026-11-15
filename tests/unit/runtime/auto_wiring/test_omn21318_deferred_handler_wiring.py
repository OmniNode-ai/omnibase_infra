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
from collections.abc import Awaitable, Callable, Collection, Sequence
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import pytest

from omnibase_core.models.errors import ModelOnexError
from omnibase_infra.enums import EnumConsumerGroupPurpose
from omnibase_infra.event_bus.enum_contract_attach_exclusion_reason import (
    EnumContractAttachExclusionReason,
)
from omnibase_infra.event_bus.enum_contract_attach_status import (
    EnumContractAttachStatus,
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
from omnibase_infra.handlers.boot_readiness_checks import (
    record_attach_gate,
    verify_roster_partition,
)
from omnibase_infra.handlers.boot_wiring_deferral import (
    DEFERRED_WIRING_REASON,
    BootWiringDeferral,
    DeferredContractPreparer,
    DeferredContractWirer,
    make_deferred_contract_preparer,
    make_deferred_contract_wirer,
    plan_boot_wiring_deferral,
)
from omnibase_infra.models import ModelNodeIdentity
from omnibase_infra.models.operational_timeouts import TIMEOUTS
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
from omnibase_infra.runtime.health.contract_attach_readiness_gate import (
    ContractAttachReadinessGate,
    derive_required_contract_names,
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

DeferredPreparerFactory = Callable[
    [ModelAutoWiringManifest, MessageDispatchEngine, RecordingKafkaBus],
    DeferredContractPreparer,
]
"""Builds the boot's strict-verification preparation step the same way."""


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


def _real_preparer(
    manifest: ModelAutoWiringManifest,
    engine: MessageDispatchEngine,
    bus: RecordingKafkaBus,
) -> DeferredContractPreparer:
    """The production strict-verification preparation step, bound the same way."""
    return make_deferred_contract_preparer(
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


async def _ticks_until(predicate: Callable[[], bool], *, limit: int = 500) -> None:
    """Yield to the event loop until *predicate* holds; fail loudly, never hang."""
    for _ in range(limit):
        if predicate():
            return
        await asyncio.sleep(0)
    pytest.fail("the expected async work never completed within the tick budget")


async def _plan(
    manifest: ModelAutoWiringManifest,
    demand: FakeDemandSource,
    wire: DeferredContractWirer | None,
    priority: tuple[str, ...] = (),
    *,
    prepare: DeferredContractPreparer | None = None,
    core_runtime_topics: frozenset[str] = frozenset(),
) -> BootWiringDeferral:
    return await plan_boot_wiring_deferral(
        manifest,
        demand_source=cast("HandlerKafkaTopicDemandSource", demand),
        readiness_config=ModelTopicReadinessConfig(priority_contracts=priority),
        result_appliers_by_contract=None,
        wire=wire,
        prepare=prepare,
        core_runtime_topics=core_runtime_topics,
        wiring=handler_wiring,
    )


async def _boot(
    demand: FakeDemandSource,
    make_wire_deferred: DeferredWirerFactory,
    *,
    first_record_before_freeze: bool = False,
    make_prepare_deferred: DeferredPreparerFactory | None = None,
    on_attempt: Callable[[Sequence[ModelContractAttachResult]], None] | None = None,
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
    contract's through *make_wire_deferred*'s step. *make_prepare_deferred*
    binds the strict-verification preparation step the same way, and
    *on_attempt* travels to ``deferral.watch`` so a readiness gate observes
    every first-request attach attempt.
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
    deferral = await _plan(
        manifest,
        demand,
        make_wire_deferred(manifest, engine, bus),
        prepare=(
            make_prepare_deferred(manifest, engine, bus)
            if make_prepare_deferred is not None
            else None
        ),
    )
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
        on_attempt=on_attempt,
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
async def test_error_chain_a_first_request_wiring_that_fails_attaches_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A FAILED first-request wiring attaches nothing, raises nothing, and the
    FAILED row reaches the attempt observer with the row's reason as detail."""
    monkeypatch.setattr(handler_wiring, "DEFAULT_NOT_READY_RETRY_BACKOFF_SECONDS", 0.0)
    demand = FakeDemandSource(frozenset({ACTIVE}))
    attempts: list[tuple[ModelContractAttachResult, ...]] = []
    imports: list[tuple[str, str]] = []
    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        side_effect=_recording_import(imports),
    ):
        bus, engine, _report, _exclusions, _results = await _boot(
            demand, _failed_wirer, on_attempt=attempts.append
        )
        assert demand.on_first_record is not None

        await demand.on_first_record(IDLE)
        await _ticks_until(lambda: bool(attempts))

    assert bus.joins == [(ACTIVE, None)]
    assert imports == [(FAKE_MODULE, FAKE_HANDLER)]
    assert engine.dispatcher_count == 1
    # Every attempt reported the deferred contract FAILED with the wiring row's
    # reason as the detail: a readiness gate sees WHY the contract is not ready.
    failed = [
        result
        for batch in attempts
        for result in batch
        if result.contract_name == "node_idle"
    ]
    assert failed
    assert all(result.status is EnumContractAttachStatus.FAILED for result in failed)
    assert all(
        "first-request handler import refused (test)" in result.detail
        for result in failed
    )


@pytest.mark.asyncio
async def test_error_chain_a_failed_first_request_retries_and_shows_not_ready_then_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The seam manifest -> wiring -> health gate: a FAILED first attempt keeps
    the real readiness gate not ready, and the retry that attaches flips it."""
    monkeypatch.setattr(handler_wiring, "DEFAULT_NOT_READY_RETRY_BACKOFF_SECONDS", 0.0)
    demand = FakeDemandSource(frozenset({ACTIVE}))
    manifest = ModelAutoWiringManifest(
        contracts=(_contract("node_in_use", (ACTIVE,)), _contract("node_idle", (IDLE,)))
    )
    # Both contracts subscribe onex.cmd.* topics, so the gate requires both.
    gate = ContractAttachReadinessGate(derive_required_contract_names(manifest))
    assert gate.required_contract_names == frozenset({"node_in_use", "node_idle"})
    attempts: list[tuple[ModelContractAttachResult, ...]] = []

    def record_attempt(batch: Sequence[ModelContractAttachResult]) -> None:
        attempts.append(tuple(batch))
        gate.record(batch)

    def _flaky_wirer(
        manifest: ModelAutoWiringManifest,
        engine: MessageDispatchEngine,
        bus: RecordingKafkaBus,
    ) -> DeferredContractWirer:
        """FAILED row on the first call, the production wiring step after it."""
        real = _real_wirer(manifest, engine, bus)
        calls = {"count": 0}

        async def wire(contract: ModelDiscoveredContract) -> ModelContractWiringResult:
            calls["count"] += 1
            if calls["count"] == 1:
                return ModelContractWiringResult(
                    contract_name=contract.name,
                    package_name=contract.package_name,
                    outcome=EnumWiringOutcome.FAILED,
                    reason="first-request wiring not possible yet (test)",
                )
            return await real(contract)

        return wire

    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        side_effect=_recording_import([]),
    ):
        bus, engine, _report, exclusions, results = await _boot(
            demand, _flaky_wirer, on_attempt=record_attempt
        )
        # The kernel folds the boot interleave's exclusions and results into the
        # gate right after the watch: the in-use contract is ATTACHED, the
        # deferred one excluded until a result says otherwise.
        record_attach_gate(gate, results, exclusions, None)
        bus.joined.clear()
        assert demand.on_first_record is not None

        await demand.on_first_record(IDLE)
        await _ticks_until(
            lambda: any(
                result.contract_name == "node_idle"
                and result.status is EnumContractAttachStatus.FAILED
                for batch in attempts
                for result in batch
            )
        )

        # After the first failed attempt the gate names the failed contract and
        # the probe answers not ready.
        assert gate.status().failed_contracts == ("node_idle",)
        assert gate.probe()[0] is False

        # The retry wires through the production step and attaches.
        await asyncio.wait_for(bus.joined.wait(), timeout=2)
        await _ticks_until(lambda: gate.status().ready)

    assert [join for join in bus.joins if join[0] == IDLE] == [(IDLE, "earliest")]
    assert engine.dispatcher_count == 2
    assert [result.status for batch in attempts for result in batch] == [
        EnumContractAttachStatus.FAILED,
        EnumContractAttachStatus.ATTACHED,
    ]
    assert gate.status().ready is True
    assert gate.probe()[0] is True


@pytest.mark.asyncio
async def test_error_chain_a_failed_first_request_never_attaching_keeps_the_gate_not_ready(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A contract whose every attempt fails keeps the gate not ready; the spent
    budget releases the name, so the NEXT first record starts it again."""
    monkeypatch.setattr(handler_wiring, "DEFAULT_NOT_READY_RETRY_BACKOFF_SECONDS", 0.0)
    demand = FakeDemandSource(frozenset({ACTIVE}))
    manifest = ModelAutoWiringManifest(
        contracts=(_contract("node_in_use", (ACTIVE,)), _contract("node_idle", (IDLE,)))
    )
    gate = ContractAttachReadinessGate(derive_required_contract_names(manifest))
    wire_calls = {"count": 0}

    def _always_failing_wirer(
        _manifest: ModelAutoWiringManifest,
        _engine: MessageDispatchEngine,
        _bus: RecordingKafkaBus,
    ) -> DeferredContractWirer:
        async def wire(contract: ModelDiscoveredContract) -> ModelContractWiringResult:
            wire_calls["count"] += 1
            return ModelContractWiringResult(
                contract_name=contract.name,
                package_name=contract.package_name,
                outcome=EnumWiringOutcome.FAILED,
                reason="first-request handler import refused (test)",
            )

        return wire

    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        side_effect=_recording_import([]),
    ):
        _bus, _engine, _report, exclusions, results = await _boot(
            demand, _always_failing_wirer, on_attempt=gate.record
        )
        record_attach_gate(gate, results, exclusions, None)
        assert demand.on_first_record is not None
        with caplog.at_level("ERROR"):
            await demand.on_first_record(IDLE)
            await _ticks_until(
                lambda: any(
                    "did not activate after" in record.getMessage()
                    for record in caplog.records
                )
            )

    # The attempt budget is spent on exactly its count of wirings, and the gate
    # keeps naming the failed contract: the attached in-use contract cannot
    # mask it, and the demand-deferred exclusion cannot exclude it into ready.
    status = gate.status()
    assert wire_calls["count"] == TIMEOUTS.not_ready_retry_max_attempts
    assert status.failed_contracts == ("node_idle",)
    assert status.attached_contracts == ("node_in_use",)
    assert status.ready is False
    assert gate.probe()[0] is False

    # The name left ``started``: a second record on the idle topic starts the
    # contract again instead of leaving it wedged until a restart.
    await demand.on_first_record(IDLE)
    await _ticks_until(
        lambda: wire_calls["count"] > TIMEOUTS.not_ready_retry_max_attempts
    )
    assert wire_calls["count"] > TIMEOUTS.not_ready_retry_max_attempts
    assert gate.status().failed_contracts == ("node_idle",)
    assert gate.probe()[0] is False


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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exception out of a first-request start is caught per attempt: logged
    as 'raised', retried, and never fatal to the running runtime."""
    monkeypatch.setattr(handler_wiring, "DEFAULT_NOT_READY_RETRY_BACKOFF_SECONDS", 0.0)
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
            await _ticks_until(
                lambda: any(
                    "Demand start: activating contract" in record.getMessage()
                    and "raised" in record.getMessage()
                    for record in caplog.records
                )
            )

    # The bus still joined only the in-use contract on the first attempt, the
    # per-attempt 'raised' log appears, and the start task never 'ended with an
    # exception': the exception is one failed attempt, not a lost task.
    assert bus.joins == [(ACTIVE, None)]
    assert any(
        "Demand start: activating contract" in record.getMessage()
        and "raised" in record.getMessage()
        for record in caplog.records
    )
    assert not any(
        "ended with an exception" in record.getMessage() for record in caplog.records
    )


@pytest.mark.asyncio
async def test_strict_boot_refuses_a_broken_deferred_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ONEX_WIRING_STRICT_MODE prepares the deferred contracts: a handler a
    strict boot would refuse does not hide behind the deferral."""
    monkeypatch.setenv("ONEX_WIRING_STRICT_MODE", "1")
    monkeypatch.delenv("ONEX_STRICT_DISPATCHER_COVERAGE", raising=False)
    manifest = ModelAutoWiringManifest(
        contracts=(_contract("node_in_use", (ACTIVE,)), _contract("node_idle", (IDLE,)))
    )
    engine = MessageDispatchEngine()
    bus = RecordingKafkaBus()
    deferral = await _plan(
        manifest,
        FakeDemandSource(frozenset({ACTIVE})),
        None,
        prepare=_real_preparer(manifest, engine, bus),
    )
    assert deferral.deferred_names == {"node_idle"}

    imports: list[tuple[str, str]] = []

    def _import(module_path: str, class_name: str) -> type:
        # Both contracts declare the same fake handler, so key on call count:
        # the FIRST import is the in-use contract wiring at boot, the second is
        # the deferred contract's strict preparation, which must refuse.
        imports.append((module_path, class_name))
        if len(imports) == 1:
            return _fake_handler_cls()
        raise ImportError("handler module is gone (test)")

    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        side_effect=_import,
    ):
        # The wired manifest holds only the in-use contract, so the boot passes.
        wired_report = await wire_from_manifest(
            deferral.wired_manifest,
            engine,
            event_bus=bus,
            environment="local",
            subscribe_immediately=False,
        )
        assert wired_report.total_wired == 1
        with pytest.raises(ModelOnexError) as excinfo:
            await deferral.verify_strict(wired_report)

    assert "node_idle" in str(excinfo.value)
    assert "ImportError" in str(excinfo.value)
    assert [c.name for c in deferral.wired_manifest.contracts] == ["node_in_use"]
    # The refusal came from the deferred contract's preparation only, and that
    # preparation registered nothing on the engine.
    assert imports == [(FAKE_MODULE, FAKE_HANDLER)] * 2
    assert engine.dispatcher_count == 1


@pytest.mark.asyncio
async def test_non_strict_boot_does_not_refuse_a_broken_deferred_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the strict flags the same broken deferred handler refuses
    nothing: verify_strict returns None and the breakage is only a FAILED
    first-request row later."""
    monkeypatch.delenv("ONEX_WIRING_STRICT_MODE", raising=False)
    monkeypatch.delenv("ONEX_STRICT_DISPATCHER_COVERAGE", raising=False)
    monkeypatch.setattr(handler_wiring, "DEFAULT_NOT_READY_RETRY_BACKOFF_SECONDS", 0.0)
    manifest = ModelAutoWiringManifest(
        contracts=(_contract("node_in_use", (ACTIVE,)), _contract("node_idle", (IDLE,)))
    )
    engine = MessageDispatchEngine()
    bus = RecordingKafkaBus()
    demand = FakeDemandSource(frozenset({ACTIVE}))
    attempts: list[tuple[ModelContractAttachResult, ...]] = []
    deferral = await _plan(
        manifest,
        demand,
        _real_wirer(manifest, engine, bus),
        prepare=_real_preparer(manifest, engine, bus),
    )
    imports: list[tuple[str, str]] = []

    def _import(module_path: str, class_name: str) -> type:
        imports.append((module_path, class_name))
        if len(imports) == 1:
            return _fake_handler_cls()  # the in-use contract wires at boot
        raise ImportError("handler module is gone (test)")

    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        side_effect=_import,
    ):
        wired_report = await wire_from_manifest(
            deferral.wired_manifest,
            engine,
            event_bus=bus,
            environment="local",
            subscribe_immediately=False,
        )
        assert wired_report.total_wired == 1
        assert await deferral.verify_strict(wired_report) is None

        engine.freeze()
        results: list[ModelContractAttachResult] = []
        exclusions: list[ModelContractAttachExclusion] = []
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
            on_attempt=attempts.append,
        )

        # The boot refused nothing: the broken handler surfaces only when the
        # deferred contract's first request runs.
        assert demand.on_first_record is not None
        await demand.on_first_record(IDLE)
        await _ticks_until(lambda: bool(attempts))

    assert bus.joins == [(ACTIVE, None)]
    assert engine.dispatcher_count == 1
    failed = [
        result
        for batch in attempts
        for result in batch
        if result.contract_name == "node_idle"
    ]
    assert failed
    assert all(result.status is EnumContractAttachStatus.FAILED for result in failed)
    assert all("ImportError" in result.detail for result in failed)
    assert imports == [(FAKE_MODULE, FAKE_HANDLER)] * 2


@pytest.mark.asyncio
async def test_strict_dispatcher_coverage_runs_on_deferred_orchestrators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ONEX_STRICT_DISPATCHER_COVERAGE covers deferred orchestrators too: a
    start topic whose dispatcher cannot even be prepared refuses the boot."""
    monkeypatch.delenv("ONEX_WIRING_STRICT_MODE", raising=False)
    monkeypatch.setenv("ONEX_STRICT_DISPATCHER_COVERAGE", "1")
    start_topic = "onex.cmd.test.orchestration-start.v1"
    # _contract already declares node_type="ORCHESTRATOR_GENERIC".
    manifest = ModelAutoWiringManifest(
        contracts=(
            _contract("node_in_use", (ACTIVE,)),
            _contract("node_orchestrator_idle", (start_topic,)),
        )
    )
    engine = MessageDispatchEngine()
    bus = RecordingKafkaBus()
    deferral = await _plan(
        manifest,
        FakeDemandSource(frozenset({ACTIVE})),
        None,
        prepare=_real_preparer(manifest, engine, bus),
    )
    assert deferral.deferred_names == {"node_orchestrator_idle"}

    imports: list[tuple[str, str]] = []

    def _import(module_path: str, class_name: str) -> type:
        imports.append((module_path, class_name))
        if len(imports) == 1:
            return _fake_handler_cls()  # the in-use contract wires at boot
        raise ImportError("handler module is gone (test)")

    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        side_effect=_import,
    ):
        wired_report = await wire_from_manifest(
            deferral.wired_manifest,
            engine,
            event_bus=bus,
            environment="local",
            subscribe_immediately=False,
        )
        assert wired_report.total_wired == 1
        with pytest.raises(ModelOnexError) as excinfo:
            await deferral.verify_strict(wired_report)

    assert "missing dispatcher" in str(excinfo.value)
    assert "node_orchestrator_idle" in str(excinfo.value)
    assert "orchestration-start" in str(excinfo.value)


@pytest.mark.asyncio
async def test_core_runtime_topics_keep_their_contract_wired_but_defer_the_rest() -> (
    None
):
    """A contract subscribing a core-runtime topic stays in the wired manifest;
    only the contracts off the core loop defer to their first request."""
    other_idle = "onex.cmd.test.other-idle.v1"
    manifest = ModelAutoWiringManifest(
        contracts=(
            _contract("node_a", (ACTIVE,)),
            _contract("node_core", (IDLE,)),
            _contract("node_other", (other_idle,)),
        )
    )

    deferral = await _plan(
        manifest,
        FakeDemandSource(frozenset({ACTIVE})),
        None,
        core_runtime_topics=frozenset({IDLE}),
    )

    assert deferral.deferred_names == {"node_other"}
    assert [c.name for c in deferral.wired_manifest.contracts] == [
        "node_a",
        "node_core",
    ]

    # Control: without core_runtime_topics both idle contracts defer.
    control = await _plan(manifest, FakeDemandSource(frozenset({ACTIVE})), None)
    assert control.deferred_names == {"node_core", "node_other"}


@pytest.mark.asyncio
async def test_error_chain_a_preflight_that_raises_is_a_failed_row() -> None:
    """The real wirer turns a preflight refusal into a FAILED row, not a crash."""
    manifest = ModelAutoWiringManifest(contracts=(_contract("node_idle", (IDLE,)),))
    engine = MessageDispatchEngine()
    engine.freeze()
    wire = _real_wirer(manifest, engine, RecordingKafkaBus())
    with (
        patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
            return_value=_fake_handler_cls(),
        ),
        patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring"
            "._preflight_prepared_registration_ids",
            side_effect=ModelOnexError("duplicate prepared dispatcher IDs (test)"),
        ),
    ):
        row = await wire(manifest.contracts[0])

    assert row.outcome is EnumWiringOutcome.FAILED
    assert "ModelOnexError" in row.reason
    assert "duplicate prepared dispatcher IDs (test)" in row.reason
    assert engine.dispatcher_count == 0


@pytest.mark.asyncio
async def test_error_chain_a_commit_that_raises_is_a_failed_row() -> None:
    """A commit that raises after nothing registered is a FAILED row too: the
    engine has no rollback, so the row is what the next attempt reads."""
    manifest = ModelAutoWiringManifest(contracts=(_contract("node_idle", (IDLE,)),))
    engine = MessageDispatchEngine()
    engine.freeze()
    wire = _real_wirer(manifest, engine, RecordingKafkaBus())

    async def _commit_raises(*args: object, **kwargs: object) -> object:
        raise RuntimeError("commit exploded (test)")

    with (
        patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
            return_value=_fake_handler_cls(),
        ),
        patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._commit_contract_wiring",
            new=_commit_raises,
        ),
    ):
        row = await wire(manifest.contracts[0])

    assert row.outcome is EnumWiringOutcome.FAILED
    assert "RuntimeError" in row.reason
    assert "commit exploded (test)" in row.reason
    assert engine.dispatcher_count == 0
