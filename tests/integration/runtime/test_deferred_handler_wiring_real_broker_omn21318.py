# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21318: a demand-deferred contract's handler is wired by its first record.

Real Redpanda broker (testcontainers); the demand measure, the wiring split and
the subscription seam are the production classes. The seam under test is the
one between the boot wiring pass and the demand split (OMN-18087): the broker
is read, not the code, for which contracts cost nothing at boot.

Golden chain: at boot the in-use contract's handler module is imported and the
deferred contract's is not (its wiring row is DEFERRED, its attach exclusion
DEMAND_DEFERRED); the first record on the deferred contract's subscribe topic
imports the module exactly once and its consumer group joins from earliest, so
that record is the first one its group reads. Error chain: topics the broker
cannot show count as in use, so nothing defers; a FAILED first-request wiring
retries within its attempt budget and shows the readiness gate not ready until
it attaches; a broker that cannot be read at all refuses the boot.
"""

from __future__ import annotations

import asyncio
import socket
import time
import uuid
from collections.abc import (
    AsyncGenerator,
    Awaitable,
    Callable,
    Iterator,
    Sequence,
)
from pathlib import Path
from unittest.mock import patch

import pytest
from aiokafka import AIOKafkaProducer
from aiokafka.admin import AIOKafkaAdminClient, NewTopic
from aiokafka.errors import KafkaConnectionError, KafkaError
from testcontainers.core.container import DockerContainer
from testcontainers.core.docker_client import DockerClient

from omnibase_infra.enums.enum_consumer_group_purpose import EnumConsumerGroupPurpose
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
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.event_bus.service_topic_manager import TopicProvisioner
from omnibase_infra.handlers.boot_wiring_deferral import (
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
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
    ModelHandlerRef,
    ModelHandlerRouting,
    ModelHandlerRoutingEntry,
)
from omnibase_infra.runtime.auto_wiring.report import (
    EnumWiringOutcome,
    ModelContractWiringResult,
)
from omnibase_infra.runtime.health.contract_attach_readiness_gate import (
    ContractAttachReadinessGate,
    derive_required_contract_names,
)
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine
from omnibase_infra.runtime.models.model_contract_attach_gate_status import (
    ModelContractAttachGateStatus,
)
from omnibase_infra.topics.topic_namespace import apply_topic_namespace
from omnibase_infra.utils import compute_consumer_group_id
from tests.helpers.kafka_demand_source_container import (
    demand_source_for,
    demand_source_over_bus,
)

pytestmark = [pytest.mark.integration, pytest.mark.kafka]

BROKER_IMAGE = "redpandadata/redpanda:v24.2.7"
BROKER_READY_SECONDS = 90.0
DEMAND_START_SECONDS = 60.0
HANDLED_ONCE_SETTLE_SECONDS = 3.0
BUS_ENVIRONMENT = "local"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


async def _wait_for_broker(bootstrap: str) -> None:
    deadline = time.monotonic() + BROKER_READY_SECONDS
    last: Exception | None = None
    while time.monotonic() < deadline:
        admin = AIOKafkaAdminClient(bootstrap_servers=bootstrap)
        try:
            await admin.start()
            await admin.describe_cluster()
            return
        except (KafkaError, OSError) as exc:
            last = exc
            await asyncio.sleep(1.0)
        finally:
            await admin.close()
    raise AssertionError(f"broker at {bootstrap} not ready: {last!r}")


@pytest.fixture(scope="module")
def broker_bootstrap() -> Iterator[str]:
    """A throwaway Redpanda broker on a free port, advertised at that port."""
    port = _free_port()
    host = DockerClient().host()
    container = (
        DockerContainer(BROKER_IMAGE)
        .with_bind_ports(port, port)
        .with_command(
            "redpanda start "
            f"--kafka-addr PLAINTEXT://0.0.0.0:{port} "
            f"--advertise-kafka-addr PLAINTEXT://{host}:{port} "
            "--mode dev-container --smp 1 --memory 1G --reserve-memory 0M "
            "--reactor-backend=epoll --default-log-level=warn"
        )
    )
    with container:
        bootstrap = f"{host}:{port}"
        asyncio.run(_wait_for_broker(bootstrap))
        yield bootstrap


@pytest.fixture
async def bus(broker_bootstrap: str) -> AsyncGenerator[EventBusKafka, None]:
    started = EventBusKafka(
        config=ModelKafkaEventBusConfig(
            bootstrap_servers=broker_bootstrap,
            environment=BUS_ENVIRONMENT,
            timeout_seconds=30,
            max_retry_attempts=0,
        )
    )
    await started.start()
    yield started
    await started.close()


class _Names:
    """Topic names of one scenario; the suffix keeps scenarios apart on one broker."""

    def __init__(self) -> None:
        suffix = uuid.uuid4().hex[:8]
        self.in_use_subscribe = f"onex.evt.platform.hotload-rb-in-use-{suffix}.v1"
        self.in_use_publish = f"onex.evt.platform.hotload-rb-in-use-out-{suffix}.v1"
        self.deferred_subscribe = f"onex.cmd.platform.hotload-rb-deferred-{suffix}.v1"
        self.deferred_publish = f"onex.evt.platform.hotload-rb-deferred-out-{suffix}.v1"


async def _broker_topics(bootstrap: str) -> set[str]:
    admin = AIOKafkaAdminClient(bootstrap_servers=bootstrap)
    await admin.start()
    try:
        return set(await admin.list_topics())
    finally:
        await admin.close()


async def _create_topic(bootstrap: str, logical: str) -> None:
    admin = AIOKafkaAdminClient(bootstrap_servers=bootstrap)
    await admin.start()
    try:
        await admin.create_topics(
            [
                NewTopic(
                    name=apply_topic_namespace(logical),
                    num_partitions=1,
                    replication_factor=1,
                )
            ]
        )
    finally:
        await admin.close()


async def _write_record(bootstrap: str, logical: str) -> None:
    producer = AIOKafkaProducer(bootstrap_servers=bootstrap)
    await producer.start()
    try:
        await producer.send_and_wait(
            apply_topic_namespace(logical), key=b"k", value=b"{}"
        )
    finally:
        await producer.stop()


_HANDLED: list[object] = []
"""Every envelope a deferred contract's handler was called with (cleared per test)."""


class _FakeHandler:
    async def handle(self, envelope: object) -> None:
        _HANDLED.append(envelope)


def _contract(
    name: str,
    subscribe: tuple[str, ...],
    publish: tuple[str, ...],
    module: str = "fake.module",
) -> ModelDiscoveredContract:
    routing = (
        ModelHandlerRouting(
            routing_strategy="payload_type_match",
            handlers=(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(name="FakeHandler", module=module),
                    event_model=None,
                    operation=None,
                ),
            ),
        )
        if subscribe
        else None
    )
    return ModelDiscoveredContract(
        name=name,
        node_type="ORCHESTRATOR_GENERIC" if subscribe else "EFFECT_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path(f"/fake/{name}/contract.yaml"),
        entry_point_name=name,
        package_name="test-package",
        event_bus=ModelEventBusWiring(
            subscribe_topics=subscribe, publish_topics=publish
        ),
        handler_routing=routing,
    )


def _manifest(names: _Names) -> ModelAutoWiringManifest:
    """The boot manifest: one in-use subscriber, one demand-deferred subscriber.

    The two handler modules differ so the import recorder can tell the contract
    wired at boot from the one wired on its first request.
    """
    return ModelAutoWiringManifest(
        contracts=(
            _contract(
                "node_in_use", (names.in_use_subscribe,), (names.in_use_publish,)
            ),
            _contract(
                "node_deferred",
                (names.deferred_subscribe,),
                (names.deferred_publish,),
                module="fake.deferred_module",
            ),
        )
    )


def _recorder(imported_modules: list[str]) -> Callable[[str, str], type[_FakeHandler]]:
    """A stand-in for ``_import_handler_class`` that records every module asked for."""

    def _recording_import(module_path: str, class_name: str) -> type[_FakeHandler]:
        imported_modules.append(module_path)
        return _FakeHandler

    return _recording_import


def _fail_first_wirer(
    wirer: DeferredContractWirer, calls: list[str]
) -> DeferredContractWirer:
    """A wirer whose first call returns a FAILED row; every later call delegates.

    A FAILED wiring row is not structural (OMN-21318): the demand start forgets
    it and its attempt budget retries, so the same first record still wires.
    """

    async def wire(contract: ModelDiscoveredContract) -> ModelContractWiringResult:
        calls.append(contract.name)
        if len(calls) == 1:
            return ModelContractWiringResult(
                contract_name=contract.name,
                package_name=contract.package_name,
                outcome=EnumWiringOutcome.FAILED,
                reason="the first request found the contract unprepared",
            )
        return await wirer(contract)

    return wire


@pytest.fixture(autouse=True)
def _lane_declares_on_demand(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The lane's overlay row declares on_demand (OMN-21318); a control removes it."""
    root = tmp_path / "overlays"
    (root / "runtime_contract_hotload").mkdir(parents=True)
    (root / "runtime_contract_hotload" / "overlay.yaml").write_text(
        "schema_version: 1\nlanes:\n  dogfood:\n    enabled: true\n"
        "    mode: on_demand\n    allowed_environments: [local]\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ONEX_SKILL_OVERLAY_ROOTS", str(root))
    monkeypatch.setenv("ONEX_RUNTIME_LANE", "dogfood")


async def _prepare(bootstrap: str, names: _Names) -> None:
    """In-use: its subscribe topic holds a record. Deferred: its topic is empty."""
    await _create_topic(bootstrap, names.in_use_subscribe)
    await _write_record(bootstrap, names.in_use_subscribe)
    await _create_topic(bootstrap, names.deferred_subscribe)


async def test_golden_the_first_record_wires_the_deferred_contract_and_its_group_reads_it_once(
    broker_bootstrap: str, bus: EventBusKafka, tmp_path: Path
) -> None:
    _HANDLED.clear()
    names = _Names()
    await _prepare(broker_bootstrap, names)
    provisioner = TopicProvisioner(
        bootstrap_servers=broker_bootstrap, contracts_root=tmp_path
    )
    manifest = _manifest(names)
    engine = MessageDispatchEngine()
    demand_source = demand_source_over_bus(bus)
    exclusions: list[ModelContractAttachExclusion] = []
    stops: list[Callable[[], Awaitable[None]]] = []
    imported_modules: list[str] = []

    try:
        with patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
            side_effect=_recorder(imported_modules),
        ):
            # The boot order the kernel follows (OMN-21318): measure demand on
            # the real broker and name the deferred contracts, wire the rest,
            # freeze the engine, subscribe the booted ones, then arm the first
            # request of each deferred one.
            readiness = ModelTopicReadinessConfig(max_concurrent_contract_attach=1)
            deferral = await plan_boot_wiring_deferral(
                manifest,
                demand_source=demand_source,
                readiness_config=readiness,
                result_appliers_by_contract=None,
                wire=make_deferred_contract_wirer(
                    manifest=manifest,
                    dispatch_engine=engine,
                    event_bus=bus,
                    environment=BUS_ENVIRONMENT,
                    container=None,
                    result_appliers_by_contract=None,
                    materialized_explicit_dependencies=None,
                    topology=None,
                    wiring=handler_wiring,
                ),
                wiring=handler_wiring,
            )
            assert deferral.deferred_names == frozenset({"node_deferred"})
            wired_report = await wire_from_manifest(
                deferral.wired_manifest,
                engine,
                event_bus=bus,
                environment=BUS_ENVIRONMENT,
                subscribe_immediately=False,
            )
            engine.freeze()
            await subscribe_wired_contract_topics(
                manifest=deferral.wired_manifest,
                report=wired_report,
                dispatch_engine=engine,
                event_bus=bus,
                environment=BUS_ENVIRONMENT,
                provisioner=provisioner,
                readiness_config=readiness,
                exclusions_out=exclusions,
            )
            await deferral.watch(
                dispatch_engine=engine,
                event_bus=bus,
                environment=BUS_ENVIRONMENT,
                provisioner=provisioner,
                exclusions_out=exclusions,
                stops_out=stops,
            )

            # At boot the deferred contract was named, not wired: its module
            # was never imported, and it is reported as watched, not attached.
            assert "fake.deferred_module" not in imported_modules
            assert [
                e.contract_name
                for e in exclusions
                if e.reason is EnumContractAttachExclusionReason.DEMAND_DEFERRED
            ] == ["node_deferred"]
            assert wired_report.total_wired == 1

            await _write_record(broker_bootstrap, names.deferred_subscribe)
            deadline = time.monotonic() + DEMAND_START_SECONDS
            while time.monotonic() < deadline:
                if "fake.deferred_module" in imported_modules:
                    break
                await asyncio.sleep(0.25)
            else:
                raise AssertionError(
                    "the first record never wired the deferred contract; "
                    f"imported_modules={imported_modules!r}"
                )
            assert imported_modules.count("fake.deferred_module") == 1

            # The record that started the contract is the first its group
            # reads: its handler is called for it, once, and not again.
            deadline = time.monotonic() + DEMAND_START_SECONDS
            while time.monotonic() < deadline and not _HANDLED:
                await asyncio.sleep(0.25)
            assert len(_HANDLED) == 1, (
                "the deferred handler did not handle the first record"
            )
            await asyncio.sleep(HANDLED_ONCE_SETTLE_SECONDS)
            assert len(_HANDLED) == 1, "the first record was handled more than once"
    finally:
        for stop in stops:
            await stop()


async def test_error_chain_a_failed_first_request_shows_not_ready_then_attaches_on_the_real_broker(
    broker_bootstrap: str,
    bus: EventBusKafka,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _HANDLED.clear()
    names = _Names()
    await _prepare(broker_bootstrap, names)
    provisioner = TopicProvisioner(
        bootstrap_servers=broker_bootstrap, contracts_root=tmp_path
    )
    manifest = _manifest(names)
    engine = MessageDispatchEngine()
    demand_source = demand_source_over_bus(bus)
    exclusions: list[ModelContractAttachExclusion] = []
    stops: list[Callable[[], Awaitable[None]]] = []
    imported_modules: list[str] = []
    wire_calls: list[str] = []
    gate = ContractAttachReadinessGate(derive_required_contract_names(manifest))
    if not gate.required_contract_names:
        # The contracts' topics stopped being command topics: the gate then
        # requires nothing, so pin the requirement to the deferred contract
        # for the chain this test follows.
        gate = ContractAttachReadinessGate({"node_deferred"})
    gate_snapshots: list[ModelContractAttachGateStatus] = []

    def on_attempt(results: Sequence[ModelContractAttachResult]) -> None:
        gate.record(results)
        gate_snapshots.append(gate.status())

    # The failed first request must retry within the same first record's
    # attempt budget, not a minute later.
    monkeypatch.setattr(handler_wiring, "DEFAULT_NOT_READY_RETRY_BACKOFF_SECONDS", 0)

    try:
        with patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
            side_effect=_recorder(imported_modules),
        ):
            # The boot order the kernel follows, with the gate folded in: the
            # watch reports every attach attempt to it, FAILED ones included.
            readiness = ModelTopicReadinessConfig(max_concurrent_contract_attach=1)
            deferral = await plan_boot_wiring_deferral(
                manifest,
                demand_source=demand_source,
                readiness_config=readiness,
                result_appliers_by_contract=None,
                wire=_fail_first_wirer(
                    make_deferred_contract_wirer(
                        manifest=manifest,
                        dispatch_engine=engine,
                        event_bus=bus,
                        environment=BUS_ENVIRONMENT,
                        container=None,
                        result_appliers_by_contract=None,
                        materialized_explicit_dependencies=None,
                        topology=None,
                        wiring=handler_wiring,
                    ),
                    wire_calls,
                ),
                wiring=handler_wiring,
            )
            assert deferral.deferred_names == frozenset({"node_deferred"})
            wired_report = await wire_from_manifest(
                deferral.wired_manifest,
                engine,
                event_bus=bus,
                environment=BUS_ENVIRONMENT,
                subscribe_immediately=False,
            )
            engine.freeze()
            await subscribe_wired_contract_topics(
                manifest=deferral.wired_manifest,
                report=wired_report,
                dispatch_engine=engine,
                event_bus=bus,
                environment=BUS_ENVIRONMENT,
                provisioner=provisioner,
                readiness_config=readiness,
                exclusions_out=exclusions,
            )
            await deferral.watch(
                dispatch_engine=engine,
                event_bus=bus,
                environment=BUS_ENVIRONMENT,
                provisioner=provisioner,
                exclusions_out=exclusions,
                stops_out=stops,
                on_attempt=on_attempt,
            )

            await _write_record(broker_bootstrap, names.deferred_subscribe)
            deadline = time.monotonic() + DEMAND_START_SECONDS
            while time.monotonic() < deadline:
                if len(wire_calls) >= 2:
                    break
                await asyncio.sleep(0.25)
            else:
                raise AssertionError(
                    "a FAILED first-request wiring was never retried; "
                    f"wire_calls={wire_calls!r}"
                )

            # The retried wiring attaches before the group joins from earliest,
            # so the same first record reaches the handler once the gate is
            # ready again; anything asserted after it arrived is settled.
            deadline = time.monotonic() + DEMAND_START_SECONDS
            while time.monotonic() < deadline and not _HANDLED:
                await asyncio.sleep(0.25)
            shown = [snapshot.summary_line() for snapshot in gate_snapshots]
            assert any(
                not snapshot.ready and "node_deferred" in snapshot.failed_contracts
                for snapshot in gate_snapshots
            ), (
                "the FAILED first request never showed the gate not ready; "
                f"statuses={shown!r}"
            )
            assert gate.status().ready is True, (
                "the retried wiring never attached; "
                f"status={gate.status().summary_line()!r}"
            )
            assert imported_modules.count("fake.deferred_module") == 1
            assert len(_HANDLED) == 1, (
                "the deferred handler did not handle the first record"
            )
            await asyncio.sleep(HANDLED_ONCE_SETTLE_SECONDS)
            assert len(_HANDLED) == 1, "the first record was handled more than once"
            assert wire_calls == ["node_deferred", "node_deferred"], (
                "the failed first request was retried the wrong number of times"
            )
    finally:
        for stop in stops:
            await stop()


async def test_error_chain_topics_the_broker_cannot_show_defer_nothing(
    broker_bootstrap: str, bus: EventBusKafka
) -> None:
    names = _Names()
    # No topic of the manifest is ever created: every topic is unreadable, so
    # every topic counts as in use and the boot path is kept whole.
    manifest = _manifest(names)
    engine = MessageDispatchEngine()
    demand_source = demand_source_over_bus(bus)
    imported_modules: list[str] = []

    deferral = await plan_boot_wiring_deferral(
        manifest,
        demand_source=demand_source,
        readiness_config=ModelTopicReadinessConfig(),
        result_appliers_by_contract=None,
        wire=None,
        wiring=handler_wiring,
    )
    assert deferral.deferred_names == frozenset()
    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        side_effect=_recorder(imported_modules),
    ):
        report = await wire_from_manifest(
            deferral.wired_manifest,
            engine,
            event_bus=bus,
            environment=BUS_ENVIRONMENT,
            subscribe_immediately=False,
        )
    assert report.total_wired == 2
    assert set(imported_modules) == {"fake.module", "fake.deferred_module"}


async def test_error_chain_an_unreachable_broker_refuses_the_boot() -> None:
    names = _Names()
    manifest = _manifest(names)
    # Nothing listens on the port: the demand measure cannot read the broker
    # at all, and the boot is refused instead of deferring anything.
    demand_source = demand_source_for(f"127.0.0.1:{_free_port()}")
    with pytest.raises(KafkaConnectionError):
        async with asyncio.timeout(60):
            await plan_boot_wiring_deferral(
                manifest,
                demand_source=demand_source,
                readiness_config=ModelTopicReadinessConfig(),
                result_appliers_by_contract=None,
                wire=None,
                wiring=handler_wiring,
            )
