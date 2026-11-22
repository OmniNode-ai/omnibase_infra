# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A dormant contract starts on its first request, once, through the front door (OMN-21592).

Plan task A4 (contract hot-loading plan, sections 3.11.5 and 3.11.9). Twenty
fixture contracts carry an import sentinel in their handler module. A lane
whose hot-load overlay row is ``on_demand`` with a ``lazy_start`` block boots
only the front door and the declared boot-priority set; the rest stay dormant
until a request for them reaches the front door, which builds, provisions,
confirms and attaches the node in that order and then delivers the request to
the node's own dispatcher.

RED at the merge base: ``omnibase_infra.handlers.lazy_start_boot_split``
and ``lazy_start_front_door`` do not exist there, and the lane row
has no ``lazy_start`` key (``extra="forbid"`` refuses it), so every test fails
at collection or at the overlay read.
"""

from __future__ import annotations

import asyncio
import sys
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import cast
from uuid import UUID, uuid4

import pytest
import yaml

from omnibase_core.container import ModelONEXContainer
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.protocols.event_bus.protocol_event_bus_subscriber import (
    ProtocolEventBusSubscriber,
)
from omnibase_infra.enums.enum_lazy_start_outcome import EnumLazyStartOutcome
from omnibase_infra.errors import InfraUnavailableError
from omnibase_infra.event_bus.enum_topic_readiness_status import (
    EnumTopicReadinessStatus,
)
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.model_topic_readiness_config import (
    ModelTopicReadinessConfig,
)
from omnibase_infra.event_bus.model_topic_set_readiness import ModelTopicSetReadiness
from omnibase_infra.event_bus.models.model_event_headers import ModelEventHeaders
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.handlers import lazy_start_boot_split as lazy
from omnibase_infra.handlers.handler_lazy_start_ingress import HandlerLazyStartIngress
from omnibase_infra.handlers.lazy_start_boot_split import (
    StoreLazyStartDormant,
    boot_manifest,
    boot_universe_warm_enabled,
)
from omnibase_infra.handlers.lazy_start_front_door import LazyStartFrontDoor
from omnibase_infra.handlers.lazy_start_runtime_transaction import (
    LazyStartRuntimeTransaction,
)
from omnibase_infra.models.lazy_start import (
    ModelLazyStartPolicy,
    ModelLazyStartRequest,
)
from omnibase_infra.protocols.protocol_lazy_start_bus import (
    ProtocolLazyStartBus,
)
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts_from_paths
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
from omnibase_infra.runtime.auto_wiring.report import EnumWiringOutcome
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

pytestmark = [pytest.mark.unit]

FIXTURE_COUNT = 20
INGRESS = "node_lazy_fixture_ingress"
BOOT_PRIORITY = "node_lazy_fixture_00"
TARGET = "node_lazy_fixture_07"
SENTINEL_MODULE = "lazy_start_fixture_sentinel"


def _name(i: int) -> str:
    return f"node_lazy_fixture_{i:02d}"


def _cmd(name: str) -> str:
    return f"onex.cmd.platform.{name.replace('_', '-')}.v1"


def _cmd_b(name: str) -> str:
    return f"onex.cmd.platform.{name.replace('_', '-')}-b.v1"


def _evt(name: str) -> str:
    return f"onex.evt.platform.{name.replace('_', '-')}-done.v1"


A_ORCH = "node_wf_a_orch"
A_FX = "node_wf_z_fx"
SHARED_P = "node_wf_shared_p"
SHARED_Q = "node_wf_shared_q"
B_ORCH = "node_wf_b_orch"
B_GX = "node_wf_b_gx"
B_DEEP = "node_wf_zz_deep"
OUTSIDE = "node_wf_outside"

# name -> (subscribe topics, publish topics). Written as contract.yaml files and read
# back through the runtime's own discovery; the closure code sees only those files.
# Workflow A's orchestrator publishes to three contracts; workflow B's publishes to
# the same two shared ones and to one that publishes onward (a transitive hop).
CLOSURE_SPEC: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    A_ORCH: ((_cmd(A_ORCH),), (_cmd(A_FX), _cmd(SHARED_P), _cmd(SHARED_Q))),
    A_FX: ((_cmd(A_FX),), ()),
    SHARED_P: ((_cmd(SHARED_P),), (_evt(SHARED_P),)),
    SHARED_Q: ((_cmd(SHARED_Q),), ()),
    B_ORCH: ((_cmd(B_ORCH),), (_cmd(SHARED_P), _cmd(SHARED_Q), _cmd(B_GX))),
    B_GX: ((_cmd(B_GX),), (_cmd(B_DEEP),)),
    B_DEEP: ((_cmd(B_DEEP),), ()),
    OUTSIDE: ((_cmd(OUTSIDE),), ()),
}
WORKFLOW_A = {A_ORCH, A_FX, SHARED_P, SHARED_Q}
WORKFLOW_B = {B_ORCH, B_GX, B_DEEP, SHARED_P, SHARED_Q}


@pytest.fixture
def fixture_handlers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Twenty handler modules; importing one appends its name to the sentinel."""
    pkg = tmp_path / "lazyfix"
    pkg.mkdir()
    (tmp_path / f"{SENTINEL_MODULE}.py").write_text(
        "IMPORTED: list[str] = []\nHANDLED: list[str] = []\nEMIT: set[str] = set()\n",
        encoding="utf-8",
    )
    for name in [INGRESS, *(_name(i) for i in range(FIXTURE_COUNT)), *CLOSURE_SPEC]:
        (pkg / f"{name}.py").write_text(
            "from pydantic import BaseModel\n"
            f"import {SENTINEL_MODULE} as _s\n"
            f"_s.IMPORTED.append({name!r})\n\n\n"
            "class FixtureRequest(BaseModel):\n"
            '    ask: str = ""\n\n\n'
            "class FixtureHandler:\n"
            "    async def handle(self, envelope: object) -> FixtureRequest | None:\n"
            f"        _s.HANDLED.append({name!r})\n"
            f"        return FixtureRequest(ask='hop') if {name!r} in _s.EMIT else None\n",
            encoding="utf-8",
        )
    monkeypatch.syspath_prepend(str(tmp_path))
    for mod in [
        m for m in sys.modules if m.startswith("lazyfix") or m == SENTINEL_MODULE
    ]:
        monkeypatch.delitem(sys.modules, mod)
    sentinel = __import__(SENTINEL_MODULE)
    sentinel.IMPORTED.clear()
    sentinel.HANDLED.clear()
    sentinel.EMIT.clear()
    return cast("list[str]", sentinel.IMPORTED)


def _sentinel() -> ModuleType:
    return sys.modules[SENTINEL_MODULE]


def _contract(name: str, *, two_topics: bool = False) -> ModelDiscoveredContract:
    subscribe = (_cmd(name), _cmd_b(name)) if two_topics else (_cmd(name),)
    return ModelDiscoveredContract(
        name=name,
        node_type="EFFECT_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path(f"/fixture/{name}/contract.yaml"),
        entry_point_name=name,
        package_name="lazy-fixture",
        event_bus=ModelEventBusWiring(
            subscribe_topics=subscribe, publish_topics=(_evt(name),)
        ),
        handler_routing=ModelHandlerRouting(
            routing_strategy="payload_type_match",
            handlers=(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(
                        name="FixtureHandler", module=f"lazyfix.{name}"
                    ),
                    event_model=ModelHandlerRef(
                        name="FixtureRequest", module=f"lazyfix.{name}"
                    ),
                    operation=None,
                ),
            ),
        ),
    )


def _manifest() -> ModelAutoWiringManifest:
    return ModelAutoWiringManifest(
        contracts=(
            _contract(INGRESS),
            *(_contract(_name(i), two_topics=True) for i in range(FIXTURE_COUNT)),
        )
    )


@pytest.fixture(autouse=True)
def lane_overlay(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """The lane's hot-load row: on_demand, with the boot set as data."""
    root = tmp_path / "overlays"
    (root / "runtime_contract_hotload").mkdir(parents=True)
    (root / "runtime_contract_hotload" / "overlay.yaml").write_text(
        "schema_version: 1\nlanes:\n  dogfood:\n    enabled: true\n"
        "    mode: on_demand\n    allowed_environments: [local]\n"
        "    lazy_start:\n"
        f"      ingress_contract: {INGRESS}\n"
        f"      boot_contracts: [{BOOT_PRIORITY}]\n"
        "      budget_seconds: 30\n      max_retries: 0\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ONEX_SKILL_OVERLAY_ROOTS", str(root))
    monkeypatch.setenv("ONEX_RUNTIME_LANE", "dogfood")
    monkeypatch.delenv("ONEX_BOOT_UNIVERSE_PROVISION", raising=False)
    StoreLazyStartDormant.clear()
    return root


class CallLog(list[tuple[str, str]]):
    """One ordered log of provision, confirm, attach, detach and deliver."""


class RecordingProvisioner:
    def __init__(self, log: CallLog) -> None:
        self.log = log

    async def ensure_topic_exists(
        self,
        topic_name: str,
        spec: object | None = None,
        correlation_id: UUID | None = None,
    ) -> bool:
        self.log.append(("provision", topic_name))
        return True

    async def confirm_topics_ready(
        self,
        topics: Sequence[str],
        *,
        expected_specs: object | None = None,
        config: ModelTopicReadinessConfig | None = None,
        correlation_id: UUID | None = None,
    ) -> ModelTopicSetReadiness:
        self.log.extend(("confirm", t) for t in topics)
        return ModelTopicSetReadiness(
            topics=tuple(topics),
            status=EnumTopicReadinessStatus.READY,
            ready_topics=tuple(topics),
            attempts=1,
        )


class RecordingBus(EventBusKafka):
    """Kafka-typed stand-in; never opens a transport, records every subscription."""

    def __init__(self, log: CallLog, *, fail_attach: set[str] | None = None) -> None:
        self._log = log
        self._started = False
        self._environment = "local"
        self.fail_attach = set(fail_attach or ())
        self.live: set[str] = set()
        self.attach_delay = 0.0
        self.topic_delay: dict[str, float] = {}
        self.consumers: dict[str, list[Callable[..., Awaitable[None]]]] = {}
        self.published: list[str] = []

    async def publish(
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
        headers: ModelEventHeaders | None = None,
    ) -> None:
        """The bus's one publish: record it, then hand it to every live subscription."""
        self._log.append(("publish", topic))
        self.published.append(topic)
        assert headers is not None
        message = ModelEventMessage(topic=topic, key=key, value=value, headers=headers)
        for on_message in tuple(self.consumers.get(topic, ())):
            await on_message(message)

    def get_consumer_groups(self) -> dict[tuple[str, str], str]:
        return {}

    async def subscribe(
        self,
        *,
        topic: str,
        node_identity: object,
        on_message: object,
        auto_offset_reset: str | None = None,
    ) -> object:
        if self.attach_delay or topic in self.topic_delay:
            await asyncio.sleep(self.attach_delay + self.topic_delay.get(topic, 0.0))
        if topic in self.fail_attach:
            self.fail_attach.discard(topic)
            self._log.append(("attach_failed", topic))
            raise ConnectionError(f"join refused for {topic}")
        self._log.append(("attach", topic))
        self.live.add(topic)
        callback = cast("Callable[..., Awaitable[None]]", on_message)
        self.consumers.setdefault(topic, []).append(callback)

        async def _stop() -> None:
            self._log.append(("detach", topic))
            self.live.discard(topic)
            self.consumers[topic].remove(callback)

        return _stop


class CountingTransaction(LazyStartRuntimeTransaction):
    """The runtime transaction, counting builds and logging deliveries."""

    def __init__(
        self,
        log: CallLog,
        contracts: dict[str, ModelDiscoveredContract],
        bus: RecordingBus,
    ) -> None:
        super().__init__(
            contracts,
            cast("ProtocolLazyStartBus", bus),
            "local",
            RecordingProvisioner(log),
            None,
            ModelTopicReadinessConfig(),
        )
        self.log = log
        self.builds = 0
        self.built: Counter[str] = Counter()
        self.rolled_back: list[str] = []

    async def build(self, contract_name: str) -> None:
        self.builds += 1
        self.built[contract_name] += 1
        await super().build(contract_name)

    async def rollback(self, contract_name: str) -> None:
        self.rolled_back.append(contract_name)
        await super().rollback(contract_name)

    async def deliver(self, contract_name: str, topic: str, envelope: object) -> object:
        self.log.append(("deliver", topic))
        return await super().deliver(contract_name, topic, envelope)


def _front_door(
    log: CallLog,
    bus: RecordingBus,
    *,
    budget: float = 30.0,
    retries: int = 0,
) -> tuple[LazyStartFrontDoor, CountingTransaction]:
    _, dormant = _booted()
    return _door_over({c.name: c for c in dormant}, log, bus, budget, retries)


def _door_over(
    contracts: dict[str, ModelDiscoveredContract],
    log: CallLog,
    bus: RecordingBus,
    budget: float = 30.0,
    retries: int = 0,
    wired: dict[str, ModelDiscoveredContract] | None = None,
) -> tuple[LazyStartFrontDoor, CountingTransaction]:
    transaction = CountingTransaction(log, contracts, bus)
    policy = ModelLazyStartPolicy(
        ingress_contract=INGRESS, budget_seconds=budget, max_retries=retries
    )
    return LazyStartFrontDoor(transaction, contracts, policy, wired), transaction


def _booted() -> tuple[ModelAutoWiringManifest, tuple[ModelDiscoveredContract, ...]]:
    boot = boot_manifest(_manifest())
    return boot, tuple(StoreLazyStartDormant.contracts().values())


def _request(name: str = TARGET) -> ModelLazyStartRequest:
    envelope = ModelEventEnvelope[object](
        payload={"ask": "first"}, correlation_id=uuid4(), event_type="FixtureRequest"
    )
    return ModelLazyStartRequest(
        target_contract=name,
        topic=_cmd(name),
        correlation_id=uuid4(),
        envelope=envelope,
    )


class TestBootWiresOnlyTheBootSet:
    """(a) AC1: twenty admitted contracts, no handler imported, no topic created."""

    @pytest.mark.asyncio
    async def test_boot_wires_only_the_boot_set(
        self, fixture_handlers: list[str]
    ) -> None:
        log = CallLog()
        boot, dormant = _booted()
        engine = MessageDispatchEngine()
        bus = RecordingBus(log)
        report = await wire_from_manifest(
            boot,
            engine,
            event_bus=bus,
            environment="local",
            subscribe_immediately=False,
        )
        engine.freeze()
        await subscribe_wired_contract_topics(
            manifest=boot,
            report=report,
            dispatch_engine=engine,
            event_bus=bus,
            environment="local",
            provisioner=RecordingProvisioner(log),
            readiness_config=ModelTopicReadinessConfig(),
        )

        wired = [r.contract_name for r in report.results]
        assert report.total_wired == 2
        assert sorted(wired) == sorted([INGRESS, BOOT_PRIORITY])
        assert len(dormant) == FIXTURE_COUNT - 1
        assert sorted(fixture_handlers) == sorted([INGRESS, BOOT_PRIORITY])
        dormant_topics = {
            t for c in dormant for t in (_cmd(c.name), _cmd_b(c.name), _evt(c.name))
        }
        assert not dormant_topics.intersection(t for _, t in log)
        assert boot_universe_warm_enabled() is False

    @pytest.mark.asyncio
    async def test_failing_control_a_lane_without_lazy_start_wires_all(
        self, fixture_handlers: list[str], lane_overlay: Path
    ) -> None:
        """Control: the same boot with no lazy_start block imports all twenty-one."""
        path = lane_overlay / "runtime_contract_hotload" / "overlay.yaml"
        path.write_text(path.read_text().split("    lazy_start:")[0], encoding="utf-8")
        manifest = boot_manifest(_manifest())
        report = await wire_from_manifest(
            manifest,
            MessageDispatchEngine(),
            event_bus=RecordingBus(CallLog()),
            environment="local",
            subscribe_immediately=False,
        )
        assert report.total_wired == FIXTURE_COUNT + 1
        assert len(fixture_handlers) == FIXTURE_COUNT + 1
        assert boot_universe_warm_enabled() is True


class TestFirstRequestStartsTheNode:
    """(b) AC2: provision, confirm, attach, deliver; within budget; built once."""

    @pytest.mark.asyncio
    async def test_first_request_provisions_confirms_attaches_then_delivers(
        self, fixture_handlers: list[str]
    ) -> None:
        log = CallLog()
        door, transaction = _front_door(log, RecordingBus(log))
        assert TARGET not in fixture_handlers

        started = time.monotonic()
        first = await door.handle(_request())
        elapsed = time.monotonic() - started

        assert first.outcome is EnumLazyStartOutcome.DELIVERED, first.detail
        assert elapsed < 30.0
        kinds = [kind for kind, _ in log]
        last_provision = max(i for i, k in enumerate(kinds) if k == "provision")
        first_confirm = kinds.index("confirm")
        last_confirm = max(i for i, k in enumerate(kinds) if k == "confirm")
        first_attach = kinds.index("attach")
        last_attach = max(i for i, k in enumerate(kinds) if k == "attach")
        assert last_provision < first_confirm
        assert last_confirm < first_attach
        assert last_attach < kinds.index("deliver")
        assert {t for k, t in log if k == "attach"} == {_cmd(TARGET), _cmd_b(TARGET)}
        assert TARGET in fixture_handlers
        assert _sentinel().HANDLED == [TARGET]

        second = await door.handle(_request())
        assert second.outcome is EnumLazyStartOutcome.DELIVERED
        assert transaction.builds == 1
        assert _sentinel().HANDLED == [TARGET, TARGET]

    @pytest.mark.asyncio
    async def test_an_unknown_name_builds_nothing(
        self, fixture_handlers: list[str]
    ) -> None:
        log = CallLog()
        door, transaction = _front_door(log, RecordingBus(log))
        result = await door.handle(_request("node_never_announced"))
        assert result.outcome is EnumLazyStartOutcome.CONTRACT_NOT_ANNOUNCED
        assert transaction.builds == 0
        assert log == []


class TestConcurrentFirstRequests:
    """(c) AC3: ten concurrent first requests build once and all ten are answered."""

    @pytest.mark.asyncio
    async def test_ten_concurrent_first_requests_build_once(
        self, fixture_handlers: list[str]
    ) -> None:
        log = CallLog()
        bus = RecordingBus(log)
        bus.attach_delay = 0.05
        door, transaction = _front_door(log, bus)

        results = await asyncio.gather(*(door.handle(_request()) for _ in range(10)))

        assert transaction.builds == 1
        assert [r.outcome for r in results] == [EnumLazyStartOutcome.DELIVERED] * 10
        assert len(_sentinel().HANDLED) == 10
        assert fixture_handlers.count(TARGET) == 1


class TestFailedAttachLeavesNothing:
    """(d) AC4: the second of two attaches fails; nothing is left; a retry starts it."""

    @pytest.mark.asyncio
    async def test_second_attach_failure_leaves_no_subscription_or_dispatcher(
        self, fixture_handlers: list[str]
    ) -> None:
        log = CallLog()
        bus = RecordingBus(log, fail_attach={_cmd_b(TARGET)})
        bus.attach_delay = 0.01
        door, transaction = _front_door(log, bus)

        failed = await door.handle(_request())

        assert failed.outcome is EnumLazyStartOutcome.START_FAILED
        assert bus.live == set()
        assert ("detach", _cmd(TARGET)) in log
        assert transaction.node_engine(TARGET) is None
        assert TARGET not in door.serving
        assert _sentinel().HANDLED == []

        retried = await door.handle(_request())

        assert retried.outcome is EnumLazyStartOutcome.DELIVERED, retried.detail
        assert bus.live == {_cmd(TARGET), _cmd_b(TARGET)}
        engine = transaction.node_engine(TARGET)
        assert engine is not None and engine.dispatcher_count == 1
        assert transaction.builds == 2


class TestRestart:
    """AC11: a restart starts nothing until a request arrives; no resend is needed."""

    @pytest.mark.asyncio
    async def test_a_restart_starts_nothing_until_a_request(
        self, fixture_handlers: list[str]
    ) -> None:
        log = CallLog()
        door, _ = _front_door(log, RecordingBus(log))
        assert (await door.handle(_request())).outcome is (
            EnumLazyStartOutcome.DELIVERED
        )

        # Restart: a new process boots the same lane; nothing of the target is loaded.
        for mod in [m for m in sys.modules if m.startswith("lazyfix.")]:
            del sys.modules[mod]
        fixture_handlers.clear()
        _sentinel().HANDLED.clear()
        log.clear()
        restarted, transaction = _front_door(log, RecordingBus(log))
        assert fixture_handlers == []
        assert log == []
        assert restarted.serving == frozenset()

        answer = await restarted.handle(_request())

        assert answer.outcome is EnumLazyStartOutcome.DELIVERED
        assert transaction.builds == 1
        assert _sentinel().HANDLED == [TARGET]


class TestStartTimeout:
    """AC12: a request that outlasts the budget is START_TIMEOUT and is not lost."""

    @pytest.mark.asyncio
    async def test_a_request_over_budget_is_held_and_delivered(
        self, fixture_handlers: list[str]
    ) -> None:
        log = CallLog()
        bus = RecordingBus(log)
        bus.attach_delay = 0.3
        door, _ = _front_door(log, bus, budget=0.05)

        answer = await door.handle(_request())

        assert answer.outcome is EnumLazyStartOutcome.START_TIMEOUT
        assert answer.held is True
        held = await door.drain_held()
        assert [r.outcome for r in held] == [EnumLazyStartOutcome.DELIVERED]
        assert _sentinel().HANDLED == [TARGET]


def _closure_contracts(tmp_path: Path) -> dict[str, ModelDiscoveredContract]:
    """Write the workflow fixtures as contract.yaml files; read them via discovery."""
    paths: list[Path] = []
    for name, (subscribe, publish) in CLOSURE_SPEC.items():
        path = tmp_path / "workflow_contracts" / name / "contract.yaml"
        path.parent.mkdir(parents=True)
        path.write_text(
            yaml.safe_dump(
                {
                    "name": name,
                    "node_type": "EFFECT_GENERIC",
                    "contract_version": {"major": 1, "minor": 0, "patch": 0},
                    "event_bus": {
                        "subscribe_topics": list(subscribe),
                        "publish_topics": list(publish),
                    },
                    # What the handler returns goes to the contract's last publish
                    # topic: the next hop of the chain.
                    "published_events": (
                        [{"event_type": "FixtureRequest", "topic": publish[-1]}]
                        if publish
                        else []
                    ),
                    "handler_routing": {
                        "routing_strategy": "payload_type_match",
                        "handlers": [
                            {
                                "handler": {
                                    "name": "FixtureHandler",
                                    "module": f"lazyfix.{name}",
                                },
                                "event_model": {
                                    "name": "FixtureRequest",
                                    "module": f"lazyfix.{name}",
                                },
                            }
                        ],
                    },
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)
    manifest = discover_contracts_from_paths(paths)
    assert manifest.errors == ()
    return {c.name: c for c in manifest.contracts}


def _workflow_door(
    tmp_path: Path, log: CallLog, bus: RecordingBus, retries: int = 0
) -> tuple[LazyStartFrontDoor, CountingTransaction]:
    return _door_over(_closure_contracts(tmp_path), log, bus, retries=retries)


def _topics_of(names: set[str]) -> set[str]:
    return {t for n in names for t in CLOSURE_SPEC[n][0]}


async def _boot_ingress(tmp_path: Path, log: CallLog, bus: RecordingBus) -> None:
    """Boot as the kernel does: split the manifest, then wire the ingress node."""
    contracts = _closure_contracts(tmp_path)
    manifest = ModelAutoWiringManifest(
        contracts=(_contract(INGRESS), _contract(BOOT_PRIORITY), *contracts.values())
    )
    lazy.boot_manifest(manifest, RecordingProvisioner(log))
    container = ModelONEXContainer()
    assert container.service_registry is not None
    key: type[object] = ProtocolEventBusSubscriber
    await container.service_registry.register_instance(key, bus)
    HandlerLazyStartIngress(container)


def _hop_envelope() -> ModelEventEnvelope[object]:
    return ModelEventEnvelope[object](
        payload={"ask": "hop"}, correlation_id=uuid4(), event_type="FixtureRequest"
    )


class TestClosureWake:
    """A first request wakes the workflow's whole closure; a shared one wakes once."""

    @pytest.mark.asyncio
    async def test_closure_wake(
        self, fixture_handlers: list[str], tmp_path: Path
    ) -> None:
        """AC1: the orchestrator and the three contracts it publishes to all wake."""
        log = CallLog()
        door, transaction = _workflow_door(tmp_path, log, RecordingBus(log))

        # A contract already served is not woken again by the closure wake.
        assert (await door.handle(_request(A_FX))).outcome is (
            EnumLazyStartOutcome.DELIVERED
        )
        answer = await door.handle(_request(A_ORCH))

        assert answer.outcome is EnumLazyStartOutcome.DELIVERED, answer.detail
        assert set(transaction.built) == WORKFLOW_A
        assert set(transaction.built.values()) == {1}
        assert set(fixture_handlers) == WORKFLOW_A
        assert {t for k, t in log if k == "attach"} == _topics_of(WORKFLOW_A)
        assert door.serving == WORKFLOW_A
        assert _sentinel().HANDLED == [A_FX, A_ORCH]

    @pytest.mark.asyncio
    async def test_a_transitive_hop_is_in_the_closure(
        self, fixture_handlers: list[str], tmp_path: Path
    ) -> None:
        """Workflow B reaches a contract two hops away through B_GX."""
        log = CallLog()
        door, transaction = _workflow_door(tmp_path, log, RecordingBus(log))

        assert (await door.handle(_request(B_ORCH))).outcome is (
            EnumLazyStartOutcome.DELIVERED
        )

        assert set(transaction.built) == WORKFLOW_B
        assert OUTSIDE not in fixture_handlers

    @pytest.mark.asyncio
    async def test_shared_contract_skipped(
        self, fixture_handlers: list[str], tmp_path: Path
    ) -> None:
        """AC2: the second workflow neither rewires nor resubscribes a shared one."""
        log = CallLog()
        door, transaction = _workflow_door(tmp_path, log, RecordingBus(log))

        assert (await door.handle(_request(A_ORCH))).outcome is (
            EnumLazyStartOutcome.DELIVERED
        )
        assert (await door.handle(_request(B_ORCH))).outcome is (
            EnumLazyStartOutcome.DELIVERED
        )

        shared = {SHARED_P, SHARED_Q}
        assert {transaction.built[n] for n in shared} == {1}
        assert transaction.built[B_ORCH] == 1
        attached = Counter(t for k, t in log if k == "attach")
        assert {attached[t] for t in _topics_of(shared)} == {1}
        assert set(transaction.built) == WORKFLOW_A | WORKFLOW_B
        assert not [t for k, t in log if k == "detach"]
        assert fixture_handlers.count(SHARED_P) == 1

    @pytest.mark.asyncio
    async def test_concurrent_single_flight(
        self, fixture_handlers: list[str], tmp_path: Path
    ) -> None:
        """AC3: two first requests together wake each shared contract exactly once."""
        log = CallLog()
        bus = RecordingBus(log)
        bus.attach_delay = 0.05
        door, transaction = _workflow_door(tmp_path, log, bus)

        first, second = await asyncio.wait_for(
            asyncio.gather(
                door.handle(_request(A_ORCH)), door.handle(_request(B_ORCH))
            ),
            timeout=10.0,
        )

        assert first.outcome is EnumLazyStartOutcome.DELIVERED, first.detail
        assert second.outcome is EnumLazyStartOutcome.DELIVERED, second.detail
        assert set(transaction.built) == WORKFLOW_A | WORKFLOW_B
        assert set(transaction.built.values()) == {1}
        attached = Counter(t for k, t in log if k == "attach")
        assert set(attached.values()) == {1}
        assert sorted(_sentinel().HANDLED) == sorted([A_ORCH, B_ORCH])
        assert bus.live == _topics_of(WORKFLOW_A | WORKFLOW_B)
        # One stable order: both workflows reach the shared pair in the same sequence.
        order = [t for k, t in log if k == "attach"]
        assert order.index(_cmd(SHARED_P)) < order.index(_cmd(SHARED_Q))

    @pytest.mark.asyncio
    async def test_concurrent_requests_for_one_workflow_share_one_wake(
        self, fixture_handlers: list[str], tmp_path: Path
    ) -> None:
        log = CallLog()
        bus = RecordingBus(log)
        bus.attach_delay = 0.02
        door, transaction = _workflow_door(tmp_path, log, bus)

        results = await asyncio.wait_for(
            asyncio.gather(*(door.handle(_request(B_ORCH)) for _ in range(5))),
            timeout=10.0,
        )

        assert [r.outcome for r in results] == [EnumLazyStartOutcome.DELIVERED] * 5
        assert set(transaction.built.values()) == {1}
        assert _sentinel().HANDLED == [B_ORCH] * 5

    @pytest.mark.asyncio
    async def test_closure_walks_through_a_contract_wired_at_boot(
        self, fixture_handlers: list[str], tmp_path: Path
    ) -> None:
        """A wired contract between two hops carries the closure on; it is not woken."""
        log = CallLog()
        contracts = _closure_contracts(tmp_path)
        router = _contract("node_wf_wired_router").model_copy(
            update={
                "event_bus": ModelEventBusWiring(
                    subscribe_topics=(_cmd(A_FX),), publish_topics=(_cmd(OUTSIDE),)
                )
            }
        )
        door, transaction = _door_over(
            contracts, log, RecordingBus(log), wired={router.name: router}
        )

        assert (await door.handle(_request(A_ORCH))).outcome is (
            EnumLazyStartOutcome.DELIVERED
        )

        assert set(transaction.built) == WORKFLOW_A | {OUTSIDE}
        assert router.name not in transaction.built

    @pytest.mark.asyncio
    async def test_a_closure_request_over_budget_is_held_and_delivered(
        self, fixture_handlers: list[str], tmp_path: Path
    ) -> None:
        """The budget stays: START_TIMEOUT holds the request, the wake finishes it."""
        log = CallLog()
        bus = RecordingBus(log)
        bus.attach_delay = 0.2
        door = _door_over(_closure_contracts(tmp_path), log, bus, budget=0.05)[0]

        answer = await door.handle(_request(B_ORCH))

        assert answer.outcome is EnumLazyStartOutcome.START_TIMEOUT
        assert answer.held is True
        held = await door.drain_held()
        assert [r.outcome for r in held] == [EnumLazyStartOutcome.DELIVERED]
        assert _sentinel().HANDLED == [B_ORCH]
        assert door.serving == WORKFLOW_B


class TestFailedWakeRollback:
    """A failed wake leaves nothing half-wired and never costs another workflow."""

    @pytest.mark.asyncio
    async def test_failed_wake_rollback_leaves_nothing(
        self, fixture_handlers: list[str], tmp_path: Path
    ) -> None:
        """AC4: the last member's attach fails; every member this wake started goes."""
        log = CallLog()
        bus = RecordingBus(log, fail_attach={_cmd(SHARED_Q)})
        door, transaction = _workflow_door(tmp_path, log, bus)

        failed = await door.handle(_request(A_ORCH))

        assert failed.outcome is EnumLazyStartOutcome.START_FAILED
        assert bus.live == set()
        assert door.serving == frozenset()
        assert {
            n for n in WORKFLOW_A if transaction.node_engine(n) is not None
        } == set()
        assert set(transaction.rolled_back) == {A_ORCH, SHARED_P, SHARED_Q}
        assert _sentinel().HANDLED == []
        # A topic the failed wake created stays: the runtime never deletes one.
        assert {t for k, t in log if k == "provision"} >= _topics_of(WORKFLOW_A)
        assert not [k for k, _ in log if k.startswith("delete")]

        retried = await door.handle(_request(A_ORCH))

        assert retried.outcome is EnumLazyStartOutcome.DELIVERED, retried.detail
        assert bus.live == _topics_of(WORKFLOW_A)
        assert door.serving == WORKFLOW_A
        assert _sentinel().HANDLED == [A_ORCH]

    @pytest.mark.asyncio
    async def test_failed_wake_rollback_keeps_what_another_workflow_serves(
        self, fixture_handlers: list[str], tmp_path: Path
    ) -> None:
        """AC4: a contract another workflow already serves keeps serving."""
        log = CallLog()
        bus = RecordingBus(log, fail_attach={_cmd(A_FX)})
        door, transaction = _workflow_door(tmp_path, log, bus)
        assert (await door.handle(_request(B_ORCH))).outcome is (
            EnumLazyStartOutcome.DELIVERED
        )
        served = _topics_of(WORKFLOW_B)
        assert bus.live == served

        failed = await door.handle(_request(A_ORCH))

        assert failed.outcome is EnumLazyStartOutcome.START_FAILED
        assert transaction.rolled_back.count(A_ORCH) == 1
        assert SHARED_P not in transaction.rolled_back
        assert SHARED_Q not in transaction.rolled_back
        assert bus.live == served
        assert door.serving == WORKFLOW_B
        assert transaction.built[SHARED_P] == 1
        assert (await door.handle(_request(B_ORCH))).outcome is (
            EnumLazyStartOutcome.DELIVERED
        )

        retried = await door.handle(_request(A_ORCH))

        assert retried.outcome is EnumLazyStartOutcome.DELIVERED, retried.detail
        assert bus.live == served | _topics_of({A_ORCH, A_FX})
        assert transaction.built[SHARED_P] == 1
        assert transaction.built[A_ORCH] == 2

    @pytest.mark.asyncio
    async def test_failed_wake_rollback_spares_a_member_a_peer_wake_joined(
        self, fixture_handlers: list[str], tmp_path: Path
    ) -> None:
        """A member the failing wake started, but a live peer wake joined, stays up."""
        log = CallLog()
        bus = RecordingBus(log, fail_attach={_cmd(A_FX)})
        bus.attach_delay = 0.02
        bus.topic_delay[_cmd(B_DEEP)] = 0.3  # B is still waking when A's wake fails
        door, transaction = _workflow_door(tmp_path, log, bus)

        failed, peer = await asyncio.wait_for(
            asyncio.gather(
                door.handle(_request(A_ORCH)), door.handle(_request(B_ORCH))
            ),
            timeout=10.0,
        )

        assert failed.outcome is EnumLazyStartOutcome.START_FAILED
        assert peer.outcome is EnumLazyStartOutcome.DELIVERED, peer.detail
        assert sorted(transaction.rolled_back) == sorted([A_FX, A_ORCH])
        assert bus.live == _topics_of(WORKFLOW_B)
        assert door.serving == WORKFLOW_B


class TestInnerHopWake:
    """A bus message to a dormant consumer wakes it first and is then consumed."""

    @pytest.mark.asyncio
    async def test_inner_hop_wake(
        self, fixture_handlers: list[str], tmp_path: Path
    ) -> None:
        """AC5: a publish to a dormant consumer's topic wakes it, and it handles it.

        The runtime wires the ingress at boot; that installs the wake on the bus's
        publish path, so the publish below is the same call an orchestrator's
        result applier makes for an inner hop.
        """
        log = CallLog()
        bus = RecordingBus(log)
        await _boot_ingress(tmp_path, log, bus)

        await bus.publish_envelope(_hop_envelope(), topic=_cmd(B_ORCH), key=None)

        assert _sentinel().HANDLED == [B_ORCH]
        kinds = [k for k, t in log if t == _cmd(B_ORCH)]
        assert kinds == ["provision", "confirm", "attach", "publish"]
        assert bus.published.count(_cmd(B_ORCH)) == 1
        # The consumer's own closure woke with it; nothing else did.
        assert set(fixture_handlers) == WORKFLOW_B
        assert bus.live == _topics_of(WORKFLOW_B)

        await bus.publish_envelope(_hop_envelope(), topic=_cmd(B_ORCH), key=None)
        assert _sentinel().HANDLED == [B_ORCH, B_ORCH]
        assert Counter(t for k, t in log if k == "attach")[_cmd(B_ORCH)] == 1

    @pytest.mark.asyncio
    async def test_inner_hop_wake_holds_a_message_published_while_the_wake_runs(
        self, fixture_handlers: list[str], tmp_path: Path
    ) -> None:
        """Messages sent while the wake is in flight wait for it; each is consumed once."""
        log = CallLog()
        bus = RecordingBus(log)
        bus.attach_delay = 0.05
        await _boot_ingress(tmp_path, log, bus)
        envelope = _hop_envelope()

        await asyncio.wait_for(
            asyncio.gather(
                *(bus.publish_envelope(envelope, topic=_cmd(B_ORCH)) for _ in range(3))
            ),
            timeout=10.0,
        )

        assert _sentinel().HANDLED == [B_ORCH] * 3
        assert bus.published == [_cmd(B_ORCH)] * 3
        assert Counter(t for k, t in log if k == "attach")[_cmd(B_ORCH)] == 1
        first_publish = log.index(("publish", _cmd(B_ORCH)))
        assert log.index(("attach", _cmd(B_ORCH))) < first_publish

    @pytest.mark.asyncio
    async def test_inner_hop_chain_is_answered_on_the_first_request(
        self, fixture_handlers: list[str], tmp_path: Path
    ) -> None:
        """Each hop's output reaches the next node's topic on the first request."""
        log = CallLog()
        bus = RecordingBus(log)
        door, _ = _workflow_door(tmp_path, log, bus)
        _sentinel().EMIT.update({B_ORCH, B_GX})

        answer = await asyncio.wait_for(door.handle(_request(B_ORCH)), timeout=10.0)

        assert answer.outcome is EnumLazyStartOutcome.DELIVERED, answer.detail
        assert _sentinel().HANDLED == [B_ORCH, B_GX, B_DEEP]
        assert bus.published == [_cmd(B_GX), _cmd(B_DEEP)]

    @pytest.mark.asyncio
    async def test_inner_hop_to_a_woken_or_unrelated_topic_wakes_nothing(
        self, fixture_handlers: list[str], tmp_path: Path
    ) -> None:
        log = CallLog()
        bus = RecordingBus(log)
        await _boot_ingress(tmp_path, log, bus)

        nobody = _evt("node_nobody")

        await bus.publish_envelope(_hop_envelope(), topic=nobody)

        assert fixture_handlers == []
        assert log == [("publish", nobody)]

    @pytest.mark.asyncio
    async def test_inner_hop_to_a_consumer_that_fails_to_wake_is_not_published(
        self, fixture_handlers: list[str], tmp_path: Path
    ) -> None:
        """A failed wake fails the publish loudly; nothing is published into the void."""
        log = CallLog()
        bus = RecordingBus(log, fail_attach={_cmd(B_ORCH)})
        await _boot_ingress(tmp_path, log, bus)

        with pytest.raises(InfraUnavailableError, match=B_ORCH):
            await bus.publish_envelope(_hop_envelope(), topic=_cmd(B_ORCH))

        assert bus.published == []
        assert bus.live == set()
        await bus.publish_envelope(_hop_envelope(), topic=_cmd(B_ORCH))
        assert _sentinel().HANDLED == [B_ORCH]


@pytest.mark.asyncio
async def test_publish_gate_runs_first_once_and_takes_the_topic_by_keyword() -> None:
    from omnibase_infra.handlers.lazy_start_publish_gate import install_publish_gate

    seen: list[str] = []

    class Bus:
        async def publish(self, topic: str, key: bytes | None, value: bytes) -> str:
            seen.append(f"publish:{topic}")
            return "receipt"

    async def gate(topic: str) -> None:
        seen.append(f"gate:{topic}")

    bus = Bus()
    assert install_publish_gate(bus, gate) is True
    assert install_publish_gate(bus, gate) is True  # a second install keeps one gate

    assert await bus.publish("t1", None, b"") == "receipt"
    assert await bus.publish(topic="t2", key=None, value=b"") == "receipt"

    assert seen == ["gate:t1", "publish:t1", "gate:t2", "publish:t2"]
    assert install_publish_gate(object(), gate) is False


def test_wiring_outcome_enum_is_the_runtime_one() -> None:
    """Guards the import the runtime transaction compares against."""
    assert EnumWiringOutcome.WIRED.value
