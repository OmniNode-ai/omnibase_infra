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
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import cast
from uuid import UUID, uuid4

import pytest

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums.enum_lazy_start_outcome import EnumLazyStartOutcome
from omnibase_infra.event_bus.enum_topic_readiness_status import (
    EnumTopicReadinessStatus,
)
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.model_topic_readiness_config import (
    ModelTopicReadinessConfig,
)
from omnibase_infra.event_bus.model_topic_set_readiness import ModelTopicSetReadiness
from omnibase_infra.handlers.lazy_start_boot_split import (
    RegistryLazyStartDormant,
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
from omnibase_infra.protocols.protocol_lazy_start_transaction import (
    ProtocolLazyStartBus,
)
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


@pytest.fixture
def fixture_handlers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Twenty handler modules; importing one appends its name to the sentinel."""
    pkg = tmp_path / "lazyfix"
    pkg.mkdir()
    (tmp_path / f"{SENTINEL_MODULE}.py").write_text(
        "IMPORTED: list[str] = []\nHANDLED: list[str] = []\n", encoding="utf-8"
    )
    for name in [INGRESS, *(_name(i) for i in range(FIXTURE_COUNT))]:
        (pkg / f"{name}.py").write_text(
            "from pydantic import BaseModel\n"
            f"import {SENTINEL_MODULE} as _s\n"
            f"_s.IMPORTED.append({name!r})\n\n\n"
            "class FixtureRequest(BaseModel):\n"
            '    ask: str = ""\n\n\n'
            "class FixtureHandler:\n"
            "    async def handle(self, envelope: object) -> None:\n"
            f"        _s.HANDLED.append({name!r})\n",
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
    RegistryLazyStartDormant.clear()
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
        self.fail_attach = set(fail_attach or ())
        self.live: set[str] = set()
        self.attach_delay = 0.0

    async def publish_envelope(
        self, envelope: object, topic: str, *, key: bytes | None = None
    ) -> None:
        return None

    async def publish(self, topic: str, key: bytes | None, value: bytes) -> None:
        return None

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
        if self.attach_delay:
            await asyncio.sleep(self.attach_delay)
        if topic in self.fail_attach:
            self.fail_attach.discard(topic)
            self._log.append(("attach_failed", topic))
            raise ConnectionError(f"join refused for {topic}")
        self._log.append(("attach", topic))
        self.live.add(topic)

        async def _stop() -> None:
            self._log.append(("detach", topic))
            self.live.discard(topic)

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

    async def build(self, contract_name: str) -> None:
        self.builds += 1
        await super().build(contract_name)

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
    transaction = CountingTransaction(log, {c.name: c for c in dormant}, bus)
    policy = ModelLazyStartPolicy(
        ingress_contract=INGRESS, budget_seconds=budget, max_retries=retries
    )
    door = LazyStartFrontDoor(transaction, [c.name for c in dormant], policy)
    return door, transaction


def _booted() -> tuple[ModelAutoWiringManifest, tuple[ModelDiscoveredContract, ...]]:
    boot = boot_manifest(_manifest())
    return boot, tuple(RegistryLazyStartDormant.contracts().values())


def _request(name: str = TARGET) -> ModelLazyStartRequest:
    envelope = ModelEventEnvelope[object](
        payload={"ask": "first"}, correlation_id=uuid4(), event_type="FixtureRequest"
    )
    return ModelLazyStartRequest(
        contract_name=name, topic=_cmd(name), correlation_id=uuid4(), envelope=envelope
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


def test_wiring_outcome_enum_is_the_runtime_one() -> None:
    """Guards the import the runtime transaction compares against."""
    assert EnumWiringOutcome.WIRED.value
